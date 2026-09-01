## 9. From RoCE to Software RDMA: A KV-Cache-over-Wi-Fi Transport Prototype

### 9.1 Motivation

Section 7.3 identified Linux/memfd, NPU coherency, and demand-paged
eviction as the primary directions for a full evaluation, but left
open a fourth question raised directly by MemLLM's own design
philosophy: what happens to the zero-copy argument once the
application process and the inference server are no longer on the
same machine? Multi-device and edge-offload deployments — a phone
streaming context to a nearby workstation, or a wearable offloading
generation to a paired device — replace the shared virtual address
space with a wireless hop, and the natural analogue to reach for is
RDMA over Converged Ethernet (RoCE), the same family of techniques
memif itself descends from (Section 2.1).

RoCE's reliability model assumes a near-lossless fabric maintained by
link-layer flow control (PFC/ECN). Wi-Fi provides neither: 802.11's
CSMA/CA contention model means loss is a normal operating condition,
not an exception, and prior work applying RDMA directly to wireless
links has documented exactly this failure mode — a single dropped
packet can produce a livelock where the link stays fully utilized but
the receiver never completes a message [12]. This section reports a
small, from-scratch prototype built to test the alternative: an
eRPC-style [13] software transport that assumes loss as a first-class
condition rather than an exception, evaluated first under controlled
synthetic loss and then on a real, physical Wi-Fi link between two
independent devices.

### 9.2 Transport Design

The prototype (`memllm_udp_rpc.py`) reuses MemLLM's own descriptor
philosophy — a fixed-size header plus a payload pool (Section 3.2) —
but serializes it onto plain UDP datagrams instead of a shared ring,
since no directly-mapped address space exists across a wireless hop.
Three design choices, all inherited from eRPC rather than from RoCE,
were made deliberately:

1. **Reliability in userspace, not the fabric.** Each logical message
   is fragmented into ≤1200-byte datagrams and tracked with a
   per-fragment selective-repeat ACK (a bitmap, not a cumulative
   sequence number), so a single lost fragment costs one
   retransmission rather than a full-message resend.
2. **A single-threaded, run-to-completion polling engine.** Both
   sender and receiver busy-poll a non-blocking socket rather than
   blocking on `recv()` — the same latency-vs-CPU tradeoff MemLLM's
   Windows prototype already makes with `threading.Event` polling in
   place of Linux `eventfd` (Section 5.1).
3. **Exponential backoff with jitter on retransmission** (added after
   the finding in 9.3.3 below) rather than a fixed retry interval —
   the one point where the design diverges from a naive first pass
   and converges on what TCP and real eRPC already do.

### 9.3 What Building It Actually Found

The prototype was validated in three stages: a controlled mock
backend isolating transport from compute, synthetic loss injection on
loopback, and finally two independent physical devices on a real home
Wi-Fi network. Each stage surfaced a genuine defect or design gap that
loopback-only testing at 0% loss would not have exposed.

**9.3.1 Duplicate delivery under ACK loss.** A lost ACK causes the
sender to retransmit an already-fully-received message. Without
dedup, the receiver's reassembly table re-queued the duplicate,
producing two failure modes: a stale reply silently misdelivered to
the *next*, unrelated caller, and — once the duplicate was queued a
second time for a message already consumed — a `KeyError` that
silently killed the server thread. The fix tracks a bounded set of
already-delivered message IDs and re-acknowledges duplicates
idempotently without re-queueing them.

**9.3.2 Session-scoped identity.** A long-lived server plus a freshly
restarted client reusing small integer message IDs produced genuine
collisions with the previous session's dedup cache: the server
silently ACKed the request datagram (recognizing the ID as
"already delivered") without ever invoking the backend or sending a
real response, leaving the client blocked for its full receive
timeout. Real RPC systems scope sequence numbers per session for
exactly this reason; the fix was to key delivery state by identity,
not by a raw counter alone.

**9.3.3 Fixed retry intervals cause retry storms.** Under synthetic
loopback loss, a fixed 20 ms retransmit timer produced *worse* than
predicted failure rates at high loss (≈80% message failure at a
nominal 30% loss, versus a first-order independent-loss estimate near
zero) — both endpoints' retransmissions piled up faster than the
busy-poll loop could drain them. Adding exponential backoff with
jitter, the same mechanism TCP and real eRPC already use, restored
100% delivery at the same 30% loss with negligible latency cost. This
is the most paper-relevant finding: it is direct, measured evidence
for *why* RoCE's lossless-fabric assumption is the wrong starting
point for Wi-Fi, and why a viable software transport needs real
congestion control, not merely "add retries."

**9.3.4 Fault isolation is not free.** Even after the above fixes, one
response that exhausted its retry budget raised an uncaught exception
inside the server's request loop and took the *entire* endpoint down
— every subsequent turn failed, not just the one message. HTTP/TCP
gives per-connection fault isolation for free at the kernel layer; a
hand-rolled UDP transport does not, and has to be built to survive
one failed message without crashing the request loop.

**9.3.5 A fast synthetic backend hides a real one's failure mode.**
The transport was validated end-to-end against a synthetic backend
before being pointed at a live vLLM server (Section 9.4b). Real 7B
decode latency (6–14 s/turn) exceeded the client's 30 s receive
timeout margin often enough that the client would occasionally give
up and issue the next turn's request while the server was still
blocked inside the previous turn's `generate()` call — which does not
poll its socket while blocked, so the new request went unacknowledged
for the server's entire remaining compute time. The result looked
identical to a reliability failure ("gave up after N attempts") but
had nothing to do with loss; it was a client/server pipeline desync
caused by a timeout tuned for a fast backend and left unchanged for a
slow one. Widening the client's receive timeout to comfortably exceed
the real backend's observed worst case resolved it. The general
lesson: a transport's timeout and retry parameters are tuned against
whatever backend validated them, and re-tuning is required, not
optional, when the backend's latency profile changes character.

### 9.4 Real Wi-Fi Results

After the fixes above, the transport was deployed across two physical
devices on an ordinary home Wi-Fi network — a Linux workstation
(server) and a Windows laptop (client), reachable only after
reconnecting a dropped Wi-Fi adapter and opening LAN-scoped firewall
rules, itself a small reminder that "just point it at the other
device" carries real deployment friction HTTP-over-Ethernet mostly
lets users forget about. Baseline link RTT measured 18 ms (`ping`,
n=2), consistent with ordinary 802.11 latency.

*(a) Controlled backend (isolating transport from compute).* A
matched mock backend held generation time statistically identical
across both transports so any latency difference is attributable to
the transport alone:

| Condition | N | Mean | Median | stdev | p95 | Retransmits |
|---|---|---|---|---|---|---|
| UDP-RPC, loopback | 10 | 1485 ms | 1507 ms | 327 ms | 2006 ms | 0 |
| UDP-RPC, real Wi-Fi | 10 | 1509 ms | 1538 ms | 333 ms | 2020 ms | 3 |
| HTTP/JSON, real Wi-Fi | 10 | 1590 ms | 1604 ms | 308 ms | 2096 ms | — (TCP-internal) |

The real-Wi-Fi transport overhead over loopback was ≈25 ms mean —
consistent with the measured 18 ms link RTT plus one retransmission
round, not a multiplicative blowup. UDP-RPC ran ≈80 ms lower than the
HTTP/JSON baseline in both mean and p95 on the same link, the same
direction as MemLLM's own local-shared-memory-vs-HTTP result
(Section 6.3), now reproduced across an actual wireless hop instead
of a shared address space.

*(b) Real inference backend.* The same two transports were then
pointed at a live, unmodified vLLM server already running on the
workstation (Qwen2.5-7B-Instruct, OpenAI-compatible API,
`vllm_backend.py`), replacing the synthetic sleep with genuine GPU
decode while holding the transport and benchmark code fixed:

| Condition | N | Mean | Median | stdev | p95 | Retransmits |
|---|---|---|---|---|---|---|
| UDP-RPC, real Wi-Fi + real vLLM | 10 | 10613 ms | 12071 ms | 3570 ms | 13720 ms | 2 |
| HTTP/JSON, real Wi-Fi + real vLLM | 10 | 10769 ms | 12557 ms | 3809 ms | 13712 ms | — |

UDP-RPC still ran marginally lower on mean and p95 and absorbed two
genuine Wi-Fi packet losses transparently, with no failed turns. The
more interesting result is negative: on this hardware, a 7B-parameter
model's decode latency (6–14 s/turn, growing with context) is slow
enough that transport overhead is once again a rounding error against
compute — the same regime as the paper's own CPU-bound Phi-3 baseline
in Section 6, for a different underlying reason. This revised an
initial hypothesis going into this run (that faster GPU decode would
make transport choice *more* visible); the honest result is that
"faster than a laptop CPU" was not, on this particular accelerator,
fast enough to flip that balance. A backend closer to a batched,
high-throughput serving regime — or a scenario with many small,
low-latency turns rather than long generations — would be a more
direct test of that hypothesis, and is a natural next step rather
than a settled conclusion. (Getting a clean run against this real
backend also required diagnosing and fixing the timeout-desync issue
in Finding 9.3.5.)

### 9.5 Honest Limitations

This remains a small prototype, not a systems paper's worth of
evaluation: loss injection was tested on loopback only (synthetic, not
a real contended channel); the real-Wi-Fi run used a single link, a
single pair of devices, and n=10 turns; and RoCE itself was not
implemented or benchmarked directly — the comparison is to its
documented failure mode in prior work [12], not to a reproduction of
it. A rigorous follow-up would add `tc netem`-shaped loss/jitter on a
real link, multiple independent runs, and a direct RoCE-over-Wi-Fi
(or soft-RoCE) baseline rather than relying on the literature's
account of its behavior.

### References (additions)

[12] *WIP: When RDMA Meets Wireless*, WoWMoM 2022.
[13] Kalia, A., Kaminsky, M., Andersen, D. — *Datacenter RPCs can be
General and Fast*, NSDI 2019 (eRPC).
