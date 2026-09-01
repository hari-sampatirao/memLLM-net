## 9. Beyond a Single Windows Machine: Wi-Fi, Real Linux Shared Memory, and RoCE

### 9.1 Motivation

Section 7.3 named Linux/`memfd`, NPU coherency, and demand-paged
eviction as the primary directions for a full evaluation, but left a
fourth, more basic question implicit: MemLLM's central argument is
zero-copy shared memory, and that argument is scoped to a single
machine by construction. Two natural extensions probe the edges of
that scope directly: (1) what happens when the application and the
inference server are *not* on the same machine — the multi-device and
edge-offload case, where the shared address space is replaced by a
wireless hop — and (2) what does the Linux path the paper always
described but never built (`memfd_create` + `SCM_RIGHTS` + `eventfd`,
Section 3.1/3.3) actually measure, now that only the Windows named-
shared-memory prototype has real numbers (Table 2)? This section
reports both, plus a third result that fell out of asking the second
question honestly: where does RDMA/RoCE — the mechanism MemLLM's own
`memif` inspiration descends from (Section 2.1) — actually sit
relative to everything else measured here.

All three investigations ran on the same platform: a Linux
workstation (NVIDIA GB10) as server, a Windows laptop as client for
the Wi-Fi case, connected over an ordinary home Wi-Fi network, with a
live, unmodified vLLM server (Qwen2.5-7B-Instruct, OpenAI-compatible
API) available as a real inference backend alongside the paper's own
synthetic-latency mock.

### 9.2 The Cross-Device Case: An eRPC-Style Transport for Wi-Fi

#### 9.2.1 Why not RoCE directly

RoCE's reliability model assumes a near-lossless fabric maintained by
link-layer flow control (PFC/ECN). Wi-Fi is close to the opposite of
that environment, for reasons that are physical, not incidental.
802.11's CSMA/CA medium-access scheme is contention-based and only
avoids collisions probabilistically — it cannot detect them the way
wired Ethernet's now-obsolete CSMA/CD could, so a busy channel means
real, ongoing frame loss rather than an edge case. On top of that,
consumer Wi-Fi shares an already-crowded unlicensed band: neighboring
access points on overlapping channels (especially in apartments and
offices), Bluetooth devices, microwave ovens, and baby monitors on
2.4GHz, and simple distance-driven signal attenuation and multipath
fading all degrade the link independently of anything the two
endpoints are doing. Rate adaptation then compounds this into latency
variance as well as loss: a station backs off to a slower, more
robust modulation under interference, which shows up as jitter, not
just dropped frames. None of this is a corner case to be engineered
around — it is Wi-Fi's normal operating condition, which is precisely
why prior work applying RDMA directly to wireless links documents a
livelock failure mode: a single dropped packet can leave the link
fully utilized while the receiver never completes a message [12].
Section 9.4 below returns to this claim directly with a real Soft-RoCE
measurement; this subsection first reports the practical alternative
built to run without RDMA-capable hardware, and validated — Section
9.2.3, Findings 3 and 5 — against exactly this kind of loss and
latency variance, not merely an idealized lossless channel.

#### 9.2.2 Design

The prototype (`memllm_udp_rpc.py`) reuses MemLLM's own descriptor
philosophy — a fixed-size header plus a payload pool (Section 3.2) —
serialized onto plain UDP datagrams instead of a shared ring, since no
directly-mapped address space exists across a wireless hop. Three
choices, inherited from eRPC [13] rather than RoCE:

1. **Reliability in userspace, not the fabric** — per-fragment
   selective-repeat ACKs (a bitmap, not a cumulative sequence number),
   so one lost fragment costs one retransmission, not a full resend.
2. **A single-threaded, run-to-completion polling engine** — both
   sides busy-poll a non-blocking socket, the same latency-vs-CPU
   tradeoff the Windows prototype already makes with `threading.Event`
   polling in place of `eventfd` (Section 5.1).
3. **Exponential backoff with jitter on retransmission** (added after
   Finding 3 below) rather than a fixed retry interval.

#### 9.2.3 What Building It Actually Found

Five genuine defects surfaced across three validation stages
(synthetic backend, synthetic loss on loopback, then two independent
physical devices on a real home Wi-Fi network) that loopback-only
testing at 0% loss would not have exposed:

1. **Duplicate delivery under ACK loss.** A lost ACK causes the
   sender to retransmit an already-delivered message. Without dedup,
   this produced both a stale reply silently misdelivered to a later,
   unrelated caller, and a `KeyError` that silently killed the server
   thread once the duplicate was queued twice. Fixed by tracking
   delivered message IDs and re-acknowledging duplicates idempotently
   without re-queueing them.
2. **Session-scoped identity.** A long-lived server plus a freshly
   restarted client reusing small integer message IDs collided with
   the previous session's dedup cache — the server silently ACKed the
   request without ever generating a real response. Real RPC systems
   scope sequence numbers per session for exactly this reason.
3. **Fixed retry intervals cause retry storms.** Under synthetic
   loopback loss, a fixed 20ms retransmit timer produced far worse
   than predicted failure at high loss (≈80% message failure at a
   nominal 30% loss) — both endpoints' retransmissions outpaced the
   busy-poll drain rate. Exponential backoff with jitter restored
   100% delivery at the same 30% loss. This is the most directly
   relevant finding to the RoCE question: concrete evidence that a
   viable software transport needs real congestion control, not
   merely "add retries."
4. **Fault isolation is not free.** One response exhausting its retry
   budget raised an uncaught exception that took the entire server
   down, not just that message. HTTP/TCP gives per-connection fault
   isolation for free at the kernel layer; a hand-rolled UDP transport
   has to be built to survive one failed message.
5. **A fast synthetic backend hides a real one's failure mode.**
   Once pointed at the live vLLM server, real 7B decode latency
   (6–14s/turn) exceeded the client's 30s receive-timeout margin often
   enough that the client would give up and issue the next turn's
   request while the server was still blocked inside the previous
   turn's (non-polling) `generate()` call — a pipeline desync that
   looked identical to a reliability failure but had nothing to do
   with loss. Widening the timeout to exceed the backend's observed
   worst case resolved it. The lesson: transport timeouts tuned
   against a fast synthetic backend are not automatically correct for
   a slow real one.

#### 9.2.4 Real Wi-Fi Results

Baseline link RTT measured 18ms (`ping`, n=2), consistent with
ordinary 802.11 latency. Reaching this measurement also required
reconnecting a dropped Wi-Fi adapter and opening LAN-scoped firewall
rules on the server host — a reminder that "just point it at the
other device" carries real deployment friction that HTTP-over-Ethernet
mostly lets users forget about.

*(a) Controlled backend, isolating transport from compute:*

| Condition | N | Mean | Median | stdev | p95 | Retransmits |
|---|---|---|---|---|---|---|
| UDP-RPC, loopback | 10 | 1485ms | 1507ms | 327ms | 2006ms | 0 |
| UDP-RPC, real Wi-Fi | 10 | 1509ms | 1538ms | 333ms | 2020ms | 3 |
| HTTP/JSON, real Wi-Fi | 10 | 1590ms | 1604ms | 308ms | 2096ms | — |

Real-Wi-Fi overhead over loopback was ≈25ms mean — consistent with
the measured 18ms link RTT plus one retransmission round, not a
multiplicative blowup. UDP-RPC ran ≈80ms lower than HTTP/JSON on both
mean and p95, the same direction as MemLLM's local-shared-memory-vs-
HTTP result (Section 6.3), now reproduced across a wireless hop.

*(b) Real inference backend (Qwen2.5-7B-Instruct via vLLM):*

| Condition | N | Mean | Median | stdev | p95 | Retransmits |
|---|---|---|---|---|---|---|
| UDP-RPC, real Wi-Fi + real vLLM | 10 | 10613ms | 12071ms | 3570ms | 13720ms | 2 |
| HTTP/JSON, real Wi-Fi + real vLLM | 10 | 10769ms | 12557ms | 3809ms | 13712ms | — |

UDP-RPC still ran marginally lower and absorbed two genuine Wi-Fi
packet losses with no failed turns, but the gap shrinks to a rounding
error against 6–14s of real decode. This revised an initial hypothesis
(that faster GPU decode would make transport choice *more* visible);
the honest result is that this particular accelerator's decode speed
was not fast enough to flip that balance, the same regime as the
paper's own CPU-bound Phi-3 baseline for a different underlying
reason.

#### 9.2.5 Why the KV Cache Itself Should Usually Never Cross the Link

Every result in 9.2.4 moved short text — prompts and replies of a few
hundred bytes, always fitting in a single ≤1200-byte fragment. That was
not a simplification made for convenience; it is the correct design.
The client and server in this prototype already follow the same
principle the paper states for the local, shared-memory case in
Section 7.1 — "only the new tokens and response need to traverse the
ring" — over the network as well: the server-side `history` (standing
in for the KV cache) accumulates across turns, and the client only
ever sends the newest user message and receives the newest reply.
Nothing about a growing conversation increases what crosses the wire,
because the cache never leaves the machine that computed it.

Given that, the KV cache itself — potentially tens of gigabytes for a
long context (Section 2.2) — never needs to touch Wi-Fi's bandwidth
ceiling (typically 50–300Mbps of *usable* real-world throughput on
802.11ac/ax, well short of advertised link rates, for the reasons in
9.2.1) at all. The client's role is genuinely light: hold the
conversation as plain text for its own UI (kilobytes, trivial on any
device with no GPU whatsoever), and forward only what changed. For the
common, linear, append-only chat case, "what changed" is trivial to
identify — it is simply whatever the user just typed, which the client
already has for free. No transfer of cache contents, and no complex
diffing, is required.

This does mean the client needs to *track identity*, not *maintain
state*, once the conversation stops being perfectly linear:

- **Edits or branches.** If an earlier turn is edited, the client
  needs to tell the server which turns are now invalid (a sequence
  number or per-turn hash is enough) so the server can discard the
  stale suffix of its cache and recompute from that point — still
  bookkeeping, not cache transfer.
- **Multiple devices sharing one context.** If two devices contribute
  to the same conversation, each needs to know how much of the
  server's cached prefix is still valid for what it's about to send.
  This is exactly the problem RadixAttention's prefix hashing [4]
  solves server-side; the client's part of that exchange is a hash or
  ID, never the tensor data itself.

Real KV cache *migration* — the cache physically moving across a
network — is a genuine technique, but it belongs to a narrower and
different problem than "a thin client talks to a fixed remote GPU,"
which is the scenario this section actually tested:

1. **Prefill/decode disaggregation** (Mooncake, DistServe) moves cache
   between differently-optimized serving nodes *within a datacenter*,
   over RDMA/RoCE on wired infrastructure built for it — not a home
   Wi-Fi problem at all.
2. **Compute handoff mid-conversation** (e.g. phone to laptop) is the
   one case that plausibly touches Wi-Fi, and even there, transferring
   the cache is one option, not the only one: the raw conversation
   text is kilobytes, the cache is megabytes-to-gigabytes, so simply
   resending the text and letting the new device recompute its own
   cache is usually cheaper than shipping the cache itself — unless
   recomputing prefill is prohibitively slow (very long context, a
   weak destination device), which is a real but narrower condition.
3. **Layer-split pipeline parallelism**, where a single model's layers
   run on two devices, requires activations to cross the link on
   *every token*, continuously — a categorically harder and more
   frequent transfer problem than a one-time migration, and, if
   anything, a stronger argument against attempting it over Wi-Fi.

None of these three is what 9.2.4 tested, and none of them needed to
be for the scenario this paper is actually about. The result to take
from this section is narrower and more defensible than "we validated
moving KV cache over Wi-Fi": it is that a delta-only transport,
carrying only new tokens in and generated text out, survives real
Wi-Fi conditions (loss, jitter, firewall/reconnection friction) with
low and bounded overhead — which is precisely sufficient for a fixed
remote inference server, and is the only case that needed testing.

### 9.3 The Same-Machine Case: Completing the Linux Evaluation

The paper's Section 3.1/3.3 always described the Linux path —
`memfd_create(2)` for the shared region, `SCM_RIGHTS` over `AF_UNIX`
for fd transfer, `eventfd` for notification — but Table 2's numbers
came entirely from the Windows named-shared-memory prototype. This
gap is now closed: `memllm_linux_shm.py` implements the real Linux
path, reusing the identical `ControlBlock`/`TokenDescriptor` layout
from the Windows implementation (Section 4.2), so the "same physical
page layout and ring protocol" claim (Section 3.1) is now literally
true rather than asserted for a path that didn't exist yet. Building
it surfaced two real bugs — a `socket.recv_fds()` return-tuple
ordering mistake, and a `ctypes`-created buffer view blocking
`mmap.close()` until explicitly released — both straightforward once
found, both the kind of mistake that only shows up by actually
building the thing.

With a synthetic backend fast enough to remove compute from the
measurement entirely, pure transport overhead separates cleanly:

| Mechanism | Mean | p95 | Implementation |
|---|---|---|---|
| MemLLM-Linux (`memfd`+`eventfd`) | 0.1ms | 0.2ms | Python |
| UDP-RPC (loopback) | 0.4ms | 0.7ms | Python |
| HTTP/JSON (loopback) | 1.9ms | 14.8ms | Python |

Real shared memory is ≈4x faster than the eRPC-style transport and
≈19x faster than HTTP/JSON on mean, with a far tighter tail — HTTP's
p95 balloons to 14.8ms, plausibly from per-request TCP connection
overhead that shared memory and even UDP never pay. This is the
missing Linux/`memfd` result Section 7.3 called future work, and it
confirms the paper's central argument on the platform it was always
meant for, not only on Windows.

### 9.4 Where RoCE Actually Fits

Testing RoCE directly needs RDMA verbs semantics on both ends. The
Windows laptop used for the Wi-Fi test has no compatible RDMA stack,
so a genuine two-device RoCE-over-Wi-Fi test was not attempted;
instead, Soft-RoCE (`rdma_rxe`, a kernel module emulating real RDMA
verbs — queue pairs, memory registration, completion queues — over an
ordinary NIC with no specialized hardware) was brought up on the
Linux workstation alone, bound to its Wi-Fi interface, and measured
with `ib_send_lat` from the standard `perftest` suite.

**Baseline.** A tiny (2-byte) RDMA send, no loss:

| Mechanism | Mean latency |
|---|---|
| **Soft-RoCE** (`ib_send_lat`, C) | **2.8µs** |
| MemLLM-Linux (`memfd`+`eventfd`, Python) | 100µs |
| UDP-RPC (Python) | 400µs |
| HTTP/JSON (Python) | 1900µs |

Real RDMA verbs — even software-emulated, no specialized NIC — beat
the paper's own Python `memfd`/`eventfd` implementation by roughly
36x. This is a result worth stating plainly rather than letting the
memfd/eventfd number stand unchallenged as "the fast one": most of
that 36x gap is very likely C-versus-Python and kernel-bypass-style
completion-queue polling versus per-operation syscalls, not something
fundamental to shared memory as a mechanism. It is a real, measured
ceiling this paper's Python prototypes fall well short of.

**Behavior under synthetic loss.** `tc netem` loss was applied to the
loopback interface carrying this self-to-self RDMA traffic (traffic to
a host's own address is delivered via `lo` regardless of which
physical NIC owns that address, so this is where the packets actually
flowed):

| Loss | Result |
|---|---|
| 0% | 2.8µs mean |
| 20% | 4.7µs mean (+67%), 1000/1000 iterations completed, no failures |
| 40% | Connection could not be established (reproduced twice) |

At 20% loss, the RDMA Reliable Connection (RC) transport's own
retransmission logic absorbed the loss with a graceful latency
increase and zero failures — a more resilient result than a naive
reading of "RoCE assumes a lossless fabric" would predict. The failure
that does appear at 40% is more precise than "RoCE breaks under loss":
the server log shows `ib_send_lat` failing during its connection
*bootstrap* — a plain, non-redundant TCP side-channel used to exchange
queue-pair parameters before any RDMA traffic begins — not the RC
data path itself timing out mid-transfer. This distinction matters:
the vulnerability demonstrated here is in a benchmark tool's naive
setup handshake, not necessarily in RDMA's steady-state reliability
mechanism, which handled the loss rate it did get to exercise (20%)
without incident. It nuances, rather than settles, the paper's earlier
framing (Section 9.2.1) that RoCE simply livelocks on lossy links —
the honest finding is "the control plane is fragile under loss in
naive tooling; the data plane, where actually exercised, was not."

### 9.5 Synthesis

Three questions, three different answers, and none of them "MemLLM
helps everywhere":

- **Does MemLLM's own mechanism help over Wi-Fi?** No — shared memory
  requires the same physical RAM, which no longer exists once a
  second device is involved. This is not a limitation to engineer
  around; it is definitional.
- **Does MemLLM's *design philosophy*, reincarnated as a wire
  protocol, help over Wi-Fi?** Modestly, yes — the eRPC-style
  transport beat HTTP/JSON by ≈80ms on a real link with a controlled
  backend, and survived real packet loss without failures. That gap
  shrinks to noise once real model compute (6–14s/turn) dominates,
  which is the common case; it matters most for fast backends or
  many-small-turn workloads, not for large, slow generations.
- **Does extending MemLLM with the real Linux `memfd`/`eventfd` path
  help?** Yes, substantially — ≈4x over the Wi-Fi-capable UDP
  transport and ≈19x over HTTP/JSON — but strictly on one machine.
  This is the paper's original argument, now actually measured on
  Linux instead of only asserted for it.
- **Where does RoCE fit?** Above all of the above by another order of
  magnitude (2.8µs, ≈36x faster than this paper's own memfd/eventfd
  implementation), achievable today via Soft-RoCE with no specialized
  hardware, on the same machine. Its cross-device viability over Wi-Fi
  remains untested here for lack of a compatible peer, but the
  synthetic-loss result complicates the standard "RoCE + Wi-Fi =
  livelock" narrative: the RC data path degraded gracefully at 20%
  loss, and the actual failure found at 40% was in a benchmark tool's
  naive connection setup, not the reliability mechanism the RDMA
  transport itself provides.
- **Does any of this require moving a real KV cache over Wi-Fi?** No
  — and this is the most important clarification in the section
  (9.2.5). For a thin client talking to a fixed remote inference
  server, the cache should never leave the machine that computed it;
  only the newest turn's text needs to cross the link, which is
  exactly what every Wi-Fi number above measured. Real KV cache
  migration is a real technique, but it belongs to narrower problems
  this scenario doesn't have — disaggregated prefill/decode serving
  within a datacenter, or handing an in-progress conversation off to a
  different compute device — not to "using a remote GPU over Wi-Fi" in
  general. The transport-reliability result in 9.2.4 is therefore not
  a partial answer to a harder question still open; it is a complete
  answer to the question this architecture actually poses.

### 9.6 Honest Limitations

Loss injection was synthetic and loopback/local-interface only, not a
real contended wireless channel — real Wi-Fi loss also comes with
correlated bursts and rate-adaptation-driven jitter (9.2.1) that
independent-probability packet drop does not reproduce; the real-Wi-Fi
run used a single link, a single device pair, and n=10 turns; the RoCE
result is same-machine only, since no second RDMA-capable peer was
available on the Wi-Fi network; and the Soft-RoCE loss test used only
two data points (20%, 40%) rather than a swept curve. A rigorous
follow-up would add: a second Linux device (physically separate, real
Soft-RoCE peer) to test RoCE across an actual Wi-Fi hop rather than
self-to-self over `lo`; `tc netem`-shaped loss *and* jitter on a real
link rather than loopback; multiple independent runs per condition;
and a finer-grained loss sweep to localize the RC transport's own
actual breaking point (not merely the benchmark tool's control-channel
fragility) if one exists at all.

One item deliberately does *not* appear on this list: transferring a
real KV cache tensor over the Wi-Fi link. Section 9.2.5 argues this is
not a missing test but an absent requirement — for the thin-client,
fixed-remote-server architecture this section actually validates, the
cache correctly never leaves the compute node, so there is no cache
payload to benchmark. That test would only become meaningful for the
narrower disaggregated-serving or compute-handoff scenarios named in
9.2.5, which are different problems than the one this paper poses, and
are left as exactly that: a different, future problem, not an
unfinished part of this one.

### References (additions)

[12] *WIP: When RDMA Meets Wireless*, WoWMoM 2022.
[13] Kalia, A., Kaminsky, M., Andersen, D. — *Datacenter RPCs can be
General and Fast*, NSDI 2019 (eRPC).
