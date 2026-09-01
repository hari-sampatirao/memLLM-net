"""
memllm_udp_rpc.py
==================
A from-scratch, simplified reimplementation of eRPC's core ideas
(Kalia et al., NSDI'19) as an alternative MemLLM transport for the
across-the-network case (e.g. Wi-Fi), where a directly mapped shared
memory region is not available.

This is NOT the eRPC library. It borrows three of its design choices
that matter for the RoCE-over-Wi-Fi question raised in the paper:

  1. Reliability implemented in userspace over plain UDP, so it needs
     no lossless fabric (PFC/ECN) and no special NIC — unlike RoCE,
     it runs on any commodity Wi-Fi chipset.
  2. A single-threaded, run-to-completion polling engine (busy-poll
     recvfrom, no blocking calls, no interrupts) — the same latency
     vs. CPU tradeoff MemLLM's Windows prototype already makes with
     threading.Event polling instead of eventfd.
  3. Batched fragment transmission with selective-repeat ACKs, so a
     single lost fragment costs one retransmission, not a whole
     message resend — the property that matters once the link is
     lossy (Wi-Fi) instead of the effectively lossless loopback/PCIe
     path MemLLM's shared-memory ring assumes.

Wire format
-----------
Every UDP datagram starts with a fixed 32-byte header:

    msg_id       Q (8)   monotonic id, unique per logical request/response
    frag_idx     I (4)   this fragment's index
    frag_count   I (4)   total fragments in the message
    flags        I (4)   REQUEST | RESPONSE | ACK | FIN
    payload_len  I (4)   bytes of payload in *this* datagram
    corr_ts_ns   Q (8)   client's original send timestamp, propagated
                         unchanged through the response so round-trip
                         latency can be measured without cross-machine
                         clock sync (valid on loopback / same-host
                         tests; a real cross-device deployment would
                         need PTP/NTP-disciplined clocks for this
                         field to mean anything).

followed by up to MAX_CHUNK bytes of payload. For an ACK, the payload
is a bitmap (one byte per fragment, 1 = received) so the sender only
retransmits the fragments actually missing.
"""

import random
import socket
import struct
import time
from collections import deque
from typing import Optional, Tuple

HEADER_FMT  = "!QIIIQ"          # note: payload_len folded out, see below
# msg_id(Q) frag_idx(I) frag_count(I) flags(I) corr_ts_ns(Q)
HEADER_FMT  = "!QIIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FLAG_REQUEST  = 0x01
FLAG_RESPONSE = 0x02
FLAG_ACK      = 0x04
FLAG_FIN      = 0x08

MAX_CHUNK   = 1200          # safe under a 1500-byte Ethernet/Wi-Fi MTU
DEFAULT_RTO = 0.02          # 20 ms retransmit timeout
MAX_RETRIES = 20            # ~420ms worst-case budget; still fails visibly
                            # under sustained double-digit loss, which is
                            # the point — see memllm_mock_server.py's
                            # per-message failure handling.
POLL_SLEEP  = 0.0002        # 0.2 ms busy-poll granularity


def _pack_header(msg_id: int, frag_idx: int, frag_count: int,
                  flags: int, corr_ts_ns: int) -> bytes:
    return struct.pack(HEADER_FMT, msg_id, frag_idx, frag_count, flags, corr_ts_ns)


def _unpack_header(buf: bytes):
    return struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])


class _OutMsg:
    __slots__ = ("dest", "fragments", "unacked", "last_send", "attempts", "corr_ts_ns")

    def __init__(self, dest, fragments, corr_ts_ns):
        self.dest       = dest
        self.fragments  = fragments               # list[bytes] (full datagrams incl. header)
        self.unacked    = set(range(len(fragments)))
        self.last_send  = 0.0
        self.attempts   = 0
        self.corr_ts_ns = corr_ts_ns


class _InMsg:
    __slots__ = ("src", "frag_count", "parts", "flags", "corr_ts_ns")

    def __init__(self, src, frag_count, flags, corr_ts_ns):
        self.src        = src
        self.frag_count = frag_count
        self.parts      = {}
        self.flags      = flags
        self.corr_ts_ns = corr_ts_ns

    def complete(self) -> bool:
        return len(self.parts) == self.frag_count

    def assemble(self) -> bytes:
        return b"".join(self.parts[i] for i in range(self.frag_count))


class ErpcEndpoint:
    """
    One side of a reliable request/response channel over a plain UDP
    socket. Both the client and the server construct one of these
    around their own socket.
    """

    def __init__(self, sock: socket.socket, loss_pct: float = 0.0,
                 rto: float = DEFAULT_RTO, max_retries: int = MAX_RETRIES):
        self.sock        = sock
        self.sock.setblocking(False)
        self.loss_pct    = loss_pct
        self.rto         = rto
        self.max_retries = max_retries
        self._out: dict  = {}          # msg_id -> _OutMsg awaiting ACK
        self._in: dict   = {}          # msg_id -> _InMsg being reassembled
        self._ready      = deque()     # completed inbound messages
        self.retransmits = 0           # counter for reporting

        # Dedup for inbound messages already delivered to recv_message().
        # A lost ACK makes the sender retransmit a message we already
        # finished reassembling; without this, poll_once() would either
        # re-queue it (misdelivering a stale reply to a later, unrelated
        # recv_message() call) or, once already popped from self._in,
        # crash on a duplicate _ready entry. Capped so a long session
        # doesn't grow this unboundedly.
        self._completed: set   = set()
        self._completed_order  = deque()
        self._completed_cap    = 4096

    # ── internal: one iteration of the event loop ──────────────────
    def poll_once(self):
        try:
            data, addr = self.sock.recvfrom(65535)
        except BlockingIOError:
            return
        except OSError:
            return
        if len(data) < HEADER_SIZE:
            return
        msg_id, frag_idx, frag_count, flags, corr_ts_ns = _unpack_header(data)
        payload = data[HEADER_SIZE:]

        if flags & FLAG_ACK:
            out = self._out.get(msg_id)
            if out is None:
                return
            for i, bit in enumerate(payload):
                if bit and i in out.unacked:
                    out.unacked.discard(i)
            if not out.unacked:
                del self._out[msg_id]
            return

        # data fragment (request or response)
        if msg_id in self._completed:
            # Already fully delivered to the application; this is a
            # retransmit racing a lost ACK. Re-ack it (idempotently) so
            # the sender stops retrying, but do not re-queue it.
            bitmap = bytes([1] * frag_count)
            ack = _pack_header(msg_id, 0, frag_count, FLAG_ACK, 0) + bitmap
            self._send_raw(ack, addr)
            return

        im = self._in.get(msg_id)
        if im is None:
            im = _InMsg(addr, frag_count, flags, corr_ts_ns)
            self._in[msg_id] = im
        im.parts[frag_idx] = payload

        if im.complete():
            bitmap = bytes([1] * im.frag_count)
            ack = _pack_header(msg_id, 0, im.frag_count, FLAG_ACK, 0) + bitmap
            self._send_raw(ack, im.src)
            self._completed.add(msg_id)
            self._completed_order.append(msg_id)
            if len(self._completed_order) > self._completed_cap:
                self._completed.discard(self._completed_order.popleft())
            self._ready.append(msg_id)

    def _send_raw(self, datagram: bytes, dest) -> bool:
        """Actually put a datagram on the wire, unless loss injection eats it."""
        if self.loss_pct > 0 and random.random() < self.loss_pct:
            return False
        try:
            self.sock.sendto(datagram, dest)
        except OSError:
            return False
        return True

    # ── public: send a logical message reliably ────────────────────
    def send_message(self, dest, msg_id: int, flags: int, payload: bytes,
                      corr_ts_ns: Optional[int] = None) -> None:
        if corr_ts_ns is None:
            corr_ts_ns = time.perf_counter_ns()
        if not payload:
            fragments = [_pack_header(msg_id, 0, 1, flags, corr_ts_ns)]
        else:
            chunks = [payload[i:i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
            fragments = [
                _pack_header(msg_id, idx, len(chunks), flags, corr_ts_ns) + c
                for idx, c in enumerate(chunks)
            ]
        out = _OutMsg(dest, fragments, corr_ts_ns)
        self._out[msg_id] = out

        # Exponential backoff with jitter. A fixed retry interval causes a
        # retry storm under heavy loss: both endpoints' retransmissions pile
        # up faster than the busy-poll loop (and the OS UDP recv buffer)
        # can drain them, pushing *effective* loss well above the nominal
        # rate — this was measured directly (see prototype notes), not
        # assumed. Backing off, like TCP/eRPC do, relieves that pressure.
        cur_rto = self.rto
        max_rto = self.rto * 8
        deadline = time.monotonic() + max_rto * (self.max_retries + 2)
        while out.unacked:
            now = time.monotonic()
            if now - out.last_send >= cur_rto:
                if out.attempts > 0:
                    self.retransmits += len(out.unacked)
                    cur_rto = min(max_rto, cur_rto * 1.7) * random.uniform(0.85, 1.15)
                for idx in sorted(out.unacked):
                    self._send_raw(out.fragments[idx], dest)
                out.last_send = now
                out.attempts += 1
                if out.attempts > self.max_retries:
                    del self._out[msg_id]
                    raise TimeoutError(f"msg {msg_id}: gave up after {out.attempts} attempts")
            self.poll_once()
            if now > deadline:
                raise TimeoutError(f"msg {msg_id}: hard deadline exceeded")
            if out.unacked:
                time.sleep(POLL_SLEEP)

    # ── public: block (with busy-poll) for the next fully-reassembled message ──
    def recv_message(self, timeout: float) -> Optional[Tuple]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready:
                msg_id = self._ready.popleft()
                im = self._in.pop(msg_id)
                return (im.src, msg_id, im.flags, im.assemble(), im.corr_ts_ns)
            self.poll_once()
            if not self._ready:
                time.sleep(POLL_SLEEP)
        return None
