"""
memllm_wifi_client.py
======================
Self-contained (stdlib-only) client for the real over-Wi-Fi MemLLM
transport prototype. Run this on the peer device (e.g. a laptop) while
memllm_mock_server.py runs on the other machine, both on the same
Wi-Fi network.

This is a standalone copy of the eRPC-style transport from
memllm_udp_rpc.py plus the UdpRpcClient/HttpMockClient benchmark
clients from memllm_benchmark.py, bundled into one file so it can be
copied to a machine without the rest of the repo. No third-party
packages required — only the Python standard library.

Usage
-----
  python memllm_wifi_client.py --mode udp-rpc --host 192.168.1.114
  python memllm_wifi_client.py --mode http   --host 192.168.1.114
  python memllm_wifi_client.py --mode both   --host 192.168.1.114
"""

import argparse
import itertools
import json
import random
import socket
import statistics
import struct
import time
import urllib.request
from collections import deque
from typing import Optional, Tuple

# ── eRPC-style transport (identical to memllm_udp_rpc.py) ─────────────────────

HEADER_FMT  = "!QIIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FLAG_REQUEST  = 0x01
FLAG_RESPONSE = 0x02
FLAG_ACK      = 0x04
FLAG_FIN      = 0x08

MAX_CHUNK   = 1200
DEFAULT_RTO = 0.02
MAX_RETRIES = 20
POLL_SLEEP  = 0.0002


def _pack_header(msg_id, frag_idx, frag_count, flags, corr_ts_ns) -> bytes:
    return struct.pack(HEADER_FMT, msg_id, frag_idx, frag_count, flags, corr_ts_ns)


def _unpack_header(buf: bytes):
    return struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])


class _OutMsg:
    __slots__ = ("dest", "fragments", "unacked", "last_send", "attempts", "corr_ts_ns")

    def __init__(self, dest, fragments, corr_ts_ns):
        self.dest, self.fragments = dest, fragments
        self.unacked   = set(range(len(fragments)))
        self.last_send = 0.0
        self.attempts  = 0
        self.corr_ts_ns = corr_ts_ns


class _InMsg:
    __slots__ = ("src", "frag_count", "parts", "flags", "corr_ts_ns")

    def __init__(self, src, frag_count, flags, corr_ts_ns):
        self.src, self.frag_count = src, frag_count
        self.parts = {}
        self.flags = flags
        self.corr_ts_ns = corr_ts_ns

    def complete(self) -> bool:
        return len(self.parts) == self.frag_count

    def assemble(self) -> bytes:
        return b"".join(self.parts[i] for i in range(self.frag_count))


class ErpcEndpoint:
    def __init__(self, sock, loss_pct: float = 0.0, rto: float = DEFAULT_RTO,
                 max_retries: int = MAX_RETRIES):
        self.sock = sock
        self.sock.setblocking(False)
        self.loss_pct, self.rto, self.max_retries = loss_pct, rto, max_retries
        self._out, self._in = {}, {}
        self._ready = deque()
        self.retransmits = 0
        self._completed: set = set()
        self._completed_order = deque()
        self._completed_cap = 4096

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

        if msg_id in self._completed:
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
        if self.loss_pct > 0 and random.random() < self.loss_pct:
            return False
        try:
            self.sock.sendto(datagram, dest)
        except OSError:
            return False
        return True

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


# ── benchmark clients ───────────────────────────────────────────────────────

TURN_PROMPTS = [
    "Explain what a transformer neural network is in two sentences.",
    "What are the main components you just described?",
    "How does the attention mechanism work?",
    "What is the computational complexity of self-attention?",
    "How does positional encoding solve the sequence order problem?",
    "What is the difference between encoder-only and decoder-only transformers?",
    "Name three real-world applications of decoder-only models.",
    "What are the main memory bottlenecks in LLM inference?",
    "How does KV caching reduce redundant computation?",
    "What is the main challenge of KV caching for very long contexts?",
]


class UdpRpcClient:
    def __init__(self, host: str, port: int, loss_pct: float = 0.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        self.ep = ErpcEndpoint(sock, loss_pct=loss_pct / 100.0)
        self.addr = (host, port)
        self._ids = itertools.count(1)

    def send(self, prompt: str):
        msg_id = next(self._ids)
        t0 = time.perf_counter_ns()
        try:
            self.ep.send_message(self.addr, msg_id, FLAG_REQUEST, prompt.encode("utf-8"),
                                  corr_ts_ns=t0)
            result = self.ep.recv_message(timeout=90)
        except TimeoutError as e:
            return f"ERROR: {e}", 0, 0
        t1 = time.perf_counter_ns()
        if result is None:
            return "ERROR: timeout", 0, 0
        _src, _mid, _flags, payload, corr_ts_ns = result
        return payload.decode("utf-8"), t1 - t0, t1 - corr_ts_ns


class HttpMockClient:
    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/")

    def send(self, prompt: str):
        body = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            t1 = time.perf_counter_ns()
            return data.get("reply", "").strip(), t1 - t0, t1 - t0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


def run(client, label):
    print(f"\n{'='*60}\n  {label}  ({len(TURN_PROMPTS)} turns)\n{'='*60}")
    lats = []
    for i, prompt in enumerate(TURN_PROMPTS):
        reply, rt_ns, ipc_ns = client.send(prompt)
        ok = not reply.startswith("ERROR")
        rt_ms = rt_ns / 1e6
        print(f"  Turn {i+1:2d} [{'OK ' if ok else 'ERR'}]  {rt_ms:8.1f}ms  {reply[:55]!r}")
        if ok:
            lats.append(rt_ms)
    if lats:
        lats.sort()
        n = len(lats)
        print(f"\n  n={n}  mean={statistics.mean(lats):.1f}ms  "
              f"median={statistics.median(lats):.1f}ms  "
              f"stdev={statistics.stdev(lats) if n > 1 else 0:.1f}ms  "
              f"p95={lats[int(n*0.95)] if n > 1 else lats[0]:.1f}ms  "
              f"min={lats[0]:.1f}ms  max={lats[-1]:.1f}ms")
    else:
        print("\n  no successful turns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Server's LAN IP, e.g. 192.168.1.114")
    ap.add_argument("--udp-port", type=int, default=57301)
    ap.add_argument("--http-port", type=int, default=57380)
    ap.add_argument("--mode", choices=["udp-rpc", "http", "both"], default="both")
    ap.add_argument("--loss-pct", type=float, default=0.0,
                     help="Client-side simulated loss (must match server's --udp-loss-pct)")
    args = ap.parse_args()

    if args.mode in ("udp-rpc", "both"):
        client = UdpRpcClient(args.host, args.udp_port, loss_pct=args.loss_pct)
        run(client, "UDP-RPC (real Wi-Fi)")
        print(f"  [udp-rpc] retransmitted fragments: {client.ep.retransmits}")

    if args.mode in ("http", "both"):
        client = HttpMockClient(f"http://{args.host}:{args.http_port}")
        run(client, "HTTP-Mock (real Wi-Fi)")


if __name__ == "__main__":
    main()
