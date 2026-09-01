"""
memllm_mock_server.py
======================
Hosts the MockLLM backend behind two transports at once, each on its
own thread with its own conversation history so the two benchmark
runs (udp-rpc-mock, http-mock) don't interfere with each other:

  * UDP-RPC listener   (memllm_udp_rpc.ErpcEndpoint)   — default port 57301
  * HTTP/JSON listener (http.server, ThreadingHTTPServer) — default port 57380

Both threads construct their own MockLLM with the same tokens_per_sec,
so compute time is drawn from the same distribution on both paths —
the only thing that should differ between the two benchmark runs is
transport overhead and variance.

Usage
-----
  python memllm_mock_server.py
  python memllm_mock_server.py --udp-loss-pct 5 --udp-port 57301 --http-port 57380
"""

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memllm_udp_rpc import ErpcEndpoint, FLAG_REQUEST, FLAG_RESPONSE
from mock_llm import MockLLM
from vllm_backend import VLLMBackend


def make_backend(kind: str, tokens_per_sec: float):
    if kind == "vllm":
        return VLLMBackend()
    return MockLLM(tokens_per_sec=tokens_per_sec)


def serve_udp(port: int, loss_pct: float, tokens_per_sec: float, backend: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    ep = ErpcEndpoint(sock, loss_pct=loss_pct / 100.0)
    llm = make_backend(backend, tokens_per_sec)
    history: list = []
    print(f"[udp-rpc]  listening on :{port}  (loss_pct={loss_pct}%)")
    while True:
        result = ep.recv_message(timeout=600)
        if result is None:
            print("[udp-rpc]  idle timeout — shutting down")
            return
        src, msg_id, flags, payload, corr_ts_ns = result
        if not (flags & FLAG_REQUEST):
            continue
        text = payload.decode("utf-8")
        if text.strip() == "__SHUTDOWN__":
            print("[udp-rpc]  shutdown received")
            return
        reply = llm.generate(text, history)
        try:
            ep.send_message(src, msg_id, FLAG_RESPONSE, reply.encode("utf-8"),
                             corr_ts_ns=corr_ts_ns)
        except TimeoutError as e:
            # One message failing to land shouldn't take the whole server
            # down — a naive port of the shared-memory protocol doesn't
            # get HTTP/TCP's per-connection fault isolation for free;
            # this is exactly the isolation that has to be added by hand.
            print(f"[udp-rpc]  seq={msg_id}  response delivery failed: {e}")


def make_http_handler(llm: MockLLM, history: list, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep benchmark output clean

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = body.get("prompt", "")
            with lock:
                reply = llm.generate(prompt, history)
            out = json.dumps({"reply": reply}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    return Handler


def serve_http(port: int, tokens_per_sec: float, backend: str):
    llm = make_backend(backend, tokens_per_sec)
    history: list = []
    lock = threading.Lock()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_http_handler(llm, history, lock))
    print(f"[http-mock] listening on :{port}")
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int, default=57301)
    ap.add_argument("--http-port", type=int, default=57380)
    ap.add_argument("--udp-loss-pct", type=float, default=0.0,
                     help="Fraction (0-100) of outbound UDP fragments to drop, "
                          "simulating a lossy Wi-Fi link.")
    ap.add_argument("--tokens-per-sec", type=float, default=12.0)
    ap.add_argument("--backend", choices=["mock", "vllm"], default="mock",
                     help="'vllm' calls a live OpenAI-compatible vLLM server "
                          "(see vllm_backend.py) for real generation latency "
                          "instead of the synthetic MockLLM sleep.")
    args = ap.parse_args()

    t_udp = threading.Thread(
        target=serve_udp, args=(args.udp_port, args.udp_loss_pct, args.tokens_per_sec, args.backend),
        daemon=True)
    t_http = threading.Thread(
        target=serve_http, args=(args.http_port, args.tokens_per_sec, args.backend),
        daemon=True)
    t_udp.start()
    t_http.start()
    print("[mock-server] both transports up. Ctrl+C to stop.")
    try:
        while t_udp.is_alive() or t_http.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[mock-server] stopping.")


if __name__ == "__main__":
    main()
