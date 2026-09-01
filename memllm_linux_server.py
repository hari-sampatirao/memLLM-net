"""
memllm_linux_server.py
========================
Server for the real Linux memif-style path (memfd_create + SCM_RIGHTS
+ eventfd, see memllm_linux_shm.py). Same backend abstraction as
memllm_mock_server.py — MockLLM or a live vLLM server — so results are
directly comparable to both the Windows-prototype numbers (Table 2)
and the Wi-Fi UDP-RPC numbers from this session.

Usage
-----
  python memllm_linux_server.py --backend mock
  python memllm_linux_server.py --backend vllm
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memllm_linux_shm import server_handshake
from mock_llm import MockLLM
from vllm_backend import VLLMBackend


def make_backend(kind: str, tokens_per_sec: float):
    if kind == "vllm":
        return VLLMBackend()
    return MockLLM(tokens_per_sec=tokens_per_sec)


def serve(region, backend, max_idle: float = 300.0):
    history: list = []
    print("[server] ready — signalling client via control block.")
    region.ctrl.server_ready = 1
    while True:
        result = region.dequeue_request(timeout=max_idle)
        if result is None:
            print("[server] idle timeout — shutting down.")
            break
        seq, text, req_ns = result
        if text.strip() == "__SHUTDOWN__":
            print("[server] shutdown received.")
            break
        preview = text[:60] + ("..." if len(text) > 60 else "")
        print(f"[server] seq={seq}  prompt={preview!r}")
        t0 = time.perf_counter_ns()
        reply = backend.generate(text, history)
        ms = (time.perf_counter_ns() - t0) / 1e6
        print(f"[server] seq={seq}  {ms:.0f}ms  reply={reply[:50]!r}...")
        region.enqueue_response(reply, seq)
    region.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    ap.add_argument("--tokens-per-sec", type=float, default=12.0)
    args = ap.parse_args()

    region = server_handshake()
    backend = make_backend(args.backend, args.tokens_per_sec)
    serve(region, backend)


if __name__ == "__main__":
    main()
