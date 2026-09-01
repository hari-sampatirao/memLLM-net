"""
memllm_server.py  (Windows-native edition)
==========================================
MemLLM inference server for Windows.

  1. Waits for the client to connect (TCP handshake on 127.0.0.1:57201)
  2. Opens the named shared memory region the client already created
  3. Loads the GGUF model via llama-cpp-python
  4. Services requests from the zero-copy ring

Usage
-----
  python memllm_server.py --model models\phi-3-mini-4k-instruct-Q4_K_M.gguf

Install llama-cpp-python on Windows (CPU-only, no CUDA required):
  pip install llama-cpp-python --prefer-binary
  # if that fails (no pre-built wheel for your Python version):
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memllm_region import server_handshake, HANDSHAKE_HOST, HANDSHAKE_PORT


def load_model(model_path: str, ctx_size: int, gpu_layers: int):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[server] llama-cpp-python not installed.")
        print("         pip install llama-cpp-python --prefer-binary")
        sys.exit(1)

    print(f"[server] Loading: {model_path}")
    t0  = time.monotonic()
    llm = Llama(
        model_path   = model_path,
        n_ctx        = ctx_size,
        n_gpu_layers = gpu_layers,
        verbose      = False,
    )
    print(f"[server] Loaded in {time.monotonic()-t0:.1f}s")
    return llm


def run_inference(llm, prompt: str, history: list, max_tokens: int) -> str:
    history.append({"role": "user", "content": prompt})
    full = ""
    for m in history:
        full += f"<|{m['role'].upper()}|>\n{m['content']}\n"
    full += "<|ASSISTANT|>\n"
    out   = llm(full, max_tokens=max_tokens,
                stop=["<|USER|>", "<|SYSTEM|>", "\n\n\n"], echo=False)
    reply = out["choices"][0]["text"].strip()
    history.append({"role": "assistant", "content": reply})
    return reply


def serve(region, llm, max_tokens: int = 256):
    history: list = []
    print("[server] Ready — signalling client.")
    region.ctrl.server_ready = 1

    while True:
        result = region.dequeue_request(timeout=300)
        if result is None:
            print("[server] Idle timeout — shutting down.")
            break
        seq, text, req_ns = result
        if text.strip() == "__SHUTDOWN__":
            print("[server] Shutdown received.")
            break

        preview = text[:60] + ("..." if len(text) > 60 else "")
        print(f"[server] seq={seq}  prompt={preview!r}")
        t0    = time.perf_counter_ns()
        reply = run_inference(llm, text, history, max_tokens)
        ms    = (time.perf_counter_ns() - t0) / 1e6
        print(f"[server] seq={seq}  {ms:.0f}ms  reply={reply[:50]!r}...")
        region.enqueue_response(reply, seq)

    region.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True)
    ap.add_argument("--ctx",        type=int, default=4096)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port",       type=int, default=HANDSHAKE_PORT)
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"[server] Model not found: {args.model}")
        print("  Run:  python memllm_setup.py --download")
        sys.exit(1)

    region = server_handshake(port=args.port)
    llm    = load_model(args.model, args.ctx, args.gpu_layers)
    serve(region, llm, args.max_tokens)


if __name__ == "__main__":
    main()
