"""
memllm_setup.py  (Windows-native edition)
==========================================
  python memllm_setup.py              # check environment + print instructions
  python memllm_setup.py --install    # pip install all dependencies
  python memllm_setup.py --download   # download Phi-3-mini GGUF (~2.2 GB)
  python memllm_setup.py --all        # install + download
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "phi-3-mini-4k-instruct-Q4_K_M.gguf"
MODEL_URL  = (
    "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF"
    "/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
)


def check_env():
    v = sys.version_info
    print(f"[setup] Python {v.major}.{v.minor}.{v.micro}", end="  ")
    print("OK" if v >= (3, 9) else "NEED >= 3.9")

    print(f"[setup] Platform: {sys.platform}", end="  ")
    print("OK" if sys.platform == "win32" else "(not Windows — use Linux edition)")

    # Check named shared memory works on this Python/Windows build
    try:
        import mmap
        m = mmap.mmap(-1, 4096, tagname="MemLLM_test")
        m.close()
        print("[setup] Named shared memory (mmap tagname=)  OK")
    except Exception as e:
        print(f"[setup] Named shared memory  FAILED: {e}")
        print("        This feature requires Windows 10 + CPython (not PyPy).")

    # Check TCP loopback
    try:
        import socket
        s = socket.create_connection(("127.0.0.1", 1), timeout=0.01)
        s.close()
    except (ConnectionRefusedError, OSError):
        print("[setup] TCP loopback  OK")
    except Exception as e:
        print(f"[setup] TCP loopback issue: {e}")

    # Check llama-cpp-python
    try:
        from llama_cpp import Llama
        print("[setup] llama-cpp-python  OK")
    except ImportError:
        print("[setup] llama-cpp-python  NOT installed  (run --install)")

    # Check matplotlib
    try:
        import matplotlib
        print(f"[setup] matplotlib {matplotlib.__version__}  OK")
    except ImportError:
        print("[setup] matplotlib  NOT installed  (run --install)")

    print()


def install():
    print("[setup] Installing dependencies ...")

    # llama-cpp-python: try pre-built wheel first (much faster than compiling)
    print("  Trying pre-built llama-cpp-python wheel ...")
    r = subprocess.run([
        sys.executable, "-m", "pip", "install", "llama-cpp-python",
        "--prefer-binary",
        "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        "--quiet",
    ])
    if r.returncode != 0:
        print("  Pre-built failed — compiling from source (needs cmake + VS Build Tools) ...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "llama-cpp-python", "--quiet"
        ])

    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"])
    print("[setup] Done.\n")


def download():
    MODELS_DIR.mkdir(exist_ok=True)
    if MODEL_FILE.exists():
        gb = MODEL_FILE.stat().st_size / 1024**3
        print(f"[setup] Model already present: {MODEL_FILE} ({gb:.2f} GB)")
        return

    print(f"[setup] Downloading Phi-3-mini-4k Q4_K_M (~2.2 GB) ...")
    print(f"        {MODEL_URL}\n")

    def progress(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, reporthook=progress)
        print(f"\n[setup] Saved to {MODEL_FILE}")
    except Exception as e:
        print(f"\n[setup] Download failed: {e}")
        print("\n  Manual download:")
        print(f"  Open this URL in a browser and save to {MODEL_FILE}:")
        print(f"  {MODEL_URL}")


INSTRUCTIONS = r"""
╔══════════════════════════════════════════════════════════════════════╗
║         MemLLM Prototype — Windows Quick Start                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Install deps + download model (~2.2 GB):                         ║
║     python memllm_setup.py --all                                     ║
║                                                                      ║
║  2. Open TWO Command Prompt (or PowerShell) windows                  ║
║     in this folder.                                                  ║
║                                                                      ║
║  Window 1 — start the MemLLM inference server:                       ║
║     python memllm_server.py                                \         ║
║       --model models\phi-3-mini-4k-instruct-Q4_K_M.gguf             ║
║                                                                      ║
║  Window 2 — run benchmarks (after server prints "Ready"):            ║
║                                                                      ║
║     # MemLLM shared-memory path:                                     ║
║     python memllm_benchmark.py --mode memllm                         ║
║                                                                      ║
║     # HTTP baseline (Ollama must be running — see below):            ║
║     python memllm_benchmark.py --mode http-ollama                    ║
║                                                                      ║
║     # Claude API baseline (cloud round-trip):                        ║
║     set ANTHROPIC_API_KEY=sk-ant-...                                 ║
║     python memllm_benchmark.py --mode http-claude                    ║
║                                                                      ║
║     # Re-plot from saved results:                                     ║
║     python memllm_benchmark.py --mode plot-only                      ║
║                                                                      ║
║  Results:                                                            ║
║     results\benchmark_raw.csv          every turn's latency          ║
║     results\benchmark_summary.json     mean / p95 / p99              ║
║     results\plot_latency.png           bar chart                     ║
║     results\plot_per_turn.png          per-turn timeline             ║
║     results\plot_context_scaling.png   latency vs context length     ║
║                                                                      ║
║  Optional — Ollama local HTTP baseline:                              ║
║     https://ollama.com/download  → install → run:                    ║
║     ollama pull phi3                                                 ║
║     ollama serve                                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install",  action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--all",      action="store_true")
    args = ap.parse_args()

    print("\n[setup] MemLLM — environment check")
    print("─" * 50)
    check_env()

    if args.install or args.all:
        install()
    if args.download or args.all:
        download()
    if not any([args.install, args.download, args.all]):
        print(INSTRUCTIONS)
    else:
        print("[setup] Done. Run  python memllm_setup.py  for usage instructions.")


if __name__ == "__main__":
    main()
