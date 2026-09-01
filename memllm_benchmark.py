"""
memllm_benchmark.py  (Windows-native edition)
==============================================
Measures and compares:

  MemLLM      : named shared memory ring  (this paper's contribution)
  HTTP-Ollama : localhost HTTP to Ollama   (local HTTP baseline)
  HTTP-Claude : Anthropic API over HTTPS  (cloud/serialization baseline)

Usage (run each in its own CMD/PowerShell window)
-------------------------------------------------
  # Window 1 — start inference server:
  python memllm_server.py --model models/phi-3-mini-4k-instruct-Q4_K_M.gguf

  # Window 2 — benchmarks:
  python memllm_benchmark.py --mode memllm
  python memllm_benchmark.py --mode http-ollama
  python memllm_benchmark.py --mode http-claude   # needs ANTHROPIC_API_KEY

  # After all runs, regenerate plots:
  python memllm_benchmark.py --mode plot-only
"""

import argparse
import csv
import json
import os
import socket
import sys
import time
import statistics
import urllib.request
import urllib.error
import datetime as dt
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memllm_region import (
    MemLLMRegion, client_handshake,
    HANDSHAKE_HOST, HANDSHAKE_PORT, DEFAULT_SIZE
)
from memllm_udp_rpc import ErpcEndpoint, FLAG_REQUEST
from memllm_linux_shm import MemLLMRegionLinux, client_handshake as linux_client_handshake
import itertools

RESULTS_DIR  = Path("results")
RAW_CSV      = RESULTS_DIR / "benchmark_raw.csv"
SUMMARY_JSON = RESULTS_DIR / "benchmark_summary.json"

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


# ── MemLLM client ─────────────────────────────────────────────────────────────

class MemLLMClient:
    def __init__(self, size_mb: int = 128, session_id: str = "bench01"):
        self.region = MemLLMRegion(size=size_mb * 1024 * 1024, session_id=session_id)
        self.region.open()
        self.region.init_control()
        print(f"[client] Shared region '{session_id}' created ({size_mb} MB)")

    def connect(self, port: int = HANDSHAKE_PORT):
        client_handshake(self.region, port=port)

    def send(self, prompt: str):
        t0  = time.perf_counter_ns()
        self.region.enqueue_request(prompt)
        text, recv_ns, send_ns = self.region.dequeue_response(timeout=120)
        t1  = time.perf_counter_ns()
        rt  = t1 - t0
        ipc = recv_ns - send_ns if recv_ns and send_ns else rt
        return text or "", rt, ipc

    def shutdown(self):
        try:
            self.region.enqueue_request("__SHUTDOWN__")
            time.sleep(0.5)
        except Exception:
            pass
        self.region.close()


# ── Real Linux memif-style client (memfd_create + SCM_RIGHTS + eventfd) ───────
#
# The actual Linux path from Section 3.1/3.3 of the paper, never previously
# implemented — only the Windows named-shared-memory prototype was benchmarked.
# Same ring/descriptor layout as MemLLMClient below; the difference is the
# region is a real memfd (not Windows named paging-file memory) and waits are
# genuine blocking eventfd reads, not a 0.5ms poll loop.

class MemLLMLinuxClient:
    def __init__(self, size_mb: int = 128, session_id: str = "linuxbench01"):
        self.region = MemLLMRegionLinux(size=size_mb * 1024 * 1024, session_id=session_id)
        self.region.create()
        print(f"[client] memfd region '{session_id}' created ({size_mb} MB)")

    def connect(self):
        linux_client_handshake(self.region)

    def send(self, prompt: str):
        t0 = time.perf_counter_ns()
        self.region.enqueue_request(prompt)
        text, recv_ns, send_ns = self.region.dequeue_response(timeout=120)
        t1 = time.perf_counter_ns()
        rt = t1 - t0
        ipc = recv_ns - send_ns if recv_ns and send_ns else rt
        return text or "", rt, ipc

    def shutdown(self):
        try:
            self.region.enqueue_request("__SHUTDOWN__")
            time.sleep(0.5)
        except Exception:
            pass
        self.region.close()


# ── UDP-RPC client (eRPC-style transport, KV-cache-over-Wi-Fi prototype) ───────
#
# Talks to memllm_mock_server.py's UDP listener. Same reliable, fragmenting,
# selective-repeat RPC engine that a real Wi-Fi deployment would use — on
# loopback there's normally no loss, but --udp-loss-pct on both this client
# and the server lets you inject synthetic fragment loss to see the
# retransmission cost that a real lossy link would impose.

class UdpRpcClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 57301,
                 loss_pct: float = 0.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        self.ep      = ErpcEndpoint(sock, loss_pct=loss_pct / 100.0)
        self.addr    = (host, port)
        self._ids    = itertools.count(1)

    def send(self, prompt: str):
        msg_id = next(self._ids)
        t0     = time.perf_counter_ns()
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
        reply  = payload.decode("utf-8")
        rt_ns  = t1 - t0
        ipc_ns = t1 - corr_ts_ns   # wire-only latency (loopback-clock, see memllm_udp_rpc.py)
        return reply, rt_ns, ipc_ns

    def shutdown(self):
        try:
            self.ep.send_message(self.addr, next(self._ids), FLAG_REQUEST,
                                  b"__SHUTDOWN__")
        except Exception:
            pass


# ── HTTP-mock client (JSON/HTTP baseline against the same MockLLM backend) ────
#
# Structurally identical to OllamaClient below, but talks to
# memllm_mock_server.py's HTTP listener instead of a real Ollama instance,
# so the compute-time distribution is controlled to be the same as the
# UDP-RPC run above — isolating transport overhead as the only variable.

class HttpMockClient:
    def __init__(self, base_url: str = "http://127.0.0.1:57380"):
        self.url = base_url.rstrip("/")

    def send(self, prompt: str):
        body = json.dumps({"prompt": prompt}).encode()
        req  = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            t1    = time.perf_counter_ns()
            reply = data.get("reply", "").strip()
            return reply, t1 - t0, t1 - t0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


# ── Ollama HTTP client ────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3"):
        self.url     = base_url.rstrip("/")
        self.model   = model
        self.history = []

    def send(self, prompt: str):
        self.history.append({"role": "user", "content": prompt})
        full = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)
        full += "\nASSISTANT:"
        body = json.dumps({"model": self.model, "prompt": full, "stream": False}).encode()
        req  = urllib.request.Request(
            f"{self.url}/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            t1    = time.perf_counter_ns()
            reply = data.get("response", "").strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply, t1 - t0, t1 - t0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


# ── Claude API client ─────────────────────────────────────────────────────────

class ClaudeAPIClient:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model   = model
        self.history = []
        self.key     = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

    def send(self, prompt: str):
        # Keep only this turn's message — avoids sending assistant
        # turns with stale content that triggers a 400
        self.history.append({"role": "user", "content": prompt})
        # Anthropic API requires alternating user/assistant roles.
        # Rebuild history ensuring strict alternation.
        clean = []
        for msg in self.history:
            if clean and clean[-1]["role"] == msg["role"]:
                clean[-1]["content"] += "\n" + msg["content"]
            else:
                clean.append({"role": msg["role"], "content": msg["content"]})

        body = json.dumps({
            "model":      self.model,
            "max_tokens": 256,
            "messages":   clean,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json",
                     "x-api-key":         self.key,
                     "anthropic-version": "2023-06-01"},
            method="POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            t1    = time.perf_counter_ns()
            reply = data["content"][0]["text"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply, t1 - t0, t1 - t0
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")
            print(f"\n  [API error {e.code}] {body_err[:200]}")
            return f"ERROR: HTTP {e.code}", 0, 0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


# ── Groq API client (free tier — Llama-3 / Mixtral) ──────────────────────────
#
# Free account: https://console.groq.com
# Free limits : 30 req/min, 6000 req/day — more than enough for benchmarks
# Groq uses the OpenAI-compatible /v1/chat/completions endpoint.

class GroqClient:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model   = model
        self.history = []
        self.key     = os.environ.get("GROQ_API_KEY", "")
        if not self.key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "  1. Sign up free at https://console.groq.com\n"
                "  2. Create an API key\n"
                "  3. set GROQ_API_KEY=gsk_...")

    def send(self, prompt: str):
        self.history.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model":       self.model,
            "max_tokens":  256,
            "messages":    self.history,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data    = body,
            headers = {"Content-Type":  "application/json",
                       "Authorization": f"Bearer {self.key}",
                       "User-Agent":    "python-memllm/1.0"},
            method  = "POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            t1    = time.perf_counter_ns()
            reply = data["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply, t1 - t0, t1 - t0
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")
            print(f"\n  [Groq error {e.code}] {body_err[:200]}")
            return f"ERROR: HTTP {e.code}", 0, 0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


# ── Google Gemini API client (free tier) ──────────────────────────────────────
#
# Free account: https://aistudio.google.com  → Get API key
# Free limits : 15 req/min on Gemini 1.5 Flash — fine for benchmarks
# Uses Google's generateContent REST endpoint (no SDK needed).

class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model   = model
        self.history = []
        self.key     = os.environ.get("GEMINI_API_KEY", "")
        if not self.key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set.\n"
                "  1. Go to https://aistudio.google.com\n"
                "  2. Click 'Get API key' (free, no credit card)\n"
                "  3. set GEMINI_API_KEY=AIza...")

    def send(self, prompt: str):
        # Gemini uses "user"/"model" roles (not "assistant")
        self.history.append({
            "role":  "user",
            "parts": [{"text": prompt}]
        })
        body = json.dumps({
            "contents":         self.history,
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.7},
        }).encode()
        url = (f"https://generativelanguage.googleapis.com/v1/models/"
               f"{self.model}:generateContent?key={self.key}")
        req = urllib.request.Request(
            url,
            data    = body,
            headers = {"Content-Type": "application/json",
                       "User-Agent":   "python-memllm/1.0"},
            method  = "POST")
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            t1    = time.perf_counter_ns()
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Gemini needs "model" role for assistant turns
            self.history.append({
                "role":  "model",
                "parts": [{"text": reply}]
            })
            return reply, t1 - t0, t1 - t0
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")
            print(f"\n  [Gemini error {e.code}] {body_err[:200]}")
            return f"ERROR: HTTP {e.code}", 0, 0
        except Exception as e:
            return f"ERROR: {e}", 0, 0


# ── runner ────────────────────────────────────────────────────────────────────

# Inter-turn delay (seconds) per backend — respects free-tier rate limits.
# Gemini free: 15 req/min = 4s min gap. Groq free: 30 req/min = 2s min gap.
# MemLLM / Ollama are local so 0 delay is fine.
INTER_TURN_DELAY = {
    "HTTP-Gemini": 5.0,   # 15 req/min → wait 5s to be safe
    "HTTP-Groq":   2.5,   # 30 req/min → wait 2.5s
    "HTTP-Claude": 1.0,
    "HTTP-Ollama": 0.0,
    "MemLLM":      0.0,
    "UDP-RPC":     0.0,
    "HTTP-Mock":   0.0,
}
MAX_RETRIES = 3


def _send_with_retry(client, prompt: str, label: str):
    """Send with exponential backoff on 429 rate-limit responses."""
    delay = INTER_TURN_DELAY.get(label, 0.0)
    for attempt in range(MAX_RETRIES):
        reply, rt_ns, ipc_ns = client.send(prompt)
        if not reply.startswith("ERROR: HTTP 429"):
            return reply, rt_ns, ipc_ns
        wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
        print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/{MAX_RETRIES} ...")
        time.sleep(wait)
        # Remove the failed user turn from history so we don't double-append
        if hasattr(client, "history") and client.history:
            if client.history[-1].get("role") in ("user", None):
                client.history.pop()
            # Gemini uses "parts" structure
            if client.history and "parts" in client.history[-1]:
                client.history.pop()
    return reply, rt_ns, ipc_ns   # return last attempt regardless


def run_turns(client, prompts, label):
    results = []
    print(f"\n{'='*60}\n  {label}  ({len(prompts)} turns)\n{'='*60}")
    delay      = INTER_TURN_DELAY.get(label, 0.0)
    ctx_tokens = 0
    for i, prompt in enumerate(prompts):
        if i > 0 and delay > 0:
            print(f"  [rate-limit pause {delay:.0f}s] ", end="", flush=True)
            time.sleep(delay)
        reply, rt_ns, ipc_ns = _send_with_retry(client, prompt, label)
        rt_ms  = rt_ns  / 1e6
        ipc_ms = ipc_ns / 1e6
        ok     = bool(reply) and not reply.startswith("ERROR")
        ctx_tokens += len(prompt.split()) + len((reply or "").split())
        results.append({
            "label":          label,
            "turn":           i + 1,
            "prompt_len":     len(prompt),
            "reply_len":      len(reply or ""),
            "round_trip_ms":  round(rt_ms,  3),
            "ipc_latency_ms": round(ipc_ms, 3),
            "context_tokens": ctx_tokens,
            "ok":             ok,
            "timestamp":      dt.datetime.now(timezone.utc).isoformat(),
        })
        tag = "OK " if ok else "ERR"
        print(f"  Turn {i+1:2d} [{tag}]  {rt_ms:7.1f}ms  {(reply or '')[:55]!r}")
    return results


def summarise(results, label):
    lats = [r["round_trip_ms"] for r in results if r["ok"] and r["round_trip_ms"] > 0]
    if not lats:
        return {"label": label, "error": "no valid samples"}
    lats.sort()
    n = len(lats)
    return {
        "label":  label,
        "n":      n,
        "mean":   round(statistics.mean(lats), 2),
        "median": round(statistics.median(lats), 2),
        "stdev":  round(statistics.stdev(lats) if n > 1 else 0, 2),
        "p95":    round(lats[int(n * 0.95)], 2),
        "p99":    round(lats[min(int(n * 0.99), n - 1)], 2),
        "min":    round(lats[0], 2),
        "max":    round(lats[-1], 2),
    }


def save(all_results):
    RESULTS_DIR.mkdir(exist_ok=True)
    if all_results:
        with open(RAW_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader(); w.writerows(all_results)
        print(f"\n[bench] CSV  → {RAW_CSV}")

    labels    = sorted(set(r["label"] for r in all_results))
    summaries = [summarise([r for r in all_results if r["label"] == lb], lb)
                 for lb in labels]
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[bench] JSON → {SUMMARY_JSON}")
    return summaries


def print_table(summaries):
    print("\n" + "=" * 70)
    print(f"  {'Label':<25} {'N':>4}  {'Mean':>8}  {'Median':>8}  {'p95':>8}  {'p99':>8}")
    print("  " + "-" * 66)
    for s in summaries:
        if "error" in s:
            print(f"  {s['label']:<25}  ERROR: {s['error']}")
        else:
            print(f"  {s['label']:<25} {s['n']:>4}  "
                  f"{s['mean']:>7.1f}ms  {s['median']:>7.1f}ms  "
                  f"{s['p95']:>7.1f}ms  {s['p99']:>7.1f}ms")
    print("=" * 70)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(all_results, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bench] pip install matplotlib  to enable plots")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    COLORS = {"memllm": "#1A6B3C", "ollama": "#C0392B", "claude": "#2E4DA7",
              "groq": "#E67E22", "gemini": "#8E44AD"}

    def col(label):
        lb = label.lower()
        for k, v in COLORS.items():
            if k in lb:
                return v
        return "#555555"

    valid = [s for s in summaries if "error" not in s]

    # ── bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    labels  = [s["label"] for s in valid]
    means   = [s["mean"]  for s in valid]
    errs    = [s["stdev"] for s in valid]
    bars    = ax.bar(labels, means, yerr=errs, capsize=5,
                     color=[col(lb) for lb in labels],
                     edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean round-trip latency (ms)", fontsize=12)
    ax.set_title("MemLLM vs HTTP — inter-turn latency", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}ms", ha="center", va="bottom", fontsize=10)
    if len(valid) >= 2:
        bmax = max(s["mean"] for s in valid)
        bmin = min(s["mean"] for s in valid)
        if bmin > 0:
            ax.annotate(f"{bmax/bmin:.1f}x faster",
                        xy=(0, bmin), xytext=(0.4, bmax * 0.6),
                        arrowprops=dict(arrowstyle="->", color="#333"),
                        fontsize=11, color="#1A6B3C", fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "plot_latency.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[bench] plot → {out}")

    # ── per-turn timeline ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label in sorted(set(r["label"] for r in all_results)):
        sub = [r for r in all_results if r["label"] == label and r["ok"]]
        if sub:
            ax.plot([r["turn"] for r in sub],
                    [r["round_trip_ms"] for r in sub],
                    marker="o", label=label, color=col(label),
                    linewidth=2, markersize=5)
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Per-turn latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = RESULTS_DIR / "plot_per_turn.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[bench] plot → {out}")

    # ── context scaling ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in sorted(set(r["label"] for r in all_results)):
        sub = sorted([r for r in all_results if r["label"] == label and r["ok"]],
                     key=lambda r: r["context_tokens"])
        if sub:
            ax.plot([r["context_tokens"] for r in sub],
                    [r["round_trip_ms"]  for r in sub],
                    marker="o", label=label, color=col(label),
                    linewidth=2, markersize=5)
    ax.set_xlabel("Approx. context tokens", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency vs context length", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = RESULTS_DIR / "plot_context_scaling.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[bench] plot → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
        choices=["memllm", "memllm-linux", "http-ollama", "http-claude",
                 "http-groq", "http-gemini", "udp-rpc-mock", "http-mock",
                 "plot-only"],
        default="memllm")
    ap.add_argument("--port",         type=int, default=HANDSHAKE_PORT)
    ap.add_argument("--ollama-url",   default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="phi3")
    ap.add_argument("--groq-model",   default="llama-3.1-8b-instant")
    ap.add_argument("--gemini-model", default="gemini-2.0-flash")
    ap.add_argument("--turns",        type=int, default=len(TURN_PROMPTS))
    ap.add_argument("--region-mb",    type=int, default=128)
    ap.add_argument("--session-id",   default="bench01")
    ap.add_argument("--udp-host",     default="127.0.0.1")
    ap.add_argument("--udp-port",     type=int, default=57301)
    ap.add_argument("--udp-loss-pct", type=float, default=0.0,
                     help="Client-side simulated fragment loss, mirrored "
                          "with memllm_mock_server.py's --udp-loss-pct for "
                          "a symmetric lossy-link test.")
    ap.add_argument("--http-mock-url", default="http://127.0.0.1:57380")
    args = ap.parse_args()

    all_results = []

    if args.mode == "plot-only":
        if not RAW_CSV.exists():
            print(f"No results at {RAW_CSV}. Run a benchmark first.")
            sys.exit(1)
        with open(RAW_CSV) as f:
            for row in csv.DictReader(f):
                row["round_trip_ms"]  = float(row["round_trip_ms"])
                row["ipc_latency_ms"] = float(row["ipc_latency_ms"])
                row["context_tokens"] = int(row["context_tokens"])
                row["turn"]           = int(row["turn"])
                row["ok"]             = row["ok"] == "True"
                all_results.append(row)
        labels    = sorted(set(r["label"] for r in all_results))
        summaries = [summarise([r for r in all_results if r["label"] == lb], lb)
                     for lb in labels]
        print_table(summaries)
        plot(all_results, summaries)
        return

    prompts = TURN_PROMPTS[:args.turns]

    if args.mode == "memllm":
        client = MemLLMClient(size_mb=args.region_mb, session_id=args.session_id)
        client.connect(port=args.port)
        all_results = run_turns(client, prompts, "MemLLM")
        client.shutdown()

    elif args.mode == "http-ollama":
        client      = OllamaClient(args.ollama_url, args.ollama_model)
        all_results = run_turns(client, prompts, "HTTP-Ollama")

    elif args.mode == "http-claude":
        try:
            client      = ClaudeAPIClient()
            all_results = run_turns(client, prompts, "HTTP-Claude")
        except EnvironmentError as e:
            print(f"[bench] {e}")
            sys.exit(1)

    elif args.mode == "http-groq":
        try:
            client      = GroqClient(model=args.groq_model)
            all_results = run_turns(client, prompts, "HTTP-Groq")
        except EnvironmentError as e:
            print(f"[bench] {e}")
            sys.exit(1)

    elif args.mode == "memllm-linux":
        client = MemLLMLinuxClient(size_mb=args.region_mb, session_id=args.session_id)
        client.connect()
        all_results = run_turns(client, prompts, "MemLLM-Linux")
        client.shutdown()

    elif args.mode == "udp-rpc-mock":
        client = UdpRpcClient(args.udp_host, args.udp_port, loss_pct=args.udp_loss_pct)
        label  = "UDP-RPC" if args.udp_loss_pct == 0 else f"UDP-RPC-{args.udp_loss_pct:g}pctloss"
        all_results = run_turns(client, prompts, label)
        print(f"  [udp-rpc] retransmitted fragments: {client.ep.retransmits}")

    elif args.mode == "http-mock":
        client      = HttpMockClient(args.http_mock_url)
        all_results = run_turns(client, prompts, "HTTP-Mock")

    elif args.mode == "http-gemini":
        try:
            client      = GeminiClient(model=args.gemini_model)
            all_results = run_turns(client, prompts, "HTTP-Gemini")
        except EnvironmentError as e:
            print(f"[bench] {e}")
            sys.exit(1)

    # Merge with any previous results from other modes
    existing = []
    if RAW_CSV.exists():
        with open(RAW_CSV) as f:
            for row in csv.DictReader(f):
                if all_results and row["label"] == all_results[0]["label"]:
                    continue   # skip old runs of same mode — replace them
                row["round_trip_ms"]  = float(row["round_trip_ms"])
                row["ipc_latency_ms"] = float(row["ipc_latency_ms"])
                row["context_tokens"] = int(row["context_tokens"])
                row["turn"]           = int(row["turn"])
                row["ok"]             = row["ok"] == "True"
                existing.append(row)

    combined  = existing + all_results
    summaries = save(combined)
    print_table(summaries)
    plot(combined, summaries)


if __name__ == "__main__":
    main()
