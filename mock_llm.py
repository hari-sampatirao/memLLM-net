"""
mock_llm.py
===========
A deterministic stand-in for the real Phi-3 backend, used to isolate
*transport* overhead from model-compute variance when comparing MemLLM
transports. No GGUF model or llama-cpp-python is required.

Every backend (UDP-RPC, HTTP-mock) that talks to a MockLLM instance
with the same `tokens_per_sec` gets statistically identical "compute"
time for the same prompt/history length, so any latency or variance
difference measured between transports is attributable to the
transport, not to the model. This is the deliberate experimental
control for the KV-cache-over-Wi-Fi prototype: we are validating the
transport theory, not re-benchmarking Phi-3.
"""

import random
import time

# Matches the paper's measured Phi-3-mini CPU decode range (Section 6.3).
DEFAULT_TOKENS_PER_SEC = 12.0

_CANNED_REPLIES = [
    "A transformer processes sequences with self-attention instead of recurrence, "
    "letting every token attend to every other token in parallel.",
    "The main components are the embedding layer, stacked attention blocks, "
    "feed-forward layers, and a final projection back to vocabulary logits.",
    "Attention computes a weighted sum over value vectors, where the weights come "
    "from the similarity between query and key projections of each token.",
    "Self-attention is O(n^2) in sequence length because every token compares "
    "against every other token to form the attention matrix.",
    "Positional encoding injects order information as an additive or rotary "
    "signal, since attention itself is permutation-invariant.",
    "Encoder-only models attend bidirectionally and suit understanding tasks; "
    "decoder-only models attend causally and suit generation.",
    "Chatbots, code assistants, and document summarizers are common decoder-only "
    "applications.",
    "The KV cache and model weights dominate memory; both scale with context "
    "length and hidden size.",
    "KV caching stores past key/value projections so each new token only computes "
    "attention against cached history instead of recomputing it.",
    "Very long contexts make the KV cache grow linearly until it exceeds device "
    "memory, which is exactly the constraint MemLLM targets.",
]


class MockLLM:
    def __init__(self, tokens_per_sec: float = DEFAULT_TOKENS_PER_SEC, seed: int = 7):
        self.tokens_per_sec = tokens_per_sec
        self._rng = random.Random(seed)
        self._turn = 0

    def generate(self, prompt: str, history: list) -> str:
        history.append({"role": "user", "content": prompt})
        reply = _CANNED_REPLIES[self._turn % len(_CANNED_REPLIES)]
        self._turn += 1

        # Simulate CPU-bound autoregressive decode: latency scales with
        # both the reply length and the accumulated history (context
        # re-processing), matching the super-linear growth pattern the
        # paper measures for HTTP-Ollama (Section 6.3, Figure 3).
        out_tokens   = max(1, len(reply.split()))
        context_toks = sum(len(m["content"].split()) for m in history)
        think_s = (out_tokens / self.tokens_per_sec) * (1 + context_toks / 4000.0)
        think_s *= self._rng.uniform(0.9, 1.1)   # small jitter, like real decode
        time.sleep(think_s)

        history.append({"role": "assistant", "content": reply})
        return reply
