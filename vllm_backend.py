"""
vllm_backend.py
================
Real-inference backend for the MemLLM transport prototype, calling an
already-running vLLM OpenAI-compatible server instead of the synthetic
MockLLM sleep. Same generate(prompt, history) -> str interface as
MockLLM, so it's a drop-in swap in memllm_mock_server.py — the point
is to hold the transport comparison methodology fixed while replacing
simulated compute with a real 7B model's real GPU decode latency.
"""

import json
import urllib.request


class VLLMBackend:
    def __init__(self, api_base: str = "http://127.0.0.1:8100/v1",
                 model: str = "Qwen/Qwen2.5-7B-Instruct", max_tokens: int = 150):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, history: list) -> str:
        history.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": history,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        reply = data["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": reply})
        return reply
