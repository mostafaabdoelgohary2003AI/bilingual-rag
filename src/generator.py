"""Answer generation.

* ``ExtractiveGenerator`` - offline fallback: stitches the retrieved
  passages into an answer with explicit citations. Needs no API key, so
  the whole pipeline runs on a laptop with no network access.
* ``OpenAICompatibleClient`` - any OpenAI-compatible chat endpoint
  (OpenAI, NVIDIA NIM / build.nvidia.com, a local Ollama server, ...).
  Uses only the standard library, so there is no SDK dependency.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .retriever import ScoredChunk

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the "
    "provided context passages. The passages may be in Arabic or English; "
    "answer in the language of the question. Cite every claim with the "
    "passage number in square brackets, e.g. [1]. If the passages do not "
    "contain the answer, say so plainly instead of guessing."
)


def build_prompt(question: str, passages: list[ScoredChunk]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {p.chunk.doc_id})\n{p.chunk.text}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"


class ExtractiveGenerator:
    """Offline generator: quote the evidence instead of hallucinating."""

    def generate(self, question: str, passages: list[ScoredChunk]) -> str:
        if not passages:
            return "No relevant passages were retrieved for this question."
        lines = ["Answer (extractive mode, cite the sources below):"]
        for i, p in enumerate(passages, start=1):
            snippet = " ".join(p.chunk.text.split())[:280]
            lines.append(f"[{i}] {snippet}  (source: {p.chunk.doc_id}, score {p.score:.3f})")
        return "\n".join(lines)



def _urlopen_json(request, timeout):
    """urlopen + JSON, retrying 429/5xx with backoff (free tiers return transient 503s)."""
    import time, urllib.error
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)


class OpenAICompatibleClient:
    """Minimal chat-completions client for any OpenAI-style endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("an API key is required (LLM_API_KEY or OPENAI_API_KEY)")
        self.temperature = temperature
        self.timeout = timeout

    def _chat(self, messages: list[dict]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {"model": self.model, "messages": messages, "temperature": self.temperature}
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        payload = _urlopen_json(request, self.timeout)
        return payload["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, passages: list[ScoredChunk]) -> str:
        return self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, passages)},
            ]
        )
