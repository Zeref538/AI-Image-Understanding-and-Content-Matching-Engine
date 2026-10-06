"""The only module that talks to the model server. Everything else gets plain results."""
import time
from dataclasses import dataclass

import httpx

from app import config


class ModelError(Exception):
    """The call itself failed: server down, timeout, HTTP error."""


@dataclass
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    duration_ms: int
    logprobs: list | None


def chat(model: str, prompt: str, *, images: list[str] | None = None, schema: dict | None = None,
         logprobs: bool = False, temperature: float = 0.0) -> ChatResult:
    body = {"model": model, "stream": False, "think": False,  # thinking off: slower, no better for tags
            "options": {"temperature": temperature},
            "messages": [{"role": "user", "content": prompt, **({"images": images} if images else {})}]}
    if schema:
        body["format"] = schema          # the server constrains output to this JSON schema
    if logprobs:
        body["logprobs"], body["top_logprobs"] = True, 3
    start = time.monotonic()
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=config.MODEL_TIMEOUT_S)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelError(f"{type(exc).__name__}: {exc}") from exc
    d = r.json()
    return ChatResult(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                      int((time.monotonic() - start) * 1000), d.get("logprobs"))


def embed(model: str, texts: list[str]) -> tuple[list[list[float]], int, int]:
    """Returns (vectors, input_tokens, duration_ms)."""
    start = time.monotonic()
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/embed", json={"model": model, "input": texts},
                       timeout=config.MODEL_TIMEOUT_S)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelError(f"{type(exc).__name__}: {exc}") from exc
    d = r.json()
    return d["embeddings"], d.get("prompt_eval_count", 0), int((time.monotonic() - start) * 1000)
