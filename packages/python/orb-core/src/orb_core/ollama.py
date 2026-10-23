"""A small async client for the Ollama chat API, with streaming."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger("orb.ollama")

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e2b"


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, options: dict | None = None,
                 keep_alive: str = "30m", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = {"temperature": 0.7, "num_ctx": 16384, **(options or {})}
        self.keep_alive = keep_alive
        self.timeout = timeout

    def _payload(self, messages: list[dict], options: dict | None, stream: bool) -> dict:
        return {"model": self.model, "stream": stream, "think": False, "keep_alive": self.keep_alive,
                "options": {**self.options, **(options or {})}, "messages": messages}

    async def stream(self, messages: list[dict], options: dict | None = None) -> AsyncIterator[str]:
        """Yields the reply text piece by piece. Raises OllamaError when the server fails."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=self._payload(messages, options, True)) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")[:300]
                    raise OllamaError(f"Ollama returned HTTP {r.status_code}: {body}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise OllamaError(str(msg["error"]))
                    piece = msg.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if msg.get("done"):  # Ollama's own timings: counts and seconds only, never words
                        log.info("ollama %s: load %.2f s, prompt %s tokens in %.2f s, reply %s tokens in %.2f s",
                                 self.model, msg.get("load_duration", 0) / 1e9, msg.get("prompt_eval_count"),
                                 msg.get("prompt_eval_duration", 0) / 1e9, msg.get("eval_count"),
                                 msg.get("eval_duration", 0) / 1e9)

    async def warm(self, messages: list[dict]) -> None:
        """Read the prompt once, so that the first real reply does not wait for it. Errors are ignored."""
        payload = self._payload(messages, {"num_predict": 1}, False)
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError:
            pass

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{self.base_url}/api/tags")
            names = [m["name"] for m in r.json().get("models", [])]
            return {"ok": self.model in names, "model": self.model, "available": names}
        except (httpx.HTTPError, ValueError) as e:
            return {"ok": False, "model": self.model, "error": f"{type(e).__name__}: {e}"}
