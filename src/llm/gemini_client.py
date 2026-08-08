"""Thin wrapper around the Gemini API that every LLM/embedding call in the
project goes through, so token usage and cost are captured in one place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src import config

RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_DEFAULT_BACKOFF_SECONDS = 15


def _retry_delay_seconds(exc: genai_errors.ClientError, attempt: int) -> float:
    """Prefer the server's suggested RetryInfo delay; fall back to a
    linear backoff so a burst of 429s (free-tier per-minute quotas) doesn't
    just fail the whole run."""
    details = getattr(exc, "details", None) or {}
    for item in details.get("error", {}).get("details", []):
        if item.get("@type", "").endswith("RetryInfo"):
            delay = item.get("retryDelay", "")
            if delay.endswith("s"):
                try:
                    return float(delay[:-1]) + 1
                except ValueError:
                    pass
    return RATE_LIMIT_DEFAULT_BACKOFF_SECONDS * attempt


def _call_with_rate_limit_retry(fn):
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return fn()
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            time.sleep(_retry_delay_seconds(exc, attempt))


@dataclass
class GenerationResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    parsed: Optional[Any] = None


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    tokens_in: int
    cost_usd: float
    latency_ms: float


def _cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (tokens_in / 1_000_000) * rates["input"] + (tokens_out / 1_000_000) * rates["output"]


def _estimate_tokens(text: str) -> int:
    """Rough fallback estimate (~4 chars/token) when the API doesn't return usage."""
    return max(1, len(text) // 4)


class GeminiClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or config.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        model: str,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: Any | None = None,
        temperature: float = 0.2,
    ) -> GenerationResult:
        start = time.perf_counter()
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        if response_schema is not None:
            gen_config.response_mime_type = "application/json"
            gen_config.response_schema = response_schema

        response = _call_with_rate_limit_retry(
            lambda: self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) or _estimate_tokens(
            (system_instruction or "") + prompt
        )
        tokens_out = getattr(usage, "candidates_token_count", None) or _estimate_tokens(
            response.text or ""
        )

        parsed = None
        if response_schema is not None:
            parsed = getattr(response, "parsed", None)
            if parsed is None and response.text:
                try:
                    import json

                    parsed = json.loads(response.text)
                except (ValueError, TypeError):
                    parsed = None

        return GenerationResult(
            text=response.text or "",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_cost(model, tokens_in, tokens_out),
            latency_ms=latency_ms,
            parsed=parsed,
        )

    def embed(self, texts: list[str], model: str = config.EMBEDDING_MODEL) -> EmbeddingResult:
        start = time.perf_counter()
        result = _call_with_rate_limit_retry(
            lambda: self._client.models.embed_content(model=model, contents=texts)
        )
        latency_ms = (time.perf_counter() - start) * 1000

        vectors = [list(e.values) for e in result.embeddings]
        tokens_in = sum(_estimate_tokens(t) for t in texts)

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            tokens_in=tokens_in,
            cost_usd=_cost(model, tokens_in, 0),
            latency_ms=latency_ms,
        )


_default_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
