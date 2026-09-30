"""Configurable LLM wrapper for OpenAI-compatible chat APIs."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from http.client import IncompleteRead
import hashlib
import json
import math
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()


_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "envfactory_llm_trace", default=None
)


@contextmanager
def capture_llm_trace():
    """Capture response metadata in the current task context, never payload text."""
    records: list[dict[str, Any]] = []
    token = _TRACE.set(records)
    try:
        yield records
    finally:
        _TRACE.reset(token)


def summarize_llm_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models: dict[str, int] = {}
    finishes: dict[str, int] = {}
    usage: dict[str, float | int] = {}
    response_ids = []
    request_seconds = 0.0
    network_retries = 0
    for record in records:
        model = str(record.get("model", ""))
        finish = str(record.get("finish_reason", ""))
        if model:
            models[model] = models.get(model, 0) + 1
        if finish:
            finishes[finish] = finishes.get(finish, 0) + 1
        response_id = record.get("response_id")
        if isinstance(response_id, str) and response_id:
            response_ids.append(hashlib.sha256(response_id.encode("utf-8")).hexdigest())
        values = record.get("usage")
        if isinstance(values, Mapping):
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[str(name)] = usage.get(str(name), 0) + value
        seconds = record.get("request_seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds >= 0:
            request_seconds += float(seconds)
        retries = record.get("network_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries >= 0:
            network_retries += retries
    return {
        "version": "1.1",
        "responses": len(records),
        "models": dict(sorted(models.items())),
        "finish_reasons": dict(sorted(finishes.items())),
        "usage": dict(sorted(usage.items())),
        "response_id_sha256": sorted(response_ids),
        "request_seconds": round(request_seconds, 3),
        "network_retries": network_retries,
    }


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an invalid response."""


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from an OpenAI-compatible chat completion."""

    content: str
    model: str
    id: str | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: Mapping[str, Any] | None = None


class LLMClient:
    """Call a configurable OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        api_key_env: str = "LLM_API_KEY",
        base_url_env: str = "LLM_BASE_URL",
        model_env: str = "LLM_MODEL",
        default_params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        network_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} is not configured")
        self.base_url = (
            base_url or os.getenv(base_url_env, "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv(model_env)
        if not self.model:
            raise ValueError(f"{model_env} is not configured")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if network_retries < 0:
            raise ValueError("network_retries must not be negative")
        self.timeout = timeout
        self.network_retries = network_retries
        self.default_params = dict(default_params or {})
        self.headers = dict(headers or {})

    @classmethod
    def from_env(cls, prefix: str = "LLM", **kwargs: Any) -> "LLMClient":
        """Create a client from ``<PREFIX>_API_KEY/BASE_URL/MODEL`` variables."""

        prefix = prefix.rstrip("_").upper()
        return cls(
            api_key_env=f"{prefix}_API_KEY",
            base_url_env=f"{prefix}_BASE_URL",
            model_env=f"{prefix}_MODEL",
            **kwargs,
        )

    def chat(self, messages: Sequence[Mapping[str, Any]], **parameters: Any) -> LLMResponse:
        """Generate a response and pass arbitrary provider parameters through."""

        if not messages:
            raise ValueError("messages must not be empty")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            **self.default_params,
            **parameters,
        }
        body["model"] = self.model
        body["messages"] = [dict(message) for message in messages]
        body["stream"] = False
        # K3 always uses its own reasoning mode and accepts temperature=1 only.
        # Generic generation callers pass thinking=False and lower temperatures.
        if (
            self.model == "kimi-k3"
            and urlparse(self.base_url).hostname in {"api.moonshot.ai", "api.moonshot.cn"}
        ):
            body.pop("thinking", None)
            body.pop("temperature", None)
            effort = os.getenv("KIMI_K3_REASONING_EFFORT", "").strip()
            if effort:
                if effort not in {"low", "high", "max"}:
                    raise ValueError("KIMI_K3_REASONING_EFFORT must be low, high, or max")
                body.setdefault("reasoning_effort", effort)
        if isinstance(body.get("thinking"), bool) and "deepseek" in self.base_url.lower():
            body["thinking"] = {
                "type": "enabled" if body["thinking"] else "disabled"
            }
        if isinstance(body.get("response_format"), str):
            body["response_format"] = {"type": body["response_format"]}

        request_data = json.dumps(body).encode("utf-8")
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.headers,
        }
        request_started = time.monotonic()
        for attempt in range(self.network_retries + 1):
            request = Request(
                f"{self.base_url}/chat/completions",
                data=request_data,
                headers=request_headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except HTTPError as exc:
                detail = self._error_detail(exc)
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.network_retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        parsed_delay = float(retry_after)
                        if not math.isfinite(parsed_delay):
                            raise ValueError("Retry-After must be finite")
                        delay = min(max(parsed_delay, 0.0), 5.0)
                    except (TypeError, ValueError):
                        delay = 0.5 * (2**attempt)
                    time.sleep(delay)
                    continue
                raise LLMError(f"LLM returned HTTP {exc.code}: {detail}") from exc
            except (IncompleteRead, URLError, TimeoutError, OSError) as exc:
                if attempt >= self.network_retries:
                    raise LLMError(
                        f"Unable to read a complete LLM response from {self.base_url} "
                        f"after {attempt + 1} attempts"
                    ) from exc
                delay = 0.5 * (2**attempt)
                time.sleep(delay)
            except json.JSONDecodeError as exc:
                raise LLMError("LLM returned invalid JSON") from exc
        else:
            raise LLMError("LLM request failed unexpectedly")

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            normalized = LLMResponse(
                content=message.get("content") or "",
                model=str(payload.get("model", self.model)),
                id=payload.get("id"),
                finish_reason=choice.get("finish_reason"),
                reasoning_content=message.get("reasoning_content"),
                usage=payload.get("usage"),
            )
            trace = _TRACE.get()
            if trace is not None:
                trace.append({
                    "model": normalized.model,
                    "response_id": normalized.id,
                    "finish_reason": normalized.finish_reason,
                    "usage": dict(normalized.usage or {}),
                    "request_seconds": time.monotonic() - request_started,
                    "network_retries": attempt,
                })
            return normalized
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response has an unexpected shape") from exc

    def complete(self, prompt: str, *, system_prompt: str | None = None, **parameters: Any) -> LLMResponse:
        """Generate a response from a single prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **parameters)

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            payload = json.load(error)
            if isinstance(payload, dict):
                error_value = payload.get("error")
                if isinstance(error_value, dict) and error_value.get("message"):
                    return str(error_value["message"])
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return str(error.reason or "request failed")
