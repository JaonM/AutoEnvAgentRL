"""Small, dependency-free runtime adapter for generated sandboxes.

The adapter is intentionally separate from task-generation prompts.  A
generated sandbox may copy this module (or implement the same contract) so
User Simulator and reward evaluators share timeout, retry, JSON validation and
credential-redaction behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
import time
from dataclasses import dataclass
from http.client import IncompleteRead
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RuntimeLLMError(RuntimeError):
    pass


_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "envfactory_runtime_llm_trace", default=None
)


@contextmanager
def capture_runtime_llm_trace():
    """Capture non-secret provider response metadata for one runtime request."""
    records: list[dict[str, Any]] = []
    token = _TRACE.set(records)
    try:
        yield records
    finally:
        _TRACE.reset(token)


def summarize_runtime_llm_trace(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models: dict[str, int] = {}
    usage: dict[str, float | int] = {}
    for record in records:
        model = str(record.get("model", ""))
        if model:
            models[model] = models.get(model, 0) + 1
        values = record.get("usage")
        if isinstance(values, Mapping):
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[str(name)] = usage.get(str(name), 0) + value
    return {
        "version": "1.0",
        "responses": len(records),
        "mock_responses": sum(record.get("mock") is True for record in records),
        "models": dict(sorted(models.items())),
        "usage": dict(sorted(usage.items())),
        "response_id_sha256": sorted(
            str(record["response_id_sha256"])
            for record in records
            if isinstance(record.get("response_id_sha256"), str)
        ),
    }


@dataclass(frozen=True)
class RuntimeLLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0
    max_retries: int = 3
    mock: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeLLMConfig":
        # Per-field overrides; empty values count as unconfigured. Keep this
        # dependency-free so copied runtimes work with container env injection.
        api_key = os.getenv("SANDBOX_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")
        base_url = (os.getenv("SANDBOX_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("SANDBOX_LLM_MODEL") or os.getenv("LLM_MODEL", "")
        timeout = float(os.getenv("SANDBOX_LLM_TIMEOUT_SECONDS") or os.getenv("LLM_TIMEOUT") or "60")
        retries = int(os.getenv("SANDBOX_LLM_MAX_RETRIES") or "3")
        mock = os.getenv("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"}
        if not api_key and not mock:
            raise RuntimeLLMError("neither SANDBOX_LLM_API_KEY nor LLM_API_KEY is configured")
        if not model and not mock:
            raise RuntimeLLMError("neither SANDBOX_LLM_MODEL nor LLM_MODEL is configured")
        if timeout <= 0 or retries < 0:
            raise RuntimeLLMError("invalid runtime LLM timeout or retry configuration")
        return cls(api_key, base_url, model, timeout, retries, mock)


class RuntimeLLMClient:
    def __init__(self, config: RuntimeLLMConfig | None = None, *, mock_handler: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None) -> None:
        self.config = config or RuntimeLLMConfig.from_env()
        self.mock_handler = mock_handler

    def json_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        response_schema: Mapping[str, Any],
        semantic_validator: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not messages:
            raise RuntimeLLMError("messages must not be empty")
        if self.config.mock:
            if self.mock_handler is None:
                raise RuntimeLLMError("mock mode requires an injected deterministic evaluator")
            value = self.mock_handler(messages, response_schema)
            if not isinstance(value, dict):
                raise RuntimeLLMError("mock evaluator must return an object")
            self._validate_response(value, response_schema)
            if semantic_validator is not None:
                semantic_validator(value)
            trace = _TRACE.get()
            if trace is not None:
                trace.append({"model": self.config.model, "usage": {}, "mock": True})
            return value
        body = {
            "model": self.config.model,
            "messages": [dict(item) for item in messages],
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.config.model == "kimi-k3":
            effort = os.getenv("KIMI_K3_REASONING_EFFORT", "").strip()
            if effort:
                if effort not in {"low", "high", "max"}:
                    raise RuntimeLLMError(
                        "KIMI_K3_REASONING_EFFORT must be low, high, or max"
                    )
                body["reasoning_effort"] = effort
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                request = Request(f"{self.config.base_url}/chat/completions", data=encoded, headers=headers, method="POST")
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.load(response)
                trace = _TRACE.get()
                if trace is not None:
                    response_id = payload.get("id")
                    trace.append({
                        "model": str(payload.get("model") or self.config.model),
                        "usage": dict(payload.get("usage") or {}),
                        **({
                            "response_id_sha256": hashlib.sha256(
                                response_id.encode("utf-8")
                            ).hexdigest(),
                        } if isinstance(response_id, str) and response_id else {}),
                        "mock": False,
                    })
                content = payload["choices"][0]["message"]["content"]
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise RuntimeLLMError("runtime LLM JSON response must be an object")
                self._validate_response(value, response_schema)
                if semantic_validator is not None:
                    semantic_validator(value)
                return value
            except HTTPError as exc:
                if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise RuntimeLLMError(f"runtime LLM HTTP failure: {exc.code}") from exc
            except RuntimeLLMError as exc:
                if attempt >= self.config.max_retries:
                    raise
                body["messages"] = [*body["messages"], {"role": "user", "content": f"Your JSON failed validation: {exc}. Return a corrected complete object matching this schema: {json.dumps(response_schema, ensure_ascii=False)}"}]
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            except (IncompleteRead, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                if attempt >= self.config.max_retries:
                    raise RuntimeLLMError("runtime LLM request or response failed") from exc
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeLLMError("runtime LLM request failed")

    @staticmethod
    def _validate_response(value: dict[str, Any], schema: Mapping[str, Any]) -> None:
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise RuntimeLLMError(f"runtime LLM response missing fields: {missing}")
        # Reuse the runtime's supported schema subset; import lazily to avoid
        # the runtime adapter's module initialization cycle.
        try:
            from .sandbox_runtime import validate_json_schema, SandboxError
        except ImportError:
            from sandbox_runtime import validate_json_schema, SandboxError
        try:
            validate_json_schema(schema, value, "response")
        except SandboxError as exc:
            raise RuntimeLLMError(str(exc)) from exc
