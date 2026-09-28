"""LLM stage execution, retry, envelope normalization and caching.

This module owns transport concerns. Domain-specific task generation remains
in :mod:`env_factory.task_pipeline`.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from ..llm import LLMClient
from .stage_cache import StageCache


def parse_json_object(content: str) -> Any:
    """Parse a JSON response, accepting a single fenced JSON block."""
    import re

    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    return json.loads(match.group(1) if match else text)


def normalize_stage_result(
    value: dict[str, Any], *, payload: dict[str, Any]
) -> dict[str, Any]:
    """Normalize common provider envelopes against the declared output example."""
    expected = payload.get("output")
    expected_keys = set(expected) if isinstance(expected, dict) else set()
    current: Any = value
    for _ in range(4):
        if not isinstance(current, dict):
            break
        if expected_keys and expected_keys.issubset(current):
            return current
        nested: Any = None
        for envelope in ("output", "result", "data"):
            candidate = current.get(envelope)
            if isinstance(candidate, str):
                try:
                    candidate = parse_json_object(candidate)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            if isinstance(candidate, dict) and (
                not expected_keys
                or expected_keys.intersection(candidate)
                or any(key in candidate for key in ("output", "result", "data"))
            ):
                nested = candidate
                break
        if nested is None:
            break
        current = nested
    return current if isinstance(current, dict) else value


def validate_stage_result_shape(
    value: dict[str, Any], *, payload: dict[str, Any]
) -> None:
    """Validate the output envelope before a stage result leaves its retry boundary."""
    expected = payload.get("output")
    if not isinstance(expected, dict) or not expected:
        return
    missing = sorted(key for key in expected if key not in value)
    if missing:
        raise ValueError(f"stage response is missing required output fields: {missing}")
    for key, example in expected.items():
        actual = value.get(key)
        if isinstance(example, bool) and not isinstance(actual, bool):
            raise ValueError(f"stage response field {key!r} must be boolean")
        if isinstance(example, str) and (
            not isinstance(actual, str) or not actual.strip()
        ):
            raise ValueError(f"stage response field {key!r} must be a non-empty string")
        if isinstance(example, dict) and not isinstance(actual, dict):
            raise ValueError(f"stage response field {key!r} must be an object")
        if isinstance(example, list):
            if not isinstance(actual, list):
                raise ValueError(f"stage response field {key!r} must be an array")
            required_non_empty_arrays = {
                "actions", "user_profiles", "user_scripts", "entities",
                "tables", "rows", "capabilities", "metrics", "decisions",
            }
            if example and key in required_non_empty_arrays and not actual:
                raise ValueError(f"stage response field {key!r} must be a non-empty array")


def is_transient_llm_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in (
        "http 429", "http 502", "http 503", "http 504",
        "too many requests", "service is too busy", "temporarily unavailable",
        "timed out", "timeout", "connection reset",
    ))


class StageExecutor:
    """Execute one structured LLM stage with cache and feedback-driven retry."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        retries: int,
        cache: StageCache,
        logger: logging.Logger,
        error_type: type[Exception],
    ) -> None:
        self.llm = llm
        self.retries = retries
        self.cache = cache
        self.logger = logger
        self.error_type = error_type

    def call(
        self,
        stage: str,
        system: str,
        payload: dict[str, Any],
        *,
        jitter: Callable[[float, float], float],
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cache_key = self.cache.key(stage, system, payload)
        cached = self.cache.read(cache_key)
        if cached is not None:
            validate_stage_result_shape(cached, payload=payload)
            self.logger.info("pipeline stage cache hit: stage=%s", stage)
            return cached
        self.logger.info("pipeline stage started: stage=%s", stage)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                effective_system = self._system_prompt(stage, system)
                effective_payload = payload
                if last_error is not None:
                    effective_system += (
                        f"\n上一轮输出未通过本阶段结构校验，具体错误是：{last_error}。"
                        "请修复该错误，严格按照 output 示例返回完整 JSON 对象；不要返回 output/type 包装对象，"
                        "不要省略必需字段。"
                    )
                    if stage.startswith("environment_table_data."):
                        effective_system += (
                            "本阶段只允许返回一个顶层 rows 数组；不要返回 table、columns、schema、"
                            "markdown 或解释文字。每行必须是 JSON object，所有字符串中的双引号、反斜杠和换行"
                            "必须正确转义；保持字段简短，避免长篇文本。"
                        )
                    effective_payload = dict(payload)
                    effective_payload["previous_validation_error"] = str(last_error)
                response = self.llm.complete(
                    json.dumps(effective_payload, ensure_ascii=False, indent=2),
                    system_prompt=effective_system + "\n只输出合法 JSON 对象，不要解释。",
                    thinking=False,
                    temperature=0.2 if attempt > 1 else 0.5,
                    max_tokens=(
                        8_000 if stage.startswith("observations_rewards")
                        else 8_000 if stage.startswith("environment_table_data.")
                        else 16_000
                    ),
                    response_format="json_object",
                )
                if getattr(response, "finish_reason", None) in {"length", "max_tokens"}:
                    raise ValueError("LLM response was truncated by the token limit")
                value = parse_json_object(response.content)
                if not isinstance(value, dict):
                    raise TypeError("stage result must be a JSON object")
                value = normalize_stage_result(value, payload=payload)
                self._validate_stage_semantics(stage, value, payload)
                validate_stage_result_shape(value, payload=payload)
                self.logger.info(
                    "pipeline stage completed: stage=%s attempt=%d duration_ms=%.1f result_keys=%s",
                    stage, attempt, (time.perf_counter() - started) * 1000,
                    sorted(value.keys()),
                )
                self.cache.write(cache_key, stage, value)
                return value
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "pipeline stage attempt failed: stage=%s attempt=%d/%d duration_ms=%.1f error=%s",
                    stage, attempt, self.retries,
                    (time.perf_counter() - started) * 1000, exc,
                )
                if attempt < self.retries and is_transient_llm_error(exc):
                    delay = min(8.0, float(2 ** attempt)) + jitter(0.0, 0.5)
                    self.logger.info(
                        "transient LLM failure; backing off before retry: stage=%s delay_seconds=%.2f",
                        stage, delay,
                    )
                    sleep(delay)
        self.logger.error(
            "pipeline stage failed: stage=%s attempts=%d duration_ms=%.1f error=%s",
            stage, self.retries, (time.perf_counter() - started) * 1000,
            last_error,
        )
        raise self.error_type(f"stage {stage} failed: {last_error}") from last_error

    @staticmethod
    def _system_prompt(stage: str, system: str) -> str:
        if stage.startswith("capability_plan"):
            system += (
                "\n每个 capability 必须声明 dependencies 数组，元素只能是输入动作原名。"
                "这是信息或状态的真实依赖，不是书写顺序。multi_step_agentic 必须具有至少两个环境操作，"
                "并明确一个环境操作如何消费前序环境操作产生的信息；其 dependencies 必须引用前序环境动作。"
                "推理和最终回答不能用来凑环境依赖。该图与是否给予过程奖励无关。"
            )
        if stage == "agent_actions" or stage.startswith("agent_actions."):
            system += (
                "\n本阶段的输入隔离规则优先级最高：只允许使用 payload 中的 task_description、"
                "business_model 和 environment_plan。business_model 只有实体、表、字段、关系和约束定义，"
                "不得索取、猜测或使用业务数据行、数据文档、隐藏真值、内部记录、具体字段值或数据库 ID；"
                "如果旧提示提到完整业务环境，以本条输入隔离规则和实际 payload 为准。"
            )
        return system

    @staticmethod
    def _validate_stage_semantics(
        stage: str, value: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        if stage == "task_description":
            task_value = value.get("task")
            if not isinstance(task_value, str) or not task_value.strip():
                raise ValueError("task_description response must contain a non-empty task string")
            if value.get("complexity") not in {"simple", "standard", "complex"}:
                raise ValueError("task_description.complexity must be simple, standard or complex")
            expected_intent = payload.get("task_intent")
            if expected_intent is not None and value.get("task_intent") != expected_intent:
                raise ValueError(
                    f"task_description.task_intent must be {expected_intent!r}"
                )
        if stage.startswith("environment_table_data."):
            rows = value.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{stage} response must contain a non-empty rows array")
