"""Portable, secret-free provenance for model-generated task contracts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _valid_provider(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"host", "model", "identity_sha256"}
        and isinstance(value.get("host"), str) and bool(value["host"])
        and isinstance(value.get("model"), str) and bool(value["model"])
        and isinstance(value.get("identity_sha256"), str)
        and DIGEST.fullmatch(value["identity_sha256"]) is not None
    )


def _valid_trace(value: Any, *, require_response: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    common_fields = {
        "version", "responses", "models", "finish_reasons", "usage",
        "response_id_sha256",
    }
    version = value.get("version")
    if version == "1.0":
        if set(value) != common_fields:
            return False
    elif version == "1.1":
        if set(value) != common_fields | {"request_seconds", "network_retries"}:
            return False
        seconds = value.get("request_seconds")
        retries = value.get("network_retries")
        if (
            not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(float(seconds)) or seconds < 0
            or not isinstance(retries, int) or isinstance(retries, bool) or retries < 0
        ):
            return False
    else:
        return False
    responses = value.get("responses")
    models = value.get("models")
    finishes = value.get("finish_reasons")
    usage = value.get("usage")
    response_ids = value.get("response_id_sha256")
    return (
        isinstance(responses, int) and not isinstance(responses, bool)
        and responses >= (1 if require_response else 0)
        and isinstance(models, Mapping)
        and all(isinstance(name, str) and name and isinstance(count, int) and count > 0
                for name, count in models.items())
        and sum(models.values()) == responses
        and isinstance(finishes, Mapping)
        and all(isinstance(name, str) and name and isinstance(count, int) and count > 0
                for name, count in finishes.items())
        and sum(finishes.values()) <= responses
        and isinstance(usage, Mapping)
        and all(
            isinstance(name, str) and name
            and isinstance(number, (int, float)) and not isinstance(number, bool)
            and math.isfinite(float(number)) and number >= 0
            for name, number in usage.items()
        )
        and isinstance(response_ids, list)
        and all(isinstance(item, str) and DIGEST.fullmatch(item) is not None
                for item in response_ids)
    )


def generation_provenance_snapshot(
    manifest: Mapping[str, Any], task: Mapping[str, Any], *,
    expected_provider: Mapping[str, Any] | None = None,
    expected_task_sha256: str | None = None,
) -> dict[str, Any]:
    if manifest.get("version") != "2.0" or manifest.get("status") != "completed":
        raise ValueError("sample manifest is not a completed v2 generation record")
    provider = manifest.get("generator_provider")
    deterministic = manifest.get("generation_backend") == "spec" and provider is None
    code_agent = manifest.get("generation_backend") == "code_agent"
    agent_provenance = manifest.get("code_agent_provenance")
    compiler = manifest.get("compiler_provenance")
    if deterministic:
        pipeline = task.get("artifacts", {}).get("generation_pipeline", {})
        if (expected_provider is not None or pipeline.get("backend") != "spec"
                or not isinstance(compiler, Mapping)
                or set(compiler) != {"prototype", "spec_sha256", "compiler_sha256"}
                or not isinstance(compiler.get("prototype"), str) or not compiler["prototype"]
                or any(not isinstance(compiler.get(key), str) or not DIGEST.fullmatch(compiler[key])
                       for key in ("spec_sha256", "compiler_sha256"))
                or any(pipeline.get(key) != value for key, value in compiler.items())
                or pipeline.get("llm_calls") != 0):
            raise ValueError("sample manifest compiler provenance is invalid")
    elif not _valid_provider(provider) or (
        expected_provider is not None and dict(provider) != dict(expected_provider)
    ):
        raise ValueError("sample manifest generator provider is invalid")
    if code_agent:
        pipeline = task.get("artifacts", {}).get("generation_pipeline", {})
        if (not isinstance(agent_provenance, Mapping)
                or set(agent_provenance) != {"model", "agent_invocations", "completed_turns",
                    "events_sha256", "source_sha256", "request_sha256", "compiler_sha256"}
                or agent_provenance.get("model") != provider.get("model")
                or any(not isinstance(agent_provenance.get(key), int)
                    or isinstance(agent_provenance[key], bool) or agent_provenance[key] < 1
                    for key in ("agent_invocations", "completed_turns"))
                or any(not isinstance(agent_provenance.get(key), str)
                    or DIGEST.fullmatch(agent_provenance[key]) is None
                    for key in ("events_sha256", "source_sha256", "request_sha256", "compiler_sha256"))
                or pipeline.get("backend") != "code_agent"
                or any(pipeline.get(key) != value for key, value in agent_provenance.items())):
            raise ValueError("sample manifest Code Agent provenance is invalid")
    for name in ("batch_index", "run_seed", "sample_seed", "hops", "successful_attempt"):
        if not isinstance(manifest.get(name), int) or isinstance(manifest.get(name), bool):
            raise ValueError(f"sample manifest {name} is invalid")
    if (
        manifest["batch_index"] <= 0
        or manifest["sample_seed"] < 0
        or manifest["hops"] < 0
    ):
        raise ValueError("sample manifest index/hops is invalid")
    category = manifest.get("training_category")
    if not isinstance(category, str) or not category or task.get("training_category") != category:
        raise ValueError("sample manifest training category does not match task")
    task_sha256 = manifest.get("task_sha256")
    if (
        not isinstance(task_sha256, str)
        or DIGEST.fullmatch(task_sha256) is None
        or expected_task_sha256 is not None
        and task_sha256 != expected_task_sha256
    ):
        raise ValueError("sample manifest task digest is invalid")
    intent = manifest.get("resolved_task_intent")
    if not isinstance(intent, str) or not intent:
        raise ValueError("sample manifest resolved task intent is invalid")
    settings = manifest.get("generation_settings")
    if not (
        isinstance(settings, Mapping)
        and set(settings) == {"route_attempt_limit", "timeout_seconds", "network_retries"}
        and isinstance(settings.get("route_attempt_limit"), int)
        and settings["route_attempt_limit"] > 0
        and isinstance(settings.get("timeout_seconds"), (int, float))
        and not isinstance(settings["timeout_seconds"], bool)
        and math.isfinite(float(settings["timeout_seconds"]))
        and (settings["timeout_seconds"] >= 0 if deterministic else settings["timeout_seconds"] > 0)
        and isinstance(settings.get("network_retries"), int)
        and settings["network_retries"] >= 0
    ):
        raise ValueError("sample manifest generation settings are invalid")
    attempts = manifest.get("attempts")
    successful = manifest["successful_attempt"]
    if not isinstance(attempts, list) or not attempts or successful != len(attempts):
        raise ValueError("sample manifest attempt sequence is invalid")
    portable_attempts = []
    completed = 0
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise ValueError("sample manifest attempt is invalid")
        status = attempt.get("status")
        trace = attempt.get("llm_trace")
        if (
            attempt.get("attempt") != index
            or attempt.get("seed") != manifest["sample_seed"] + index - 1
            or status not in {"rejected", "completed"}
            or not _valid_trace(trace, require_response=status == "completed" and not deterministic and not code_agent)
            or deterministic and trace.get("responses") != 0
            or code_agent and trace.get("responses") != 0
        ):
            raise ValueError("sample manifest attempt provenance is invalid")
        completed += status == "completed"
        portable_attempts.append({
            "attempt": index,
            "seed": attempt["seed"],
            "status": status,
            "llm_trace": dict(trace),
        })
    if completed != 1 or portable_attempts[-1]["status"] != "completed":
        raise ValueError("sample manifest must have one final completed attempt")
    task_id = manifest.get("task_id")
    if not isinstance(task_id, str) or re.fullmatch(r"task-\d+", task_id) is None:
        raise ValueError("sample manifest task_id is invalid")
    return {
        "version": "1.2" if code_agent else "1.1" if deterministic else "1.0",
        **({"generation_backend": "spec", "compiler_provenance": dict(compiler)} if deterministic else {}),
        **({"generation_backend": "code_agent", "code_agent_provenance": dict(agent_provenance)} if code_agent else {}),
        "task_id": task_id,
        "batch_index": manifest["batch_index"],
        "run_seed": manifest["run_seed"],
        "sample_seed": manifest["sample_seed"],
        "task_sha256": task_sha256,
        "training_category": category,
        "resolved_task_intent": intent,
        "hops": manifest["hops"],
        "generator_provider": None if deterministic else dict(provider),
        "generation_settings": dict(settings),
        "successful_attempt": successful,
        "attempts": portable_attempts,
    }


def valid_generation_provenance(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    extra = ({"generation_backend", "code_agent_provenance"} if value.get("version") == "1.2"
        else {"generation_backend", "compiler_provenance"} if value.get("version") == "1.1" else set())
    if set(value) != {
        "version", "task_id", "batch_index", "run_seed", "sample_seed",
        "task_sha256", "training_category", "resolved_task_intent", "hops",
        "generator_provider",
        "generation_settings", "successful_attempt", "attempts",
    } | extra:
        return False
    task = {"training_category": value.get("training_category")}
    if extra and isinstance(value.get("compiler_provenance"), Mapping):
        task["artifacts"] = {"generation_pipeline": {"backend": "spec", "llm_calls": 0, **value["compiler_provenance"]}}
    if value.get("version") == "1.2" and isinstance(value.get("code_agent_provenance"), Mapping):
        task["artifacts"] = {"generation_pipeline": {"backend": "code_agent", **value["code_agent_provenance"]}}
    manifest = dict(value)
    manifest["version"] = "2.0"
    manifest["status"] = "completed"
    try:
        return generation_provenance_snapshot(manifest, task) == dict(value)
    except (TypeError, ValueError):
        return False
