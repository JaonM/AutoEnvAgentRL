"""Boundary cases for the shared model-stage response parser."""

import unittest
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

from env_factory.generation.pipeline_stage import StageExecutor, normalize_stage_result, validate_stage_result_shape
from env_factory.generation.stage_cache import StageCache


class PipelineStageContractTest(unittest.TestCase):
    def test_semantically_invalid_cache_entry_is_regenerated(self):
        class LLM:
            calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                return SimpleNamespace(content='{"task":"有效任务","complexity":"simple"}', finish_reason="stop")

        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(Path(directory), revision="test", model="test")
            payload = {"output": {"task": "string", "complexity": "simple"}}
            key = cache.key("task_description", "system", payload)
            cache.write(key, "task_description", {"task": "旧任务", "complexity": "unknown"})
            llm = LLM()
            executor = StageExecutor(llm, retries=1, cache=cache,
                                     logger=logging.getLogger(__name__), error_type=ValueError)
            result = executor.call("task_description", "system", payload, jitter=lambda *_: 0)
            self.assertEqual(result["complexity"], "simple")
            self.assertEqual(llm.calls, 1)
            self.assertEqual(executor.call("task_description", "system", payload,
                                           jitter=lambda *_: 0), result)
            self.assertEqual(llm.calls, 1)

    def test_nested_json_string_is_unwrapped(self):
        normalized = normalize_stage_result(
            {"result": '{"output":{"metrics":[],"observation_schema":{},"reward_formula":{}}}'},
            payload={"output": {
                "metrics": [], "observation_schema": {}, "reward_formula": {},
            }},
        )
        self.assertEqual(set(normalized), {"metrics", "observation_schema", "reward_formula"})

    def test_deterministic_fallback_stage_stops_after_structural_error(self):
        class LLM:
            calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                return SimpleNamespace(content='{}', finish_reason="stop")

        with tempfile.TemporaryDirectory() as directory:
            llm = LLM()
            executor = StageExecutor(
                llm, retries=3,
                cache=StageCache(Path(directory), revision="test", model="test"),
                logger=logging.getLogger(__name__), error_type=ValueError,
            )
            with self.assertRaisesRegex(ValueError, "missing required output fields"):
                executor.call(
                    "acceptance_contract", "system",
                    {"output": {"acceptance_contract": {"scenarios": []}}},
                    jitter=lambda *_: 0, stop_on_structural_error=True,
                )
            self.assertEqual(llm.calls, 1)

    def test_required_and_optional_arrays_have_distinct_empty_rules(self):
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            validate_stage_result_shape(
                {"actions": []},
                payload={"output": {"actions": [{"name": "string"}]}},
            )
        validate_stage_result_shape(
            {"tool_bindings": []},
            payload={"output": {"tool_bindings": [{"tool_name": "string"}]}},
        )
