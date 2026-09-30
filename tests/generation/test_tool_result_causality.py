import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import (ContractRewardGate, ContractToolRegistry,
    DeclarativeMetricEvaluator, DeclarativeToolCompiler, EpisodeStore, ManifestDataStore)
from .test_agent_authoring import fixture


class ToolResultCausalityTest(unittest.TestCase):
    def test_invalid_tool_result_cannot_earn_outcome_and_valid_retry_recovers(self):
        source, request = fixture("simple_agentic")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = compile_source(source, root=root, request=request)
            store = EpisodeStore(root / "episode.sqlite3")
            data = ManifestDataStore(artifacts["data_manifest"], root / "data/business_data", store)
            registry = ContractToolRegistry(artifacts["tools"], DeclarativeToolCompiler(data).compile_all(
                artifacts["tool_implementations"]), event_recorder=store.event,
                tool_contracts=artifacts["task_spec"]["tool_contracts"])
            store.reset(episode_id="invalid-result", seed=1)
            data.reset()
            def scores():
                context = {"business_state": {"zones": data.table("zones")},
                    "initial_business_state": data.baseline, "trajectory": store.replay(),
                    "final_agent_response": "温度：18摄氏度。"}
                raw = DeclarativeMetricEvaluator().evaluate_all(artifacts["metric_implementations"], context)
                self.assertEqual(raw["temperature_answer"], 1)
                return ContractRewardGate(artifacts["task_spec"], artifacts["metrics"]).apply(raw, context)
            with patch.dict(os.environ, {"SANDBOX_MUTATION_MODE": "constant_tool_result"}):
                registry.execute("read_temperature", {"name": "青松库"})
            self.assertEqual(scores()["temperature_answer"], 0)
            with patch.dict(os.environ, {"SANDBOX_MUTATION_MODE": "disabled"}):
                registry.execute("read_temperature", {"name": "青松库"})
            self.assertEqual(scores()["temperature_answer"], 1)
