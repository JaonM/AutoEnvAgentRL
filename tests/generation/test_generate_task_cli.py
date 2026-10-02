import tempfile
import random
import json
import hashlib
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import examples.generate_task as generator_cli

from examples.generate_task import (
    _existing_task_numbers,
    _generation_failure_class,
    _reserve_task_directories,
    _retry_seed_blocks,
    _reset_reserved_directory,
    _validate_generated_candidate,
    _write_task_artifact,
)
from env_factory.task_pipeline import PipelineGenerationError
from env_factory.tasks.task_routing import (
    allocate_training_routes,
    compatible_training_categories,
    parse_training_mix,
    select_training_intent,
)


class IncrementalTaskDirectoryTest(unittest.TestCase):
    def test_batch_deliverables_are_all_under_task_directory(self):
        class FakeStore:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakeTaskGenerator:
            STYLES = ("plain",)
            INTENTS = ("query",)

            def __init__(self, *_args, **_kwargs):
                pass

            def generate(self, *_args, artifact_dir, **_kwargs):
                (artifact_dir / "data").mkdir()
                (artifact_dir / "data" / "rows.jsonl").write_text("{}\n", encoding="utf-8")
                (artifact_dir / "tools.json").write_text("[]\n", encoding="utf-8")
                return SimpleNamespace(
                    desc="查询记录", task_type=SimpleNamespace(value="Chat"),
                    task_intent="query", complexity="simple", env={}, metrics=[],
                    artifacts={"training_category": "simple_agentic"},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = SimpleNamespace(
                base_url="https://example.invalid", model="test-model",
                timeout=10, network_retries=0,
            )
            argv = ["generate_task.py", "--count", "1", "--route-attempts", "1",
                    "--training-category", "simple_agentic", "--output", str(root),
                    "--log-file", str(root / "generation.log")]
            with patch.object(sys, "argv", argv), \
                    patch.object(generator_cli, "Neo4jGraphStore", FakeStore), \
                    patch.object(generator_cli, "TaskGenerator", FakeTaskGenerator), \
                    patch.object(generator_cli.LLMClient, "from_env", return_value=provider), \
                    patch.object(generator_cli, "_validate_generated_candidate"):
                self.assertEqual(generator_cli.main(), 0)
            sample = root / "task" / "task-1"
            task_path = sample / "task.json"
            self.assertTrue(task_path.is_file())
            self.assertTrue((sample / "tools.json").is_file())
            self.assertTrue((sample / "data" / "rows.jsonl").is_file())
            manifest = json.loads((sample / "sample_manifest.json").read_text())
            self.assertEqual(manifest["task_sha256"], hashlib.sha256(task_path.read_bytes()).hexdigest())
            self.assertGreaterEqual(manifest["generation_seconds"], 0)
            self.assertFalse((root / "task_artifacts").exists())

    def test_retry_seed_reservations_do_not_overlap(self):
        bases = _retry_seed_blocks(random.Random(42), 300, 3)
        attempted = [base + offset for base in bases for offset in range(3)]
        self.assertEqual(len(attempted), len(set(attempted)))
        self.assertEqual(bases, _retry_seed_blocks(random.Random(42), 300, 3))
        self.assertEqual(len(_retry_seed_blocks(random.Random(42), 3, 1)), 3)

    def test_noise_resistance_is_not_an_active_training_category(self):
        with self.assertRaisesRegex(ValueError, "invalid training mix entry"):
            parse_training_mix(
                "direct_response=.2,simple_agentic=.3,multi_step_agentic=.4,noise_resistance=.1"
            )

    def test_route_intent_mapping_keeps_explanations_out_of_agentic_routes(self):
        self.assertEqual(compatible_training_categories("explain"), ("direct_response",))
        self.assertEqual(select_training_intent("direct_response", "explain"), "explain")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            select_training_intent("multi_step_agentic", "explain")

    def test_explicit_intent_allocates_only_compatible_routes(self):
        routes = allocate_training_routes(
            12,
            parse_training_mix(None),
            allowed_categories=compatible_training_categories("recommend"),
        )
        self.assertEqual(set(routes), {"simple_agentic"})

    def test_default_training_mix_allocates_expected_twenty_task_quota(self):
        routes = allocate_training_routes(20, parse_training_mix(None))
        self.assertEqual(routes.count("direct_response"), 4)
        self.assertEqual(routes.count("simple_agentic"), 6)
        self.assertEqual(routes.count("multi_step_agentic"), 10)

    def test_five_task_batch_preserves_every_curriculum_category(self):
        routes = allocate_training_routes(5, parse_training_mix(None))
        self.assertEqual(routes.count("direct_response"), 1)
        self.assertEqual(routes.count("simple_agentic"), 2)
        self.assertEqual(routes.count("multi_step_agentic"), 2)
        self.assertEqual(set(routes), {
            "direct_response", "simple_agentic", "multi_step_agentic",
        })

    def test_reservation_appends_after_highest_existing_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task-1").mkdir()
            (root / "task-3").mkdir()
            (root / "task-draft").mkdir()

            reserved = _reserve_task_directories(root, 2)

            self.assertEqual([number for number, _ in reserved], [4, 5])
            self.assertEqual(_existing_task_numbers(root), [1, 3, 4, 5])
            self.assertTrue((root / "task-1").is_dir())
            self.assertTrue((root / "task-3").is_dir())

    def test_sequential_runs_never_reuse_previous_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            first = _reserve_task_directories(root, 1)
            second = _reserve_task_directories(root, 2)

            self.assertEqual([number for number, _ in first], [1])
            self.assertEqual([number for number, _ in second], [2, 3])

    def test_concurrent_reservations_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ThreadPoolExecutor(max_workers=2) as executor:
                batches = list(executor.map(
                    lambda _: _reserve_task_directories(root, 3),
                    range(2),
                ))

            numbers = [number for batch in batches for number, _ in batch]
            self.assertEqual(len(numbers), 6)
            self.assertEqual(len(set(numbers)), 6)

    def test_retry_cleanup_preserves_sample_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample_manifest.json").write_text("{}")
            (root / "partial.json").write_text("partial")
            (root / "rows").mkdir()
            (root / "rows/data.jsonl").write_text("{}")
            _reset_reserved_directory(root)
            self.assertTrue((root / "sample_manifest.json").is_file())
            self.assertFalse((root / "partial.json").exists())
            self.assertFalse((root / "rows").exists())

    def test_generation_failure_taxonomy_is_structured(self):
        self.assertEqual(
            _generation_failure_class(PipelineGenerationError("tool schema is invalid")),
            "GEN_SCHEMA",
        )
        self.assertEqual(
            _generation_failure_class(PipelineGenerationError("external capability unavailable")),
            "TASK_BUILDABILITY",
        )
        self.assertEqual(
            _generation_failure_class(PipelineGenerationError(
                "stage tools failed: LLM returned HTTP 429: rate limit"
            )),
            "INFRA",
        )

    def test_unavailable_review_is_not_a_candidate_defect(self):
        for reason in (
            "SOURCE_REVIEW_INVALID: incomplete checks",
            "SOURCE_REVIEW_TIMEOUT: reviewer deadline exceeded",
            "SOURCE_REVIEW_EXEC_FAILED: exit 1",
            "SOURCE_REVIEW_UNAVAILABLE: invalid schema",
            "SEMANTIC_CALIBRATION_UNAVAILABLE: no real judgment",
        ):
            with self.subTest(reason=reason):
                error = PipelineGenerationError("CODE_AGENT_GENERATION_FAILED: " + reason)
                self.assertEqual(_generation_failure_class(error), "INFRA")
        self.assertEqual(_generation_failure_class(PipelineGenerationError(
            "SOURCE_SEMANTIC_REVIEW_FAILED: invalid business schema")), "GEN_SEMANTIC")

    def test_candidate_gate_rejects_unbuildable_task_before_publication(self):
        with patch("scripts.sandbox.assess_task_buildability.assess", return_value={
            "buildable": False,
            "issues": [{"code": "TASK_ACCEPTANCE_FIXTURE"}, {"code": "TASK_QUALITY_GATE"}],
        }):
            with self.assertRaisesRegex(Exception, "TASK_ACCEPTANCE_FIXTURE, TASK_QUALITY_GATE"):
                _validate_generated_candidate(Path("unused-task-directory"))

    def test_candidate_is_materialized_before_buildability_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = SimpleNamespace(
                desc="筛选数据", task_type=SimpleNamespace(value="Chat"),
                task_intent="query", complexity="simple", env={}, metrics=[],
                artifacts={"training_category": "simple_agentic"},
            )
            path = _write_task_artifact(root, task, "simple_agentic")
            self.assertTrue(path.is_file())
            with self.assertRaisesRegex(Exception, "task buildability gate failed") as raised:
                _validate_generated_candidate(root)
            self.assertNotIn("TASK_ARTIFACT_INVALID", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
