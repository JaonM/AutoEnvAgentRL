import tempfile
import random
import json
import hashlib
import sys
import unittest
import pytest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import env_factory.generate_task as generator_cli

from env_factory.generate_task import (
    _existing_task_numbers,
    _generation_failure_class,
    _reserve_task_directories,
    _retry_seed_blocks,
    _archive_rejected_candidate,
    _validate_generated_candidate,
    _write_task_artifact,
)
from env_factory.generation.contract_compiler import PipelineGenerationError
from env_factory.tasks.task_routing import (
    allocate_training_routes,
    compatible_training_categories,
    parse_training_mix,
    select_training_intent,
)


@pytest.mark.parametrize('last_rejected', [False, True])
def test_buildability_rejection_after_resume_starts_fresh_and_keeps_evidence(tmp_path, last_rejected):
    from contextlib import nullcontext
    from env_factory.generation.authoring_checkpoint import load_checkpoint, save_checkpoint
    from tests.generation.test_business_blueprint import blueprint, request
    calls = []
    task = SimpleNamespace(desc='Fresh task', task_type=SimpleNamespace(value='Chat'),
        task_intent='query', complexity='standard', env={}, metrics=[], artifacts={})

    class Generator:
        STYLES, INTENTS = ('plain',), ('query',)
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, artifact_dir, seed, **kwargs):
            calls.append(('generate', seed))
            if len(calls) == 1:
                (artifact_dir / 'original_failure.txt').write_text('timeout evidence')
                save_checkpoint(artifact_dir, request=dict(request(), seed=seed), blueprint=blueprint(),
                    source=None, evidence={'status': 'failed', 'author_seconds': 600,
                        'attempts': [{'phase': 'implementation'}]})
                raise PipelineGenerationError('CODE_AGENT_TIMEOUT: fixed task interrupted')
            assert not (artifact_dir / 'authoring_checkpoint').exists()
            (artifact_dir / 'fresh_source.json').write_text('{}')
            return task
        def resume(self, checkpoint, *, artifact_dir):
            calls.append(('resume', load_checkpoint(checkpoint)['request']['seed']))
            artifact_dir.mkdir(parents=True)
            (artifact_dir / 'rejected_source.json').write_text('{}')
            return task

    rejection = generator_cli.TaskGenerationError('task buildability gate failed: rejected source')
    argv = ['generate_task.py', '--count', '1', '--route-attempts', '3',
        '--training-category', 'multi_step_agentic', '--output', str(tmp_path),
        '--log-file', str(tmp_path / 'generation.log')]
    with patch.object(sys, 'argv', argv), \
         patch.object(generator_cli, 'Neo4jGraphStore', return_value=nullcontext(None)), \
         patch.object(generator_cli, 'TaskGenerator', Generator), \
         patch.object(generator_cli, '_validate_generated_candidate',
                      side_effect=[rejection, rejection if last_rejected else None]):
        assert generator_cli.main() == int(last_rejected)
    task_dir = tmp_path / 'artifacts/artifact-1/task'
    manifest = json.loads((task_dir / 'sample_manifest.json').read_text())
    assert [call[0] for call in calls] == ['generate', 'resume', 'generate']
    assert calls[0][1] == calls[1][1]
    assert calls[2][1] == calls[0][1] + 2
    assert manifest['attempts'][-1]['continued_fixed_design'] is False
    assert manifest['status'] == ('failed' if last_rejected else 'completed')
    from datetime import datetime
    assert datetime.fromisoformat(manifest['completed_at']) >= datetime.fromisoformat(manifest['reserved_at'])
    archive = task_dir.parent / 'rejected_generations/attempt-2'
    assert (archive / 'original_failure.txt').read_text() == 'timeout evidence'
    assert (archive / 'authoring_checkpoint/manifest.json').is_file()
    assert (archive / 'generation_retries/attempt-2/rejected_source.json').is_file()
    assert (task_dir / 'fresh_source.json').is_file()
    assert (task_dir / 'task.json').is_file()
    if last_rejected:
        assert 'FileNotFoundError' not in (task_dir / 'failure.json').read_text()


class IncrementalTaskDirectoryTest(unittest.TestCase):
    def test_timeout_continues_checkpoint_without_resampling_or_erasing_failure(self):
        from contextlib import nullcontext
        from env_factory.generation.authoring_checkpoint import save_checkpoint
        from tests.generation.test_business_blueprint import blueprint, request
        calls = []
        class Generator:
            STYLES = ('plain',)
            INTENTS = ('query',)
            def __init__(self, *args, **kwargs):
                pass
            def generate(self, *args, artifact_dir, seed, **kwargs):
                calls.append('generate')
                (artifact_dir / 'original_failure.txt').write_text('timeout evidence')
                (artifact_dir / 'data').mkdir()
                (artifact_dir / 'data/old.json').write_text('{}')
                save_checkpoint(artifact_dir, request=dict(request(), seed=seed), blueprint=blueprint(),
                    source=None, evidence={'status': 'failed', 'author_seconds': 600,
                        'attempts': [{'phase': 'implementation'}]})
                raise PipelineGenerationError('CODE_AGENT_TIMEOUT: fixed task interrupted')
            def resume(self, checkpoint, *, artifact_dir):
                calls.append('resume')
                self_checkpoint = json.loads(Path(checkpoint).read_text())
                assert self_checkpoint['previous_status'] == 'failed'
                artifact_dir.mkdir(parents=True)
                (artifact_dir / 'data').mkdir()
                (artifact_dir / 'data/resumed.json').write_text('{}')
                (artifact_dir / 'tools.json').write_text('[]')
                return SimpleNamespace(desc='Fixed task', task_type=SimpleNamespace(value='Chat'),
                    task_intent='query', complexity='standard', env={}, metrics=[], artifacts={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['generate_task.py', '--count', '1', '--route-attempts', '2',
                    '--training-category', 'multi_step_agentic', '--output', str(root),
                    '--log-file', str(root / 'generation.log')]
            with patch.object(sys, 'argv', argv), \
                 patch.object(generator_cli, 'Neo4jGraphStore', return_value=nullcontext(None)), \
                 patch.object(generator_cli, 'TaskGenerator', Generator), \
                 patch.object(generator_cli, '_validate_generated_candidate'):
                self.assertEqual(generator_cli.main(), 0)
            task_dir = root / 'artifacts/artifact-1/task'
            manifest = json.loads((task_dir / 'sample_manifest.json').read_text())
            self.assertEqual(calls, ['generate', 'resume'])
            self.assertEqual([a['status'] for a in manifest['attempts']], ['rejected', 'completed'])
            self.assertEqual(manifest['attempts'][0]['seed'], manifest['attempts'][1]['seed'])
            self.assertTrue(manifest['attempts'][1]['continued_fixed_design'])
            self.assertTrue((task_dir / 'original_failure.txt').exists())
            self.assertTrue((task_dir / 'data/resumed.json').exists())
            self.assertTrue((task_dir / 'generation_retries/attempt-2/previous_delivery/data/old.json').exists())

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
            argv = ["generate_task.py", "--count", "1", "--route-attempts", "1",
                    "--training-category", "simple_agentic", "--output", str(root),
                    "--log-file", str(root / "generation.log")]
            with patch.object(sys, "argv", argv), \
                    patch.object(generator_cli, "Neo4jGraphStore", FakeStore), \
                    patch.object(generator_cli, "TaskGenerator", FakeTaskGenerator), \
                    patch.object(generator_cli, "_validate_generated_candidate"):
                self.assertEqual(generator_cli.main(), 0)
            sample = root / "artifacts" / "artifact-1" / "task"
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
            (root / "artifact-1").mkdir()
            (root / "artifact-3").mkdir()
            (root / "artifact-draft").mkdir()

            reserved = _reserve_task_directories(root, 2)

            self.assertEqual([number for number, _ in reserved], [4, 5])
            self.assertEqual(_existing_task_numbers(root), [1, 3, 4, 5])
            self.assertTrue((root / "artifact-1").is_dir())
            self.assertTrue((root / "artifact-3").is_dir())

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

    def test_retry_archive_preserves_sample_identity_and_rejected_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "task"
            root.mkdir()
            (root / "sample_manifest.json").write_text("{}")
            (root / "partial.json").write_text("partial")
            (root / "rows").mkdir()
            (root / "rows/data.jsonl").write_text("{}")
            archive = _archive_rejected_candidate(root, 2)
            self.assertEqual((archive / "partial.json").read_text(), "partial")
            self.assertEqual((archive / "rows/data.jsonl").read_text(), "{}")
            self.assertTrue((archive / "sample_manifest.json").is_file())
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
