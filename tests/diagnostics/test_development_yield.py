import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnostics/report_development_yield.py"
spec = importlib.util.spec_from_file_location("report_development_yield", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class DevelopmentYieldTest(unittest.TestCase):
    def test_five_minute_latency_uses_manifest_reservation_and_job_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "sample_manifest.json"
            manifest_path.write_text(json.dumps({
                "reserved_at": "2026-09-29T10:00:00+00:00",
                "generation_seconds": 112.5,
                "attempts": [{"llm_trace": {
                    "version": "1.1", "request_seconds": 90.25,
                }}],
            }), encoding="utf-8")
            history = {"rounds": [{"summary": {"requested": 1}, "jobs": [{
                "id": 1, "completed_at": "2026-09-29T10:04:30+00:00",
                "result": {
                    "sample_manifest": str(manifest_path), "task_path": "task.json",
                    "build_completed_at": "2026-09-29T10:03:50+00:00",
                    "task_score": {"eligible": True, "score": 9},
                    "sandbox_score": {"passed": True}, "passed": True,
                },
            }]}]}
            report = reporter.summarize(history)
            self.assertEqual(report["generation_seconds"]["median"], 112.5)
            self.assertEqual(report["generation_recorded_llm_seconds"]["median"], 90.25)
            self.assertEqual(report["generation_unattributed_seconds"]["median"], 22.25)
            self.assertEqual(report["sample_latency_seconds"]["median"], 270)
            self.assertEqual(report["generation_build_seconds"]["median"], 230)
            self.assertEqual(report["generation_build_seconds"]["within_300_seconds_rate"], 1)
            self.assertEqual(report["generation_build_seconds"]["offline_qualified_within_300_seconds"], 1)
            self.assertEqual(report["generation_build_seconds"]["final_qualified_within_300_seconds"], 1)
            self.assertEqual(report["generation_build_seconds"]["final_qualified_within_300_seconds_rate"], 1)
            self.assertEqual(report["sample_latency_seconds"]["within_300_seconds_rate"], 1)
            self.assertEqual(report["sample_latency_seconds"]["qualified_within_300_seconds"], 1)

    def test_phase_duration_summary_uses_persisted_build_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "status.json").write_text(json.dumps({"phase_timings": [
                {"phase": "acceptance", "seconds": 2.0},
                {"phase": "acceptance", "seconds": 4.0},
                {"phase": "defect_repair", "seconds": 8.0},
            ]}), encoding="utf-8")
            history = {"rounds": [{"summary": {"requested": 1}, "jobs": [{
                "id": 1, "result": {"output": str(output), "passed": False},
            }]}]}
            phases = reporter.summarize(history)["build_phase_seconds"]
            self.assertEqual(phases["acceptance"]["total"], 6.0)
            self.assertEqual(phases["acceptance"]["median"], 3.0)
            self.assertEqual(phases["defect_repair"]["count"], 1)

    def test_quality_and_five_minutes_use_requested_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "sample_manifest.json"
            manifest.write_text(json.dumps({
                "reserved_at": "2026-09-29T10:00:00+00:00",
            }), encoding="utf-8")
            history = {"rounds": [{"summary": {"requested": 2}, "jobs": [
                {"id": 1, "result": {
                    "sample_manifest": str(manifest), "task_path": "task.json",
                    "build_completed_at": "2026-09-29T10:04:00+00:00",
                    "task_score": {"eligible": True, "score": 9},
                    "sandbox_score": {"passed": True}, "passed": True,
                }},
                {"id": 2, "result": {
                    "task_path": "rejected.json", "failure_code": "TASK_QUALITY",
                    "passed": False,
                }},
            ]}]}
            report = reporter.summarize(history)["generation_build_seconds"]
            self.assertEqual(report["final_qualified_within_300_seconds"], 1)
            self.assertEqual(report["final_qualified_within_300_seconds_rate"], 0.5)


    def test_stage_denominators_and_failures(self):
        history = {"config": {"threshold": 8}, "rounds": [{
            "summary": {"requested": 3}, "jobs": [
                {"id": 1, "result": {"category": "simple_agentic", "task_path": "a",
                 "task_score": {"eligible": True, "score": 9},
                 "sandbox_score": {"passed": True}, "passed": True,
                 "build": {"seconds": 4}}},
                {"id": 2, "result": {"category": "simple_agentic", "task_path": "b",
                 "task_score": {"eligible": True, "score": 8},
                 "failure_code": "BUILD_RUNTIME_INTEGRITY", "passed": False}},
                {"id": 3, "result": {"category": "multi_step_agentic",
                 "failure_code": "GEN_SEMANTIC", "passed": False}},
            ],
        }]}
        report = reporter.summarize(history)
        self.assertEqual(report["overall"]["task_yield"], 2 / 3)
        self.assertEqual(report["overall"]["conditional_sandbox_yield"], .5)
        self.assertEqual(report["overall"]["end_to_end_yield"], 1 / 3)
        self.assertLess(report["overall"]["end_to_end_yield_ci95_lower"], 1 / 3)
        self.assertEqual(report["failure_codes"], {
            "BUILD_RUNTIME_INTEGRITY": 1, "GEN_SEMANTIC": 1,
        })
        self.assertEqual(report["build_seconds"]["median"], 4)
        self.assertEqual(report["by_task_score_band"]["9_to_10"]["conditional_sandbox_yield"], 1)
        self.assertEqual(report["by_task_score_band"]["threshold_to_9"]["conditional_sandbox_yield"], 0)

    def test_rejects_non_monotonic_claims(self):
        history = {"rounds": [{"summary": {"requested": 1}, "jobs": [
            {"id": 1, "result": {"passed": True}},
        ]}]}
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            reporter.summarize(history)


if __name__ == "__main__":
    unittest.main()
