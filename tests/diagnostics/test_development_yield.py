import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnostics/report_development_yield.py"
spec = importlib.util.spec_from_file_location("report_development_yield", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class DevelopmentYieldTest(unittest.TestCase):
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

    def test_rejects_non_monotonic_claims(self):
        history = {"rounds": [{"summary": {"requested": 1}, "jobs": [
            {"id": 1, "result": {"passed": True}},
        ]}]}
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            reporter.summarize(history)


if __name__ == "__main__":
    unittest.main()
