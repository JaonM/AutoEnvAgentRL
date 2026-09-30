import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from env_factory.sandbox_scoring import review_source_hashes


SCRIPT = Path(__file__).parents[2] / "scripts" / "sandbox" / "extract_delivery_defects.py"
SPEC = importlib.util.spec_from_file_location("extract_delivery_defects", SCRIPT)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extractor)


class ExtractDeliveryDefectsTest(unittest.TestCase):
    def test_current_review_is_used_but_changed_implementation_invalidates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task_impl.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "last_delivery_error.txt").write_text("current failure\n", encoding="utf-8")
            report = {
                "status": "fail", "review_run_id": "current",
                "source_hashes": review_source_hashes(root),
                "findings": [{"id": "DEF-003", "category": "semantic", "evidence": "wrong formula"}],
            }
            (root / "review_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(extractor.extract(root, "semantic_review")[0]["evidence"], "wrong formula")

            (root / "task_impl.py").write_text("VALUE = 2\n", encoding="utf-8")
            defect = extractor.extract(root, "acceptance")[0]
            self.assertEqual(defect["category"], "delivery_failure")
            self.assertIn("current failure", defect["evidence"])
            self.assertNotIn("wrong formula", defect["evidence"])

    def test_fallback_limits_prompt_to_current_gate_and_keeps_full_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = "old context\n" * 2000 + "first current error\nsecond current error\n"
            (root / "last_delivery_error.txt").write_text(log, encoding="utf-8")
            defect = extractor.extract(root, "mutation_testing")[0]
            self.assertIn("failed_phase=mutation_testing", defect["evidence"])
            self.assertIn("second current error", defect["evidence"])
            self.assertLess(len(defect["evidence"]), 4100)
            self.assertEqual((root / "last_delivery_error.txt").read_text(encoding="utf-8"), log)


if __name__ == "__main__":
    unittest.main()
