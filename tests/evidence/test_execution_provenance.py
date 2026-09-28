from pathlib import Path
import unittest

from env_factory.evidence.execution_provenance import (
    collect_execution_provenance,
    valid_execution_provenance,
    verify_execution_provenance,
)


ROOT = Path(__file__).resolve().parents[2]


class ExecutionProvenanceTest(unittest.TestCase):
    def test_snapshot_is_complete_stable_and_contains_no_local_paths(self):
        first = collect_execution_provenance(ROOT)
        second = collect_execution_provenance(ROOT)
        self.assertEqual(first, second)
        self.assertTrue(first["complete"])
        self.assertEqual(len(first["identity_sha256"]), 64)
        self.assertTrue(valid_execution_provenance(first))
        rendered = str(first)
        self.assertNotIn(str(ROOT), rendered)
        self.assertNotIn(str(Path.home()), rendered)

    def test_verifier_detects_lock_and_runtime_tampering(self):
        recorded = collect_execution_provenance(ROOT)
        recorded["uv_lock_sha256"] = "0" * 64
        report = verify_execution_provenance(ROOT, recorded)
        self.assertFalse(report["verified"])
        self.assertIn("uv_lock_sha256", report["differences"])
        self.assertFalse(valid_execution_provenance(recorded))


if __name__ == "__main__":
    unittest.main()
