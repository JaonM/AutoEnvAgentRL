import unittest

from scripts.diagnostics.compare_development_batches import compare


def sample(*, passed: bool, seed: int = 7, audited: bool = True):
    task_id = "task-1"
    return {
        "root": "sample", "rows": {task_id: {
            "task_id": task_id, "status": "completed" if passed else "failed",
            "buildable": passed, "failure_class": None if passed else "GEN_SCHEMA",
        }},
        "manifests": {task_id: {
            "run_seed": 42, "sample_seed": seed,
            "training_category": "multi_step_agentic", "hops": 3,
        }},
        "selection": {"selected": {"multi_step_agentic": [task_id]}},
        "probes": {"rows": [{
            "task_id": task_id, "build_success": passed,
            "independent_reward_audit": audited,
        }]},
    }


class PairedDevelopmentComparisonTest(unittest.TestCase):
    def test_reports_improvement_and_audit_loss_separately(self):
        before = sample(passed=False)
        after = sample(passed=True, audited=False)
        report = compare(before, after)
        self.assertEqual(report["generation_transitions"], {"0->1": 1})
        self.assertEqual(report["build_probes"]["raw_build_transitions"], {"0->1": 1})
        self.assertEqual(report["build_probes"]["audited_end_to_end_transitions"], {"0->0": 1})

    def test_rejects_changed_sampling_identity(self):
        with self.assertRaisesRegex(ValueError, "sampling identity"):
            compare(sample(passed=False), sample(passed=True, seed=8))


if __name__ == "__main__":
    unittest.main()
