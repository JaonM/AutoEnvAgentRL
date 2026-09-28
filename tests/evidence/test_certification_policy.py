import unittest

from env_factory.evidence.certification_policy import (
    canonical_certification_policy,
    policy_for_experiment,
    valid_certification_policy,
)


class CertificationPolicyTest(unittest.TestCase):
    def test_canonical_policy_is_exact_and_threshold_bound(self):
        policy = canonical_certification_policy()
        self.assertTrue(valid_certification_policy(policy))
        stricter = canonical_certification_policy(score_threshold=8.5)
        self.assertTrue(valid_certification_policy(stricter))
        self.assertFalse(valid_certification_policy(
            stricter, expected_threshold=8.0
        ))

    def test_policy_cannot_be_weakened_or_extended(self):
        policy = canonical_certification_policy()
        policy["min_tasks"] = 1
        self.assertFalse(valid_certification_policy(policy))
        policy = canonical_certification_policy()
        policy["unreviewed_override"] = True
        self.assertFalse(valid_certification_policy(policy))
        with self.assertRaises(ValueError):
            canonical_certification_policy(score_threshold=7.99)

    def test_experiment_threshold_must_be_numeric(self):
        policy, valid = policy_for_experiment({"threshold": 8.5})
        self.assertTrue(valid)
        self.assertEqual(policy["score_threshold"], 8.5)
        fallback, valid = policy_for_experiment({"threshold": "8.5"})
        self.assertFalse(valid)
        self.assertEqual(fallback, canonical_certification_policy())
        fallback, valid = policy_for_experiment(None)
        self.assertFalse(valid)
        self.assertEqual(fallback, canonical_certification_policy())


if __name__ == "__main__":
    unittest.main()
