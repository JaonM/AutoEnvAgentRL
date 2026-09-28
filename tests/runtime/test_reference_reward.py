import unittest

from scripts.sandbox.reference_reward import evaluate


class ReferenceRewardTest(unittest.TestCase):
    def test_independent_formula(self):
        task = {"metrics": [
            {"id": "process", "category": "process", "weight": 0.3, "score_range": [0, 1]},
            {"id": "outcome", "category": "outcome", "weight": 0.7, "score_range": [0, 1]},
            {"id": "penalty", "category": "penalty", "weight": 1.0, "score_range": [-1, 0]},
        ]}
        result = evaluate(task, {"process": 1, "outcome": 0.5, "penalty": -0.2})
        self.assertAlmostEqual(result["reward"], 0.3 + 0.35 - 0.2)


if __name__ == "__main__":
    unittest.main()
