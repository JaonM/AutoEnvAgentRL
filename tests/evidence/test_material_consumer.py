import unittest

from env_factory.evidence.material_consumer import (
    BUNDLE_VERSION,
    DATASET_SPLITS,
    assign_dataset_splits,
    consumer_contract,
    consumer_record_errors,
)
from env_factory.tasks.task_similarity import task_family_ids


class MaterialConsumerTest(unittest.TestCase):
    def test_split_assignment_is_deterministic_and_stratified(self):
        items = [
            {"item_id": f"{category}-{index:03d}", "category": category}
            for category in ("direct_response", "simple_agentic", "multi_step_agentic")
            for index in range(20)
        ]
        first = assign_dataset_splits(items)
        second = assign_dataset_splits(list(reversed(items)))
        self.assertEqual(first, second)
        for category in {item["category"] for item in items}:
            counts = {
                split: sum(
                    first[item["item_id"]] == split
                    for item in items if item["category"] == category
                )
                for split in DATASET_SPLITS
            }
            self.assertEqual(counts, {"train": 16, "validation": 2, "test": 2})

    def test_tiny_category_stays_train_only(self):
        assignments = assign_dataset_splits([
            {"item_id": "one", "category": "simple_agentic"},
            {"item_id": "two", "category": "simple_agentic"},
        ])
        self.assertEqual(set(assignments.values()), {"train"})

    def test_near_duplicate_family_is_kept_in_one_split(self):
        family_inputs = [
            {"item_id": "variant-a", "task": "为客户 100 创建订单并核验库存"},
            {"item_id": "variant-b", "task": "为客户 200 创建订单并核验库存"},
            {"item_id": "different", "task": "汇总航班延误并发送报告"},
        ]
        families = task_family_ids(family_inputs)
        self.assertEqual(families["variant-a"], families["variant-b"])
        self.assertNotEqual(families["variant-a"], families["different"])
        items = [
            {
                "item_id": item["item_id"],
                "category": "multi_step_agentic",
                "task_family_id": families[item["item_id"]],
            }
            for item in family_inputs
        ]
        assignments = assign_dataset_splits(items)
        self.assertEqual(assignments["variant-a"], assignments["variant-b"])

    def test_consumer_schema_makes_split_part_of_every_record(self):
        contract = consumer_contract()
        records = contract["records"]
        self.assertEqual(contract["bundle_version"], BUNDLE_VERSION)
        self.assertEqual(contract["experiment_contract"], "experiment_contract.json")
        self.assertIn(
            "frozen_experiment_configuration_is_portable_and_bound",
            contract["invariants"],
        )
        self.assertIn("task_lineage", contract["environments"])
        self.assertIn("dockerignore", contract["environments"])
        self.assertIn("generated_task_equals_runtime_task", contract["invariants"])
        self.assertEqual(records["split_unit"], "item_id")
        self.assertIn("split", records["required_fields"])
        self.assertIn("task_family_id", records["required_fields"])
        self.assertIn("generation_model", records["required_fields"])
        self.assertIn("trajectory_role", records["required_fields"])
        self.assertEqual(
            contract["material_roles"]["exported_trajectories"]
                ["direct_policy_optimization"],
            "not_certified",
        )
        self.assertEqual(records["near_duplicate_split_unit"], "task_family_id")
        self.assertEqual(
            records["json_schema"]["properties"]["split"]["enum"],
            list(DATASET_SPLITS),
        )
        self.assertFalse(
            records["json_schema"]["properties"]["transition"]["properties"]
            ["agent_input"]["items"]["additionalProperties"]
        )

    @staticmethod
    def valid_record():
        return {
            "schema_version": "2.0",
            "item_id": "a" * 16 + "-" + "b" * 16,
            "task_sha256": "c" * 64,
            "category": "simple_agentic",
            "episode_index": 0,
            "episode_seed": 7,
            "episode_success": True,
            "episode_termination": "completed",
            "episode_initial_reward": 0.0,
            "episode_final_reward": 1.0,
            "agent_model": "policy-model",
            "runtime_model": "simulator-model",
            "agent_usage": {},
            "trajectory_role": "certification_evidence",
            "direct_training_status": "not_certified",
            "split": "train",
            "task_family_id": "d" * 64,
            "generation_model": "generator-model",
            "generation_provider_identity_sha256": "e" * 64,
            "generation_sample_seed": 9,
            "transition": {
                "step": 0,
                "agent_input": [{"role": "user", "content": "help"}],
                "assistant_output": '{"kind":"respond","content":"done"}',
                "observation": {},
                "action": {"kind": "respond", "content": "done"},
                "result": {"status": 200},
                "next_observation": {},
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
            },
        }

    def test_consumer_record_validator_rejects_nested_policy_message_fields(self):
        record = self.valid_record()
        self.assertEqual(consumer_record_errors(record), [])
        record["transition"]["agent_input"][0]["trainer_hint"] = "hidden"
        self.assertIn(
            "transition:agent_input[0]:fields",
            consumer_record_errors(record),
        )

    def test_consumer_record_validator_rejects_invalid_numeric_values(self):
        record = self.valid_record()
        record["episode_final_reward"] = float("nan")
        record["transition"]["step"] = -1
        errors = consumer_record_errors(record)
        self.assertIn("record:episode_final_reward", errors)
        self.assertIn("transition:step", errors)


if __name__ == "__main__":
    unittest.main()
