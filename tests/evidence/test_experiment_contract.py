import unittest

from env_factory.evidence.experiment_contract import (
    build_experiment_contract,
    valid_experiment_contract,
)
from env_factory.evidence.material_artifacts import digest_json


def production_config():
    return {
        "certification_profile": "production",
        "threshold": 8.0,
        "validation": "live",
        "sandbox_runtime": "docker",
        "holdout_count": 300,
        "holdout_batches": 3,
        "holdout_rollout_episodes": 10,
        "source_digest": "1" * 64,
        "input_digests": ["2" * 64],
        "project": "/private/build/EnvFactory",
        "task_paths": ["/private/tasks/task-1.json"],
        "bundle_attestation_key_identity_sha256": "3" * 64,
        "generation_provider": {
            "host": "generation.example", "model": "generator",
            "identity_sha256": "4" * 64,
        },
        "rollout_provider": {
            "host": "agent.example", "model": "agent",
            "identity_sha256": "5" * 64,
        },
        "runtime_provider": {
            "host": "runtime.example", "model": "judge",
            "identity_sha256": "6" * 64,
        },
        "generation_model": "generator",
        "rollout_model": "agent",
        "runtime_model": "judge",
        "generation_allowed_response_models": ["generator"],
        "rollout_allowed_response_models": ["agent"],
        "runtime_allowed_response_models": ["judge"],
    }


class ExperimentContractTest(unittest.TestCase):
    def test_contract_removes_local_paths_and_binds_source(self):
        config = production_config()
        contract = build_experiment_contract(config)
        self.assertNotIn("project", contract["configuration"])
        self.assertNotIn("task_paths", contract["configuration"])
        self.assertTrue(valid_experiment_contract(
            contract,
            expected_source_config_sha256=digest_json(config),
        ))

    def test_contract_rejects_semantic_and_privacy_drift(self):
        config = production_config()
        contract = build_experiment_contract(config)
        contract["configuration"]["holdout_count"] = 1
        contract["configuration_sha256"] = digest_json(
            contract["configuration"]
        )
        self.assertFalse(valid_experiment_contract(contract))

        config = production_config()
        config["hypothesis"] = "contact operator@example.com"
        self.assertFalse(valid_experiment_contract(
            build_experiment_contract(config)
        ))

        config = production_config()
        config["runtime_provider"]["host"] = config["rollout_provider"]["host"]
        self.assertFalse(valid_experiment_contract(
            build_experiment_contract(config)
        ))


if __name__ == "__main__":
    unittest.main()
