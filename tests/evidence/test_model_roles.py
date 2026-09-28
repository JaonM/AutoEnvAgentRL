import unittest

from env_factory.evidence.model_roles import resolve_model_roles


class ModelRolesTest(unittest.TestCase):
    def test_unconfigured_roles_fall_back_field_by_field(self):
        roles = resolve_model_roles({
            "LLM_API_KEY": "generation-key",
            "LLM_BASE_URL": "https://generation.example/v1/",
            "LLM_MODEL": "generation-model",
            "LLM_TIMEOUT": "42",
            "ROLLOUT_LLM_MODEL": "policy-model",
            "SANDBOX_LLM_BASE_URL": "https://runtime.example/v1",
        })
        self.assertEqual(roles["agent"], {
            "api_key": "generation-key",
            "base_url": "https://generation.example/v1",
            "model": "policy-model",
            "timeout_seconds": "42",
            "allowed_response_models": ["policy-model"],
        })
        self.assertEqual(roles["runtime"], {
            "api_key": "generation-key",
            "base_url": "https://runtime.example/v1",
            "model": "generation-model",
            "timeout_seconds": "42",
            "allowed_response_models": ["generation-model"],
        })

    def test_all_roles_can_be_independent(self):
        roles = resolve_model_roles({
            "LLM_API_KEY": "generation-key",
            "LLM_MODEL": "generation-model",
            "ROLLOUT_LLM_API_KEY": "policy-key",
            "ROLLOUT_LLM_BASE_URL": "https://policy.example/v1",
            "ROLLOUT_LLM_MODEL": "policy-model",
            "ROLLOUT_LLM_TIMEOUT_SECONDS": "30",
            "SANDBOX_LLM_API_KEY": "runtime-key",
            "SANDBOX_LLM_BASE_URL": "https://runtime.example/v1",
            "SANDBOX_LLM_MODEL": "runtime-model",
            "SANDBOX_LLM_TIMEOUT_SECONDS": "90",
        })
        self.assertEqual(roles["agent"]["api_key"], "policy-key")
        self.assertEqual(roles["agent"]["model"], "policy-model")
        self.assertEqual(roles["runtime"]["api_key"], "runtime-key")
        self.assertEqual(roles["runtime"]["model"], "runtime-model")

    def test_response_model_allowlists_are_explicit_and_role_scoped(self):
        roles = resolve_model_roles({
            "LLM_MODEL": "generation-alias",
            "LLM_ALLOWED_RESPONSE_MODELS": "gen-v1, gen-v2,gen-v1",
            "ROLLOUT_LLM_MODEL": "policy-alias",
            "ROLLOUT_LLM_ALLOWED_RESPONSE_MODELS": "policy-v1,policy-v2",
            "SANDBOX_LLM_MODEL": "runtime-alias",
        })
        self.assertEqual(
            roles["generation"]["allowed_response_models"], ["gen-v1", "gen-v2"]
        )
        self.assertEqual(
            roles["agent"]["allowed_response_models"], ["policy-v1", "policy-v2"]
        )
        self.assertEqual(
            roles["runtime"]["allowed_response_models"], ["runtime-alias"]
        )


if __name__ == "__main__":
    unittest.main()
