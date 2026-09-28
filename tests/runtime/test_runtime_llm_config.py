import os
import unittest
from unittest.mock import patch

from env_factory.runtime_llm import RuntimeLLMConfig, RuntimeLLMError


class RuntimeConfigTest(unittest.TestCase):
    def test_generation_defaults(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_MODEL": "generation-model",
                                   "LLM_BASE_URL": "https://provider.test/v1/", "LLM_TIMEOUT": "45"}, clear=True):
            config = RuntimeLLMConfig.from_env()
        self.assertEqual(config.api_key, "test")
        self.assertEqual(config.model, "generation-model")
        self.assertEqual(config.base_url, "https://provider.test/v1")
        self.assertEqual(config.timeout_seconds, 45)
        self.assertFalse(config.mock)

    def test_explicit_overrides_and_empty_values(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_MODEL": "generation-model",
                                   "LLM_BASE_URL": "https://provider.test/v1", "SANDBOX_LLM_API_KEY": "",
                                   "SANDBOX_LLM_MODEL": "runtime-model", "SANDBOX_LLM_BASE_URL": "",
                                   "SANDBOX_LLM_TIMEOUT_SECONDS": "10"}, clear=True):
            config = RuntimeLLMConfig.from_env()
        self.assertEqual(config.api_key, "test")
        self.assertEqual(config.model, "runtime-model")
        self.assertEqual(config.base_url, "https://provider.test/v1")
        self.assertEqual(config.timeout_seconds, 10)

    def test_no_configuration_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeLLMError):
            RuntimeLLMConfig.from_env()

    def test_mock_does_not_require_credentials(self):
        with patch.dict(os.environ, {"SANDBOX_EVALUATOR_MOCK": "1"}, clear=True):
            self.assertTrue(RuntimeLLMConfig.from_env().mock)
