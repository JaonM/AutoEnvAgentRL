import io
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from env_factory.llm import LLMClient, LLMError, capture_llm_trace, summarize_llm_trace


class LLMNetworkRetryTest(unittest.TestCase):
    def test_kimi_k3_omits_unsupported_generation_parameters(self):
        client = LLMClient(
            api_key="test", base_url="https://api.moonshot.cn/v1", model="kimi-k3",
        )
        response = io.BytesIO(json.dumps({
            "model": "kimi-k3",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }).encode())
        with patch("env_factory.llm.urlopen", return_value=response) as request:
            client.chat(
                [{"role": "user", "content": "json"}], thinking=False,
                temperature=0.5, response_format="json_object",
            )
        body = json.loads(request.call_args.args[0].data)
        self.assertNotIn("thinking", body)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_kimi_k3_reasoning_effort_is_explicit_when_configured(self):
        client = LLMClient(
            api_key="test", base_url="https://api.moonshot.cn/v1", model="kimi-k3",
        )
        response = io.BytesIO(json.dumps({
            "model": "kimi-k3",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }).encode())
        with patch.dict(os.environ, {"KIMI_K3_REASONING_EFFORT": "low"}), \
                patch("env_factory.llm.urlopen", return_value=response) as request:
            client.chat([{"role": "user", "content": "json"}])
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "low")

    def client(self):
        return LLMClient(
            api_key="test", base_url="https://example.invalid/v1", model="test-model",
            network_retries=1,
        )

    def test_retryable_http_error_keeps_one_candidate_and_records_latency(self):
        unavailable = HTTPError(
            "https://example.invalid/v1/chat/completions", 429, "rate limit",
            {"Retry-After": "0"}, io.BytesIO(b'{"error":"rate limit"}'),
        )
        response = io.BytesIO(json.dumps({
            "model": "test-model", "id": "secret-response-id",
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        }).encode())
        with patch("env_factory.llm.urlopen", side_effect=[unavailable, response]) as request, \
                patch("env_factory.llm.time.sleep") as sleep, capture_llm_trace() as trace:
            result = self.client().chat([{"role": "user", "content": "hello"}])
        self.assertEqual(result.content, "done")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(0.0)
        summary = summarize_llm_trace(trace)
        self.assertEqual(summary["responses"], 1)
        self.assertEqual(summary["network_retries"], 1)
        self.assertGreaterEqual(summary["request_seconds"], 0)
        self.assertNotIn("secret-response-id", str(summary))

    def test_nonretryable_http_error_fails_immediately(self):
        invalid = HTTPError(
            "https://example.invalid/v1/chat/completions", 400, "bad request",
            {}, io.BytesIO(b'{"error":"bad request"}'),
        )
        with patch("env_factory.llm.urlopen", side_effect=invalid) as request, \
                patch("env_factory.llm.time.sleep") as sleep:
            with self.assertRaisesRegex(LLMError, "HTTP 400"):
                self.client().chat([{"role": "user", "content": "hello"}])
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
