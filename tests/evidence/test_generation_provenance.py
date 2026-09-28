import unittest

from env_factory.evidence.generation_provenance import (
    generation_provenance_snapshot,
    valid_generation_provenance,
)
from env_factory.llm import summarize_llm_trace


class GenerationProvenanceTest(unittest.TestCase):
    def manifest(self):
        return {
            "version": "2.0",
            "status": "completed",
            "task_id": "task-7",
            "batch_index": 7,
            "run_seed": 100,
            "sample_seed": 200,
            "task_sha256": "f" * 64,
            "training_category": "multi_step_agentic",
            "resolved_task_intent": "fixture",
            "hops": 3,
            "successful_attempt": 2,
            "generator_provider": {
                "host": "generator.example",
                "model": "generator-model",
                "identity_sha256": "a" * 64,
            },
            "generation_settings": {
                "route_attempt_limit": 3,
                "timeout_seconds": 60.0,
                "network_retries": 2,
            },
            "attempts": [
                {
                    "attempt": 1, "seed": 200, "status": "rejected",
                    "llm_trace": {
                        "version": "1.0", "responses": 0, "models": {},
                        "finish_reasons": {}, "usage": {},
                        "response_id_sha256": [],
                    },
                },
                {
                    "attempt": 2, "seed": 201, "status": "completed",
                    "llm_trace": {
                        "version": "1.0", "responses": 1,
                        "models": {"generator-model": 1},
                        "finish_reasons": {"stop": 1},
                        "usage": {"total_tokens": 12},
                        "response_id_sha256": ["b" * 64],
                    },
                },
            ],
        }

    def test_completed_manifest_compiles_to_portable_snapshot(self):
        manifest = self.manifest()
        snapshot = generation_provenance_snapshot(
            manifest,
            {"training_category": "multi_step_agentic"},
            expected_provider=manifest["generator_provider"],
            expected_task_sha256="f" * 64,
        )
        self.assertTrue(valid_generation_provenance(snapshot))
        self.assertEqual(snapshot["successful_attempt"], 2)

    def test_seed_provider_and_trace_tampering_are_rejected(self):
        for mutation in ("seed", "provider", "trace", "task"):
            manifest = self.manifest()
            if mutation == "seed":
                manifest["attempts"][1]["seed"] = 999
            elif mutation == "provider":
                manifest["generator_provider"]["identity_sha256"] = "bad"
            else:
                if mutation == "trace":
                    manifest["attempts"][1]["llm_trace"]["responses"] = 2
                else:
                    manifest["task_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                generation_provenance_snapshot(
                    manifest, {"training_category": "multi_step_agentic"},
                    expected_task_sha256="f" * 64,
                )

    def test_trace_summary_hashes_ids_and_never_contains_payloads(self):
        summary = summarize_llm_trace([{
            "model": "model-a",
            "response_id": "provider-response-secret-id",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "prompt": "must not survive",
            "content": "must not survive",
        }])
        rendered = str(summary)
        self.assertNotIn("provider-response-secret-id", rendered)
        self.assertNotIn("must not survive", rendered)
        self.assertEqual(summary["responses"], 1)
        self.assertEqual(summary["usage"]["prompt_tokens"], 5)
        self.assertEqual(len(summary["response_id_sha256"][0]), 64)


if __name__ == "__main__":
    unittest.main()
