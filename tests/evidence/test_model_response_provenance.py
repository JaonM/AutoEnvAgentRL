import copy
import unittest

from env_factory.evidence.model_response_provenance import response_provenance


class ModelResponseProvenanceTest(unittest.TestCase):
    def rollout(self):
        return {
            "episodes": [{
                "transitions": [{"action": {"kind": "respond"}}],
                "usage": [{
                    "response_model": "actual-agent-snapshot",
                    "response_id_sha256": "a" * 64,
                    "finish_reason": "stop",
                    "token_usage": {"total_tokens": 11},
                }],
                "replay": {"events": [{
                    "event": "runtime_llm_call",
                    "payload": {"summary": {
                        "version": "1.0",
                        "responses": 1,
                        "mock_responses": 0,
                        "models": {"actual-runtime-snapshot": 1},
                        "usage": {"total_tokens": 5},
                        "response_id_sha256": ["b" * 64],
                    }},
                }]},
            }],
        }

    def test_recomputes_actual_models_from_episode_evidence(self):
        report = response_provenance(self.rollout())
        self.assertTrue(report["verified"])
        self.assertEqual(
            report["agent_response_models"], {"actual-agent-snapshot": 1}
        )
        self.assertEqual(
            report["runtime_response_models"], {"actual-runtime-snapshot": 1}
        )

    def test_rejects_missing_agent_metadata_or_mock_runtime_response(self):
        for mutation in ("agent", "runtime"):
            rollout = copy.deepcopy(self.rollout())
            if mutation == "agent":
                rollout["episodes"][0]["usage"][0] = {}
            else:
                rollout["episodes"][0]["replay"]["events"][0]["payload"][
                    "summary"
                ]["mock_responses"] = 1
            self.assertFalse(response_provenance(rollout)["verified"])

    def test_tool_only_failed_episode_need_not_invent_runtime_call(self):
        rollout = self.rollout()
        rollout["episodes"].append({
            "transitions": [{"action": {"kind": "tool"}}],
            "usage": [{
                "response_model": "actual-agent-snapshot",
                "response_id_sha256": None,
                "finish_reason": "stop",
                "token_usage": {},
            }],
            "replay": {"events": []},
        })
        report = response_provenance(rollout)
        self.assertTrue(report["verified"])
        self.assertEqual(
            report["agent_response_models"], {"actual-agent-snapshot": 2}
        )


if __name__ == "__main__":
    unittest.main()
