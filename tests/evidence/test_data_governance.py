import json
from pathlib import Path
import tempfile
import unittest

from env_factory.evidence.data_governance import (
    audit,
    provider_identity,
    valid_governance_report,
)


class DataGovernanceTest(unittest.TestCase):
    def test_platform_call_uuid_is_not_pii_but_real_identifiers_still_are(self):
        from env_factory.evidence.data_governance import scan_payloads
        value = "call-abc123456789012345678defabcdefab"
        self.assertEqual(len(value.removeprefix("call-")), 32)
        self.assertEqual(scan_payloads({"trace":{"payload":{"tool_call_id":value}}})["pii_findings"], [])
        self.assertTrue(scan_payloads({"trace":{"payload":{"tool_call_id":"123456789012345678"}}})["pii_findings"])
        self.assertTrue(scan_payloads({"text":"contact 13812345678"})["pii_findings"])

    def test_content_digests_do_not_randomly_match_pii_patterns(self):
        from env_factory.evidence.data_governance import scan_payloads

        digest = "a" * 20 + "13812345678" + "b" * 33
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            scan_payloads({"digest": digest})["pii_findings"], []
        )
        self.assertTrue(
            scan_payloads({"text": f"contact 13812345678 about {digest}"})[
                "pii_findings"
            ]
        )

    def make_root(self, root: Path, *, governance=None, material=None) -> None:
        (root / "data").mkdir()
        (root / "task.json").write_text(json.dumps({
            "public_input": {"request": "process the synthetic fixture"},
            "tools": [],
            "artifacts": {"data_manifest": governance or {
                "origin": "invalid",
            }},
        }))
        if material is not None:
            (root / "data/material.txt").write_text(material)

    @staticmethod
    def synthetic_manifest():
        return {"data_governance": {
            "origin": "model_generated_synthetic",
            "contains_real_user_data": False,
            "intended_use": "agentic_rl_training_material",
        }}

    def run_audit(self, root: Path):
        return audit(
            root,
            agent_base_url="https://agent.example/v1",
            agent_model="policy-model",
            runtime_base_url="https://runtime.example/v1",
            runtime_model="simulator-model",
        )

    def test_declared_synthetic_payload_is_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance=self.synthetic_manifest(), material="fixture row")
            report = self.run_audit(root)
            self.assertTrue(report["eligible_for_external_model_processing"])
            self.assertEqual(report["credential_findings"], [])
            self.assertEqual(report["providers"]["agent"]["host"], "agent.example")

    def test_non_synthetic_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance={"data_governance": {
                "origin": "external", "contains_real_user_data": "undetermined",
                "intended_use": "agentic_rl_training_material",
            }}, material="fixture row")
            self.assertFalse(self.run_audit(root)["eligible_for_external_model_processing"])

    def test_credential_is_reported_by_location_without_leaking_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
            self.make_root(
                root, governance=self.synthetic_manifest(), material=f"api_key={secret}"
            )
            report = self.run_audit(root)
            self.assertFalse(report["eligible_for_external_model_processing"])
            self.assertTrue(report["credential_findings"])
            self.assertNotIn(secret, json.dumps(report))

    def test_synthetic_pii_fixture_is_ineligible_for_external_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(
                root,
                governance=self.synthetic_manifest(),
                material="fictional contact: fixture@example.test",
            )
            report = self.run_audit(root)
            self.assertFalse(report["eligible_for_external_model_processing"])
            self.assertEqual(report["pii_findings"][0]["kind"], "email")
            task = json.loads((root / "task.json").read_text())
            self.assertFalse(valid_governance_report(report, root, task))

    def test_csv_fixture_cannot_bypass_pii_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance=self.synthetic_manifest())
            (root / "data/contacts.csv").write_text(
                "name,email\nfixture,fixture@example.test\n"
            )
            report = self.run_audit(root)
            self.assertFalse(report["eligible_for_external_model_processing"])
            self.assertEqual(report["pii_findings"][0]["kind"], "email")

    def test_unreadable_binary_fixture_fails_governance_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance=self.synthetic_manifest())
            report = self.run_audit(root)
            (root / "data/blob.bin").write_bytes(b"\xff\xfe\x00")
            task = json.loads((root / "task.json").read_text())
            self.assertFalse(valid_governance_report(report, root, task))

    def test_missing_synthetic_declaration_is_ineligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, material="fixture@example.test")
            report = self.run_audit(root)
            self.assertFalse(report["eligible_for_external_model_processing"])

    def test_reward_rubric_is_inside_the_scanned_outbound_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance=self.synthetic_manifest())
            task = json.loads((root / "task.json").read_text())
            task["metrics"] = [{
                "rubric": "Use Bearer abcdefghijklmnopqrstuvwxyz123456",
                "criteria": ["correct"],
                "evaluation_inputs": ["final_agent_response"],
            }]
            (root / "task.json").write_text(json.dumps(task))
            report = self.run_audit(root)
            self.assertFalse(report["eligible_for_external_model_processing"])
            self.assertIn("reward_contract", report["credential_findings"][0]["path"])

    def test_provider_models_are_required_before_external_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_root(root, governance=self.synthetic_manifest())
            report = audit(
                root,
                agent_base_url="https://agent.example/v1",
                agent_model="",
                runtime_base_url="https://runtime.example/v1",
                runtime_model="simulator-model",
            )
            self.assertFalse(report["eligible_for_external_model_processing"])

    def test_provider_identity_is_stable_and_does_not_contain_base_url(self):
        first = provider_identity("https://provider.example/v1/", "model-a")
        second = provider_identity("https://provider.example/v1", "model-a")
        self.assertEqual(first, second)
        self.assertEqual(len(first["identity_sha256"]), 64)
        self.assertNotIn("base_url", first)


if __name__ == "__main__":
    unittest.main()
