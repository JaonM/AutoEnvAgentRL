import unittest

from env_factory.evidence.portable_metadata import (
    audit_portable_metadata,
    sanitize_portable_metadata,
)


class PortableMetadataTest(unittest.TestCase):
    def test_sanitizer_removes_paths_without_touching_urls(self):
        value = {
            "task_path": "/Users/person/task.json",
            "diagnostic": ["/tmp/run", "https://provider.example/v1"],
            "nested": {"score": 9.0},
        }
        sanitized = sanitize_portable_metadata(value)
        self.assertNotIn("task_path", sanitized)
        self.assertEqual(sanitized["diagnostic"][0], "<redacted-local-path>")
        self.assertEqual(
            sanitized["diagnostic"][1], "https://provider.example/v1"
        )
        self.assertTrue(audit_portable_metadata(sanitized)["safe"])

    def test_audit_reports_locations_without_echoing_secrets(self):
        value = {
            "diagnostic": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "output": "/private/tmp/run",
            "owner": "private.person@example.com",
        }
        report = audit_portable_metadata(value)
        self.assertFalse(report["safe"])
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", str(report))
        self.assertTrue(report["credential_findings"])
        self.assertTrue(report["local_path_findings"])
        self.assertTrue(report["pii_findings"])


if __name__ == "__main__":
    unittest.main()
