import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from env_factory.tasks.task_portability import prepare_sandbox_task, valid_task_lineage


class TaskPortabilityTest(unittest.TestCase):
    def portable_task(self, root: Path) -> Path:
        business = root / "data" / "business_data"
        simulation = root / "data" / "user_simulation"
        business.mkdir(parents=True)
        simulation.mkdir(parents=True)
        (business / "data_document.md").write_text("fixture\n")
        (simulation / "user_profiles.json").write_text("[]\n")
        task = {
            "task": "use the fixture",
            "artifacts": {
                "data_manifest": {
                    "root": "data/business_data",
                    "document_file": "data_document.md",
                },
                "user_simulation_manifest": {
                    "root": "data/user_simulation",
                    "profiles_file": "user_profiles.json",
                },
            },
            "environment": [],
        }
        path = root / "task.json"
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
        return path

    def test_portable_roots_preserve_exact_task_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "sandbox"
            source.mkdir()
            task = self.portable_task(source)
            report = prepare_sandbox_task(task, output)
            self.assertTrue(report["identity_preserved"])
            self.assertEqual(report["legacy_manifest_relocations"], [])
            self.assertEqual(task.read_bytes(), (output / "task.json").read_bytes())
            self.assertTrue((output / "data/business_data/data_document.md").is_file())
            self.assertTrue((output / "data/user_simulation/user_profiles.json").is_file())
            self.assertEqual(
                report["runtime_task_sha256"],
                hashlib.sha256(task.read_bytes()).hexdigest(),
            )
            self.assertTrue(valid_task_lineage(
                report, task, output / "task.json"
            ))

    def test_lineage_report_cannot_hide_runtime_task_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "sandbox"
            source.mkdir()
            task = self.portable_task(source)
            report = prepare_sandbox_task(task, output)
            runtime = json.loads((output / "task.json").read_text())
            runtime["task"] = "different task"
            (output / "task.json").write_text(json.dumps(runtime))
            self.assertFalse(valid_task_lineage(
                report, task, output / "task.json"
            ))

    def test_parent_traversal_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            task.write_text(json.dumps({
                "artifacts": {"data_manifest": {"root": "../private"}},
            }))
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                prepare_sandbox_task(task, root / "sandbox")

    def test_legacy_absolute_root_is_relocated_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "legacy-data"
            fixture.mkdir()
            (fixture / "data_document.md").write_text("fixture\n")
            source = root / "source"
            source.mkdir()
            task = source / "task.json"
            task.write_text(json.dumps({
                "artifacts": {"data_manifest": {"root": str(fixture)}},
                "task_spec": {"environment_contract": {"initial_fixture": {
                    "manifest": {"root": str(fixture)}, "table_names": [],
                }}},
                "environment": [],
            }))
            output = root / "sandbox"
            report = prepare_sandbox_task(task, output)
            self.assertFalse(report["identity_preserved"])
            self.assertEqual(report["legacy_manifest_relocations"], ["data_manifest"])
            runtime = json.loads((output / "task.json").read_text())
            self.assertEqual(
                runtime["artifacts"]["data_manifest"]["root"],
                "data/business_data",
            )
            self.assertEqual(
                runtime["task_spec"]["environment_contract"]["initial_fixture"]["manifest"],
                runtime["artifacts"]["data_manifest"],
            )

    def test_noncanonical_relative_root_is_relocated_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            legacy = source / "fixtures"
            legacy.mkdir(parents=True)
            (legacy / "data_document.md").write_text("fixture\n")
            task = source / "task.json"
            task.write_text(json.dumps({
                "artifacts": {"data_manifest": {"root": "fixtures"}},
                "environment": [],
            }))
            output = root / "sandbox"
            report = prepare_sandbox_task(task, output)
            self.assertFalse(report["identity_preserved"])
            self.assertEqual(report["legacy_manifest_relocations"], ["data_manifest"])
            self.assertTrue(
                (output / "data/business_data/data_document.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
