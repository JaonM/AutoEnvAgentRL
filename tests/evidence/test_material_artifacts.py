from pathlib import Path
import tempfile
import unittest

from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE,
    docker_context_errors,
    portable_artifact_digests,
)


class MaterialArtifactsTest(unittest.TestCase):
    def test_portable_inventory_covers_nested_rebuild_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
            package = root / "business" / "rules"
            package.mkdir(parents=True)
            (package / "engine.py").write_text("VALUE = 1\n")
            (package / "config.json").write_text('{"enabled":true}\n')
            (root / "build.log").write_text("volatile\n")
            (root / "review_agent.stdout").write_text("volatile\n")
            (root / "live_rollout.json").write_text("{}\n")
            inventory = portable_artifact_digests(root)
            self.assertIn(".dockerignore", inventory)
            self.assertIn("business/rules/engine.py", inventory)
            self.assertIn("business/rules/config.json", inventory)
            self.assertNotIn("build.log", inventory)
            self.assertNotIn("review_agent.stdout", inventory)
            self.assertNotIn("live_rollout.json", inventory)

    def test_context_rejects_dockerignore_drift_and_sensitive_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".dockerignore").write_text(".git\n")
            (root / "credentials.json").write_text("{}\n")
            errors = docker_context_errors(root)
            self.assertIn("dockerignore_contract", errors)
            self.assertIn("sensitive_context_file:credentials.json", errors)

    def test_portable_inventory_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
            target = root / "target.py"
            target.write_text("VALUE = 1\n")
            (root / "linked.py").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "contains symlink"):
                portable_artifact_digests(root)


if __name__ == "__main__":
    unittest.main()
