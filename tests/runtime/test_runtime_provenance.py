import json
from pathlib import Path
import tempfile
import unittest

from env_factory.evidence.runtime_provenance import valid_container_rollout_execution
from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE,
    docker_build_context_digest,
)


class RuntimeProvenanceTest(unittest.TestCase):
    def evidence(self, root: Path):
        image_id = "sha256:" + "a" * 64
        (root / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
        (root / "docker_image_metadata.json").write_text(json.dumps({
            "version": "5.0",
            "image_id": image_id,
            "runtime_user": "sandbox",
            "build_context_sha256": docker_build_context_digest(root),
            "smoke_test": {
                "passed": True,
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
                "service_health": True,
                "runtime_tmpfs": {
                    "path": "/app/.runtime",
                    "uid": 10001,
                    "gid": 10001,
                    "mode": "0700",
                },
            },
        }))
        return {
            "runtime_execution": {
                "version": "1.0",
                "mode": "docker_http",
                "container_image_id": image_id,
                "transport": "loopback_http",
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
            },
        }

    def test_rollout_is_bound_to_validated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = self.evidence(root)
            self.assertTrue(valid_container_rollout_execution(rollout, root))
            rollout["runtime_execution"]["container_image_id"] = "sha256:" + "b" * 64
            self.assertFalse(valid_container_rollout_execution(rollout, root))

    def test_in_process_rollout_is_not_production_container_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.evidence(root)
            rollout = {"runtime_execution": {
                "version": "1.0", "mode": "in_process", "container_image_id": None,
            }}
            self.assertFalse(valid_container_rollout_execution(rollout, root))


if __name__ == "__main__":
    unittest.main()
