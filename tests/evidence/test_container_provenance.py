import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
from types import SimpleNamespace

from env_factory.evidence.container_provenance import inspect_local_image, verify_container_provenance
from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE,
    docker_build_context_digest,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resolve_container_image", ROOT / "scripts/sandbox/resolve_container_image.py"
)
resolver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(resolver)


class ContainerProvenanceTest(unittest.TestCase):
    def test_local_image_inspection_reads_daemon_identity(self):
        image = {
            "Id": "sha256:" + "b" * 64, "Os": "linux", "Architecture": "arm64",
            "Config": {"User": "sandbox"},
        }
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout=json.dumps(image))
        self.assertEqual(inspect_local_image("fixture", runner=runner), {
            "image_id": image["Id"],
            "platform": {"os": "linux", "architecture": "arm64"},
            "runtime_user": "sandbox",
        })
        self.assertEqual(calls[0][:4], ["docker", "image", "inspect", "fixture"])

    def test_local_image_identity_must_match_recorded_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            observed = {
                "image_id": metadata["image_id"],
                "platform": metadata["platform"],
                "runtime_user": metadata["runtime_user"],
            }
            self.assertTrue(verify_container_provenance(
                root, expected_tag="fixture", observed_image=observed,
                require_local_image=True,
            )["verified"])
            self.assertIn("local_image_unavailable", verify_container_provenance(
                root, observed_image=None, require_local_image=True,
            )["failed_gates"])
            self.assertIn("local_image_id", verify_container_provenance(
                root, observed_image={**observed, "image_id": "sha256:" + "c" * 64},
                require_local_image=True,
            )["failed_gates"])

    def test_cached_resolution_requires_matching_repository_and_platform(self):
        digest = "sha256:" + "c" * 64
        inspect = [{"Os": "linux", "Architecture": "arm64", "RepoDigests": [
            "registry.example/python@" + digest,
        ]}]
        self.assertEqual(resolver.resolve_cached_reference(
            inspect, "registry.example/python:3.14", "linux", "aarch64"
        ), "registry.example/python@" + digest)
        self.assertIsNone(resolver.resolve_cached_reference(
            inspect, "registry.example/python:3.14", "linux", "amd64"
        ))
        self.assertIsNone(resolver.resolve_cached_reference(
            inspect, "registry.example/other:3.14", "linux", "arm64"
        ))
        self.assertIsNone(resolver.resolve_cached_reference(
            inspect, "registry.example/python@sha256:" + "d" * 64, "linux", "arm64"
        ))

    def test_manifest_resolution_selects_only_the_requested_platform(self):
        manifest = [
            {"Descriptor": {
                "digest": "sha256:" + "a" * 64,
                "platform": {"os": "linux", "architecture": "amd64"},
            }},
            {"Descriptor": {
                "digest": "sha256:" + "b" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
            }},
        ]
        self.assertEqual(
            resolver.resolve_reference(
                manifest, "registry.example/python:3.14", "linux", "aarch64"
            ),
            "registry.example/python@sha256:" + "b" * 64,
        )
        self.assertIsNone(
            resolver.resolve_reference(
                manifest, "registry.example/python:3.14", "windows", "arm64"
            )
        )

    def fixture(self, root: Path):
        (root / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
        image = "registry.example/python@sha256:" + "a" * 64
        dockerfile = root / "Dockerfile"
        requirements = root / "requirements-dev.txt"
        dockerfile.write_text(f"FROM {image}\nUSER sandbox\n")
        requirements.write_text("pluggy==1.6.0\npytest==9.1.1\n")
        packages = root / "python_packages.json"
        packages.write_text(json.dumps({
            "version": "1.0",
            "packages": [
                {"name": "pluggy", "version": "1.6.0"},
                {"name": "pytest", "version": "9.1.1"},
            ],
        }))
        metadata = {
            "version": "5.0",
            "tag": "fixture",
            "base_image": image,
            "image_id": "sha256:" + "b" * 64,
            "runtime_user": "sandbox",
            "platform": {"os": "linux", "architecture": "amd64"},
            "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
            "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "python_packages_sha256": hashlib.sha256(packages.read_bytes()).hexdigest(),
            "build_context_sha256": docker_build_context_digest(root),
            "smoke_test": {
                "passed": True,
                "network": "none",
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
        }
        (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
        return metadata

    def test_verified_provenance_binds_pinned_inputs_and_security_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(verify_container_provenance(
                root, expected_tag="fixture"
            )["verified"])

    def test_pytest_only_smoke_cannot_replace_service_startup_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            metadata["smoke_test"].pop("service_health")
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("container_smoke_test", report["failed_gates"])

    def test_dockerignore_drift_breaks_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / ".dockerignore").write_text(".git\n")
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("dockerignore_contract", report["failed_gates"])

    def test_nested_source_change_breaks_build_context_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            nested = root / "business" / "rules"
            nested.mkdir(parents=True)
            (nested / "engine.py").write_text("VALUE = 2\n")
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("build_context_digest", report["failed_gates"])

    def test_floating_base_image_and_dependency_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            (root / "Dockerfile").write_text("FROM python:3.14-slim\nUSER sandbox\n")
            (root / "requirements-dev.txt").write_text("pytest>=8,<10\n")
            metadata["dockerfile_sha256"] = hashlib.sha256(
                (root / "Dockerfile").read_bytes()
            ).hexdigest()
            metadata["requirements_sha256"] = hashlib.sha256(
                (root / "requirements-dev.txt").read_bytes()
            ).hexdigest()
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("dockerfile_base_image", report["failed_gates"])
            self.assertIn("dependency_pins", report["failed_gates"])

    def test_metadata_cannot_hide_changed_inputs_or_root_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            (root / "requirements-dev.txt").write_text("pytest==9.1.0\n")
            metadata["runtime_user"] = "root"
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertIn("requirements_digest", report["failed_gates"])
            self.assertIn("non_root_user", report["failed_gates"])

    def test_expected_attempt_tag_and_base_image_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            metadata["tag"] = "other-attempt"
            metadata["base_image"] = "registry.example/other@sha256:" + "c" * 64
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root, expected_tag="fixture")
            self.assertIn("image_tag", report["failed_gates"])
            self.assertIn("base_image_mismatch", report["failed_gates"])

    def test_inventory_must_match_pinned_direct_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            inventory = root / "python_packages.json"
            inventory.write_text(json.dumps({
                "version": "1.0",
                "packages": [
                    {"name": "pluggy", "version": "1.6.0"},
                    {"name": "pytest", "version": "0.0.0"},
                ],
            }))
            metadata["python_packages_sha256"] = hashlib.sha256(
                inventory.read_bytes()
            ).hexdigest()
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("direct_dependency_version", report["failed_gates"])

    def test_inventory_rejects_undeclared_host_or_debug_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            inventory = root / "python_packages.json"
            value = json.loads(inventory.read_text())
            value["host_path"] = "/private/build/worker"
            inventory.write_text(json.dumps(value))
            metadata["python_packages_sha256"] = hashlib.sha256(
                inventory.read_bytes()
            ).hexdigest()
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("inventory_schema", report["failed_gates"])


if __name__ == "__main__":
    unittest.main()
