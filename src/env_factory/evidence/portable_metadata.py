"""Minimize and audit machine-local metadata in portable material bundles."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from env_factory.evidence.data_governance import scan_payloads


LOCAL_PATH_KEYS = {
    "path", "root", "output", "project", "task_path", "task_paths",
    "sandbox_root", "artifact_dir", "working_directory",
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _local_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    return value.startswith("file://") or WINDOWS_ABSOLUTE.match(value) is not None or (
        value.startswith("/") and Path(value).is_absolute()
    )


def sanitize_portable_metadata(value: Any) -> Any:
    """Remove path-bearing fields and redact stray absolute path strings."""
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_portable_metadata(child)
            for key, child in value.items()
            if str(key).lower() not in LOCAL_PATH_KEYS
            and not str(key).lower().endswith(("_path", "_root"))
        }
    if isinstance(value, list):
        return [sanitize_portable_metadata(child) for child in value]
    if isinstance(value, tuple):
        return [sanitize_portable_metadata(child) for child in value]
    if isinstance(value, str) and _local_path(value):
        return "<redacted-local-path>"
    return value


def audit_portable_metadata(value: Any) -> dict[str, Any]:
    """Return path-only findings; never echo credential or path contents."""
    local_paths: list[str] = []

    def walk(child: Any, location: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                key_text = str(key)
                if (
                    key_text.lower() in LOCAL_PATH_KEYS
                    or key_text.lower().endswith(("_path", "_root"))
                ):
                    local_paths.append(f"{location}.{key_text}")
                walk(nested, f"{location}.{key_text}")
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                walk(nested, f"{location}[{index}]")
        elif isinstance(child, str) and _local_path(child):
            local_paths.append(location)

    walk(value, "$")
    scan = scan_payloads({"portable_metadata": value})
    credential_paths = sorted({
        finding["path"] for finding in scan["credential_findings"]
    })
    pii_paths = sorted({
        finding["path"] for finding in scan["pii_findings"]
    })
    return {
        "safe": not local_paths and not credential_paths and not pii_paths,
        "local_path_findings": sorted(set(local_paths)),
        "credential_findings": credential_paths,
        "pii_findings": pii_paths,
    }
