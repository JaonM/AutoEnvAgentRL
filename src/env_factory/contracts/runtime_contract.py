"""Shared minimum HTTP surface for generated training environments."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_SYSTEM_ENDPOINTS = frozenset({
    ("health", "GET", "/health"),
    ("reset", "POST", "/v1/reset"),
    ("observation", "GET", "/v1/observation"),
    ("state", "GET", "/v1/state"),
    ("tools", "GET", "/v1/tools"),
    ("user_simulator", "POST", "/v1/user_simulator"),
    ("reward", "GET", "/v1/reward"),
    ("replay", "GET", "/v1/replay"),
})


def missing_system_endpoints(interface: Mapping[str, Any] | Any) -> set[tuple[str, str, str]]:
    if not isinstance(interface, Mapping):
        return set(REQUIRED_SYSTEM_ENDPOINTS)
    endpoints = interface.get("endpoints")
    if not isinstance(endpoints, list):
        return set(REQUIRED_SYSTEM_ENDPOINTS)
    actual = {
        (item["name"], item["method"], item["path"])
        for item in endpoints if isinstance(item, dict)
        if all(isinstance(item.get(key), str) for key in ("name", "method", "path"))
    }
    return set(REQUIRED_SYSTEM_ENDPOINTS - actual)
