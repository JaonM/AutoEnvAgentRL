#!/usr/bin/env python3
"""Validate sandbox runtime traces against the immutable build contract.

The validator is capability-based: it does not know task names such as
``identify_material``.  A sandbox emits one JSON object per line with at least
``event`` and may include ``capability_id`` and ``action_id``.  Required trace
events are declared by BUILD_CONTRACT.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace line {line_number} is not JSON: {exc.msg}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("event"), str):
            raise ValueError(f"trace line {line_number} must contain an event string")
        events.append(value)
    return events


def _ordered(events: list[dict[str, Any]], required: list[str]) -> bool:
    position = 0
    for event in events:
        if event.get("event") == required[position]:
            position += 1
            if position == len(required):
                return True
    return False


def validate(contract_path: Path, trace_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    events = _load_trace(trace_path)
    failures: list[dict[str, Any]] = []
    checked: list[str] = []
    for capability in contract.get("capabilities", []):
        if not capability.get("required"):
            continue
        capability_id = capability.get("id")
        required_trace = capability.get("required_trace", [])
        if not isinstance(capability_id, str) or not isinstance(required_trace, list):
            failures.append({"capability": capability_id, "error": "invalid capability contract"})
            continue
        required_trace = [item for item in required_trace if isinstance(item, str)]
        if not required_trace:
            continue
        scoped = [event for event in events if event.get("capability_id") in {None, capability_id}]
        checked.append(capability_id)
        if not _ordered(scoped, required_trace):
            failures.append({
                "capability": capability_id,
                "required_trace": required_trace,
                "observed_events": [event.get("event") for event in scoped],
            })
    return {
        "status": "ok" if not failures else "failed",
        "checked_capabilities": checked,
        "event_count": len(events),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="校验沙箱运行时 trace 是否覆盖结构化能力契约")
    parser.add_argument("--contract", required=True, type=Path, help="BUILD_CONTRACT.json 路径")
    parser.add_argument("--trace", required=True, type=Path, help="JSONL runtime trace 路径")
    args = parser.parse_args()
    try:
        result = validate(args.contract, args.trace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
