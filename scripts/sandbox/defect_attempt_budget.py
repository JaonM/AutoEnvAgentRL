#!/usr/bin/env python3
"""Persist a repair budget for each distinct sandbox defect."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def _normalized(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<id>", text)
    text = re.sub(r"\b(?:req-|sha256:)[0-9a-f]+\b", "<id>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    return " ".join(text.split())


def _field_identifiers(issue: str) -> set[str]:
    """Keep distinct snake_case fields from sharing a fuzzy repair budget."""
    return set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", issue))


def defect_identity(defect: dict[str, Any]) -> tuple[str, str]:
    """Return a stable scope plus issue text, ignoring review run IDs."""
    scope = "|".join(_normalized(defect.get(name)) for name in (
        "category", "tool_name", "file",
    ))
    evidence = str(defect.get("evidence") or "")
    if defect.get("category") == "reward_semantics":
        description = _normalized(f"{defect.get('fix_required', '')} {evidence}")
        if any(token in description for token in (
            "raw_text", "canonical csv", "exact-call", "expected-call map",
        )):
            return scope, "literal_payload_process_reward"
        if any(token in description for token in (
            "final_agent_response", "final response", "response quality", "最终回答",
        )):
            return scope, "missing_terminal_response_reward"
    if defect.get("category") == "delivery_failure":
        gates = re.findall(r'"gate"\s*:\s*"([^\"]+)"', evidence)
        messages = re.findall(r'"message"\s*:\s*"([^\"]+)"', evidence)
        if gates:
            # Outer conformance reports sometimes include repeated history.  The
            # last failure gate and its diagnostic identify the failing check.
            return scope, _normalized(" ".join(gates[-3:] + messages[-3:]))
        failure = re.findall(
            r'"failure"\s*:\s*\{[^{}]*"type"\s*:\s*"([^\"]+)"[^{}]*"message"\s*:\s*"([^\"]+)"',
            evidence, re.S,
        )
        if failure:
            # The HTTP status alone is too coarse: distinct missing argument
            # sets can fail at the same step.  Include the nearby response
            # details while stripping volatile request identifiers.
            position = evidence.rfind('"failure"')
            details = evidence[max(0, position - 450):position] if position >= 0 else ""
            return scope, _normalized(" ".join(failure[-1]) + " " + details)
        return scope, _normalized(evidence[-1200:])
    return scope, _normalized(
        f"{defect.get('fix_required', '')} {evidence}"
    )


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=state_path.parent, prefix=".defect-attempts-", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, state_path)


def reserve_attempt(
    state_path: Path, defect: dict[str, Any], maximum: int, *,
    maximum_total: int | None = None, pending: bool = False,
) -> tuple[str, int]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if maximum_total is not None and maximum_total < 1:
        raise ValueError("maximum_total must be positive")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"version": "1.0", "defects": []}
    if not isinstance(state, dict) or not isinstance(state.get("defects"), list):
        raise ValueError("invalid defect budget state")
    # A killed Code Agent leaves an uncommitted reservation. Resume must not
    # charge a business repair attempt for a network or runner interruption.
    for entry in state["defects"]:
        if entry.get("pending"):
            entry["attempts"] = max(0, int(entry.get("attempts", 0)) - 1)
            entry.pop("pending", None)
    if maximum_total is not None:
        counts = [entry.get("attempts") for entry in state["defects"]]
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("invalid defect attempt count")
        if sum(counts) >= maximum_total:
            raise RuntimeError(f"sample exhausted {maximum_total} total repair attempts")
    scope, issue = defect_identity(defect)
    candidates = [entry for entry in state["defects"] if entry.get("scope") == scope]
    matching = next((entry for entry in candidates if entry.get("issue") == issue), None)
    if matching is None and issue:
        identifiers = _field_identifiers(issue)
        scored = [
            (SequenceMatcher(None, issue, str(entry.get("issue", ""))).ratio(), entry)
            for entry in candidates
            if identifiers == _field_identifiers(str(entry.get("issue", "")))
        ]
        if scored:
            similarity, entry = max(scored, key=lambda pair: pair[0])
            if similarity >= (0.82 if identifiers else 0.95):
                matching = entry
    if matching is None:
        matching = {
            "id": f"DEFECT-{len(state['defects']) + 1:04d}",
            "scope": scope, "issue": issue, "attempts": 0,
        }
        state["defects"].append(matching)
    attempts = matching.get("attempts")
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError("invalid defect attempt count")
    if attempts >= maximum:
        raise RuntimeError(f"{matching['id']} exhausted {maximum} repair attempts")
    matching["attempts"] = attempts + 1
    if pending:
        matching["pending"] = True
    _save_state(state_path, state)
    return str(matching["id"]), matching["attempts"]


def finish_pending(state_path: Path, defect_id: str, *, release: bool) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    matching = next((entry for entry in state["defects"] if entry.get("id") == defect_id), None)
    if matching is None or not matching.get("pending"):
        raise ValueError(f"no pending attempt for {defect_id}")
    if release:
        matching["attempts"] -= 1
    matching.pop("pending")
    _save_state(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--defects", type=Path)
    parser.add_argument("--index", type=int,
                        help="specific defect index; default selects first with budget remaining")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-total-attempts", type=int)
    finish = parser.add_mutually_exclusive_group()
    finish.add_argument("--commit-id")
    finish.add_argument("--release-id")
    args = parser.parse_args()
    if args.commit_id or args.release_id:
        try:
            finish_pending(args.state, args.commit_id or args.release_id,
                           release=bool(args.release_id))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return 0
    if args.defects is None or args.max_attempts is None:
        parser.error("--defects and --max-attempts are required for reservation")
    defects = json.loads(args.defects.read_text(encoding="utf-8"))
    if not isinstance(defects, list) or not defects or any(not isinstance(item, dict) for item in defects):
        parser.error("defects must be a list of objects")
    indices = [args.index] if args.index is not None else range(len(defects))
    exhausted = []
    for index in indices:
        if index < 0 or index >= len(defects):
            parser.error("defect index out of range")
        try:
            defect_id, attempt = reserve_attempt(
                args.state, defects[index], args.max_attempts,
                maximum_total=args.max_total_attempts, pending=True,
            )
        except RuntimeError as exc:
            exhausted.append(str(exc))
            continue
        except ValueError as exc:
            parser.error(str(exc))
        print(index, defect_id, attempt)
        return 0
    parser.error("all currently reported defects exhausted their repair budgets: " + "; ".join(exhausted))


if __name__ == "__main__":
    raise SystemExit(main())
