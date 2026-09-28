"""User persona and finite-state-machine artifact construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline_errors import PipelineGenerationError


NORMAL_DIALOGUE_OUTCOMES = frozenset({
    "goal_satisfied", "information_required", "user_correction",
    "user_rejection", "user_acceptance",
})
RECOVERY_DIALOGUE_OUTCOMES = frozenset({
    "agent_off_topic", "agent_premature_completion", "unrecognized",
})


class UserSimulationContractMixin:
    @staticmethod
    def _validate_user_script_state_machine(script: dict[str, Any], index: int) -> None:
        """Validate a reusable, finite user-behavior state machine."""
        states = script.get("states")
        transitions = script.get("transitions")
        initial = script.get("initial_state")
        variables = script.get("variables", {})
        recovery_policy = script.get("recovery_policy")
        if not isinstance(states, list) or len(states) < 2:
            raise PipelineGenerationError(f"user_scripts[{index}] requires at least two states")
        if not isinstance(transitions, list) or len(transitions) < 2:
            raise PipelineGenerationError(f"user_scripts[{index}] requires at least two transitions")
        if not isinstance(variables, dict):
            raise PipelineGenerationError(f"user_scripts[{index}].variables must be an object")
        if (
            not isinstance(recovery_policy, dict)
            or not isinstance(recovery_policy.get("max_recoveries"), int)
            or not 1 <= recovery_policy["max_recoveries"] <= 3
            or not isinstance(recovery_policy.get("user_behavior"), str)
            or not recovery_policy["user_behavior"].strip()
            or set(recovery_policy.get("handled_outcomes", [])) != RECOVERY_DIALOGUE_OUTCOMES
        ):
            raise PipelineGenerationError(f"user_scripts[{index}].recovery_policy is invalid")
        by_id: dict[str, dict[str, Any]] = {}
        for state_index, state in enumerate(states):
            if not isinstance(state, dict):
                raise PipelineGenerationError(f"user_scripts[{index}].states[{state_index}] must be an object")
            state_id = state.get("state_id")
            if not isinstance(state_id, str) or not state_id.strip() or state_id in by_id:
                raise PipelineGenerationError(f"user_scripts[{index}] has invalid or duplicate state_id")
            if not isinstance(state.get("user_behavior"), str) or not state["user_behavior"].strip():
                raise PipelineGenerationError(f"user_scripts[{index}] state {state_id} requires user_behavior")
            if not isinstance(state.get("terminal"), bool):
                raise PipelineGenerationError(f"user_scripts[{index}] state {state_id} requires terminal")
            by_id[state_id] = state
        if not isinstance(initial, str) or initial not in by_id or by_id[initial]["terminal"]:
            raise PipelineGenerationError(f"user_scripts[{index}] initial_state is invalid")

        outgoing: dict[str, list[str]] = {state_id: [] for state_id in by_id}
        seen_transitions: set[str] = set()
        covered_outcomes: set[str] = set()
        for transition_index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise PipelineGenerationError(f"user_scripts[{index}].transitions[{transition_index}] must be an object")
            transition_id = transition.get("transition_id")
            source, target = transition.get("from_state"), transition.get("to_state")
            if not isinstance(transition_id, str) or not transition_id.strip() or transition_id in seen_transitions:
                raise PipelineGenerationError(f"user_scripts[{index}] has invalid or duplicate transition_id")
            if source not in by_id or target not in by_id:
                raise PipelineGenerationError(f"user_scripts[{index}] transition {transition_id} references unknown state")
            if by_id[source]["terminal"]:
                raise PipelineGenerationError(f"user_scripts[{index}] terminal state {source} cannot have outgoing transitions")
            if not isinstance(transition.get("condition"), str) or not transition["condition"].strip():
                raise PipelineGenerationError(f"user_scripts[{index}] transition {transition_id} requires condition")
            outcome = transition.get("outcome_category")
            if outcome not in NORMAL_DIALOGUE_OUTCOMES:
                raise PipelineGenerationError(
                    f"user_scripts[{index}] transition {transition_id} has invalid outcome_category"
                )
            if not isinstance(transition.get("should_end"), bool):
                raise PipelineGenerationError(f"user_scripts[{index}] transition {transition_id} requires should_end")
            if transition["should_end"] != bool(by_id[target]["terminal"]):
                raise PipelineGenerationError(f"user_scripts[{index}] transition {transition_id} end flag must match target state")
            if outcome == "user_acceptance" and not by_id[target]["terminal"]:
                raise PipelineGenerationError(f"user_scripts[{index}] user_acceptance must enter a terminal state")
            if outcome in {"information_required", "user_correction", "user_rejection"} and by_id[target]["terminal"]:
                raise PipelineGenerationError(f"user_scripts[{index}] {outcome} cannot enter a terminal state")
            updates = transition.get("updates", {})
            if not isinstance(updates, dict) or any(key not in variables for key in updates):
                raise PipelineGenerationError(f"user_scripts[{index}] transition {transition_id} has invalid updates")
            seen_transitions.add(transition_id)
            covered_outcomes.add(str(outcome))
            outgoing[source].append(target)

        reachable = {initial}
        frontier = [initial]
        while frontier:
            source = frontier.pop()
            for target in outgoing[source]:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if reachable != set(by_id):
            raise PipelineGenerationError(f"user_scripts[{index}] contains unreachable states")
        visiting: set[str] = set()
        visited: set[str] = set()

        def reject_cycle(state_id: str) -> None:
            if state_id in visiting:
                raise PipelineGenerationError(f"user_scripts[{index}] state machine must be acyclic")
            if state_id in visited:
                return
            visiting.add(state_id)
            for target in outgoing[state_id]:
                reject_cycle(target)
            visiting.remove(state_id)
            visited.add(state_id)

        reject_cycle(initial)
        terminals = {state_id for state_id, state in by_id.items() if state["terminal"]}
        if not terminals:
            raise PipelineGenerationError(f"user_scripts[{index}] requires a terminal state")
        reverse: dict[str, set[str]] = {state_id: set() for state_id in by_id}
        for source, targets in outgoing.items():
            for target in targets:
                reverse[target].add(source)
        can_terminate = set(terminals)
        frontier = list(terminals)
        while frontier:
            target = frontier.pop()
            for source in reverse[target]:
                if source not in can_terminate:
                    can_terminate.add(source)
                    frontier.append(source)
        if reachable - can_terminate:
            raise PipelineGenerationError(f"user_scripts[{index}] has states without a terminal path")
        missing_outcomes = NORMAL_DIALOGUE_OUTCOMES - covered_outcomes
        if missing_outcomes:
            raise PipelineGenerationError(
                f"user_scripts[{index}] misses dialogue outcomes: {sorted(missing_outcomes)}"
            )

    @classmethod
    def _deterministic_user_scripts(cls, *, description: dict[str, Any], count: int) -> list[dict[str, Any]]:
        """Build minimal valid FSMs when a model cannot repair script structure.

        User scripts are test drivers rather than task semantics. A structurally
        sound generic driver is therefore preferable to discarding an otherwise
        valid task after repeated formatting failures.
        """
        goal = str(description.get("description") or description.get("task") or "完成用户任务").strip()
        scripts: list[dict[str, Any]] = []
        for index in range(1, max(1, count) + 1):
            script = {
                "script_id": f"script-{index}",
                "goal": goal,
                "user_input": cls._normalize_public_input(description),
                "initial_state": "request",
                "variables": {"clarification": None},
                "recovery_policy": {
                    "max_recoveries": 2,
                    "user_behavior": "指出回复没有解决当前问题，并要求 Agent 根据已有信息重新回答。",
                    "handled_outcomes": sorted(RECOVERY_DIALOGUE_OUTCOMES),
                },
                "states": [
                    {
                        "state_id": "request",
                        "user_behavior": "提出任务请求，并仅提供完成任务所需的公开信息。",
                        "terminal": False,
                    },
                    {
                        "state_id": "clarify",
                        "user_behavior": "回答一个必要澄清问题，或要求 Agent 直接完成任务。",
                        "terminal": False,
                    },
                    {"state_id": "corrected", "user_behavior": "纠正 Agent 对需求的误解。", "terminal": False},
                    {"state_id": "rejected", "user_behavior": "拒绝不符合约束的方案并重申要求。", "terminal": False},
                    {"state_id": "review", "user_behavior": "检查 Agent 已完成的结果并决定是否接受。", "terminal": False},
                    {
                        "state_id": "done",
                        "user_behavior": "确认收到结果并结束对话。",
                        "terminal": True,
                    },
                ],
                "transitions": [
                    {
                        "transition_id": "request-to-clarify",
                        "outcome_category": "information_required",
                        "from_state": "request",
                        "to_state": "clarify",
                        "condition": "Agent 请求必要信息或开始处理任务",
                        "should_end": False,
                        "updates": {"clarification": "按任务上下文提供必要补充"},
                    },
                    {
                        "transition_id": "request-to-corrected",
                        "outcome_category": "user_correction",
                        "from_state": "request", "to_state": "corrected",
                        "condition": "Agent 误解了用户已明确的目标或约束",
                        "should_end": False, "updates": {},
                    },
                    {
                        "transition_id": "request-to-rejected",
                        "outcome_category": "user_rejection",
                        "from_state": "request", "to_state": "rejected",
                        "condition": "Agent 给出不符合约束的候选方案",
                        "should_end": False, "updates": {},
                    },
                    {
                        "transition_id": "request-to-review",
                        "outcome_category": "goal_satisfied",
                        "from_state": "request", "to_state": "review",
                        "condition": "Agent 已完成当前任务目标，等待用户确认",
                        "should_end": False, "updates": {},
                    },
                    {
                        "transition_id": "clarify-to-review",
                        "outcome_category": "goal_satisfied",
                        "from_state": "clarify",
                        "to_state": "review",
                        "condition": "Agent 给出可验收的最终结果",
                        "should_end": False,
                        "updates": {},
                    },
                    {"transition_id": "corrected-to-review", "outcome_category": "goal_satisfied", "from_state": "corrected", "to_state": "review", "condition": "Agent 按纠正后的要求完成任务", "should_end": False, "updates": {}},
                    {"transition_id": "rejected-to-review", "outcome_category": "goal_satisfied", "from_state": "rejected", "to_state": "review", "condition": "Agent 提供符合约束的新方案", "should_end": False, "updates": {}},
                    {"transition_id": "review-to-done", "outcome_category": "user_acceptance", "from_state": "review", "to_state": "done", "condition": "用户接受已完成的结果", "should_end": True, "updates": {}},
                ],
            }
            cls._validate_user_script_state_machine(script, index - 1)
            scripts.append(script)
        return scripts

    @staticmethod
    def _materialize_user_simulation(
        profiles: list[Any],
        scripts: list[Any],
        artifact_dir: Path,
        *,
        manifest_root: str | None = None,
    ) -> dict[str, Any]:
        """Persist the only inputs consumed by the runtime user simulator."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = artifact_dir / "user_profiles.json"
        scripts_path = artifact_dir / "user_scripts.json"
        profiles_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        scripts_path.write_text(json.dumps(scripts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "version": "3.0",
            "script_model": "finite_state_machine",
            "root": manifest_root or str(artifact_dir),
            "profiles_file": str(profiles_path.relative_to(artifact_dir)),
            "scripts_file": str(scripts_path.relative_to(artifact_dir)),
        }
