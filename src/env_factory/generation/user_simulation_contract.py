"""User persona and finite-state-machine artifact construction."""
from __future__ import annotations

import hashlib
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
    def _deterministic_user_profiles(count: int, *, selection_key: str = "") -> list[dict[str, Any]]:
        """Use reusable personas without spending a model call per task."""
        if count < 1:
            raise ValueError("user profile count must be positive")
        variants = (
            ("谨慎的首次使用者", "25–34", "刚开始使用相关服务的上班族", "城市通勤场景",
             "经验有限", "礼貌、简短", "先确认风险再决定", "较低", "较有耐心", "中等",
             "会逐步补充信息", "先问关键限制", "先核对证据", "重视费用", "时间适中"),
            ("熟悉领域的高效使用者", "35–44", "有实际业务经验的从业者", "工作场景",
             "经验丰富", "直接、精炼", "比较证据后快速决定", "中等", "耐心有限", "较高",
             "一次说明已知约束", "追问具体依据", "指出不一致之处", "重视效率", "时间紧"),
            ("需要解释的审慎使用者", "45–59", "兼顾家庭事务的使用者", "家庭场景",
             "只了解常用概念", "温和、详细", "先理解选项再决定", "较低", "较有耐心", "中等",
             "遇到追问才补充细节", "请求通俗解释", "会要求重新比较", "重视资源约束", "时间弹性"),
            ("独立核验的使用者", "18–24", "正在学习新领域的学生", "学习场景",
             "理论知识多于实践", "好奇、具体", "先核验来源再决定", "中等", "耐心适中", "较低",
             "明确区分事实和猜测", "追问数据来源", "会挑战未经证实的结论", "预算有限", "时间适中"),
        )
        offset = hashlib.sha256(selection_key.encode("utf-8")).digest()[0] % len(variants)
        profiles: list[dict[str, Any]] = []
        for index in range(count):
            (identity, age, occupation, location, expertise, tone, decision,
             risk, patience, trust, disclosure, questioning, feedback,
             budget, time) = variants[(offset + index) % len(variants)]
            profiles.append({
                "profile_id": f"profile-{index + 1}",
                "identity_summary": identity,
                "age_range": age,
                "occupation_or_life_stage": occupation,
                "location_context": location,
                "education_background": "能阅读常见说明，但不会假定自己掌握当前任务的私有事实。",
                "domain_knowledge": {"level": expertise, "areas": ["日常决策", "信息核对"],
                                     "evidence": "仅依赖对话中明确提供的公开信息。"},
                "goals_and_motivations": ["完成当前请求", "获得可核查的理由"],
                "communication_style": {"tone": tone, "verbosity": "按问题复杂度调整", "directness": "明确表达要求"},
                "language_habits": ["使用自然口语", "发现歧义时要求澄清"],
                "decision_style": {"pattern": decision, "needs": ["目标一致", "依据充分"]},
                "risk_tolerance": risk,
                "patience_level": patience,
                "trust_level": trust,
                "information_disclosure_style": disclosure,
                "questioning_style": questioning,
                "feedback_style": feedback,
                "budget_or_resource_sensitivity": budget,
                "time_sensitivity": time,
                "accessibility_needs": ["清楚的结构", "可理解的术语"],
                "frustration_triggers": ["重复询问已给信息", "无依据地断言"],
                "misconceptions_or_biases": ["可能高估熟悉方案", "可能忽略例外条件"],
                "known_facts": ["知道自己公开提出的目标", "知道自己已提供的约束"],
                "unknown_facts": ["不知道沙箱私有业务记录", "不知道尚未查询的工具结果"],
                "behavior_tendencies": [questioning, feedback],
            })
        return profiles

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
    def _deterministic_user_scripts(cls, *, description: dict[str, Any], count: int,
                                    finalize_on_goal: bool = False) -> list[dict[str, Any]]:
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
            if finalize_on_goal:
                # Complete fixed-goal tasks end on the user's completion decision.
                # A second confirmation cycle can repeat a non-idempotent write
                # or replace a structured final answer with a conversational ack.
                script["states"] = [state for state in script["states"] if state["state_id"] != "review"]
                for transition in script["transitions"]:
                    if transition["outcome_category"] == "goal_satisfied":
                        transition.update(to_state="done", should_end=True)
                    if transition["outcome_category"] == "user_acceptance":
                        transition.update(transition_id="request-to-done", from_state="request")
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
