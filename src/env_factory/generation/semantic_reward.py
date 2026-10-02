"""Author-declared semantic reward contracts, separate from deterministic facts."""
from __future__ import annotations

import copy
import json

from env_factory.sandbox_runtime import SandboxError, validate_json_schema


def compile_semantic_outcome(item: dict, *, schema: dict, weight: float) -> tuple[dict, list[dict]]:
    """Validate one field-scoped judge and its executable calibration cases.

    Cases are calibration evidence, never reference answers supplied to the judge.
    The caller must run them with the actual evaluator before accepting the task.
    """
    semantic = item.get("semantic")
    if "rule" in item or not isinstance(semantic, dict) or set(semantic) != {"fields", "criteria", "cases"}:
        raise ValueError("SEMANTIC_REWARD_CONTRACT: declare semantic fields, criteria and cases, without rule")
    fields, criteria, cases = (semantic[key] for key in ("fields", "criteria", "cases"))
    properties = schema.get("properties", {})
    if (not isinstance(fields, list) or not fields or any(not isinstance(x, str) for x in fields)
            or len(set(fields)) != len(fields) or any(x not in properties for x in fields)):
        raise ValueError("SEMANTIC_REWARD_FIELDS: fields must uniquely name public answer properties")
    for field in fields:
        if properties[field].get("type") != "string" or "enum" in properties[field]:
            raise ValueError("SEMANTIC_REWARD_FIELDS: open explanation fields require strings without an enum")
    if (not isinstance(criteria, list) or not criteria or len(criteria) > 16
            or any(not isinstance(x, str) or not x.strip() for x in criteria)):
        raise ValueError("SEMANTIC_REWARD_CRITERIA: provide explicit public meaning-based criteria")
    if not isinstance(cases, list) or not 4 <= len(cases) <= 16:
        raise ValueError("SEMANTIC_REWARD_CASES: provide 4 to 16 positive and negative calibration answers")
    seen, counts = set(), {"pass": 0, "fail": 0}
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"answer", "expected"}
                or case.get("expected") not in counts):
            raise ValueError("SEMANTIC_REWARD_CASES: each case requires answer and expected pass/fail")
        try:
            validate_json_schema(schema, case["answer"], "semantic_case")
        except (SandboxError, TypeError, ValueError) as exc:
            raise ValueError(f"SEMANTIC_REWARD_CASES: invalid answer shape: {exc}") from exc
        # Cases must differ in the fields this judge actually owns.
        identity = json.dumps({key: case["answer"][key] for key in fields}, sort_keys=True, ensure_ascii=False)
        if identity in seen:
            raise ValueError("SEMANTIC_REWARD_CASES: duplicate or contradictory owned-field answer")
        seen.add(identity)
        counts[case["expected"]] += 1
    if min(counts.values()) < 2:
        raise ValueError("SEMANTIC_REWARD_CASES: require two distinct paraphrase positives and two negatives")
    metric = {"id": item["id"], "category": "outcome", "type": "model-based", "scope": "terminal",
        "weight": weight, "score_range": [0, 1], "rubric": item["rubric"],
        "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
        "criteria": ["Evaluate only these JSON answer fields: " + ", ".join(fields),
                     "Accept faithful paraphrases; reject contradictions, omitted required meaning and unsupported claims.",
                     *criteria],
        "evaluator": {"kind": "external_llm_judge", "source": "external_llm",
                      "score_mapping": {"pass": 1, "fail": 0}},
        "semantic_fields": list(fields)}
    return metric, copy.deepcopy(cases)


class SemanticCalibrationUnavailable(RuntimeError):
    """Infrastructure failure cannot establish semantic acceptance or rejection."""


def calibrate_semantic_outcomes(contract: dict, calibration: dict, *, context: dict, store) -> dict:
    """Run real runtime judges against declared paraphrase and error cases.

    A failed negative judged through an infrastructure fallback is not evidence
    that the evaluator recognized the error. All cases require real judgments.
    """
    import os
    from env_factory.sandbox_runtime import ContractModelMetricEvaluator, ContractEvaluatorRuntime
    if os.getenv("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"}:
        raise SemanticCalibrationUnavailable("SEMANTIC_CALIBRATION_UNAVAILABLE: mock judge cannot certify semantics")
    metrics = [m for m in contract.get("metrics", []) if m.get("semantic_fields")]
    if set(calibration) != {m["id"] for m in metrics}:
        raise ValueError("SEMANTIC_CALIBRATION_COVERAGE: every semantic metric requires calibration")
    report = {"passed": True, "semantic_verification": True, "cases": []}
    for metric in metrics:
        judge_contract = {**contract, "metrics": [metric]}
        judge = ContractModelMetricEvaluator(judge_contract, store)
        for index, case in enumerate(calibration[metric["id"]]):
            # Calibration requires a fresh recorded judgment, not a prior cached label.
            store.set_state(ContractEvaluatorRuntime.STATE_KEY, {})
            before = len(store.replay()["events"])
            evidence = {**context, "final_agent_response": json.dumps(case["answer"], ensure_ascii=False)}
            scores = judge.evaluate_all(evidence, {})
            events = store.replay()["events"][before:]
            calls = [event for event in events if event.get("event") == "evaluator_call"]
            if (not calls or any(not call.get("payload", {}).get("judgment_obtained") for call in calls)):
                raise SemanticCalibrationUnavailable(
                    f"SEMANTIC_CALIBRATION_UNAVAILABLE: no real judgment for {metric['id']} case {index}")
            wanted = metric["evaluator"]["score_mapping"][case["expected"]]
            actual = scores[metric["id"]]
            report["cases"].append({"metric_id": metric["id"], "index": index,
                "expected_label": case["expected"], "actual_score": actual,
                "passed": actual == wanted,
                "judgment_context_hashes": [call["payload"]["context_hash"] for call in calls]})
            report["passed"] &= actual == wanted
    return report


def validate_explanation_rewards(source: dict) -> None:
    """Reject exact matching only for clearly declared open explanation fields.

    Enums, explicit quotation/extraction contracts and ordinary entity names remain
    valid deterministic targets. Ambiguous prose is left to semantic review.
    """
    import re
    properties = source.get('answer_contract', {}).get('schema', {}).get('properties', {})
    for rule in source.get('metric_implementations', []):
        if rule.get('source') != 'final_agent_response' or rule.get('operator') != 'value_targets':
            continue
        for target in rule.get('expected', {}).get('targets', []):
            key = target.get('key', '')
            field = properties.get(key, {})
            description = field.get('description', '')
            if field.get('type') != 'string' or 'enum' in field or 'const' in field:
                continue
            if re.search(r'原样|原文|逐字|直接摘录|verbatim|exact quote|copy exactly', description, re.I):
                continue
            if (re.search(r'(?:reason|explanation|rationale|justification)(?:$|_)', key, re.I)
                    and re.search(r'理由|依据|解释|说明|reason|explain|justify|rationale', description, re.I)):
                raise ValueError(
                    f'OPEN_EXPLANATION_EXACT_MATCH: answer field {key!r} asks for an explanation '
                    'but value_targets requires exact equality to one string. Use a semantic outcome '
                    'with paraphrase-positive/incorrect-negative calibration cases, or structured '
                    'evidence fields that preserve the original business reasoning requirement. '
                    'Do not replace the goal with a copy-only task merely to pass validation.')
