"""Author-declared semantic reward contracts, separate from deterministic facts."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
import tempfile

from env_factory.sandbox_runtime import SandboxError, validate_json_schema


def compile_semantic_outcome(item: dict, *, schema: dict, weight: float,
                             reference_answers: dict | None = None) -> tuple[dict, list[dict]]:
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
    expanded = []
    for case in cases:
        if not isinstance(case, dict) or 'answer_overrides' not in case:
            expanded.append(case)
            continue
        if set(case) != {'answer_overrides', 'expected', 'reference_id'}:
            raise ValueError('SEMANTIC_REWARD_CASES: answer_overrides requires reference_id and expected, without answer')
        identity = case['reference_id']
        if not isinstance(identity, str) or not identity:
            raise ValueError('SEMANTIC_REWARD_CASES: answer_overrides requires a named reference')
        name = identity
        if name not in (reference_answers or {}):
            base, separator, script = name.rpartition('-script-')
            if separator and script.isdigit() and int(script) > 0:
                name = base
        if name not in (reference_answers or {}):
            raise ValueError('SEMANTIC_REWARD_CASES: unknown answer_overrides reference: ' + identity)
        overrides = case['answer_overrides']
        if not isinstance(overrides, dict) or set(overrides) - set(fields):
            raise ValueError('SEMANTIC_REWARD_CASES: answer_overrides may replace only this metric\'s owned fields')
        try:
            answer = json.loads(reference_answers[name])
        except (ValueError, TypeError) as exc:
            raise ValueError('SEMANTIC_REWARD_CASES: reference answer must be a JSON object') from exc
        if not isinstance(answer, dict):
            raise ValueError('SEMANTIC_REWARD_CASES: reference answer must be a JSON object')
        answer.update(copy.deepcopy(overrides))
        expanded.append({'reference_id': identity, 'answer': answer, 'expected': case['expected']})
    cases = expanded
    seen, counts = set(), {"pass": 0, "fail": 0}
    for case in cases:
        if (not isinstance(case, dict) or not {"answer", "expected"} <= set(case)
                or set(case) - {"answer", "expected", "reference_id"}
                or ("reference_id" in case and (not isinstance(case["reference_id"], str)
                    or not case["reference_id"].strip()))
                or case.get("expected") not in counts):
            raise ValueError("SEMANTIC_REWARD_CASES: each case requires answer and expected pass/fail")
        try:
            validate_json_schema(schema, case["answer"], "semantic_case")
        except (SandboxError, TypeError, ValueError) as exc:
            raise ValueError(f"SEMANTIC_REWARD_CASES: invalid answer shape: {exc}") from exc
        # Cases must differ in the fields this judge actually owns.
        identity = (case.get("reference_id"), json.dumps(
            {key: case["answer"][key] for key in fields}, sort_keys=True, ensure_ascii=False))
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


def scope_calibration_references(calibration, successes):
    """A named reference family calibrates every expanded user script, never one chosen context."""
    scoped, references = {}, {None: successes[0]}
    for metric_id, cases in calibration.items():
        expanded = []
        seen = set()
        for case in cases:
            if 'reference_id' not in case:
                expanded.append(copy.deepcopy(case))
                continue
            name = case['reference_id']
            matches = [s for s in successes if s.get('scenario_id') == name]
            if not matches:
                prefix = name + '-script-'
                matches = [s for s in successes if s.get('scenario_id', '').startswith(prefix)
                           and s['scenario_id'][len(prefix):].isdigit()]
            if not matches:
                raise ValueError('SEMANTIC_CALIBRATION_REFERENCE: unknown successful reference: ' + str(name))
            if len(matches) > 8:
                raise ValueError('SEMANTIC_CALIBRATION_REFERENCE: at most eight expanded scripts per reference')
            for scenario in matches:
                identity = name if len(matches) == 1 else scenario['scenario_id']
                key = (scenario['scenario_id'], json.dumps(case['answer'], sort_keys=True, ensure_ascii=False))
                if key in seen:
                    raise ValueError('SEMANTIC_CALIBRATION_REFERENCE: overlapping or contradictory scoped cases')
                seen.add(key)
                references[identity] = scenario
                expanded.append({**copy.deepcopy(case), 'reference_id': identity})
        scoped[metric_id] = expanded
    return scoped, references


def calibrate_semantic_outcomes(contract: dict, calibration: dict, *, context: dict, store,
                                progress=None, reference_contexts: dict | None = None, judgment_cache=None,
                                workers: int = 1) -> dict:
    """Run real runtime judges against declared paraphrase and error cases.

    A failed negative judged through an infrastructure fallback is not evidence
    that the evaluator recognized the error. All cases require real judgments.
    """
    import os
    from env_factory.sandbox_runtime import ContractModelMetricEvaluator, ContractEvaluatorRuntime, EpisodeStore
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError('calibration workers must be 1 or 2')
    if os.getenv("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"}:
        raise SemanticCalibrationUnavailable("SEMANTIC_CALIBRATION_UNAVAILABLE: mock judge cannot certify semantics")
    metrics = [m for m in contract.get("metrics", []) if m.get("semantic_fields")]
    if set(calibration) != {m["id"] for m in metrics}:
        raise ValueError("SEMANTIC_CALIBRATION_COVERAGE: every semantic metric requires calibration")
    # Resolve declared contexts before asking any judge. Never pick a context by
    # whether the candidate answer happens to pass in it.
    required = {case['reference_id'] for cases in calibration.values() for case in cases
                if 'reference_id' in case}
    if required - set(reference_contexts or {}):
        raise ValueError("SEMANTIC_CALIBRATION_REFERENCE: missing executed reference context")
    report = {"passed": True, "semantic_verification": True, "cases": []}
    def evaluate_case(metric, index, case, case_store):
        judge_contract = {**contract, "metrics": [metric]}
        judge = ContractModelMetricEvaluator(judge_contract, case_store, judgment_cache=judgment_cache)
        # Every case still obtains a fresh judgment, including identical contexts.
        case_store.set_state(ContractEvaluatorRuntime.STATE_KEY, {})
        before = len(case_store.replay()["events"])
        selected = (reference_contexts[case['reference_id']]
                    if 'reference_id' in case else context)
        evidence = {**copy.deepcopy(selected), "final_agent_response": json.dumps(case["answer"], ensure_ascii=False)}
        with (progress.measure("semantic_calibration", metric_id=metric['id'], case_index=index)
              if progress is not None else nullcontext()):
            scores = judge.evaluate_all(evidence, {})
        events = case_store.replay()["events"][before:]
        calls = [event for event in events if event.get("event") == "evaluator_call"]
        if (not calls or any(not call.get("payload", {}).get("judgment_obtained") for call in calls)):
            if progress is not None:
                progress.feedback('semantic_calibration', metric_id=metric['id'], case_index=index,
                                  code='SEMANTIC_CALIBRATION_UNAVAILABLE')
            raise SemanticCalibrationUnavailable(
                f"SEMANTIC_CALIBRATION_UNAVAILABLE: no real judgment for {metric['id']} case {index}")
        wanted = metric["evaluator"]["score_mapping"][case["expected"]]
        actual = scores[metric["id"]]
        row = {"metric_id": metric["id"], "index": index,
            "expected_label": case["expected"], "actual_score": actual,
            "passed": actual == wanted, "reference_id": case.get('reference_id'),
            "criterion_evidence": [call.get('result', {}).get('criterion_results', {}) for call in calls],
            "judgment_context_hashes": [call["payload"]["context_hash"] for call in calls]}
        if actual != wanted:
            row['failed_case_evidence'] = {
                'answer': case['answer'], 'semantic_fields': metric.get('semantic_fields', []),
                'rubric': metric.get('rubric'), 'criteria': metric.get('criteria', [])}
            if progress is not None:
                progress.feedback('semantic_calibration', metric_id=metric['id'], case_index=index,
                                  code='SEMANTIC_CALIBRATION_FAILED')
        return row, judge.runtime.validation_cache_report()['conflicting_judgments']

    jobs = [(metric, index, case) for metric in metrics
            for index, case in enumerate(calibration[metric['id']])]
    if workers == 1:
        results = [evaluate_case(*job, store) for job in jobs]
    else:
        # Never reset a shared evaluator cache or SQLite episode concurrently.
        # The runtime bridge separately bounds model concurrency across pipelines.
        episode = store.current()
        with tempfile.TemporaryDirectory(prefix='envfactory-calibration-') as directory:
            def isolated_job(ordinal, job):
                isolated = EpisodeStore(Path(directory) / f'{ordinal}.sqlite3')
                isolated.reset(episode_id=f'calibration-{ordinal}', seed=episode.seed)
                result, error = None, None
                try:
                    result = evaluate_case(*job, isolated)
                except Exception as exc:
                    error = exc
                return result, isolated.replay()['events'], error

            completed = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = {executor.submit(isolated_job, ordinal, job): ordinal
                           for ordinal, job in enumerate(jobs)}
                for future in as_completed(pending):
                    completed[pending[future]] = future.result()
            # Publish in contract order, retaining each original event's identity
            # and timestamp. Completion order cannot change the report or labels.
            results, errors = [], []
            for ordinal, (metric, index, _) in enumerate(jobs):
                result, events, error = completed[ordinal]
                for event in events:
                    if event['event'] == 'reset':
                        continue
                    store.event(event['event'], {**event['payload'], 'calibration_case': {
                        'metric_id': metric['id'], 'index': index,
                        'sequence': event['sequence'], 'timestamp': event['timestamp']}}, event['result'])
                if error is not None:
                    errors.append(error)
                else:
                    results.append(result)
            if errors:
                raise errors[0]
    for row, conflicts in results:
        report['cases'].append(row)
        report['passed'] &= row['passed']
        if conflicts:
            report.setdefault('conflicting_judgments', {}).update(conflicts)
            report['passed'] = False
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
