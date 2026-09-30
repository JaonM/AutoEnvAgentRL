import copy
import pytest
from env_factory.generation.semantic_reward import compile_semantic_outcome


def fixture():
    schema = {"type": "object", "properties": {"why": {"type": "string"}},
              "required": ["why"], "additionalProperties": False}
    item = {"id": "explanation", "rubric": "Explain why the heading is broad", "semantic": {
        "fields": ["why"], "criteria": ["The heading groups ingredients with different culinary roles."],
        "cases": [{"answer": {"why": text}, "expected": label} for text, label in [
            ("It groups ingredients with different roles.", "pass"),
            ("The label covers several distinct culinary functions.", "pass"),
            ("Every ingredient has the same function.", "fail"), ("seasonings", "fail")]]}}
    return item, schema


def test_compiles_public_field_judge_without_leaking_calibration_answers():
    item, schema = fixture()
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=0.5)
    assert metric["type"] == "model-based"
    assert metric["semantic_fields"] == ["why"]
    assert "cases" not in metric
    cases[0]["answer"]["why"] = "changed"
    assert item["semantic"]["cases"][0]["answer"]["why"] != "changed"


@pytest.mark.parametrize("change", ["enum", "duplicate", "no_negative", "wrong_shape", "mixed_rule"])
def test_rejects_unusable_semantic_calibration(change):
    item, schema = fixture()
    if change == "enum": schema["properties"]["why"]["enum"] = ["canonical"]
    if change == "duplicate": item["semantic"]["cases"][1] = copy.deepcopy(item["semantic"]["cases"][0])
    if change == "no_negative":
        for case in item["semantic"]["cases"]: case["expected"] = "pass"
    if change == "wrong_shape": item["semantic"]["cases"][0]["answer"] = {"other": "answer"}
    if change == "mixed_rule": item["rule"] = {}
    with pytest.raises(ValueError, match="SEMANTIC_REWARD"):
        compile_semantic_outcome(item, schema=schema, weight=0.5)


def test_compact_semantic_outcome_publishes_criteria_but_not_calibration_cases():
    import json
    from env_factory.generation.agent_authoring import expand_source
    from env_factory.generation.answer_contract import bind_answer_contract
    item, schema = fixture()
    source = {"business_tools": [], "outcomes": [item],
        "description": {"public_input": {"initial_user_message": "Explain the heading.", "materials": []}},
        "answer_contract": {"format": "json_object", "schema": schema},
        "reference": {"calls": [], "answer": json.dumps(item["semantic"]["cases"][0]["answer"])}}
    expanded = expand_source(source, {"seed": 1})
    assert expanded["metric_implementations"] == []
    assert len(expanded["semantic_calibration"]["explanation"]) == 4
    bind_answer_contract(expanded)
    public = expanded["description"]["public_input"]
    assert public["semantic_answer_criteria"][0]["fields"] == ["why"]
    assert "faithful paraphrases" in public["initial_user_message"]
    assert "Every ingredient has the same function." not in str(public)


def test_semantic_and_deterministic_fields_cannot_overlap():
    from env_factory.generation.answer_contract import bind_answer_contract
    item, schema = fixture()
    metric, _ = compile_semantic_outcome(item, schema=schema, weight=0.5)
    source = {"metrics": [metric], "answer_contract": {"format": "json_object", "schema": schema},
        "metric_implementations": [{"source": "final_agent_response", "operator": "value_targets", "path": "$",
            "expected": {"targets": [{"key": "why", "expression": {"literal": "canonical"}}]}}]}
    with pytest.raises(ValueError, match="ANSWER_CONTRACT_COVERAGE"):
        bind_answer_contract(source)


def test_calibration_requires_actual_judgments_including_negative_cases(tmp_path, monkeypatch):
    from unittest.mock import patch
    from env_factory.generation.semantic_reward import calibrate_semantic_outcomes, SemanticCalibrationUnavailable
    from env_factory.sandbox_runtime import EpisodeStore
    from env_factory.runtime_llm import RuntimeLLMError
    item, schema = fixture()
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    contract = {"metrics": [metric], "public_input": {"initial_user_message": "Explain the broad heading."}}
    monkeypatch.setenv("SANDBOX_LLM_API_KEY", "offline-test")
    monkeypatch.setenv("SANDBOX_LLM_MODEL", "offline-test")
    monkeypatch.delenv("SANDBOX_EVALUATOR_MOCK", raising=False)
    store = EpisodeStore(tmp_path / "calibration.sqlite3")
    store.reset(episode_id="calibration", seed=1)
    with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", side_effect=[
            {"label": "pass"}, {"label": "pass"}, {"label": "fail"}, {"label": "fail"}]):
        report = calibrate_semantic_outcomes(contract, {metric["id"]: cases}, context={}, store=store)
    assert report["passed"] and len(report["cases"]) == 4
    store.reset(episode_id="infra", seed=2)
    with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", side_effect=RuntimeLLMError("offline")):
        with pytest.raises(SemanticCalibrationUnavailable):
            calibrate_semantic_outcomes(contract, {metric["id"]: cases}, context={}, store=store)
    monkeypatch.setenv("SANDBOX_EVALUATOR_MOCK", "1")
    with pytest.raises(SemanticCalibrationUnavailable, match="mock"):
        calibrate_semantic_outcomes(contract, {metric["id"]: cases}, context={}, store=store)


def test_full_compiler_executes_semantic_calibration(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    from .test_agent_authoring import fixture as task_fixture
    from env_factory.generation.agent_authoring import compile_source
    item, schema = fixture()
    source, request = task_fixture("direct_response")
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    source.update(metrics=[metric], metric_implementations=[],
        semantic_calibration={metric["id"]: cases}, answer_contract={"format": "json_object", "schema": schema})
    source["description"]["public_input"]["initial_user_message"] = "Explain why seasonings groups ingredients with distinct culinary functions."
    for scene in source["scenarios"]:
        for step in scene["steps"]:
            if step.get("operation") == "agent_response":
                step["content"] = json.dumps(cases[0 if scene.get("kind") == "goal_success" else 2]["answer"])
    monkeypatch.setenv("SANDBOX_LLM_API_KEY", "offline-test")
    monkeypatch.setenv("SANDBOX_LLM_MODEL", "offline-test")
    monkeypatch.delenv("SANDBOX_EVALUATOR_MOCK", raising=False)
    def judge(messages, **kwargs):
        evidence = json.loads(messages[1]["content"])["runtime_evidence"]
        answer = evidence["final_agent_response"]
        return {"label": "pass" if any(c["expected"] == "pass" and c["answer"]["why"] in answer for c in cases) else "fail"}
    with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", side_effect=judge):
        result = compile_source(source, root=tmp_path, request=request)
    assert result["task_readiness"]["ready"]
    report = json.loads((tmp_path / "semantic_calibration.json").read_text())
    assert report["passed"] and len(report["cases"]) == 4


def test_mixed_reward_preserves_whole_answer_shape_validation():
    import json
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
    schema = {"type": "object", "properties": {"role": {"type": "string"}, "why": {"type": "string"}},
        "required": ["role", "why"], "additionalProperties": False}
    rule = {"metric_id": "fact", "source": "final_agent_response", "path": "$", "operator": "value_targets",
        "expected": {"answer_format": "json_object", "answer_schema": schema,
            "targets": [{"key": "role", "expression": {"literal": "herb"}}]},
        "score_mapping": {"pass": 1, "fail": 0}}
    evaluator = DeclarativeMetricEvaluator()
    for answer, expected in [({"role": "herb", "why": "An explanation"}, 1),
        ({"role": "root", "why": "An explanation"}, 0), ({"role": "herb"}, 0),
        ({"role": "herb", "why": True}, 0), ({"role": "herb", "why": "x", "extra": "x"}, 0)]:
        assert evaluator.evaluate_all([rule], {"business_state": {}, "final_agent_response": json.dumps(answer)}) == {"fact": expected}


def test_semantic_judge_and_mock_are_scoped_to_owned_fields(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    from env_factory.sandbox_runtime import EpisodeStore, ContractModelMetricEvaluator
    item, schema = fixture()
    schema['properties']['role'] = {'type':'string'}
    schema['required'].append('role')
    for case in item['semantic']['cases']: case['answer']['role'] = 'herb'
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=0.5)
    reference = cases[0]['answer']
    contract = {'metrics':[metric], 'public_input':{'answer_contract':{'schema':schema}},
        'acceptance_contract':{'executable_scenarios':[{'kind':'goal_success','steps':[
            {'operation':'agent_response','content':json.dumps(reference)}]}]}}
    store = EpisodeStore(tmp_path/'scope.sqlite3');store.reset(episode_id='scope',seed=1)
    judge = ContractModelMetricEvaluator(contract, store)
    context = {'final_agent_response':json.dumps({**reference,'role':'wrong classification'})}
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK','1')
    assert judge.evaluate_all(context,{}) == {'explanation':1}
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK','0')
    monkeypatch.setenv('SANDBOX_LLM_API_KEY','offline-test')
    monkeypatch.setenv('SANDBOX_LLM_MODEL','offline-test')
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat',return_value={'label':'pass'}) as call:
        assert judge.evaluate_all(context,{}) == {'explanation':1}
    sent = json.loads(call.call_args.args[0][1]['content'])['runtime_evidence']
    assert json.loads(sent['final_agent_response']) == {'why':reference['why']}
