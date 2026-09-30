from copy import deepcopy
from env_factory.sandbox_runtime import BusinessGoalEvaluator


def test_broad_selector_does_not_allow_collateral_write_outside_goal_values():
    baseline = {"records": [{"id": "target", "amount": 10}, {"id": "other", "amount": 20}]}
    goal = {"row_predicates": [{"table": "records", "where": {}, "values": {"amount": 12}, "count": 1}],
            "table_primary_keys": {"records": ["id"]}}
    state = deepcopy(baseline)
    state["records"][0]["amount"] = 12
    assert BusinessGoalEvaluator.evaluate(goal["row_predicates"], state, baseline)
    assert BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
    state["records"][1]["amount"] = 21
    assert BusinessGoalEvaluator.evaluate(goal["row_predicates"], state, baseline)
    assert not BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)


def test_dynamic_identity_value_restricts_the_modified_record():
    baseline = {"links": [{"ticket": "T", "record": "B"}],
                "records": [{"id": "A", "amount": 10}, {"id": "B", "amount": 20}]}
    goal = {"row_predicates": [{"table": "records", "where": {}, "values": {"amount": 22},
            "value_expressions": {"id": {"initial": {"lookup": {"table": "links", "field": "record",
                                                               "where": {"ticket": "T"}}}}}, "count": 1}],
            "table_primary_keys": {"records": ["id"], "links": ["ticket"]}}
    state = deepcopy(baseline)
    state["records"][1]["amount"] = 22
    assert BusinessGoalEvaluator.evaluate(goal["row_predicates"], state, baseline)
    assert BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
    state["records"][0]["amount"] = 22
    assert BusinessGoalEvaluator.evaluate(goal["row_predicates"], state, baseline)
    assert not BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)


def test_explicit_noop_requires_observation_and_preserves_all_state():
    from env_factory.sandbox_runtime import ContractRewardGate
    from env_factory.tasks.task_spec import validate_goal_contract, TaskSpecError
    import pytest
    goal = {"allow_noop": True, "requires_state_change": False,
            "row_predicates": [{"table": "items", "where": {"id": 1},
                                "values": {"quantity": 5}, "count": 1}],
            "table_primary_keys": {"items": ["id"]}}
    tables = [{"table_name": "items", "rows": [{"id": 1, "quantity": 5}]}]
    validate_goal_contract(goal, tables)
    with pytest.raises(TaskSpecError, match="already satisfies"):
        validate_goal_contract({**goal, "allow_noop": False}, tables)
    gate = ContractRewardGate({"training_contract": {"category": "simple_agentic"},
        "capability_dag": {"nodes": ["inspect"]}, "goal_contract": goal},
        [{"id": "goal", "category": "outcome", "score_range": [0, 1]}])
    baseline = {"items": [{"id": 1, "quantity": 5}, {"id": 2, "quantity": 7}]}
    context = {"initial_business_state": baseline, "business_state": deepcopy(baseline),
               "trajectory": {"events": []}}
    assert gate.apply({"goal": 1}, context)["goal"] == 0
    context["trajectory"]["events"] = [{"event": "tool_call", "payload": {"tool_name": "inspect"}}]
    assert gate.apply({"goal": 1}, context)["goal"] == 1
    context["business_state"]["items"][1]["quantity"] = 8
    assert gate.apply({"goal": 1}, context)["goal"] == 0


def test_noop_permission_still_requires_reaching_unsatisfied_goal():
    from env_factory.sandbox_runtime import ContractRewardGate
    goal = {"allow_noop": True, "requires_state_change": False,
            "row_predicates": [{"table": "items", "where": {"id": 1}, "values": {"quantity": 5}, "count": 1}],
            "table_primary_keys": {"items": ["id"]}}
    gate = ContractRewardGate({"training_contract": {"category": "simple_agentic"},
        "capability_dag": {"nodes": ["update"]}, "goal_contract": goal},
        [{"id": "goal", "category": "outcome", "score_range": [0, 1]}])
    baseline = {"items": [{"id": 1, "quantity": 2}]}
    context = {"initial_business_state": baseline, "business_state": deepcopy(baseline),
        "trajectory": {"events": [{"event": "tool_call", "payload": {"tool_name": "update"}}]}}
    assert gate.apply({"goal": 1}, context)["goal"] == 0
    context["business_state"]["items"][0]["quantity"] = 5
    assert gate.apply({"goal": 1}, context)["goal"] == 1
