import copy

from env_factory.tasks.task_spec import _capability_edges
from env_factory.sandbox_runtime import ContractRewardGate


def test_repeated_reads_keep_capture_values_from_their_own_call():
    steps = [
        {"operation": "tool_call", "tool_name": "read", "arguments": {"sku": "A"},
         "capture": {"a": "$.available"}},
        {"operation": "tool_call", "tool_name": "read", "arguments": {"sku": "B"},
         "capture": {"b": "$.available"}},
        {"operation": "tool_call", "tool_name": "write", "arguments": {
            "sku": "B", "remaining": {"$expr": {"op": "sub", "args": [{"$ref": "b"}, 2]}}}},
    ]
    gate = ContractRewardGate({"training_contract": {"category": "multi_step_agentic"},
        "capability_dag": {"nodes": ["read", "write"],
                           "edges": _capability_edges([{"kind": "goal_success", "steps": steps}])}},
        [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}])
    events = [{"event": "tool_call", "payload": {"tool_name": name, "arguments": args},
               "result": result} for name, args, result in [
        ("read", {"sku": "A"}, {"available": 8}),
        ("read", {"sku": "B"}, {"available": 14}),
        ("write", {"sku": "B", "remaining": 12}, {"updated": 1}),
    ]]
    def score(items):
        return gate.apply({"outcome": 1}, {"trajectory": {"events": items}})["outcome"]
    assert score(events) == 1
    replayed = copy.deepcopy(events)
    replayed[2]['event'] = 'tool_call_replay'
    replayed[2]['payload']['idempotent_replay'] = True
    assert score(replayed) == 1
    assert score([replayed[0], replayed[2], replayed[1]]) == 0
    assert score([events[0], events[2]]) == 0
    wrong = copy.deepcopy(events)
    wrong[2]["payload"]["arguments"]["remaining"] = 6
    assert score(wrong) == 0
    evidence = {}
    assert gate.apply({"outcome": 1}, {"trajectory": {"events": wrong}},
                      diagnostics=evidence) == {"outcome": 0}
    assert evidence['causal_passed'] is False
    assert evidence['dependency_path_passed'] is False
    assert any(not item['established'] for item in evidence['dependency_edges'])
    wrong = copy.deepcopy(events)
    wrong[2]["payload"]["arguments"]["sku"] = "A"
    assert score(wrong) == 0
    assert score([events[0], events[2], events[1]]) == 0
