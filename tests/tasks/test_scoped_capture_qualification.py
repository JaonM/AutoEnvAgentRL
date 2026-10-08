from copy import deepcopy

from env_factory.contracts.reward_contract import ambiguous_metric_captures


def test_qualification_checks_executable_script_options_instead_of_unused_fallback():
    positional = {'captures': [{'name': 'chosen', 'tool_name': 'offers', 'path': '$.records[0].id'}]}
    keyed = {'captures': [{'name': 'chosen', 'tool_name': 'offers', 'path': '$.records[?(@.id=="A")].id'}]}
    task = {'metric_implementations': [{'metric_id': 'select_offer', 'expected': {
        **deepcopy(positional), 'by_user_script': {'script-1': [keyed], 'script-2': [keyed]}}}]}
    preview = [{'tool_name': 'offers', 'result': {'records': [{'id': 'A'}, {'id': 'B'}]}}]
    assert ambiguous_metric_captures(task, preview) == []
    # A different script or a wildcard must not bypass target-selection checks.
    for name in ['script-2', '*']:
        bad = deepcopy(task)
        bad['metric_implementations'][0]['expected']['by_user_script'][name] = [positional]
        assert ambiguous_metric_captures(bad, preview)[0]['code'] == 'AMBIGUOUS_METRIC_CAPTURE'
