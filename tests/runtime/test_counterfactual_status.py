import pytest

from env_factory.sandbox_runtime import AcceptanceScenarioRunner, SandboxError


def test_untriggered_fault_executes_remaining_steps_and_assertions():
    calls = []
    def call(method, path, body, headers):
        calls.append(path)
        return 200, {'reward': 1}, {}
    runner = AcceptanceScenarioRunner(call)
    scenario = {'steps': [{'operation': 'tool_call', 'tool_name': 'write', 'expected_status': 504},
                          {'operation': 'reward', 'step_id': 'reward'}],
                'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'lte', 'expected': 0}]}
    with pytest.raises(SandboxError, match='expected HTTP 504'):
        runner.run(scenario)
    calls.clear()
    with pytest.raises(SandboxError, match='scenario assertion failed'):
        runner.run(scenario, allow_untriggered_faults=True)
    assert calls == ['/v1/tools/write', '/v1/reward']


def test_unexpected_server_error_is_not_accepted_for_ablation():
    runner = AcceptanceScenarioRunner(lambda *args: (500, {}, {}))
    with pytest.raises(SandboxError, match='got 500'):
        runner.run({'steps': [{'operation': 'tool_call', 'tool_name': 'write'}]},
                   allow_untriggered_faults=True)


def test_shifted_fault_needs_trusted_evidence_and_still_checks_reward():
    calls = []
    def call(method, path, body, headers):
        calls.append(path)
        if path.startswith('/v1/tools/'):
            return 504, {'error': {'code': 'RESPONSE_LOST'}}, {}
        return 200, {'reward': 1}, {}
    scenario = {'steps': [{'operation': 'tool_call', 'tool_name': 'write'},
                          {'operation': 'reward', 'step_id': 'reward'}],
                'assertions': [{'source': 'step:reward', 'path': '$.reward',
                                'operator': 'lte', 'expected': 0}]}
    with pytest.raises(SandboxError, match='expected HTTP 200'):
        AcceptanceScenarioRunner(call).run(scenario, allow_untriggered_faults=True)
    runner = AcceptanceScenarioRunner(call,
        injected_fault_check=lambda name, status: name == 'write' and status == 504)
    with pytest.raises(SandboxError, match='expected HTTP 200'):
        runner.run(scenario)
    calls.clear()
    with pytest.raises(SandboxError, match='scenario assertion failed'):
        runner.run(scenario, allow_untriggered_faults=True)
    assert calls == ['/v1/tools/write', '/v1/reward']
