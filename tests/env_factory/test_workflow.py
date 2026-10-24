"""Clock-free causal progression, branching, joins, transactionality and replay."""
import copy
import json
from pathlib import Path

import pytest

from env_factory import lifecycle
from env_factory.core import spec, reward
from env_factory.core.sandbox import load_module
from env_factory.runtime import Episode, BusinessError, ToolError, run_reference, write
from env_factory.serving.server import Service
from env_factory.serving import agent_view
from env_factory.certify import gates, bundle

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples' / 'returns_workflow'


@pytest.fixture
def setup():
    return (spec.load(EXAMPLE / 'request.json'), load_module(EXAMPLE / 'env.py', role='env'),
            load_module(EXAMPLE / 'verifier.py', role='verifier'), json.loads((EXAMPLE / 'answer_schema.json').read_text()))


def episode(setup, **changes):
    request, env, _, _ = setup
    request = copy.deepcopy(request)
    request['scenario']['parameters'].update(changes)
    return Episode(env.make_tools(), request['tables'], request['user_facts'],
                   approve=lambda *_: True, **lifecycle.options(request))


def apply(api, name='request_return', key='K1', entity='R1'):
    return api.call(name, return_id=entity, request_id=key)


def status(api):
    return api.db['returns'][0]['status']


def test_one_external_activity_per_handoff_and_queries_never_advance(setup):
    api = episode(setup)
    assert api.yield_control() == {'progressed': False, 'status': 'blocked'}
    assert status(api) == 'none'
    with pytest.raises(BusinessError):
        apply(api, 'request_refund')
    apply(api)
    for target in ('in_transit', 'received', 'accepted'):
        before = api.lifecycle.snapshot()
        for _ in range(3):
            api.call('get_return', return_id='R1')
        assert before == api.lifecycle.snapshot()
        assert api.yield_control() == {'progressed': True, 'status': 'resumed'}
        assert status(api) == target and api.now == 0
    apply(api, 'request_refund', key='PAY')
    assert status(api) == 'refund_pending' and api.db['returns'][0]['refund'] == 0
    api.yield_control()
    assert status(api) == 'refunded' and api.db['returns'][0]['refund'] == 1000
    log = api.lifecycle.log
    assert [entry['sequence'] for entry in log] == list(range(1, len(log) + 1))
    assert all('time' not in entry and 'at' not in entry for entry in log)
    assert api.db['returns'][1] == setup[0]['tables']['returns'][1]
    with pytest.raises(ValueError, match='yield'):
        api.wait(10)


@pytest.mark.parametrize('params,target', [({'will_ship': False}, 'requested'),
    ({'inspection_pass': False}, 'rejected'), ({'payment_success': False}, 'refund_failed'), ({}, 'refunded')])
def test_reference_and_independent_oracle_cover_outcome_branches(setup, params, target):
    request, env, verifier, schema = setup
    request['scenario']['parameters'].update(params)
    result = run_reference(env, request['tables'], request['user_facts'], request=request)
    assert result['answer']['status'] == target
    assert not lifecycle.audit(verifier, request['tables'], request['user_facts'], 0, result, request)
    expected = lifecycle.expected(verifier, request['tables'], request['user_facts'], 0, request)
    assert reward.grade(result, expected, schema, initial=request['tables'], reference_calls=result['tool_calls'])['success']


def test_blocked_handoffs_and_customer_claims_cannot_create_delivery(setup):
    class User:
        def chat(self, *args, **kwargs):
            return {'content': '我已经寄回了，仓库已经签收。'}
    request, env, verifier, schema = setup
    request['scenario']['parameters']['will_ship'] = False
    service = Service(env, verifier, schema, request, perturbed=0, user_mode='natural', user_client=User(), audit=False)
    identifier = service.reset()['episode']
    service.step(identifier, {'type': 'chat', 'message': '是否寄回？'})
    api = service.episodes[identifier]['episode']
    api.approve = lambda *_: True
    apply(api)
    for _ in range(5):
        assert not api.yield_control()['progressed']
    assert status(api) == 'requested' and not api.lifecycle.pending


def test_cancellation_removes_pending_work_and_idempotent_retry_does_not_reactivate(setup):
    api = episode(setup)
    receipt = apply(api)
    apply(api, 'cancel_return', key='C')
    assert status(api) == 'cancelled' and not api.lifecycle.pending
    assert any(e['kind'] == 'event_cancelled' for e in api.lifecycle.log)
    before = api.lifecycle.snapshot()
    assert apply(api) == receipt and before == api.lifecycle.snapshot()
    assert not api.yield_control()['progressed']


def test_history_replays_commands_and_handoffs_without_a_clock(setup):
    request, env, verifier, _ = setup
    request['scenario']['history'] = [
        {'type': 'command', 'transition': 'apply', 'entity': 'R1', 'args': {}, 'key': 'past'},
        {'type': 'yield'}, {'type': 'yield'}]
    api = episode(setup)
    assert status(api) == 'received' and api.now == 0
    assert api.lifecycle.pending[0]['transition'] == 'accept'
    result = run_reference(env, request['tables'], request['user_facts'], request=request)
    assert not lifecycle.audit(verifier, request['tables'], request['user_facts'], 0, result, request)
    assert result['answer']['status'] == 'refunded'


@pytest.mark.parametrize('mutate', [
    lambda r: r['lifecycle_schema']['transitions']['apply']['emit'][0].update(delay=0),
    lambda r: r['scenario'].update(history=[{'offset': -1, 'transition': 'apply', 'entity': 'R1', 'args': {}, 'key': 'past'}]),
    lambda r: r['scenario'].update(history=[{'type': 'command', 'transition': 'receive', 'entity': 'R1', 'args': {}, 'key': 'past'}]),
    lambda r: r['scenario'].update(history=[{'type': 'yield'}]),
    lambda r: r['scenario'].update(dispatch_order=['refund']),
    lambda r: r['lifecycle_schema']['transitions']['handover'].update(execution='internal'),
    lambda r: r['lifecycle_schema']['transitions']['refund'].update(requires=[{'transition': 'missing', 'entity': 'R1'}]),
    lambda r: r['scenario'].update(version=1),
    lambda r: r['lifecycle_schema']['transitions']['refund'].update(requires={}),
])
def test_schema_rejects_timing_injection_and_invalid_graph(setup, mutate):
    mutate(setup[0])
    assert spec.validate(setup[0])


def test_scope_and_calendar_are_business_facts_not_progress_drivers(setup):
    request, env, verifier, schema = setup
    request['scenario']['scope'] = request['user_facts']['scope'] = 'request_only'
    request['time']['start'] = '2026-10-24T12:00'
    result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'], request=request)
    assert result['answer'] == {'status': 'requested', 'refund': 0}
    assert result['lifecycle']['pending'] and result['now'] == request['time']['start']
    assert not lifecycle.audit(verifier, request['tables'], request['user_facts'], result['now'], result, request)


def test_service_fork_restore_and_public_action_do_not_expose_scheduler(setup):
    request, env, verifier, schema = setup
    service = Service(env, verifier, schema, request, perturbed=0)
    identifier = service.reset()['episode']
    args = {'return_id': 'R1', 'request_id': 'K1'}
    actions = [{'type': 'confirm', 'key': 'approve_return', 'proposal': {'tool': 'request_return', 'args': args}},
               {'type': 'tool', 'name': 'request_return', 'args': {**args, 'consent': 'consent-1'}}, {'type': 'yield'}]
    for action in actions:
        assert service.step(identifier, action)['ok']
    fork = service.fork(identifier)['episode']
    restored = service.restore('original', [{'action': a} for a in actions])['episode']
    original = service.episodes[identifier]['episode']
    for other in (fork, restored):
        assert service.episodes[other]['episode'].result() == original.result()
    service.step(fork, {'type': 'yield'})
    assert status(original) == 'in_transit' and status(service.episodes[fork]['episode']) == 'received'
    assert not service.step(identifier, {'type': 'yield', 'transition': 'settle'})['ok']
    instructions = service.instructions
    for mode in ('fixed', 'random', 'heldout'):
        names = agent_view.naming(42, mode)
        tools = agent_view.tools(instructions, request['category'], names)
        offered = [t['function']['name'] for t in tools]
        assert names['v3_yield'] in offered and names['v3_wait'] not in offered
        assert agent_view.action(names['v3_yield'], {}, env.make_tools(), names) == {'type': 'yield'}
        with pytest.raises(ValueError):
            agent_view.action(names['v3_yield'], {'transition': 'settle'}, env.make_tools(), names)


@pytest.mark.parametrize('forgery', ['remove_handoff', 'remove_receive', 'duplicate_payment', 'change_pending', 'change_trace'])
def test_independent_audit_rejects_correct_answer_with_forged_causality(setup, forgery):
    request, env, verifier, schema = setup
    result = run_reference(env, request['tables'], request['user_facts'], request=request)
    log = result['lifecycle']['log']
    if forgery == 'remove_handoff':
        log.pop(next(i for i, e in enumerate(log) if e['kind'] == 'handoff'))
    elif forgery == 'remove_receive':
        log.pop(next(i for i, e in enumerate(log) if e.get('transition') == 'receive' and e['kind'] == 'transition'))
    elif forgery == 'duplicate_payment':
        log.append(copy.deepcopy(log[-1]))
    elif forgery == 'change_pending':
        result['lifecycle']['pending'].append({'transition': 'settle'})
    else:
        result['trace'] = [e for e in result['trace'] if e['kind'] != 'yield']
    result['lifecycle_errors'] = lifecycle.audit(verifier, request['tables'], request['user_facts'], 0, result, request)
    assert result['lifecycle_errors']
    expected = lifecycle.expected(verifier, request['tables'], request['user_facts'], 0, request)
    grade = reward.grade(result, expected, schema, initial=request['tables'], reference_calls=result['tool_calls'])
    assert grade['state_ok'] and grade['reward'] == 0


def test_full_gates_export_http_and_reload(setup, tmp_path):
    request, env, verifier, schema = setup
    report = gates.qualify(env, verifier, schema, request, perturbed=20)
    assert report['passed'], report
    bundle.export(EXAMPLE, EXAMPLE / 'request.json', report, tmp_path / 'bundle', status='ready')
    assert bundle.load_bundle(tmp_path / 'bundle')['gates']['lifecycle']['passed']


def test_author_and_verifier_selfchecks():
    from env_factory import checks
    assert checks.check_env(EXAMPLE)[0] == []
    assert checks.check_verifier(EXAMPLE)[0] == []


def graph_request():
    """Exchange: warehouse inspection and stock reservation converge before approval."""
    entity = {'path': 'entity'}
    machines = {name: {'table': name, 'field': 'status', 'initial': 'pending', 'states': ['pending', 'done']}
                for name in ('exchange', 'inspection', 'inventory')}
    data = {name: [{'id': identifier, 'status': 'pending'} for identifier in ('R1', 'R2')]
            for name in machines}
    data['control'] = [{'id': 'control', 'ready': False}]
    def rule(machine, actor, **extra):
        return {'machine': machine, 'actor': actor, 'from': ['pending'], 'to': 'done', **extra}
    transitions = {
        'start': {'machine': 'exchange', 'actor': 'agent', 'from': ['pending'], 'to': 'pending',
             'emit': [{'transition': name, 'entity': entity} for name in ('inspect', 'reserve', 'approve')]},
        'inspect': rule('inspection', 'warehouse'),
        'reserve': rule('inventory', 'system'),
        'approve': rule('exchange', 'system', execution='internal', requires=[
            {'transition': name, 'entity': entity} for name in ('inspect', 'reserve')]),
    }
    schema = {'version': 2, 'machines': machines, 'transitions': transitions, 'parameters': {}, 'scopes': ['exchange'],
              'tools': {'start': {'transition': 'start', 'entity_arg': 'identifier', 'key_arg': 'key'}}}
    scenario = {'version': 2, 'scope': 'exchange', 'parameters': {}}
    return schema, scenario, data


def graph_episode(schema, scenario, data, tools=None):
    @write
    def start(db, now, identifier: str, key: str):
        return next(dict(r) for r in db['exchange'] if r['id'] == identifier)
    return Episode({'start': start, **(tools or {})}, data, {}, approve=lambda *_: True,
                   lifecycle_schema=schema, scenario=scenario)


@pytest.mark.parametrize('order', [['inspect', 'reserve'], ['reserve', 'inspect']])
def test_parallel_branches_join_only_after_both_same_object_predecessors(order):
    schema, scenario, data = graph_request()
    scenario['dispatch_order'] = order
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    assert len(api.lifecycle.pending) == 3
    api.yield_control()
    assert api.db['exchange'][0]['status'] == 'pending'
    assert len([e for e in api.lifecycle.log if e['kind'] == 'transition']) == 2
    api.yield_control()
    assert api.db['exchange'][0]['status'] == 'done'
    transitions = [e for e in api.lifecycle.log if e['kind'] == 'transition']
    assert [e['transition'] for e in transitions] == ['start', *order, 'approve']
    witnesses = {e['id']: e for e in transitions}
    assert {witnesses[key]['transition'] for key in transitions[-1]['dependencies']} == {'inspect', 'reserve'}
    assert all(witnesses[key]['entity'] == 'R1' for key in transitions[-1]['dependencies'])
    assert not api.lifecycle.pending
    assert all(api.db[name][1] == data[name][1] for name in schema['machines'])


def test_join_cannot_reuse_other_objects_completed_work():
    schema, scenario, data = graph_request()
    # R1 completes inspection; R2 completes reservation; neither exchange may pass its join.
    schema['transitions']['start']['emit'][0]['when'] = {'op': 'eq', 'args': [{'path': 'entity'}, 'R1']}
    schema['transitions']['start']['emit'][1]['when'] = {'op': 'eq', 'args': [{'path': 'entity'}, 'R2']}
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    api.call('start', identifier='R2', key='B')
    api.yield_control(); api.yield_control()
    assert not api.yield_control()['progressed']
    assert [r['status'] for r in api.db['exchange']] == ['pending', 'pending']
    assert len(api.lifecycle.pending) == 2


def test_ready_predicate_blocks_without_discarding_and_control_write_does_not_run_external_work():
    schema, scenario, data = graph_request()
    schema['transitions']['start']['emit'][0]['ready'] = {'path': 'db.control.0.ready'}
    @write
    def enable(db, now):
        db['control'][0]['ready'] = True
        return True
    api = graph_episode(schema, scenario, data, {'enable': enable})
    api.call('start', identifier='R1', key='A')
    api.yield_control()  # reservation runs while inspection is blocked
    assert not api.yield_control()['progressed']
    api.call('enable')
    assert api.db['inspection'][0]['status'] == 'pending'
    assert api.yield_control()['progressed']
    assert api.db['exchange'][0]['status'] == 'done'


def test_explicit_cancellation_and_external_guard_failure_have_no_successors():
    schema, scenario, data = graph_request()
    schema['transitions']['start']['emit'][0]['cancel_when'] = True
    schema['transitions']['reserve']['guard'] = False
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    assert not api.yield_control()['progressed']
    assert api.db == data
    assert len([e for e in api.lifecycle.log if e['kind'] == 'event_cancelled']) == 2
    assert [e['transition'] for e in api.lifecycle.pending] == ['approve']


def test_internal_cycle_budget_rolls_back_command_and_all_consequences():
    schema, scenario, data = graph_request()
    schema['max_events'] = 3
    schema['transitions']['start']['emit'] = [{'transition': 'approve', 'entity': {'path': 'entity'}}]
    rule = schema['transitions']['approve']
    rule.update(to='pending', requires=[], emit=[{'transition': 'approve', 'entity': {'path': 'entity'}}])
    api = graph_episode(schema, scenario, data)
    before = api.lifecycle.snapshot()
    with pytest.raises(ToolError, match='budget'):
        api.call('start', identifier='R1', key='A')
    assert api.db == data and api.lifecycle.snapshot() == before


def test_handoff_rolls_back_external_transition_when_internal_consequence_fails():
    schema, scenario, data = graph_request()
    schema['max_events'] = 3
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    api.yield_control()
    # Force an infinite internal loop reached only once both branches have succeeded.
    api.lifecycle.schema['transitions']['approve'].update(to='pending', emit=[
        {'transition': 'approve', 'entity': {'path': 'entity'}}])
    before = api.result()
    with pytest.raises(ToolError, match='budget'):
        api.yield_control()
    assert api.db == before['final_state'] and api.lifecycle.snapshot() == before['lifecycle']
    assert api.now == before['now']


def test_retry_limit_bounds_external_loops_and_idempotency_does_not_spend_attempts():
    schema, scenario, data = graph_request()
    schema['transitions']['start']['emit'] = [{'transition': 'reserve', 'entity': {'path': 'entity'}}]
    schema['transitions']['reserve'].update(to='pending', max_occurrences=2, emit=[
        {'transition': 'reserve', 'entity': {'path': 'entity'}}])
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    snapshot = api.lifecycle.snapshot()
    api.call('start', identifier='R1', key='A')
    assert snapshot == api.lifecycle.snapshot()
    assert api.yield_control()['progressed'] and api.yield_control()['progressed']
    assert not api.yield_control()['progressed']
    assert not api.lifecycle.pending
    assert sum(e.get('transition') == 'reserve' and e['kind'] == 'transition' for e in api.lifecycle.log) == 2


def test_rl_adapter_fork_restore_continues_same_work_without_duplicate_payment(setup, tmp_path, monkeypatch):
    from rl.environments.sandbox import SandboxEpisode
    request, env, verifier, schema = setup
    report = gates.qualify(env, verifier, schema, request, perturbed=20)
    assert report['passed']
    root = tmp_path / 'bundle'
    bundle.export(EXAMPLE, EXAMPLE / 'request.json', report, root, status='ready')
    monkeypatch.setenv('ENVFACTORY_USER_MODE', 'structured')
    monkeypatch.setenv('ENVFACTORY_TOOL_NAMES', 'random')
    monkeypatch.setenv('ENVFACTORY_RESULT_FORMAT', 'envelope')
    runtimes = []
    try:
        original = SandboxEpisode(root)
        runtimes.append(original)
        original.reset(42, world='original')
        def send(runtime, name, args, identifier):
            runtime.step({'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]})
            result = json.loads(runtime.messages[-1]['content'])
            assert result['ok'], result
            return result['result']
        names = original.naming
        args = {'return_id': 'R1', 'request_id': 'A'}
        consent = send(original, names['v3_confirm'], {'key': 'approve_return',
                       'proposal': {'tool': 'request_return', 'args': args}}, 'confirm')
        send(original, 'request_return', {**args, names['consent']: consent}, 'apply')
        send(original, names['v3_yield'], {}, 'ship')
        saved = original.snapshot()
        fork = original.fork()
        runtimes.append(fork)
        restored = SandboxEpisode(root)
        runtimes.append(restored)
        restored.restore(saved)
        def api(runtime):
            return runtime.service.episodes[runtime.episode]['episode']
        assert api(original).result() == api(fork).result() == api(restored).result()
        for runtime in (fork, restored):
            send(runtime, names['v3_yield'], {}, 'receive')
            send(runtime, names['v3_yield'], {}, 'inspect')
            args = {'return_id': 'R1', 'request_id': 'PAY'}
            consent = send(runtime, names['v3_confirm'], {'key': 'approve_refund',
                           'proposal': {'tool': 'request_refund', 'args': args}}, 'confirm-pay')
            send(runtime, 'request_refund', {**args, names['consent']: consent}, 'pay')
            send(runtime, names['v3_yield'], {}, 'settle')
            runtime.step({'role': 'assistant', 'content': json.dumps({'status': 'refunded', 'refund': 1000})})
            assert runtime.finish()['final_reward'] == 1.0
            assert sum(e.get('transition') == 'settle' and e['kind'] == 'transition' for e in api(runtime).lifecycle.log) == 1
        assert api(fork).result() == api(restored).result()
        assert status(api(original)) == 'in_transit'
    finally:
        for runtime in reversed(runtimes):
            runtime.close()


def test_generation_brief_and_natural_interface_explain_clock_free_handoffs(setup):
    from env_factory.authoring import prompts
    request, env, verifier, schema = setup
    assert 'yield_control()' in spec.brief(request)
    assert 'explicit waits advance' not in spec.brief(request)
    assert 'lifecycle_schema version 2' in prompts.WORLD_AUTHOR
    assert 'yield_control' in prompts.AUTHOR and 'dependency' in prompts.VERIFIER
    service = Service(env, verifier, schema, request, perturbed=0)
    instructions = copy.deepcopy(service.instructions)
    instructions['user_mode'] = 'natural'
    names = agent_view.naming(42, 'heldout')
    messages = agent_view.messages(instructions, names)
    assert names['v3_yield'] in messages[0]['content']
    assert '当前时间：0' not in messages[0]['content']
    assert names['v3_wait'] not in messages[0]['content']
    for private in ('dispatch_order', 'inspection_pass', 'lifecycle_schema', 'pending'):
        assert private not in messages[0]['content']


def test_rejected_external_attempt_resumes_so_another_branch_can_continue():
    schema, scenario, data = graph_request()
    schema['transitions']['inspect']['guard'] = False
    api = graph_episode(schema, scenario, data)
    api.call('start', identifier='R1', key='A')
    assert api.yield_control() == {'progressed': False, 'status': 'resumed'}
    assert api.yield_control() == {'progressed': True, 'status': 'resumed'}
    assert api.db['inventory'][0]['status'] == 'done'
    assert api.yield_control() == {'progressed': False, 'status': 'blocked'}
    assert api.db['exchange'][0]['status'] == 'pending'


def test_sop_graph_and_trace_replay_include_handoffs(setup):
    from env_factory import sop
    request, env, _, _ = setup
    request['sop_blueprint'] = {'atoms': {
        name: {'tool': name, 'reads': ['returns'], 'writes': [] if name == 'get_return' else ['returns'],
               'inputs': {'type': 'object'}, 'output': {'type': 'object'}} for name in env.make_tools()}}
    assert sop.trace_report(env, request)['passed']
    def check(target):
        return [{'name': 'observed status', 'assert': {'op': 'eq', 'args': [{'path': 'result.status'}, target]}}]
    case = {'data': request['tables'], 'now': 0, 'graph': {'nodes': [
        {'id': 'apply', 'atom': 'request_return', 'args': {'return_id': 'R1', 'request_id': 'A'}, 'checks': check('requested')},
        {'id': 'ship', 'atom': 'get_return', 'args': {'return_id': 'R1'}, 'yield_before': True, 'checks': check('in_transit')}],
        'edges': [{'from': 'apply', 'to': 'ship', 'kind': 'state'}]}}
    assert sop.run_graph(env, request['sop_blueprint'], case, request=request)['passed']
    case['graph']['nodes'][1]['yield_before'] = 'true'
    assert not sop.run_graph(env, request['sop_blueprint'], case, request=request)['passed']
