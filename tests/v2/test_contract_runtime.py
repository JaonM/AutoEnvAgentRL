import copy
import json
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.examples import EXAMPLES, direct, simple, retrieval, transaction, lit, ref, answer
from env_factory_v2.export import export_bundle, verify_bundle
from env_factory_v2.reward import equivalent, grade
from env_factory_v2.runtime import Sandbox
from env_factory_v2.server import serve
from env_factory_v2.user_simulator import validate_language_event
from env_factory_v2.verification import qualify, run_scenario
from env_factory_v2.api_verification import verify_api


@pytest.mark.parametrize('factory', list(EXAMPLES.values()))
def test_complete_joint_execution_and_independent_replay(factory):
    compiled, report = qualify(factory())
    assert report['gates']['joint_execution_and_counterexamples']
    assert report['gates']['deterministic_reset_and_replay']
    assert not report['passed']  # Offline fixtures cannot impersonate native qualification.
    assert not report['container_verified']


@pytest.mark.parametrize('factory', list(EXAMPLES.values()))
def test_http_joint_trajectory_matches_in_process_world(factory):
    compiled, report = qualify(factory())
    evidence = verify_api(compiled['source'], report)
    assert evidence['passed'] and evidence['reward'] == 1


def test_page_collection_rejects_invented_or_incomplete_menu():
    source = simple()
    source['tables'][0]['rows'].append({'id': 'I3', 'size': 'M', 'display': '第二款'})
    source['tools'][0]['page_size'] = 1
    world = Sandbox(source)
    world.interact('size', question='尺码？')
    first = world.call('find_items', {'size': 'M'}, call_id='first')
    with pytest.raises(ContractError, match='actual end'):
        world.collect(['first'], call_id='incomplete')
    second = world.call('find_items', {'size': 'M'}, call_id='second', cursor=first['next_cursor'])
    assert second['next_cursor'] is None
    with pytest.raises(ContractError, match='unique observed pages'):
        world.collect(['first', 'first', 'second'], call_id='duplicates')
    with pytest.raises(ContractError, match='contiguous'):
        world.collect(['second', 'first'], call_id='wrong_order')
    assert len(world.collect(['first', 'second'], call_id='complete')['items']) == 2


def test_extra_eligibility_filter_cannot_be_hidden_in_tool():
    source = simple()
    source['tools'][0]['filters'] = [{'column': 'display', 'equals': '青松'}]
    with pytest.raises(ContractError, match='fields mismatch'):
        compile_contract(source)


def test_reward_cannot_rewrite_a_shared_eligibility_predicate():
    source = simple()
    source['answers'][0]['expected']['map']['items']['filter']['value'] = {
        'eq': [ref('row.size'), ref('user.size')]}
    with pytest.raises(ContractError, match='invoke named public rules'):
        compile_contract(source)


def test_rule_invocation_requires_exact_arguments_and_known_identity():
    source = simple()
    source['answers'][0]['expected']['map']['items']['filter']['value']['rule']['arguments'] = {}
    with pytest.raises(ContractError, match='rule argument mismatch'):
        compile_contract(source)


def test_public_wire_schema_matches_required_capture_object():
    compiled = compile_contract(retrieval())
    tool = next(tool for tool in compiled['public']['tools'] if tool['id'] == 'find_stock')
    assert tool['business_parameters']['properties']['brand_id']['type'] == 'string'
    assert tool['parameters']['properties']['brand_id']['type'] == 'object'
    assert set(tool['parameters']['properties']['brand_id']['properties']) == {'$result', 'path'}


def test_authored_tool_prose_cannot_override_actual_public_rules():
    source = simple()
    source['tools'][0]['description'] = 'Only garments named 青松 qualify.'
    compiled = compile_contract(source)
    assert 'Only garments named' not in compiled['public']['tools'][0]['description']
    assert '显示名称不影响资格' in compiled['public']['tools'][0]['description']


def test_descending_query_keeps_nulls_last():
    source = simple()
    source['tables'][0]['schema']['properties']['display']['nullable'] = True
    source['tables'][0]['rows'][1]['size'] = 'M'
    source['tables'][0]['rows'][1]['display'] = None
    source['tools'][0]['order_by'] = [{'field': 'display', 'direction': 'desc'}]
    world = Sandbox(source)
    world.interact('size', question='尺码？')
    result = world.call('find_items', {'size': 'M'}, call_id='found')
    assert [row['id'] for row in result['items']] == ['I1', 'I2']


def test_boolean_guard_short_circuits_unavailable_optional_fact():
    from env_factory_v2.expressions import evaluate
    assert evaluate({'and': [lit(False), ref('user.missing')]}, {'user': {}}) is False
    assert evaluate({'or': [lit(True), ref('user.missing')]}, {'user': {}}) is True


def test_even_public_extra_filter_is_exposed_by_independent_display_probe():
    source = simple()
    source['rules'].append({'id': 'wrong_display', 'public': '显示名称必须为青松。',
                           'predicate': {'eq': [ref('row.display'), lit('青松')]}})
    source['tools'][0]['rule_ids'].append('wrong_display')
    _, report = qualify(source)
    assert not report['gates']['joint_execution_and_counterexamples']
    assert next(r for r in report['records'] if r['id'] == 'display_rename')['first']['outcome']['success'] is False


def test_reward_does_not_accept_stale_record_facts():
    source = simple()
    _, report = qualify(source)
    changed = next(r for r in report['records'] if r['id'] == 'changed_size')
    stale = next(r for r in report['records'] if r['id'] == 'stale_size_answer')
    assert changed['first']['outcome']['reward'] == 1
    assert stale['first']['outcome']['reward'] == 0


def test_cannot_guess_undisclosed_user_fact():
    world = Sandbox(simple())
    with pytest.raises(ContractError, match='disclosed'):
        world.call('find_items', {'size': 'M'}, call_id='guess')
    assert not world.user.disclosed


def test_direct_answer_cannot_get_full_credit_after_unnecessary_tool_attempt():
    world = Sandbox(direct())
    with pytest.raises(ContractError, match='unknown business tool'):
        world.call('invented_tool', {}, call_id='extra')
    assert grade(world, {'temperature': 18})['reward'] == 0


def test_simple_route_cannot_be_certified_with_multiple_business_calls():
    world = Sandbox(simple())
    world.interact('size', question='尺码？')
    world.call('find_items', {'size': 'M'}, call_id='one')
    world.call('find_items', {'size': 'M'}, call_id='two')
    result = grade(world, {'ids': ['I1']})
    assert not result['category_valid'] and result['reward'] == 0


def test_literal_matching_value_does_not_replace_required_result_dependency():
    source = retrieval(4)
    world = Sandbox(source)
    world.interact('brand', question='品牌？')
    world.call('resolve_brand', {'name': '青松'}, call_id='b')
    world.interact('size', question='尺码？')
    with pytest.raises(ContractError, match='observed-result dependency'):
        world.call('find_stock', {'brand_id': 'B1', 'size': 'M'}, call_id='s')


def test_full_answer_guess_does_not_replace_pagination():
    source = retrieval(64)
    scenario = copy.deepcopy(source['scenarios'][1])
    scenario['answer'] = {'ids': lit([f'I{i:04d}' for i in range(64)])}
    run = run_scenario(compile_contract(source), scenario)
    assert run['error'] is None
    assert run['outcome']['components']['ids']['passed']
    assert run['outcome']['reward'] == 0


def test_cursors_do_not_survive_reset_or_cross_queries():
    source = simple()
    source['tables'][0]['rows'].append({'id': 'I3', 'size': 'M', 'display': '第二款'})
    source['tools'][0]['page_size'] = 1
    world = Sandbox(source)
    world.interact('size', question='尺码？')
    result = world.call('find_items', {'size': 'M'}, call_id='one')
    cursor = result['next_cursor']
    assert cursor
    world.reset()
    world.interact('size', question='尺码？')
    with pytest.raises(ContractError, match='stale cursor'):
        world.call('find_items', {'size': 'M'}, call_id='two', cursor=cursor)


def transfer_world(source=None):
    world = Sandbox(source or transaction())
    world.call('accounts', {}, call_id='balances')
    arguments = {'from_id': {'$result': 'balances', 'path': 'items.0.id'},
                 'to_id': {'$result': 'balances', 'path': 'items.1.id'}, 'amount': 20}
    world.interact('approve', question='批准转账20？', proposal={'tool': 'transfer', 'arguments': arguments})
    return world, arguments


def test_atomic_commit_and_single_use_exact_authorization():
    world, arguments = transfer_world()
    world.call('transfer', arguments, call_id='transfer')
    assert [r['balance'] for r in world.data['accounts']] == [80, 30]
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='consumed'):
        world.call('transfer', arguments, call_id='second')
    assert world.data == before


def test_changed_authorized_parameters_are_rejected():
    world, arguments = transfer_world()
    changed = {**arguments, 'from_id': arguments['to_id'], 'to_id': arguments['from_id']}
    with pytest.raises(ContractError, match='parameters changed'):
        world.call('transfer', changed, call_id='wrong')
    assert [r['balance'] for r in world.data['accounts']] == [100, 10]


def test_revocation_blocks_write():
    world, arguments = transfer_world()
    world.revoke('approve')
    with pytest.raises(ContractError, match='revoked'):
        world.call('transfer', arguments, call_id='wrong')


def test_failed_second_operation_rolls_back_first_write():
    source = transaction()
    source['tools'][1]['operations'][1]['key'] = lit('missing')
    world, arguments = transfer_world(source)
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='exactly one row'):
        world.call('transfer', arguments, call_id='broken')
    assert world.data == before


def test_negative_balance_rolls_back_whole_transaction():
    source = transaction()
    source['tables'][0]['rows'][0]['balance'] = 5
    world, arguments = transfer_world(source)
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='invariant'):
        world.call('transfer', arguments, call_id='broken')
    assert world.data == before


def test_plausible_answer_cannot_hide_wrong_persisted_state():
    source = transaction()
    for operation in source['tools'][1]['operations']:
        operation['set']['balance'] = {'add': [
            {'sub': [ref('row.balance'), ref('args.amount')]}, ref('args.amount')]}
    world, arguments = transfer_world(source)
    world.call('transfer', arguments, call_id='fake')
    result = grade(world, {'accounts': [{'id': 'A', 'balance': 100}, {'id': 'B', 'balance': 10}]})
    assert result['components']['accounts']['passed']
    assert result['goals'] == [{'id': 'actual_transfer', 'passed': False}]
    assert result['reward'] == 0


def test_virtual_time_and_events_are_deterministic_and_resettable():
    source = simple()
    source['events'] = [{'at_seconds': 10, 'table': 'items', 'key': 'I1', 'set': {'size': 'L'}}]
    world = Sandbox(source)
    assert world.time() == {'datetime': '2026-01-01T00:00:00+00:00', 'tick': 0, 'elapsed_seconds': 0}
    world.wait(9)
    assert world.data['items'][0]['size'] == 'M'
    world.wait(1)
    assert world.data['items'][0]['size'] == 'L'
    assert world.time()['tick'] == 2
    world.reset()
    assert world.data['items'][0]['size'] == 'M' and world.time()['elapsed_seconds'] == 0


def test_illegal_foreign_key_counterexample_is_not_accepted():
    world = Sandbox(retrieval(4))
    with pytest.raises(ContractError, match='foreign key'):
        world.reset([{'table': 'items', 'key': 'I0000', 'column': 'brand_id', 'value': 'missing'}])


def test_dynamic_menu_uses_observed_options_and_refuses_missing_preference():
    source = simple()
    source['user']['facts']['preferred'] = 'I1'
    source['user']['stages'].append({'id': 'choose', 'kind': 'choice', 'fact': 'preferred', 'depends_on': ['size'],
        'source_tool': 'find_items', 'source_path': 'items', 'value_field': 'id', 'fallback': 'refuse'})
    world = Sandbox(source)
    world.interact('size', question='尺码？')
    world.call('find_items', {'size': 'M'}, call_id='menu')
    result = world.interact('choose', question='选择哪个？', source_id='menu')
    assert result['value'] == 'I1' and not result['refused']
    world.reset([{'table': 'items', 'key': 'I1', 'column': 'size', 'value': 'L'}])
    world.interact('size', question='尺码？')
    world.call('find_items', {'size': 'M'}, call_id='menu')
    result = world.interact('choose', question='选择哪个？', source_id='menu')
    assert result['value'] is None and result['refused']


def test_valid_fallback_choice_is_used_by_reward_instead_of_initial_preference():
    source = simple()
    source['tables'][0]['rows'][0]['size'] = 'L'
    source['tables'][0]['rows'].append({'id': 'I3', 'size': 'M', 'display': '替代款'})
    source['user']['facts']['preferred'] = 'I1'
    source['user']['stages'].append({'id': 'choose', 'kind': 'choice', 'fact': 'preferred', 'depends_on': ['size'],
        'source_tool': 'find_items', 'source_path': 'items', 'value_field': 'id', 'fallback': 'first_available'})
    source['answers'][0]['weight'] = 0.5
    source['answers'].append(answer('chosen', '用户在实际选项中的最终选择', ref('user.preferred'), {'type': 'string'}, weight=0.5))
    for scenario in source['scenarios']:
        scenario['answer']['chosen'] = lit('I3')
    world = Sandbox(source)
    world.interact('size', question='尺码？')
    world.call('find_items', {'size': 'M'}, call_id='menu')
    choice = world.interact('choose', question='请选择？', source_id='menu')
    assert choice['value'] == 'I3' and world.user.facts['preferred'] == 'I1'
    assert grade(world, {'ids': ['I3'], 'chosen': 'I3'})['success']


def test_planned_user_correction_updates_tool_and_reward_binding():
    source = simple()
    source['user'] = {'facts': {'initial_size': 'L', 'corrected_size': 'M'}, 'stages': [
        {'id': 'size', 'kind': 'fact', 'fact': 'initial_size', 'as_fact': 'size', 'depends_on': []},
        {'id': 'correction', 'kind': 'correction', 'fact': 'corrected_size', 'as_fact': 'size', 'depends_on': ['size']}]}
    world = Sandbox(source)
    assert world.interact('size', question='尺码？')['value'] == 'L'
    assert world.interact('correction', question='最终尺码？')['value'] == 'M'
    result = world.call('find_items', {'size': 'M'}, call_id='corrected')
    assert [row['id'] for row in result['items']] == ['I1']
    assert grade(world, {'ids': ['I1']})['success']


def test_explicit_refusal_branch_is_a_success_without_any_write():
    source = transaction()
    source['user']['facts']['approve_approve'] = False
    source['success'] = [{'id': 'refused', 'public': '用户拒绝授权时，只报告原余额且不得写入。',
        'when': {'not': [ref('user.approve_approve')]}, 'required_tools': ['accounts'], 'required_stages': ['approve']}]
    source['goals'][0]['predicate'] = {'eq': [ref('data.accounts'), ref('initial.accounts')]}
    correct = copy.deepcopy(source['scenarios'][0])
    correct['steps'] = correct['steps'][:2]
    source['scenarios'] = [correct, copy.deepcopy(source['scenarios'][1])]
    source['probes'] = []
    _, report = qualify(source)
    assert report['gates']['joint_execution_and_counterexamples']
    first = report['records'][0]['first']
    assert first['outcome']['reward'] == 1
    assert first['final_state']['accounts'] == source['tables'][0]['rows']


def test_language_model_event_cannot_change_private_user_state():
    world = Sandbox(simple())
    with pytest.raises(ContractError, match='mutate'):
        validate_language_event({'stage_id': 'size', 'question': '尺码？', 'facts': {'size': 'L'}}, world.user)
    assert world.user.facts == {'size': 'M'}


def test_false_and_zero_remain_real_observations_and_numeric_equivalence():
    assert equivalent({'ready': False, 'amount': 0.0}, {'ready': False, 'amount': 0}, 'exact')
    assert not equivalent(False, 0, 'exact')
    assert equivalent([1, 1.0, False], [False, 1, 1], 'unordered')
    assert not equivalent([1, 1], [1], 'unordered')


def test_partial_reward_preserves_independent_correct_fields():
    source = direct()
    source['answers'][0]['weight'] = 0.5
    source['answers'].append(answer('unit', '温度单位', lit('摄氏度'), {'type': 'string'}, weight=0.5))
    for scenario in source['scenarios']:
        scenario['answer']['unit'] = lit('摄氏度')
    world = Sandbox(source)
    result = grade(world, {'temperature': 19, 'unit': '摄氏度'})
    assert result['reward'] == 0.5 and not result['success']


def test_explanation_judge_receives_actual_mutated_facts_and_fresh_replay_calls():
    source = simple()
    source['answers'][0]['weight'] = 0.5
    source['explanations'] = [{'id': 'reason', 'meaning': '解释尺码筛选依据', 'weight': 0.5,
        'criteria': [{'id': 'actual_ids', 'rule_id': 'size_matches', 'obligation': '解释实际合格款式及其尺码依据。'}], 'context_fields': ['ids']}]
    for scenario in source['scenarios']:
        scenario['answer']['reason'] = lit('按实际用户尺码筛选。')
    source['probes'][-1]['answer']['reason'] = lit('按实际用户尺码筛选。')
    contexts = []
    def judge(payload):
        contexts.append(payload['authoritative_facts'])
        return {'criteria': {'actual_ids': True}, 'provenance': {'id': len(contexts)}}
    _, report = qualify(source, judge=judge)
    assert report['gates']['joint_execution_and_counterexamples']
    assert len(contexts) == 2 * len(report['records'])
    assert {'ids': []} in contexts and {'ids': ['I1']} in contexts
    with pytest.raises(ContractError, match='native semantic judge required'):
        grade(Sandbox(source), {'ids': ['I1'], 'reason': '解释'})


def test_delivery_content_identity_detects_modified_runtime(tmp_path):
    compiled, report = qualify(direct())
    export_bundle(compiled, report, tmp_path / 'sandbox')
    assert verify_bundle(tmp_path / 'sandbox')['qualified'] is False
    (tmp_path / 'sandbox/env_factory_v2/runtime.py').write_text('changed')
    with pytest.raises(ContractError, match='content changed'):
        verify_bundle(tmp_path / 'sandbox')


def test_http_public_boundary_and_independent_episodes():
    server = serve(simple(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    def post(route, value, episode=None):
        request = Request(url + route, data=json.dumps(value).encode(),
            headers={'Content-Type': 'application/json', **({'X-Episode': episode} if episode else {})})
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    try:
        first = post('/reset', {})
        second = post('/reset', {})
        public = first['public']
        assert 'facts' not in public and 'rows' not in json.dumps(public)
        assert first['episode'] != second['episode']
        assert post('/user', {'stage': 'size', 'question': '尺码？'}, first['episode'])['value'] == 'M'
        with pytest.raises(HTTPError) as error:
            post('/tool', {'tool': 'find_items', 'arguments': {'size': 'M'}, 'id': 'guess'}, second['episode'])
        assert error.value.code == 400
        with pytest.raises(HTTPError):
            post('/reset', {'mutations': []})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
