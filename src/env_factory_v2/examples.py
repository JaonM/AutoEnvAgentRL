"""Independent demonstration fixtures, not a catalog used by generation."""
from __future__ import annotations

import copy


def lit(value):
    return {'literal': value}


def ref(value):
    return {'ref': value}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array(schema):
    return {'type': 'array', 'items': schema}


def base(identity, category, task):
    return {'version': '2.0', 'id': identity, 'category': category, 'task': task,
            'rules': [], 'tables': [], 'tools': [], 'user': {'facts': {}, 'stages': []},
            'answers': [], 'scenarios': [], 'probes': [], 'limits': {'max_actions': 128},
            'time': {'base': '2026-01-01T00:00:00+00:00', 'seconds_per_action': 1}}


def answer(identity, meaning, expected, schema, *, comparison='exact', weight=1):
    return {'id': identity, 'meaning': meaning, 'expected': expected, 'schema': schema,
            'comparison': comparison, 'weight': weight}


def scenario(identity, steps, answer_recipe, success=True, **extra):
    return {'id': identity, 'steps': steps, 'answer': answer_recipe, 'expect_success': success, **extra}


def query(identity, table, schema, columns, rules=(), *, page_size=64, bindings=(), user_bindings=None, complete=False):
    return {'id': identity, 'kind': 'query', 'description': f'Read {table} using the declared public rules.',
            'parameters': schema, 'rule_ids': list(rules), 'table': table, 'columns': columns,
            'order_by': [], 'page_size': page_size, 'result_bindings': list(bindings),
            'user_bindings': user_bindings or {}, 'complete_retrieval': complete}


def direct():
    source = base('fixed_temperature', 'direct_response', '库区当前温度为18摄氏度。返回 temperature 数字字段。')
    source['rules'] = [{'id': 'fixed_fact', 'public': '公开温度固定为18摄氏度。', 'predicate': lit(True)}]
    source['answers'] = [answer('temperature', '公开温度数字', lit(18), {'type': 'integer'})]
    source['scenarios'] = [scenario('correct', [], {'temperature': lit(18)}),
                           scenario('wrong', [], {'temperature': lit(19)}, False)]
    return source


def simple():
    source = base('size_lookup', 'simple_agentic', '询问用户尺码，返回库存中该尺码的全部款式ID，无需按显示名称筛选。')
    source['rules'] = [{'id': 'size_matches', 'public': '只按用户告知的尺码筛选；显示名称不影响资格。',
                        'predicate': {'eq': [ref('row.size'), ref('args.size')]}}]
    source['tables'] = [{'id': 'items', 'primary_key': 'id', 'schema': obj({'id': {'type': 'string'},
                         'size': {'type': 'string'}, 'display': {'type': 'string'}}),
                         'rows': [{'id': 'I1', 'size': 'M', 'display': '青松'}, {'id': 'I2', 'size': 'L', 'display': '白云'}]}]
    source['user'] = {'facts': {'size': 'M'}, 'stages': [{'id': 'size', 'kind': 'fact', 'fact': 'size', 'depends_on': []}]}
    source['tools'] = [query('find_items', 'items', obj({'size': {'type': 'string'}}), ['id', 'size', 'display'],
                              ['size_matches'], user_bindings={'size': 'size'})]
    selected = {'filter': {'items': ref('data.items'), 'value': {'rule': {
        'id': 'size_matches', 'arguments': {'size': ref('user.size')}}}}}
    source['answers'] = [answer('ids', '所有符合尺码的款式ID', {'map': {'items': selected, 'value': ref('row.id')}},
                                array({'type': 'string'}), comparison='unordered')]
    steps = [{'kind': 'user', 'stage': 'size', 'question': '需要哪个尺码？'},
             {'kind': 'call', 'tool': 'find_items', 'id': 'found', 'arguments': {'size': {'$user': 'size', 'path': 'value'}}}]
    correct = {'ids': {'map': {'items': ref('evidence.found.items'), 'value': ref('row.id')}}}
    source['scenarios'] = [scenario('correct', steps, correct), scenario('wrong', steps, {'ids': lit([])}, False),
                           scenario('no_tool', steps[:1], {'ids': lit(['I1'])}, False)]
    source['probes'] = [{'id': 'display_rename', 'scenario': 'correct', 'rule_ids': ['size_matches'],
                         'mutations': [{'table': 'items', 'key': 'I1', 'column': 'display', 'value': '旧名称'}], 'expect_success': True},
                        {'id': 'changed_size', 'scenario': 'correct', 'rule_ids': ['size_matches'],
                         'mutations': [{'table': 'items', 'key': 'I1', 'column': 'size', 'value': 'L'}], 'expect_success': True},
                        {'id': 'stale_size_answer', 'scenario': 'correct', 'rule_ids': ['size_matches'],
                         'mutations': [{'table': 'items', 'key': 'I1', 'column': 'size', 'value': 'L'}],
                         'answer': {'ids': lit(['I1'])}, 'expect_success': False}]
    return source


def retrieval(rows=512):
    source = base('complete_stock_lookup', 'multi_step_agentic', '询问品牌和尺码，先解析品牌ID，再完整检索该品牌该尺码的在售有库存款式，返回全部ID。显示名称不参与资格判定。')
    source['rules'] = [
        {'id': 'brand_identity', 'public': '按用户品牌名称查唯一品牌ID。', 'predicate': {'eq': [ref('row.name'), ref('args.name')]}},
        {'id': 'eligible', 'public': '款式仅按品牌ID、用户尺码、在售和库存大于0筛选，显示名称不影响资格。',
         'predicate': {'and': [{'eq': [ref('row.brand_id'), ref('args.brand_id')]},
                              {'eq': [ref('row.size'), ref('args.size')]},
                              {'eq': [ref('row.active'), lit(True)]}, {'gt': [ref('row.stock'), lit(0)]}]}}]
    source['tables'] = [
        {'id': 'brands', 'primary_key': 'id', 'schema': obj({'id': {'type': 'string'}, 'name': {'type': 'string'}}),
         'rows': [{'id': 'B1', 'name': '青松'}]},
        {'id': 'items', 'primary_key': 'id', 'schema': obj({'id': {'type': 'string'}, 'brand_id': {'type': 'string'},
            'display': {'type': 'string'}, 'size': {'type': 'string'}, 'active': {'type': 'boolean'}, 'stock': {'type': 'integer'}}),
         'foreign_keys': [{'column': 'brand_id', 'target_table': 'brands', 'target_column': 'id'}],
         'rows': [{'id': f'I{i:04d}', 'brand_id': 'B1', 'display': '青松', 'size': 'M', 'active': True, 'stock': i + 1}
                  for i in range(rows)]}]
    source['user'] = {'facts': {'brand': '青松', 'size': 'M'}, 'stages': [
        {'id': 'brand', 'kind': 'fact', 'fact': 'brand', 'depends_on': []},
        {'id': 'size', 'kind': 'fact', 'fact': 'size', 'depends_on': ['brand']}]}
    source['tools'] = [query('resolve_brand', 'brands', obj({'name': {'type': 'string'}}), ['id', 'name'],
        ['brand_identity'], user_bindings={'name': 'brand'}),
        query('find_stock', 'items', obj({'brand_id': {'type': 'string'}, 'size': {'type': 'string'}}),
            ['id', 'stock', 'display'], ['eligible'], page_size=32, complete=True, user_bindings={'size': 'size'},
            bindings=[{'argument': 'brand_id', 'tool': 'resolve_brand', 'path': 'items.0.id'}])]
    steps = [{'kind': 'user', 'stage': 'brand', 'question': '需要哪个品牌？'},
             {'kind': 'call', 'tool': 'resolve_brand', 'id': 'brand_result', 'arguments': {'name': {'$user': 'brand', 'path': 'value'}}},
             {'kind': 'user', 'stage': 'size', 'question': '需要哪个尺码？'},
             {'kind': 'traverse', 'tool': 'find_stock', 'id': 'stock_result', 'arguments': {
                 'brand_id': {'$result': 'brand_result', 'path': 'items.0.id'}, 'size': {'$user': 'size', 'path': 'value'}}}]
    # The expected business set is evaluated independently of tool output.
    selected_brand = {'at': [{'map': {'items': {'filter': {'items': ref('data.brands'),
        'value': {'rule': {'id': 'brand_identity', 'arguments': {'name': ref('user.brand')}}}}}, 'value': ref('row.id')}}, lit(0)]}
    source['answers'] = [answer('ids', '全部合格款式ID，顺序不限', {'map': {
        'items': {'filter': {'items': ref('data.items'), 'value': {'rule': {'id': 'eligible', 'arguments': {
            'brand_id': selected_brand, 'size': ref('user.size')}}}}},
        'value': ref('row.id')}}, array({'type': 'string'}), comparison='unordered')]
    correct = {'ids': {'map': {'items': ref('evidence.stock_result.items'), 'value': ref('row.id')}}}
    partial = copy.deepcopy(steps)
    partial[-1]['kind'] = 'call'
    source['scenarios'] = [scenario('correct', steps, correct), scenario('first_page_only', partial, correct, False),
                           scenario('wrong_answer', steps, {'ids': lit([])}, False)]
    source['probes'] = [{'id': 'renamed_display', 'scenario': 'correct', 'rule_ids': ['eligible'],
        'mutations': [{'table': 'items', 'key': 'I0000', 'column': 'display', 'value': '旧名称'}], 'expect_success': True},
        {'id': 'stock_changed', 'scenario': 'correct', 'rule_ids': ['eligible'],
        'mutations': [{'table': 'items', 'key': 'I0000', 'column': 'stock', 'value': 0}], 'expect_success': True}]
    return source


def transaction():
    source = base('atomic_transfer', 'multi_step_agentic', '查询账户余额，向用户展示转账参数并取得授权，再原子地从A转20到B。余额不可为负，总额不变。返回两个账户的实际余额。')
    source['rules'] = [
        {'id': 'nonnegative', 'public': '账户余额不可为负。', 'predicate': {'ge': [ref('row.balance'), lit(0)]}},
        {'id': 'conservation', 'public': '转账前后账户余额总额必须相同。', 'predicate': {'eq': [
            {'sum': [{'map': {'items': ref('data.accounts'), 'value': ref('row.balance')}}]},
            {'sum': [{'map': {'items': ref('initial.accounts'), 'value': ref('row.balance')}}]}]}}]
    source['tables'] = [{'id': 'accounts', 'primary_key': 'id',
        'schema': obj({'id': {'type': 'string'}, 'balance': {'type': 'integer'}}),
        'rows': [{'id': 'A', 'balance': 100}, {'id': 'B', 'balance': 10}]}]
    source['tools'] = [query('accounts', 'accounts', obj({}), ['id', 'balance']),
        {'id': 'transfer', 'kind': 'transaction', 'description': 'Atomically transfer exactly the approved amount.',
         'parameters': obj({'from_id': {'type': 'string'}, 'to_id': {'type': 'string'}, 'amount': {'type': 'integer', 'enum': [20]}}),
         'rule_ids': ['nonnegative'], 'authorization': 'approve',
         'result_bindings': [{'argument': 'from_id', 'tool': 'accounts', 'path': 'items.*.id'},
                             {'argument': 'to_id', 'tool': 'accounts', 'path': 'items.*.id'}],
         'operations': [{'table': 'accounts', 'key': ref('args.from_id'), 'set': {'balance': {'sub': [ref('row.balance'), ref('args.amount')]}}},
                        {'table': 'accounts', 'key': ref('args.to_id'), 'set': {'balance': {'add': [ref('row.balance'), ref('args.amount')]}}}]}]
    source['invariants'] = [
        {'rule_id': 'conservation', 'predicate': {'rule': {'id': 'conservation', 'arguments': {}}}},
        {'rule_id': 'nonnegative', 'predicate': {'eq': [{'count': [{'filter': {
            'items': ref('data.accounts'), 'value': {'not': [{'rule': {'id': 'nonnegative', 'arguments': {}}}]}}}]}, lit(0)]}}]
    source['user']['stages'] = [{'id': 'approve', 'kind': 'authorization', 'tool': 'transfer', 'depends_on': []}]
    source['user']['facts'] = {'approve_approve': True}
    args = {'from_id': {'$result': 'balances', 'path': 'items.0.id'},
            'to_id': {'$result': 'balances', 'path': 'items.1.id'}, 'amount': 20}
    steps = [{'kind': 'call', 'tool': 'accounts', 'id': 'balances', 'arguments': {}},
             {'kind': 'user', 'stage': 'approve', 'question': '是否批准从A向B转账20？', 'proposal': {'tool': 'transfer', 'arguments': args}},
             {'kind': 'call', 'tool': 'transfer', 'id': 'transfer_result', 'arguments': args}]
    # Goal targets describe the business change, not merely copying final state.
    expected = {'map': {'items': ref('initial.accounts'), 'value': {'object': {
        'id': ref('row.id'), 'balance': {'if': {'condition': {'eq': [ref('row.id'), lit('A')]},
                    'then': {'sub': [ref('row.balance'), lit(20)]}, 'else': {'add': [ref('row.balance'), lit(20)]}}}}}}}
    source['rules'].append({'id': 'transferred', 'public': '已授权的转账必须使A减少20、B增加20。',
                           'predicate': {'eq': [ref('data.accounts'), expected]}})
    source['answers'] = [answer('accounts', '转账后的实际余额', ref('data.accounts'),
        array(obj({'id': {'type': 'string'}, 'balance': {'type': 'integer'}})), comparison='unordered')]
    source['goals'] = [{'id': 'actual_transfer', 'rule_ids': ['transferred'],
                       'predicate': {'rule': {'id': 'transferred', 'arguments': {}}}}]
    correct = {'accounts': ref('data.accounts')}
    source['scenarios'] = [scenario('correct', steps, correct),
        scenario('no_authorization', [steps[0], steps[2]], correct, False, expect_error='authorization missing'),
        scenario('wrong_answer', steps, {'accounts': ref('initial.accounts')}, False)]
    source['probes'] = [{'id': 'changed_initial_balance', 'scenario': 'correct', 'rule_ids': ['nonnegative', 'conservation'],
                         'mutations': [{'table': 'accounts', 'key': 'A', 'column': 'balance', 'value': 80}], 'expect_success': True}]
    return source


def long_chain(length=12):
    source = base('dependency_chain', 'multi_step_agentic',
        '从公开入口N0逐步读取下一节点，沿实际返回的next_id到达终点，返回终点value。不可猜测节点标识跳过依赖。')
    source['rules'] = [{'id': 'node_identity', 'public': '每步按上一节点实际返回的next_id查下一节点；第一步入口为N0。',
                        'predicate': {'eq': [ref('row.id'), ref('args.node_id')]}}]
    source['tables'] = [{'id': 'nodes', 'primary_key': 'id',
        'schema': obj({'id': {'type': 'string'}, 'next_id': {'type': 'string', 'nullable': True}, 'value': {'type': 'integer'}}),
        'rows': [{'id': f'N{i}', 'next_id': f'N{i+1}' if i + 1 < length else None, 'value': i * 7} for i in range(length)]}]
    steps = []
    for i in range(length):
        tool = query(f'node_{i}', 'nodes', obj({'node_id': {'type': 'string'}}), ['id', 'next_id', 'value'], ['node_identity'],
            bindings=[] if i == 0 else [{'argument': 'node_id', 'tool': f'node_{i-1}', 'path': 'items.0.next_id'}])
        source['tools'].append(tool)
        arguments = {'node_id': 'N0' if i == 0 else {'$result': f'node_result_{i-1}', 'path': 'items.0.next_id'}}
        steps.append({'kind': 'call', 'tool': tool['id'], 'id': f'node_result_{i}', 'arguments': arguments})
    source['answers'] = [answer('value', '终点实际value', ref(f'data.nodes.{length-1}.value'), {'type': 'integer'})]
    correct = {'value': ref(f'evidence.node_result_{length-1}.items.0.value')}
    skipped = copy.deepcopy(steps)
    skipped[-1]['arguments']['node_id'] = f'N{length-1}'
    source['scenarios'] = [scenario('correct', steps, correct), scenario('wrong', steps, {'value': lit(-1)}, False),
        scenario('guessed_dependency', skipped, correct, False, expect_error='observed-result dependency')]
    source['probes'] = [{'id': 'changed_terminal_fact', 'scenario': 'correct', 'rule_ids': ['node_identity'],
        'mutations': [{'table': 'nodes', 'key': f'N{length-1}', 'column': 'value', 'value': 432}], 'expect_success': True}]
    return source


def complex_transaction():
    source = transaction()
    source['id'] = 'eight_write_atomic_transaction'
    source['task'] = '查询8个账户，展示并取得用户对全部参数的授权；原子地从每个偶数编号账户向下一个账户转20，共8次账户写入。各账户非负，总额不变。返回全部实际余额。'
    source['tables'][0]['rows'] = [{'id': f'A{i}', 'balance': 100 + i} for i in range(8)]
    tool = source['tools'][1]
    tool['parameters'] = obj({**{f'id{i}': {'type': 'string'} for i in range(8)}, 'amount': {'type': 'integer', 'enum': [20]}})
    tool['result_bindings'] = [{'argument': f'id{i}', 'tool': 'accounts', 'path': f'items.{i}.id'} for i in range(8)]
    tool['operations'] = [{'table': 'accounts', 'key': ref(f'args.id{i}'), 'set': {'balance': {
        'sub' if i % 2 == 0 else 'add': [ref('row.balance'), ref('args.amount')]}}} for i in range(8)]
    args = {**{f'id{i}': {'$result': 'balances', 'path': f'items.{i}.id'} for i in range(8)}, 'amount': 20}
    steps = [{'kind': 'call', 'tool': 'accounts', 'id': 'balances', 'arguments': {}},
        {'kind': 'user', 'stage': 'approve', 'question': '批准按展示参数执行全部四组转账？', 'proposal': {'tool': 'transfer', 'arguments': args}},
        {'kind': 'call', 'tool': 'transfer', 'id': 'transfer_result', 'arguments': args}]
    expected = {'map': {'items': ref('initial.accounts'), 'value': {'object': {'id': ref('row.id'), 'balance': {'if': {
        'condition': {'in': [ref('row.id'), lit(['A0', 'A2', 'A4', 'A6'])]},
        'then': {'sub': [ref('row.balance'), lit(20)]}, 'else': {'add': [ref('row.balance'), lit(20)]}}}}}}}
    source['rules'][-1]['predicate'] = {'eq': [ref('data.accounts'), expected]}
    source['rules'][-1]['public'] = '已授权事务必须使每个偶数编号账户减少20、下一个账户增加20，共8次写入。'
    source['scenarios'] = [scenario('correct', steps, {'accounts': ref('data.accounts')}),
        scenario('wrong', steps, {'accounts': ref('initial.accounts')}, False),
        scenario('no_authorization', [steps[0], steps[2]], {'accounts': ref('data.accounts')}, False, expect_error='authorization missing')]
    source['probes'] = [{'id': 'changed_initial_balance', 'scenario': 'correct', 'rule_ids': ['nonnegative', 'conservation'],
        'mutations': [{'table': 'accounts', 'key': 'A0', 'column': 'balance', 'value': 80}], 'expect_success': True}]
    return source


EXAMPLES = {'direct': direct, 'simple': simple, 'retrieval': retrieval, 'transaction': transaction,
            'long_chain': long_chain, 'complex_transaction': complex_transaction}
