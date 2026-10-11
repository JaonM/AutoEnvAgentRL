"""Executable partial-view tools; never relax task bindings or expose new data."""
from __future__ import annotations

import copy

from .contract import compile_contract, require
from .runtime import Sandbox


def augment(source, *, count=2):
    require(type(count) is int and 0 <= count <= 4, 'tool interference count must be 0..4')
    source = copy.deepcopy(source)
    if not count or source['category'] == 'direct_response':
        return source, {'enabled': bool(count), 'added': [], 'reason': 'disabled or direct response'}
    source['tools'] = [t for t in source['tools'] if t.get('selection_role') != 'distractor']
    # Freeze the original default completion obligations before adding optional tools.
    source = compile_contract(source)['source']
    originals = [t for t in source['tools'] if t.get('selection_role', 'task') == 'task']
    added = []
    witnesses = []
    for tool in originals:
        if tool['kind'] != 'query' or len(tool['columns']) < 2:
            continue
        for column in tool['columns']:
            if len(added) >= count:
                break
            identity = tool['id'] + '_view_' + column
            if any(t['id'] == identity for t in source['tools']):
                continue
            witness = next(((s, i) for s in source['scenarios'] if s['expect_success']
                for i, step in enumerate(s['steps'])
                if step['kind'] in {'call', 'traverse'} and step['tool'] == tool['id']), None)
            if witness is None:
                continue
            variant = copy.deepcopy(tool)
            variant.update(id=identity, columns=[column], selection_role='distractor',
                description=f"Read only {column} from {tool['table']}; retains the same filters and dependencies.")
            source['tools'].append(variant)
            added.append(identity)
            scenario, index = witness
            case = copy.deepcopy(scenario)
            case['steps'] = case['steps'][:index + 1]
            case['steps'][-1]['tool'] = identity
            case['answer'] = {}
            witnesses.append((identity, case))
    compiled = compile_contract(source)
    checks = []
    for identity, case in witnesses:
        sandbox = Sandbox(compiled)
        sandbox.reset(case.get('mutations', []))
        for step in case['steps']:
            if step['kind'] in {'call', 'traverse'}:
                function = sandbox.call if step['kind'] == 'call' else sandbox.traverse
                function(step['tool'], step['arguments'], call_id=step['id'],
                    **({'cursor': sandbox.resolve(step['cursor'])} if 'cursor' in step else {}))
            elif step['kind'] == 'user':
                sandbox.interact(step['stage'], question=step['question'],
                    proposal=step.get('proposal'), source_id=step.get('source_id'))
            elif step['kind'] == 'wait':
                sandbox.wait(step['seconds'])
            elif step['kind'] == 'time':
                sandbox.observe_time()
            elif step['kind'] == 'revoke':
                sandbox.revoke(step['stage'])
        executions = [event for event in sandbox.trace
                      if event.get('kind') == 'tool' and event.get('tool') == identity]
        require(bool(executions) and all(event.get('status') == 'ok' for event in executions),
                'interference query did not execute')
        checks.append({'tool': identity, 'passed': True, 'trace': sandbox.trace})
    return source, {'enabled': True, 'requested': count, 'added': added, 'execution_checks': checks,
                    'scope': 'read-only partial views; no unrelated tables or write tools fabricated'}
