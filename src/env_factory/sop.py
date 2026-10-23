"""Caller-owned business contracts and bounded dependency graphs.

No industry-specific rules live here. Graphs are specification/test machinery,
not a required student trace. Business outcomes still have an independent verifier.
"""
from __future__ import annotations

import copy
import itertools

from env_factory import lifecycle
from env_factory.certify.behavior import at, digest, expression
from env_factory.runtime import BusinessError, Episode, canonical

EDGE_KINDS = {'data', 'state', 'resource', 'authorization'}


def references(value):
    """Find explicit output references; never guess dependencies from equal scalars."""
    found = []
    if isinstance(value, dict):
        if set(value) == {'path'}:
            parts = value['path'].split('.') if isinstance(value['path'], str) else value['path']
            if parts and parts[0] == 'outputs':
                if len(parts) < 2:
                    raise ValueError('an output reference must name a node')
                found.append(str(parts[1]))
        else:
            for item in value.values():
                found.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(references(item))
    return found


def order(graph, *, reverse=False):
    nodes = graph['nodes']
    if not nodes or len(nodes) > 64:
        raise ValueError('SOP graph needs 1..64 nodes; unbounded loops are not supported')
    ids = [n['id'] for n in nodes]
    if len(ids) != len(set(ids)) or any(not isinstance(n, str) or not n for n in ids):
        raise ValueError('node IDs must be unique nonempty strings')
    parents = {n['id']: set() for n in nodes}
    for edge in graph.get('edges', []):
        if edge['kind'] not in EDGE_KINDS or edge['from'] not in parents or edge['to'] not in parents:
            raise ValueError('invalid typed dependency edge')
        parents[edge['to']].add(edge['from'])
    result = []
    while len(result) < len(nodes):
        ready = sorted((n for n in ids if n not in result and parents[n] <= set(result)), reverse=reverse)
        if not ready:
            raise ValueError('SOP dependency cycle')
        result.append(ready[0])
    return result


def ancestors(graph):
    result = {}
    for name in order(graph):
        direct = {e['from'] for e in graph.get('edges', []) if e['to'] == name}
        result[name] = direct | set().union(*(result[p] for p in direct)) if direct else set()
    return result


def overlap(a, b):
    """Conservative access overlap, including parent/child paths and wildcards."""
    return all(x == y or '*' in (x, y) for x, y in zip(a.split('.'), b.split('.')))


def graph_problems(graph, atoms):
    try:
        parents = ancestors(graph)
        nodes = {n['id']: n for n in graph['nodes']}
        for name, node in nodes.items():
            if node['atom'] not in atoms:
                raise ValueError(f'{name}: unknown atom')
            if not node.get('checks') and not node.get('error'):
                raise ValueError(f'{name}: independently declared postcondition required')
            for source in references([node.get('args', {}), node.get('when')]):
                if source not in parents[name]:
                    raise ValueError(f'{name}: output binding lacks a predecessor: {source}')
                if not any(e['from'] == source and e['to'] == name and e['kind'] == 'data' for e in graph.get('edges', [])):
                    raise ValueError(f'{name}: output binding requires an explicit data edge')
        for left, right in itertools.combinations(nodes, 2):
            if left in parents[right] or right in parents[left]:
                continue
            a, b = atoms[nodes[left]['atom']], atoms[nodes[right]['atom']]
            if any(overlap(x, y) for x in a['writes'] for y in b['reads'] + b['writes']) or any(
                    overlap(x, y) for x in b['writes'] for y in a['reads']):
                raise ValueError(f'{left}/{right}: unordered conflicting state/resource access')
        return []
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]


def changes(before, after, prefix=()):
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        result = []
        for key in before.keys() | after.keys():
            if key not in before or key not in after:
                result.append(prefix + (str(key),))
            else:
                result.extend(changes(before[key], after[key], prefix + (str(key),)))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix]
        return [p for i, (a, b) in enumerate(zip(before, after)) for p in changes(a, b, prefix + (str(i),))]
    return [] if before == after else [prefix]


def frame_problems(before, after, atom, args):
    problems = []
    for path in changes(before, after):
        if not any(len(p.split('.')) <= len(path) and all(a == '*' or a == b for a, b in zip(p.split('.'), path))
                   for p in atom['writes']):
            problems.append('undeclared write: ' + '.'.join(path))
    # Optional row ownership confines writes inside a declared table to the argument's object.
    for scope in atom.get('row_scopes', []):
        value = at(args, scope['argument'])
        old, new = before[scope['table']], after[scope['table']]
        old_other = [r for r in old if r[scope['key']] != value]
        new_other = [r for r in new if r[scope['key']] != value]
        if canonical(old_other) != canonical(new_other):
            problems.append('write escaped object scope: ' + scope['table'])
    return problems


def validate(blueprint, tools=None):
    errors = []
    try:
        if blueprint['version'] != 1 or not blueprint['objects'] or not blueprint['atoms']:
            raise ValueError('SOP blueprint needs version 1, objects and atoms')
        for name, atom in blueprint['atoms'].items():
            for key in ('intent', 'policy', 'inputs', 'output', 'preconditions', 'agent_obligations',
                        'reads', 'writes', 'effects', 'failure', 'retry', 'scope'):
                if key not in atom:
                    errors.append(f'{name}: missing {key}')
            if atom.get('failure') != 'atomic_rollback':
                errors.append(f'{name}: undeclared failure semantics')
            if tools is not None and atom['tool'] not in tools:
                errors.append(f'{name}: unknown tool {atom["tool"]}')
            if not isinstance(atom.get('reads'), list) or not isinstance(atom.get('writes'), list):
                errors.append(f'{name}: reads/writes must be path lists')
            if atom.get('inputs', {}).get('type') != 'object':
                errors.append(f'{name}: typed object inputs required')
        for name, method in blueprint.get('methods', {}).items():
            if not method.get('processes') or not method.get('intent'):
                errors.append(f'{name}: missing business intent/processes')
            for edge in method.get('edges', []):
                if edge['kind'] not in EDGE_KINDS or not 0 <= edge['from'] < edge['to'] < len(method['processes']):
                    errors.append(f'{name}: invalid method edge')
        return errors
    except (KeyError, TypeError, ValueError) as exc:
        return errors + [str(exc)]


def run_graph(env, blueprint, case, *, reverse=False, request=None):
    graph = case['graph']; atoms = blueprint['atoms']
    problems = graph_problems(graph, atoms)
    if problems:
        return {'passed': False, 'problems': problems}
    config = {**(request or {}), **({'scenario': case['scenario']} if 'scenario' in case else {})}
    api = Episode(env.make_tools(), case['data'], {}, start=case.get('now', '2024-05-15T15:00:00'),
                  approve=lambda *_: True, **lifecycle.options(config))
    outputs = {}; rows = []; nodes = {n['id']: n for n in graph['nodes']}
    try:
        for name in order(graph, reverse=reverse):
            node = nodes[name]; atom = atoms[node['atom']]
            if 'wait_before' in node:
                api.wait(node['wait_before'])
            before = copy.deepcopy(api.db)
            context = {'initial': case['data'], 'before': before, 'after': api.db, 'outputs': outputs}
            if 'when' in node and expression(node['when'], context) is not True:
                rows.append({'node': name, 'status': 'skipped'}); continue
            if 'now' in node:
                if api.lifecycle is not None:
                    raise ValueError('lifecycle SOPs use wait_before, not direct clock assignment')
                api.now = node['now']
            args = expression(node.get('args', {}), context)
            from jsonschema import validate as validate_json
            validate_json(args, atom['inputs'])
            try:
                result = api.call(atom['tool'], **args)
            except BusinessError as exc:
                if not node.get('error') or (node['error'] != '*' and node['error'] not in str(exc)):
                    raise
                if canonical(before) != canonical(api.db):
                    raise ValueError('failed atom changed state')
                result = {'error': str(exc)}
            else:
                if node.get('error'):
                    raise ValueError('expected rejection but atom succeeded')
                validate_json(result, atom['output'])
            outputs[name] = result
            problems += frame_problems(before, api.db, atom, args)
            context.update(result=result, after=api.db)
            for check in node.get('checks', []) + case.get('invariants', []):
                if expression(check['assert'], context) is not True:
                    problems.append(name + ': ' + check['name'])
            rows.append({'node': name, 'status': 'rejected' if 'error' in node else 'ok'})
            if problems:
                break
    except Exception as exc:
        problems.append(f'{type(exc).__name__}: {exc}')
    return {'passed': not problems, 'problems': problems, 'nodes': rows,
            'final_state_sha256': digest(api.db), 'outputs_sha256': digest(outputs)}


def task_problems(request):
    blueprint = request.get('sop_blueprint')
    if not blueprint or request.get('sop_legacy'):
        return []
    goals = request['user_facts']['goals']
    if len(goals) == 1:
        return []
    method = blueprint['methods'].get(request.get('sop_method'))
    if method is None:
        return ['multiple goals require a declared sop_method (independent work must also be explicitly modeled)']
    if [g['process'] for g in goals] != method['processes']:
        return ['goal order/processes differ from the selected SOP method']
    problems = []
    for relation in method.get('same', []):
        values = [at(goals[i]['params'], path) for i, path in relation]
        if any(v != values[0] for v in values[1:]):
            problems.append('SOP requires the same business object: ' + str(relation))
    for binding in method.get('bindings', []):
        actual = goals[binding['to']].get('bindings', {}).get(binding['target'])
        if actual != {'goal': binding['from'] + 1, 'path': binding['source']}:
            problems.append('SOP requires runtime result binding: ' + binding['target'])
    return problems


def task_report(env, verifier, request):
    """Reject invented causal labels using independent before/without-predecessor outcomes."""
    problems = task_problems(request)
    method = request.get('sop_blueprint', {}).get('methods', {}).get(request.get('sop_method'))
    if problems or not method or request.get('sop_legacy'):
        return {'passed': not problems, 'problems': problems, 'legacy': bool(request.get('sop_legacy'))}
    facts = request['user_facts']; now = request['time']['start']
    full = lifecycle.expected(verifier, copy.deepcopy(request['tables']), copy.deepcopy(facts), now, request)
    witnesses = []
    for i, allowed in method.get('statuses', {}).items():
        if full['answer']['goal_' + str(int(i) + 1)]['status'] not in allowed:
            problems.append('method outcome branch not observed at goal ' + str(int(i) + 1))
    for rule in method.get('outcome_predicates', []):
        if at(full['answer'][f"goal_{rule['goal']+1}"],rule['path']) != rule['equals']:
            problems.append('method policy branch not observed')
    for edge in method.get('edges', []):
        if edge['kind'] == 'data':
            altered = copy.deepcopy(facts)
            altered['goals'][edge['to']].pop('bindings', None)
            counter = lifecycle.expected(verifier, copy.deepcopy(request['tables']), altered, now, request)
            changed = full['answer'][f"goal_{edge['to']+1}"] != counter['answer'][f"goal_{edge['to']+1}"]
            witnesses.append({**edge, 'evidence': 'runtime_binding_ablation', 'observed': changed})
            if not changed:
                problems.append('declared result binding did not affect successor outcome')
            continue
        if not edge.get('causal'):
            witnesses.append({**edge, 'evidence': 'declared_order_only'}); continue
        # Keep the successor's static inputs fixed. For state dependencies, removal
        # must change its business answer OR the final business state.
        altered = copy.deepcopy(facts); removed = edge['from']; target = edge['to']
        altered['goals'].pop(removed)
        if any(g.get('bindings') for g in altered['goals']):
            problems.append('state ablation with mixed result bindings is unsupported'); continue
        counter = lifecycle.expected(verifier, copy.deepcopy(request['tables']), altered, now, request)
        target_changed = full['answer'][f'goal_{target+1}'] != counter['answer'][f'goal_{target}']
        changed = (target_changed or
                   canonical(full['final_state']) != canonical(counter['final_state']))
        # A persistent predecessor-only edit is not proof: its successor alone
        # must differ when started from the post-predecessor state.
        prefix = copy.deepcopy(facts); prefix['goals'] = prefix['goals'][:target]
        prior = lifecycle.expected(verifier, copy.deepcopy(request['tables']), prefix, now, request)['final_state']
        isolated = copy.deepcopy(facts); isolated['goals'] = [facts['goals'][target]]
        on_before = lifecycle.expected(verifier, copy.deepcopy(request['tables']), isolated, now, request)
        on_after = lifecycle.expected(verifier, copy.deepcopy(prior), isolated, now, request)
        effect_before = changes(request['tables'], on_before['final_state'])
        effect_after = changes(prior, on_after['final_state'])
        # Compare both changed paths and their resulting values, stripping any
        # untouched predecessor fields from the successor effect.
        def effects(state, paths):
            return [(p, at(state, list(p))) for p in paths]
        causal = target_changed or (changed and (on_before['answer'] != on_after['answer'] or
                              effects(on_before['final_state'], effect_before) != effects(on_after['final_state'], effect_after)))
        witnesses.append({**edge, 'evidence': 'counterfactual_successor_effect', 'observed': causal})
        if not causal:
            problems.append('claimed causal state dependency was not observed')
    return {'passed': not problems, 'problems': problems, 'method': request.get('sop_method'), 'witnesses': witnesses}


def trace_report(env, request):
    """Check declared write scope on the task's actual arguments, beyond canned fixtures."""
    from env_factory.runtime import run_reference
    result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'], request=request)
    atoms = {a['tool']: a for a in request['sop_blueprint']['atoms'].values()}
    api = Episode(env.make_tools(), request['tables'], request['user_facts'], start=request['time']['start'], approve=lambda *_: True, **lifecycle.options(request))
    problems = []; counts = {}
    for entry in result['trace']:
        if entry.get('kind') == 'wait':
            api.wait(entry['seconds'])
            continue
        if entry.get('kind') != 'tool' or entry.get('status') not in ('ok', 'rejected'):
            continue
        name = entry['tool']; before = copy.deepcopy(api.db)
        args = {k: v for k, v in entry['args'].items() if k != 'consent'}
        api.now = entry['time']
        try:
            actual = api.call(name, **args)
        except BusinessError:
            if entry['status'] != 'rejected' or canonical(before) != canonical(api.db):
                problems.append(name + ': invalid rejection')
        else:
            if entry['status'] != 'ok' or actual != entry['result']:
                problems.append(name + ': replay observation differs')
            problems += frame_problems(before, api.db, atoms[name], args)
            counts[name] = counts.get(name, 0) + 1
    return {'passed': not problems, 'problems': problems, 'executed_atoms': counts}


def qualify(env, verifier, request):
    blueprint = request.get('sop_blueprint')
    if blueprint is None:
        return {'passed': True, 'applicable': False}
    problems = validate(blueprint, env.make_tools())
    contract = request.get('sop_contract', {})
    if request.get('sop_contract_sha256') != digest({'blueprint': blueprint, 'contract': contract}):
        problems.append('SOP contract hash mismatch')
    if not contract.get('cases'):
        problems.append('SOP contract needs independent executable cases')
    if problems:
        return {'passed': False, 'problems': problems}
    rows = []
    for case in contract['cases']:
        forward = run_graph(env, blueprint, case, request=request)
        backward = run_graph(env, blueprint, case, reverse=True, request=request)
        equivalent = all(forward.get(k) == backward.get(k) for k in ('final_state_sha256', 'outputs_sha256'))
        rows.append({'id': case['id'], 'passed': forward['passed'] and backward['passed'] and equivalent,
                     'execution': forward, 'alternate_order': backward, 'equivalent': equivalent})
    try:
        task = task_report(env, verifier, request)
        scope = trace_report(env, request)
    except Exception as exc:
        task = {'passed': False, 'problems': [f'{type(exc).__name__}: {exc}']}
        scope = {'passed': False, 'problems': ['task execution failed']}
    return {'passed': all(r['passed'] for r in rows) and task['passed'] and scope['passed'], 'cases': rows, 'task': task, 'scope': scope,
            'contract_sha256': request['sop_contract_sha256']}
