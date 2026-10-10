"""Read actual runtime evidence without generated reward expressions or targets."""
import copy
import math


def event_time(event):
    value = event['time']['elapsed_seconds']
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Runtime evidence contains invalid virtual time')
    return value


def observed_time(world, call_id):
    """Use the exact successful business call, never the final episode clock."""
    events = [event for event in world.trace if event.get('kind') == 'tool' and event.get('id') == call_id]
    if len(events) != 1 or events[0].get('status') != 'ok':
        raise ValueError('Runtime time requires one actual successful tool call')
    return event_time(events[0])


def authorization_decisions(world):
    """Preserve approvals, refusals and revised proposals in chronological order.

    This is a decision log, not proof that authorization remains valid at a
    later write: corrections, withdrawals and consumption must also be checked.
    """
    result = []
    for index, event in enumerate(world.trace):
        if event.get('kind') != 'user':
            continue
        stage = world.user.stages.get(event['stage'])
        if stage is None or stage['kind'] != 'authorization':
            continue
        decision = event['result']
        if type(decision.get('approved')) is not bool or decision.get('proposal') != event.get('proposal'):
            raise ValueError('Authorization evidence does not match the actual proposal')
        result.append({'trace_index': index, 'stage': event['stage'], 'time': event_time(event),
                       'approved': decision['approved'], 'proposal': copy.deepcopy(event['proposal'])})
    return result


def state_before_event(world, trace_index):
    """Reconstruct historical state from initial rows and actual committed deltas.

    Never execute generated rules, reward targets or reference steps. Declared
    virtual events contain literal updates; successful transaction results
    contain the updates that actually committed. Failed writes contribute none.
    """
    if type(trace_index) is not int or not 0 <= trace_index < len(world.trace):
        raise ValueError('Historical state requires an actual trace event')
    state = copy.deepcopy(world.initial)
    keys = {table['id']: table['primary_key'] for table in world.source['tables']}
    applied = set()
    events = sorted(enumerate(world.source.get('events', [])), key=lambda pair: (pair[1]['at_seconds'], pair[0]))
    def update(table, identity, values):
        if table not in keys:
            raise ValueError('Runtime delta references an unknown table')
        found = [row for row in state[table] if type(row[keys[table]]) is type(identity)
                 and row[keys[table]] == identity]
        if len(found) != 1:
            raise ValueError('Runtime delta requires a unique actual record')
        found[0].update(copy.deepcopy(values))
    previous_time = -1
    for index, event in enumerate(world.trace[:trace_index + 1]):
        now = event_time(event)
        if now < previous_time:
            raise ValueError('Runtime evidence time moves backwards')
        previous_time = now
        for identity, due in events:
            if identity not in applied and due['at_seconds'] <= now:
                update(due['table'], due['key'], due['set'])
                applied.add(identity)
        if index == trace_index:
            return state
        if event.get('kind') == 'tool' and event.get('status') == 'ok':
            tool = world.tools[event['tool']]
            if tool['kind'] == 'transaction':
                if event['result'].get('committed') is not True:
                    raise ValueError('Successful transaction lacks commit evidence')
                for delta in event['result']['changes']:
                    if delta.get('create'):
                        table = delta['table']
                        if any(canonical_key[keys[table]] == delta['key'] for canonical_key in state[table]):
                            raise ValueError('Runtime creation duplicates an existing record')
                        state[table].append({keys[table]: copy.deepcopy(delta['key']), **copy.deepcopy(delta['set'])})
                    else:
                        update(delta['table'], delta['key'], delta['set'])


def authorized_call_context(world, call_id, authorization_index):
    """Bind one actual committed call to its exact earlier approval and state."""
    calls = [(index, event) for index, event in enumerate(world.trace)
             if event.get('kind') == 'tool' and event.get('id') == call_id]
    if len(calls) != 1 or calls[0][1].get('status') != 'ok':
        raise ValueError('Authorized context requires one successful call')
    call_index, call = calls[0]
    decisions = [d for d in authorization_decisions(world) if d['trace_index'] == authorization_index]
    if len(decisions) != 1 or not decisions[0]['approved'] or authorization_index >= call_index:
        raise ValueError('Authorized context requires an earlier actual approval')
    decision = decisions[0]
    tool = world.tools[call['tool']]
    if (tool['kind'] != 'transaction' or tool['authorization'] != decision['stage']
            or decision['proposal'] != {'tool': call['tool'], 'arguments': call['arguments']}):
        raise ValueError('Committed call differs from exact authorized proposal')
    if any(event.get('kind') == 'revoke' and event.get('stage') == decision['stage']
           for event in world.trace[authorization_index + 1:call_index]):
        raise ValueError('Authorization was revoked before the call')
    return {'call_time': event_time(call), 'authorization_time': decision['time'],
            'authorized_state': state_before_event(world, authorization_index),
            'precommit_state': state_before_event(world, call_index),
            'proposal': copy.deepcopy(decision['proposal'])}
