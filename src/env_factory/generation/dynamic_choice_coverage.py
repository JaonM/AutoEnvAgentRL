"""Executed coverage of a nonempty menu excluding a user's private preference."""
import copy
import json

from env_factory.sandbox_runtime import observed_interaction_options


def _key(item):
    return item['stage_id'], json.dumps(item['options_from'], sort_keys=True)


def requirements(scripts):
    # One witness per distinct stage/binding, not one invented user preference
    # per script. Finite executed coverage is not proof of every possible menu.
    result = {}
    for script in scripts:
        for stage in script.get('interaction_protocol', {}).get('stages', []):
            if 'options_from' in stage:
                item = {'stage_id': stage['id'], 'options_from': copy.deepcopy(stage['options_from'])}
                result[_key(item)] = item
    return list(result.values())


def observe(replay, scripts):
    scripts = {script['script_id']: script for script in scripts}
    events = replay.get('events', [])
    observations = []
    for index, event in enumerate(events):
        result = event.get('result', {})
        if (event.get('event') != 'user_turn'
                or event.get('payload', {}).get('decision_source') != 'interaction_contract'
                or result.get('fsm_transition_applied') is not True
                or not result.get('interaction_branch')
                or result.get('interaction_stage') is not None):
            continue
        script = scripts.get(result.get('fsm_script_id'), {})
        stages = script.get('interaction_protocol', {}).get('stages', [])
        after = result.get('fsm_state_after', '')
        if not isinstance(after, str) or not after.startswith('stage-'):
            continue
        position = after.removeprefix('stage-')
        if not position.isdigit() or int(position) >= len(stages):
            continue
        stage = stages[int(position)]
        if 'options_from' not in stage:
            continue
        branch = next((branch for branch in stage.get('branches', [])
                       if branch['id'] == result['interaction_branch']), None)
        if branch is None or branch.get('advance') is not False:
            continue
        menu = observed_interaction_options(stage, events[:index], require_selection=False)
        if menu is None or stage['private_fact'] in menu['options']:
            continue
        observations.append({'stage_id': stage['id'], 'options_from': copy.deepcopy(stage['options_from']),
            'script_id': script['script_id'], 'branch': branch['id'], 'outcome': result.get('outcome_category'),
            'observed_options': menu['options'], 'event_index': index})
    return observations


def coverage(artifacts, observations):
    required = artifacts.get('task_spec', {}).get('interaction_complexity', {}).get(
        'dynamic_choice_requirements', [])
    seen = {_key(item) for item in observations}
    missing = [item for item in required if _key(item) not in seen]
    return {'passed': not missing, 'required': required, 'missing': missing,
            'observations': observations,
            'scope': 'one_executed_unavailable_preference_case_per_dynamic_stage_not_exhaustive_menus'}


REPAIR_HINT = ('Provide a schema-valid route probe whose actual observed menu is nonempty but excludes '
    'the selected script preference. Execute a declared non-advancing user branch and finish its valid '
    'stop/recovery route with full reward. Empty menus, ordinary refusals while the preference remains '
    'available, and unexecuted branch declarations do not cover this case. Preserve menu membership, '
    'private preferences and the public business goal; never invent a user choice or refusal.')
