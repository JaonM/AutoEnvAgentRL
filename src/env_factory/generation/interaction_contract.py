"""Compile task-authored, executable multi-turn user protocols."""
import copy
import json
import re
import math

KINDS = {'information_required', 'option_selection', 'constraint_update', 'execution_confirmation', 'correction'}


def compile_interactions(contract, description, tools, count, *, implementations=(), scenarios=()):
    if not isinstance(contract, dict) or contract.get('version') != '1.0':
        raise ValueError('interaction_contract requires version 1.0')
    variants = contract.get('variants', [])
    if len(variants) != count:
        raise ValueError('interaction variants must match user_script_count')
    writes = {tool['tool_name'] for tool in implementations if tool['operation'] in {'insert','update','delete'}}
    references = [step for scenario in scenarios if scenario.get('kind') == 'goal_success' for step in scenario['steps'] if step.get('operation') == 'tool_call']
    names = {tool['function']['name'] for tool in tools}
    scripts, signatures, kinds = [], set(), set()
    public = description.get('public_input', {}).get('initial_user_message', '')
    for index, variant in enumerate(variants):
        stages = variant.get('stages', [])
        if not 2 <= len(stages) <= 5:
            raise ValueError('each interaction variant requires 2..5 necessary user turns')
        signature = json.dumps(stages, sort_keys=True, ensure_ascii=False)
        if signature in signatures:
            raise ValueError('interaction variants must differ beyond script identifiers')
        signatures.add(signature)
        seen, stages = set(), copy.deepcopy(stages)
        bindings = set()
        disclosed_text = public
        for stage in stages:
            key, kind = stage.get('id'), stage.get('kind')
            if not isinstance(key, str) or not key or key in seen or kind not in KINDS:
                raise ValueError('invalid interaction stage identity/kind')
            seen.add(key); kinds.add(kind)
            for field in ('user_reply', 'reference_response', 'purpose', 'retry_reply'):
                if not isinstance(stage.get(field), str) or not stage[field].strip():
                    raise ValueError(f'interaction stage requires {field}')
            required = stage.get('assistant_contains_all')
            if not isinstance(required, list) or not required or any(not isinstance(x, str) or not x.strip() for x in required):
                raise ValueError('interaction stage requires grounded assistant_contains_all')
            if not all(x in stage['reference_response'] for x in required):
                raise ValueError('reference response must satisfy interaction trigger')
            before = stage.get('before_tool', '__answer__')
            if before != '__answer__' and before not in names:
                raise ValueError('interaction stage references unknown before_tool')
            if kind == 'execution_confirmation' and before not in writes:
                raise ValueError('execution confirmation must protect an actual tool')
            evidence = stage.get('requires_tools', [])
            if not isinstance(evidence, list) or not set(evidence) <= names or before in evidence:
                raise ValueError('invalid interaction tool evidence dependencies')
            hidden = stage.get('private_fact')
            if kind in {'information_required', 'option_selection', 'constraint_update', 'correction'}:
                if not isinstance(hidden, (str, int, float, bool)) or (isinstance(hidden, str) and not hidden) or (isinstance(hidden, float) and not math.isfinite(hidden)):
                    raise ValueError('private_fact must be a finite scalar')
                fact_text = hidden if isinstance(hidden, str) else json.dumps(hidden)
                pattern = (r'(?<![\d.])' + re.escape(fact_text) + r'(?![\d.])' if not isinstance(hidden, str)
                           else re.escape(fact_text))
                if not re.search(pattern, stage['user_reply']):
                    raise ValueError('interaction requires a concrete disclosed private_fact')
                if re.search(pattern, json.dumps(description.get('public_input', {}), ensure_ascii=False)):
                    matches = []
                    def locate(value, path):
                        if isinstance(value, dict):
                            for field, child in value.items():
                                locate(child, path + "." + str(field))
                        elif isinstance(value, list):
                            for position, child in enumerate(value):
                                locate(child, path + f"[{position}]")
                        else:
                            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            match = re.search(pattern, text)
                            if match and len(matches) < 3:
                                matches.append({"path": path,
                                    "excerpt": text[max(0, match.start()-50):match.end()+50]})
                    locate(description.get('public_input', {}), 'description.public_input')
                    raise ValueError('private interaction fact already appears in initial user input: '
                        + json.dumps({"variant_index": index, "stage_id": key, "matches": matches,
                            "hint": "Compiled public input includes published answer schema and semantic criteria; inspect these as well as the original message."}, ensure_ascii=False))
                if kind != 'option_selection' and re.search(pattern, stage['reference_response']):
                    raise ValueError('reference question must not pre-disclose the private fact')
                if re.search(pattern, stage['retry_reply']):
                    raise ValueError('retry reply must not leak undisclosed private fact')
                binding = stage.get('bind_to', {})
                if not isinstance(binding, dict) or binding.get('tool') != before or not binding.get('argument'):
                    raise ValueError('private interaction fact requires bind_to tool/argument at disclosure boundary')
                binding_key = (before, binding['argument'])
                if binding_key in bindings:
                    raise ValueError('redundant interaction asks for the same argument twice')
                bindings.add(binding_key)
                if kind in {'constraint_update', 'correction'}:
                    previous = stage.get('previous_fact')
                    previous_text = previous if isinstance(previous, str) else json.dumps(previous)
                    if previous is None or previous == hidden or not previous_text or previous_text not in disclosed_text:
                        raise ValueError('update/correction requires a previously disclosed different fact')
                if not any(step['tool_name'] == before and step.get('arguments', {}).get(binding['argument']) == hidden for step in references):
                    raise ValueError('disclosed private fact does not match executable reference argument')
            if kind == 'option_selection':
                options = stage.get('options', [])
                if not isinstance(options, list) or len(set(options)) < 2 or hidden not in options:
                    raise ValueError('option selection requires distinct options and an actual selected option')
                if not all(str(option) in stage['reference_response'] for option in options):
                    raise ValueError('selection reference must present every option')
            disclosed_text += '\n' + stage['user_reply']
            stage['before_tool'] = before
        states = [{'state_id':f'stage-{i}', 'user_behavior':stage['purpose'], 'terminal':False} for i,stage in enumerate(stages)]
        states += [{'state_id':'review', 'user_behavior':'检查实际业务目标是否完成', 'terminal':False}, {'state_id':'done', 'user_behavior':'接受结果', 'terminal':True}]
        outcomes = {'correction':'user_correction', 'constraint_update':'user_correction', 'execution_confirmation':'information_required', 'option_selection':'information_required', 'information_required':'information_required'}
        transitions = [{'transition_id':stage['id'], 'from_state':f'stage-{i}', 'to_state':f'stage-{i+1}' if i+1<len(stages) else 'review', 'condition':stage['purpose'], 'outcome_category':outcomes[stage['kind']], 'should_end':False, 'updates':{}} for i,stage in enumerate(stages)]
        transitions.append({'transition_id':'complete', 'from_state':'review', 'to_state':'done', 'condition':'业务目标和所有必要交互均已满足', 'outcome_category':'goal_satisfied', 'should_end':True, 'updates':{}})
        scripts.append({'initial_state':'stage-0', 'states':states, 'transitions':transitions, 'variables':{},
                        'recovery_policy':{'max_recoveries':2, 'user_behavior':'请完成当前必要交互。', 'handled_outcomes':['agent_off_topic','agent_premature_completion','unrecognized']}, 'script_id':f'script-{index+1}', 'goal':description['task'],
                        'user_input':copy.deepcopy(description.get('public_input', {})),
                        'interaction_protocol':{'version':'1.0', 'stages':stages}})
    if len(kinds) < 2:
        raise ValueError('interaction contract requires at least two interaction kinds')
    return scripts


def interleave_scenarios(scenarios, scripts):
    """Expand business success cases into seed-selected executable dialogue cases."""
    import random
    result = []
    for scenario in scenarios:
        if scenario.get('kind') != 'goal_success':
            result.append(scenario)
            continue
        for index, script in enumerate(scripts):
            seed = next(seed for seed in range(10000) if random.Random(seed).randrange(len(scripts)) == index)
            expanded = copy.deepcopy(scenario)
            expanded['scenario_id'] += f'-{script["script_id"]}'
            pending = list(script['interaction_protocol']['stages'])
            steps = [] if any(step.get('operation') == 'reset' for step in expanded['steps']) else [{'operation':'reset', 'body':{'seed':seed}}]
            for step in expanded['steps']:
                if step.get('operation') == 'reset':
                    step.setdefault('body', {})['seed'] = seed
                anchor = step.get('tool_name') if step.get('operation') == 'tool_call' else '__answer__' if step.get('operation') == 'agent_response' else None
                while pending and pending[0]['before_tool'] == anchor:
                    stage = pending.pop(0)
                    steps.append({'operation':'dialogue_turn', 'content':stage['reference_response'],
                                  'expected_stage':stage['id'], 'expected_status':200})
                steps.append(step)
            if pending:
                raise ValueError('interaction stage ordering cannot be reached in reference tools')
            expanded['steps'] = steps
            result.append(expanded)
    return result
