"""Serial live native-user behavior checks; diagnostic, never a yield claim."""
import argparse
import copy
import json
from pathlib import Path

from env_factory_v2.native.code_agent import NativeCodeAgent
from env_factory_v2.examples import simple, transaction
from env_factory_v2.execution.free_dialogue import DialogueUser, converse
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.contracts.expressions import ContractError


def run_case(name, agent):
    source = simple() if name in {'clarification', 'composite', 'correction'} else transaction()
    if name == 'composite':
        source['user']['facts']['color'] = '蓝色'
        source['user']['stages'].append({'id': 'color', 'kind': 'fact', 'fact': 'color', 'depends_on': []})
    if name == 'correction':
        source['user']['facts']['new_size'] = 'L'
        source['user']['stages'].append({'id': 'change_size', 'kind': 'correction',
            'fact': 'new_size', 'as_fact': 'size', 'depends_on': ['size']})
    if name == 'refusal':
        source['user']['facts']['approve_approve'] = False
    if name == 'revocation':
        source['user']['facts']['revoke_approve'] = True
    if name == 'multiple_authorizations':
        other = copy.deepcopy(source['tools'][1])
        other.update(id='transfer_second', authorization='approve_second')
        source['tools'].append(other)
        source['user']['facts']['approve_approve_second'] = True
        source['user']['stages'].append({'id': 'approve_second', 'kind': 'authorization',
            'tool': 'transfer_second', 'depends_on': ['approve']})
    sandbox = Sandbox(source)
    sandbox.user = DialogueUser(sandbox.source)
    responses = []
    def chat(message, **kwargs):
        reply = converse(sandbox, agent, message, **kwargs)
        responses.append(reply)
        return reply
    if name == 'clarification':
        chat('为了查合脚的款式，你穿什么尺码？')
        chat('再确认一下，你刚才说的尺码是多少？')
        assert sandbox.user.disclosed['size'] == 'M'
    elif name == 'composite':
        chat('你穿什么尺码，喜欢什么颜色？')
        assert sandbox.user.disclosed == {'size': 'M', 'color': '蓝色'}
    elif name == 'correction':
        chat('你穿什么尺码？')
        chat('提交前再确认一下，你现在是否要更改尺码？')
        assert sandbox.user.disclosed['size'] == 'L'
    else:
        sandbox.call('accounts', {}, call_id='balances')
        proposal = copy.deepcopy(source['scenarios'][0]['steps'][1]['proposal'])
        chat('请确认是否按展示的账户和金额执行转账。', proposal=proposal)
        if name == 'refusal':
            assert sandbox.user.states['approve'] == 'consent.declined'
            assert not responses[-1]['authorizations']
        elif name == 'revocation':
            chat('执行前最后确认，你是否还要继续？')
            assert sandbox.user.states['approve'] == 'consent.revoked'
        else:
            sandbox.call('transfer', proposal['arguments'], call_id='first_write')
            proposal['tool'] = 'transfer_second'
            chat('第一笔已完成，是否另外授权第二笔相同金额的转账？', proposal=proposal)
            assert sandbox.user.states['approve_second'] == 'consent.authorized'
            sandbox.call('transfer_second', proposal['arguments'], call_id='second_write')
            assert sandbox.user.states['approve_second'] == 'consent.consumed'
            assert sandbox.data['accounts'][0]['balance'] == 60
        if name in {'refusal', 'revocation'}:
            try:
                sandbox.call('transfer', proposal['arguments'], call_id='forbidden_write')
            except ContractError as exc:
                assert 'authorization' in str(exc)
            else:
                raise AssertionError('unauthorized write accepted')
            assert sandbox.data == sandbox.initial
    return {'passed': True, 'responses': responses, 'dialogue': sandbox.user.dialogue,
            'user_state': sandbox.user.snapshot(), 'trace': sandbox.trace, 'final_state': sandbox.data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in ('clarification', 'composite', 'correction', 'refusal', 'revocation', 'multiple_authorizations'):
        agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / name / 'native')
        try:
            result = run_case(name, agent)
        except Exception as exc:
            result = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        result.update(name=name, scope='live_native_user_behavior_diagnostic_not_yield')
        (args.output / f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        records.append({'name': name, 'passed': result['passed'], 'error': result.get('error')})
        (args.output / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    return 0 if all(r['passed'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
