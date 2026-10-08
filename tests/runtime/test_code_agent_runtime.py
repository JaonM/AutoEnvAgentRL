import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from env_factory import code_agent_runtime as bridge
from env_factory.runtime_llm import RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError, capture_runtime_llm_trace, summarize_runtime_llm_trace

SCHEMA = {'type':'object','properties':{'label':{'type':'string','enum':['pass','fail']}},
          'required':['label'],'additionalProperties':False}


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script=tmp_path/'fake.py'
    script.write_text('''import json,sys,time
from pathlib import Path
prompt=sys.stdin.read()
counter=Path(__file__).with_suffix('.counter')
attempt=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(attempt))
if 'capacity_always' in prompt or 'capacity_once' in prompt and attempt == 1:
    print(json.dumps({'type':'turn.failed','error':{'message':'Selected model is at capacity. Please try a different model.'}}))
    sys.exit(1)
if 'timeout_case' in prompt: time.sleep(5)
answer=Path(sys.argv[sys.argv.index('--output-last-message')+1])
answer.write_text('not JSON' if 'bad_json' in prompt else json.dumps({'label':'fail' if 'negative' in prompt else 'pass'}))
if 'tool_case' in prompt: print(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}))
if 'missing_completion' not in prompt: print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12}}))
sys.exit(3 if 'bad_exit' in prompt else 0)
''')
    monkeypatch.setattr(bridge,'command',lambda agent,model,prompt,response,**kw:
                        [sys.executable,str(script),'--output-last-message',str(response),'-'])
    return tmp_path/'evidence'


@pytest.mark.parametrize('text,label',[('positive','pass'),('negative','fail')])
def test_native_completion_schema_and_independent_requests(fake_cli,text,label):
    results=[bridge.run_codex([{'role':'user','content':text}],SCHEMA,
        model='gpt-6.1-sol',timeout=3,evidence_root=fake_cli) for _ in range(2)]
    assert results[0]['id'] != results[1]['id']
    assert all(json.loads(r['choices'][0]['message']['content']) == {'label':label} for r in results)
    assert len(list(fake_cli.glob('*/provenance.json'))) == 2


@pytest.mark.parametrize('text',['bad_json','missing_completion','bad_exit','tool_case','timeout_case'])
def test_invalid_or_incomplete_agent_never_returns_a_verdict(fake_cli,text):
    with pytest.raises(RuntimeLLMError):
        bridge.run_codex([{'role':'user','content':text}],SCHEMA,
            model='gpt-6.1-sol',timeout=.1 if text=='timeout_case' else 3,evidence_root=fake_cli)
    provenance = json.loads(next(fake_cli.glob('*/provenance.json')).read_text())
    assert provenance['status'] == 'failed'
    assert provenance['events_sha256'] and provenance['request_sha256']


def test_capacity_retry_preserves_model_budget_and_each_failed_native_invocation(fake_cli, monkeypatch):
    monkeypatch.setattr(bridge.time, 'sleep', lambda _: None)
    result = bridge.run_codex([{'role': 'user', 'content': 'capacity_once'}], SCHEMA,
        model='gpt-6.1-sol', timeout=20, evidence_root=fake_cli)
    assert len(result['attempt_ids']) == 2
    provenance = [json.loads((fake_cli / identity / 'provenance.json').read_text()) for identity in result['attempt_ids']]
    assert [p['status'] for p in provenance] == ['failed', 'passed']
    assert provenance[0]['failure_code'] == 'CODE_AGENT_CAPACITY'
    assert all(p['requested_model'] == 'gpt-6.1-sol' for p in provenance)
    assert len({p['request_sha256'] for p in provenance}) == 1
    assert [p['completed_turns'] for p in provenance] == [0, 1]
    chain = json.loads((fake_cli / result['id'] / 'request_attempts.json').read_text())
    assert chain['attempt_ids'] == result['attempt_ids'] and chain['seconds'] < 20


@pytest.mark.parametrize('budget,attempts', [(3, 1), (20, 2)])
def test_persistent_capacity_is_bounded_and_retains_final_request_identity(fake_cli, monkeypatch, budget, attempts):
    monkeypatch.setattr(bridge.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeLLMError) as error:
        bridge.run_codex([{'role': 'user', 'content': 'capacity_always'}], SCHEMA,
            model='gpt-6.1-sol', timeout=budget, evidence_root=fake_cli)
    assert error.value.failure_code == 'CODE_AGENT_CAPACITY'
    chain = json.loads((fake_cli / error.value.request_id / 'request_attempts.json').read_text())
    assert len(chain['attempt_ids']) == attempts
    assert len(list(fake_cli.glob('*/provenance.json'))) == attempts


@pytest.mark.parametrize('budget,expected', [(20, [20, 8]), (12, [12])])
def test_scoring_retry_subtracts_execution_and_wait_from_original_deadline(tmp_path, monkeypatch, budget, expected):
    clock, timeouts = [0.0], []
    def once(messages, schema, *, timeout, model, evidence_root):
        timeouts.append(timeout)
        identity = ('a' if len(timeouts) == 1 else 'b') * 32
        (tmp_path/identity).mkdir()
        assert model == 'gpt-6.1-sol'
        clock[0] += 7 if len(timeouts) == 1 else 1
        if len(timeouts) == 1:
            raise RuntimeLLMError('unavailable', failure_code='CODE_AGENT_CAPACITY', request_id=identity)
        return {'id': identity}
    monkeypatch.setattr(bridge, '_run_codex_once', once)
    monkeypatch.setattr(bridge.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(bridge.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    if budget == 12:
        with pytest.raises(RuntimeLLMError):
            bridge.run_codex([], SCHEMA, model='gpt-6.1-sol', timeout=budget, evidence_root=tmp_path)
    else:
        bridge.run_codex([], SCHEMA, model='gpt-6.1-sol', timeout=budget, evidence_root=tmp_path)
    assert timeouts == expected


@pytest.fixture
def server(tmp_path):
    calls=[]
    def runner(messages,schema,**kwargs):
        calls.append((messages,schema,kwargs))
        return {'id':str(len(calls)),'model':kwargs['model'],'backend':'codex_cli','usage':{},
            'choices':[{'message':{'content':json.dumps({'label':'fail' if 'negative' in messages[0]['content'] else 'pass'})}}]}
    srv=bridge.CodeAgentServer(('127.0.0.1',0),token='test-token-'+'x'*32,model='gpt-6.1-sol',
                              workers=1,timeout=3,evidence_root=tmp_path,runner=runner)
    thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    yield srv,calls
    srv.shutdown();srv.server_close();thread.join()


def client_for(srv,**kwargs):
    return RuntimeLLMClient(RuntimeLLMConfig(srv.token,f'http://127.0.0.1:{srv.server_port}/v1',
        'gpt-6.1-sol',max_retries=0,backend='codex',**kwargs))


def test_runtime_uses_agent_schema_and_preserves_negative_verdict(server):
    srv,calls=server
    with capture_runtime_llm_trace() as trace:
        assert client_for(srv).json_chat([{'role':'user','content':'negative'}],response_schema=SCHEMA)=={'label':'fail'}
    assert calls[0][1]==SCHEMA and trace[0]['backend']=='codex'
    assert trace[0]['model']=='gpt-6.1-sol' and trace[0]['mock'] is False
    assert summarize_runtime_llm_trace(trace)['backends']=={'codex':1}


def test_server_rejects_unauthorized_and_other_model(server):
    srv,calls=server
    for token,model,code in [('wrong','gpt-6.1-sol',401),(srv.token,'other',400)]:
        body={'model':model,'messages':[{'role':'user','content':'hi'}],
              'response_format':{'type':'json_schema','json_schema':{'schema':SCHEMA}}}
        req=Request(f'http://127.0.0.1:{srv.server_port}/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+token})
        with pytest.raises(HTTPError) as error:urlopen(req,timeout=2)
        assert error.value.code==code
    assert calls==[]


def test_busy_agent_is_infrastructure_error_not_a_negative_label(server):
    srv,calls=server;srv.slots.acquire()
    try:
        with pytest.raises(RuntimeLLMError,match='429'):
            client_for(srv).json_chat([{'role':'user','content':'hi'}],response_schema=SCHEMA)
    finally:srv.slots.release()
    assert calls==[]


def test_http_capacity_error_exposes_only_sanitized_correlation(server):
    srv, _ = server
    def unavailable(*args, **kwargs):
        raise RuntimeLLMError('private payload must not escape', failure_code='CODE_AGENT_CAPACITY', request_id='a'*32)
    srv.runner = unavailable
    with pytest.raises(RuntimeLLMError) as error:
        client_for(srv).json_chat([{'role': 'user', 'content': 'hi'}], response_schema=SCHEMA)
    assert error.value.failure_code == 'CODE_AGENT_CAPACITY'
    assert error.value.request_id == 'a'*32
    assert '503' in str(error.value) and 'private payload' not in str(error.value)


def test_runtime_rejects_wrong_backend_without_provider_fallback(server):
    srv,_=server
    srv.runner=lambda *a,**k:{'backend':'api','model':'gpt-6.1-sol','choices':[{'message':{'content':'{"label":"pass"}'}}]}
    with pytest.raises(RuntimeLLMError,match='identity mismatch'):
        client_for(srv).json_chat([{'role':'user','content':'hi'}],response_schema=SCHEMA)


def test_backend_is_explicit_and_invalid_backend_rejected(monkeypatch):
    monkeypatch.setenv('SANDBOX_LLM_API_KEY','bridge-token')
    monkeypatch.setenv('SANDBOX_LLM_MODEL','gpt-6.1-sol')
    monkeypatch.setenv('SANDBOX_LLM_BASE_URL','http://127.0.0.1:55794/v1')
    monkeypatch.setenv('SANDBOX_LLM_BACKEND','codex')
    assert RuntimeLLMConfig.from_env().backend=='codex'
    monkeypatch.setenv('SANDBOX_LLM_BACKEND','typo')
    with pytest.raises(RuntimeLLMError,match='backend'):RuntimeLLMConfig.from_env()


@pytest.mark.parametrize('missing',['SANDBOX_LLM_API_KEY','SANDBOX_LLM_BASE_URL','SANDBOX_LLM_MODEL'])
def test_code_agent_configuration_never_inherits_external_provider(monkeypatch, missing):
    for key,value in {'SANDBOX_LLM_API_KEY':'bridge-token',
        'SANDBOX_LLM_BASE_URL':'http://127.0.0.1:55794/v1',
        'SANDBOX_LLM_MODEL':'gpt-6.1-sol','SANDBOX_LLM_BACKEND':'codex',
        'SANDBOX_EVALUATOR_MOCK':'0','LLM_API_KEY':'external-key',
        'LLM_BASE_URL':'https://external.example/v1','LLM_MODEL':'external-model'}.items():
        monkeypatch.setenv(key,value)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeLLMError,match=missing):
        RuntimeLLMConfig.from_env()


def test_optional_fields_are_restored_without_relaxing_original_validation():
    original={'type':'object','properties':{'label':{'type':'string'},'transition_id':{'type':'string'}},'required':['label']}
    encoded=bridge.cli_schema(original)
    assert encoded['additionalProperties'] is False
    assert encoded['required']==['label','transition_id']
    restored=bridge.restore_optional_fields({'label':'fail','transition_id':None},original)
    assert restored=={'label':'fail'}
    assert 'additionalProperties' not in original
    RuntimeLLMClient._validate_response(restored,original)
    with pytest.raises(RuntimeLLMError):
        RuntimeLLMClient._validate_response(bridge.restore_optional_fields({'label':None},original),original)
