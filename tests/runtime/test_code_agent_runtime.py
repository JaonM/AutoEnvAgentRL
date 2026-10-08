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
