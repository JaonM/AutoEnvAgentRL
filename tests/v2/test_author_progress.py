import json
from unittest.mock import patch
import pytest
from env_factory_v2.authoring.author_budget import ProgressBudget
from env_factory_v2.authoring.author_workbench import AuthorWorkbench
from env_factory_v2.authoring.author_mcp import AuthorMCP
from env_factory_v2.authoring.author_session import checked_source
from env_factory_v2.examples import direct
from env_factory_v2.contracts.expressions import ContractError


def test_progress_extends_idle_but_not_total_and_repeated_work_does_not_extend():
    b=ProgressBudget(0,600,240)
    def observe(tool,key,now,passed=True):
        b.observe({'tool':tool,'result':{'passed':passed,'source_identity':key}},now)
    observe('read_contract','a',230);assert b.remaining(240)==0
    observe('put_section','a',230);assert b.remaining(240)==230
    observe('put_section','a',460);assert b.remaining(470)==0
    observe('check_contract','a',460);assert b.remaining(590)==10
    observe('edit_contract','b',590);assert b.remaining(600)==0
    observe('check_contract','c',610,False);assert b.last_progress==590


def test_submit_requires_current_check_and_is_terminal(tmp_path):
    server=AuthorMCP(tmp_path/'workbench',source=direct())
    def call(name,args={}):
        result=server.dispatch({'id':1,'method':'tools/call','params':{'name':name,'arguments':args}})
        return json.loads(result['result']['content'][0]['text'])
    assert not call('submit_contract')['passed']
    assert call('check_contract')['passed']
    assert call('submit_contract')['submitted']
    assert checked_source(tmp_path)[0]==direct()
    assert not call('put_section',{'section':'task','value_json':'"changed"'})['passed']
    assert checked_source(tmp_path)[0]==direct()
    changed=direct();changed['task']='tampered'
    (tmp_path/'workbench/draft.json').write_text(json.dumps(changed))
    with pytest.raises(ContractError):checked_source(tmp_path)


def test_same_draft_check_reuse_and_change_invalidation():
    bench=AuthorWorkbench(source=direct())
    assert bench.call('check_contract',{})['passed']
    with patch('env_factory_v2.authoring.author_workbench.preflight_recipes',side_effect=AssertionError('repeated execution')):
        assert bench.call('check_contract',{})['reused']
    bench.call('put_section',{'section':'task','value_json':'"changed"'})
    assert not bench.call('submit_contract',{})['passed']


def test_check_details_are_available_without_resending_entire_report(tmp_path):
    s=AuthorMCP(tmp_path,source=direct())
    def call(name,args={}):
        return json.loads(s.dispatch({'id':1,'method':'tools/call','params':{'name':name,'arguments':args}})['result']['content'][0]['text'])
    summary=call('check_contract');assert summary['passed'] and 'compilation' not in summary
    details=call('read_check',{'path':'/checks','offset':0});assert details['passed']
    assert call('inspect_contract')['checked']


def test_explicit_submission_returns_without_waiting_for_final_model_message(tmp_path):
    from pathlib import Path
    from unittest.mock import Mock
    from env_factory_v2.authoring.author_session import generate_in_session
    def process(args,**kwargs):
        folder=Path(args[args.index('--output-last-message')+1]).parent
        server=AuthorMCP(folder/'workbench',source=direct())
        for name in ['check_contract','submit_contract']:
            server.dispatch({'id':1,'method':'tools/call','params':{'name':name,'arguments':{}}})
        (folder/'events.jsonl').write_text(json.dumps({'type':'item.completed','item':{
            'type':'mcp_tool_call','server':'envfactory_author','tool':'submit_contract','status':'completed'}})+'\n')
        return Mock(pid=123,returncode=-15)
    with patch('env_factory_v2.authoring.author_session.subprocess.Popen',side_effect=process), \
         patch('env_factory_v2.authoring.author_session.os.killpg') as kill:
        source,provenance=generate_in_session({'goal':direct()['task']},evidence_root=tmp_path)
    assert source==direct() and provenance['completion_signal']=='submit_contract'
    kill.assert_called_once()
    assert not list(tmp_path.glob('*/completion.json'))
