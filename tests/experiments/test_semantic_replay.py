import json
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from env_factory.runtime import write
from replay_world_semantics import replay


def test_replay_rejects_changed_receipt_without_rewriting_messages():
    @write
    def change(db,now,value: int):
        db['rows'][0]['value']=value
        return {'value':value}
    env=SimpleNamespace(make_tools=lambda:{'change':change},approval_key=lambda *a:'allow')
    loaded={'env':env,'request':{'tables':{'rows':[{'value':0}]},'user_facts':{'allow':True},'time':{'start':10}}}
    old={'messages':[{'role':'assistant','tool_calls':[{'id':'c','function':{'name':'change','arguments':'{"value":2}'}}]},
        {'role':'tool','tool_call_id':'c','name':'change','content':json.dumps({'ok':True,'result':{'value':1},'now':10})}]}
    before=json.dumps(old)
    verdict=replay(loaded,old)
    assert not verdict['passed'] and verdict['reason']=='tool_observation_changed'
    assert verdict['actual']['result']=={'value':2}
    assert json.dumps(old)==before
