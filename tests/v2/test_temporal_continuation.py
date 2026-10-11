import copy
import json

import pytest

from env_factory_v2 import training
from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.expressions import evaluate
from rl.environments.v2 import V2Episode


def predicate(basis):
    return {'transaction_preconditions': {'tool':'transfer','basis':basis,'arguments':{
        'from_id':lit('A'),'to_id':lit('B'),'amount':lit(20)}}}


def test_continuation_preserves_post_reset_initial_basis_and_current_clock(tmp_path, monkeypatch):
    source=transaction()
    source['rules'].append({'id':'funds','public':'Sufficient source funds required.',
        'predicate':{'ge':[ref('data.accounts.0.balance'),ref('args.amount')]}})
    source['tools'][1]['rule_ids'].append('funds')
    (tmp_path/'contract.json').write_text(json.dumps(source))
    (tmp_path/'manifest.json').write_text(json.dumps({'bundle_identity':'unit-only'}))
    monkeypatch.setattr(training,'verify_training_bundle',lambda _: {})
    environment=V2Episode(tmp_path,user_mode='structured')
    restored=V2Episode(tmp_path,user_mode='structured')
    try:
        environment.reset(7)
        world=environment.server.episodes[environment.episode]
        world.reset([{'table':'accounts','key':'A','column':'balance','value':5}])
        world.data['accounts'][0]['balance']=100
        world.wait(50)
        assert not evaluate(predicate('initial'),world.context(private=True))
        assert evaluate(predicate('current'),world.context(private=True))
        snapshot=environment.snapshot()
        restored.restore(snapshot)
        other=restored.server.episodes[restored.episode]
        assert other.time()==world.time()
        assert not evaluate(predicate('initial'),other.context(private=True))
        assert evaluate(predicate('current'),other.context(private=True))
        assert other.eligibility_initial==world.eligibility_initial
        snapshot['world']['eligibility_initial']['data']['accounts'][0]['balance']=900
        assert other.eligibility_initial['data']['accounts'][0]['balance']==5
        assert 'eligibility_initial' not in json.dumps(restored.messages)
        incomplete=environment.snapshot();incomplete['world'].pop('eligibility_initial')
        before=copy.deepcopy(other.data); episode=restored.episode
        with pytest.raises(ValueError,match='initial transaction basis'):
            restored.restore(incomplete)
        assert restored.episode==episode and other.data==before
    finally:
        environment.close();restored.close()
