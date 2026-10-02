import json,math
import pytest
from rl.replay import ReplayBuffer,validate_group,importance_diagnostics
from rl.tasks import TaskSpec


def group():
    return {'schema_version':2,'task_id':'a','task_identity':'digest','policy_version':0,
            'episodes':[{'task_id':'a','seed':1,'final_reward':1.,'terminated':True,
            'actions':[{'prompt':[1],'tokens':[2],'old_logp':[-.5]}]}]}


def test_missing_probabilities_and_eval_task_rejected():
    tasks=[TaskSpec('a','/a',identity='digest')]
    g=group();validate_group(g,tasks)
    g['episodes'][0]['actions'][0].pop('old_logp')
    with pytest.raises(ValueError,match='behavior'):validate_group(g,tasks)
    with pytest.raises(ValueError,match='eval'):validate_group(group(),[TaskSpec('a','/a',split='eval',identity='digest')])


def test_replay_has_bounded_reuse_and_restores_rng(tmp_path):
    tasks=[TaskSpec('a','/a',identity='digest')]
    p=tmp_path/'group.json';p.write_text(json.dumps(group()))
    buffer=ReplayBuffer(max_uses=2,max_age=3)
    buffer.add(p,group(),0)
    restored=ReplayBuffer(max_uses=2,max_age=3,state=json.loads(json.dumps(buffer.state())))
    assert buffer.sample(1,tasks)==restored.sample(1,tasks)
    assert buffer.sample(2,tasks)
    assert buffer.sample(2,tasks) is None
    assert restored.sample(4,tasks) is None


def test_replay_detects_changed_artifact(tmp_path):
    p=tmp_path/'group.json';p.write_text(json.dumps(group()))
    buffer=ReplayBuffer();buffer.add(p,group(),0)
    p.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        buffer.sample(1,[TaskSpec('a','/a',identity='digest')])


def test_ess_is_stable_for_extreme_log_ratios():
    stats=importance_diagnostics([0.,-1000.],[-1000.,-1000.])
    assert stats['effective_sample_fraction']==pytest.approx(.5)
    assert importance_diagnostics([-.1,-.2],[-.1,-.2])['effective_sample_fraction']==pytest.approx(1.)


def test_import_checks_policy_identity_and_rebases_age(tmp_path):
    from rl.replay import import_replay
    identity = {'base':'base-digest','temperature':1.}
    g = group();g.update(policy_identity=identity, temperature=1.,policy_version=100)
    (tmp_path/'group-0001.json').write_text(json.dumps(g))
    (tmp_path/'execution_provenance.json').write_text(json.dumps({'policy_identity':identity}))
    tasks=[TaskSpec('a','/a',identity='digest')]
    buffer=ReplayBuffer(max_age=3)
    assert import_replay(tmp_path,buffer,tasks,identity)==1
    result, entry=buffer.sample(1,tasks)
    assert result['policy_version']==100 and entry['age_origin']==0
    with pytest.raises(ValueError,match='identity mismatch'):
        import_replay(tmp_path,ReplayBuffer(),tasks,{'base':'changed','temperature':1.})


def test_ppo_replay_requires_original_values_and_preserves_advantages():
    from copy import deepcopy
    from rl.replay import validate_ppo_behavior
    from rl.trajectory import make_samples
    g=group();e=g['episodes'][0];e['bootstrap']=0.
    a=e['actions'][0];a.update(old_values=[.2],reward=1.,terminated=True)
    validate_ppo_behavior(g)
    before=make_samples(g,'ppo')
    replayed=deepcopy(g)
    replayed['episodes'][0]['actions'][0].update(target_values=[.8],target_logp=[-.1])
    assert make_samples(replayed,'ppo')==[dict(before[0],target_values=[.8],target_logp=[-.1])]
    assert before[0]['advantage']==pytest.approx([.8])
    del replayed['episodes'][0]['actions'][0]['old_values']
    with pytest.raises(ValueError,match='critic values'):validate_ppo_behavior(replayed)


def test_grpo_replays_complete_original_group_with_fixed_behavior_probabilities(tmp_path):
    from copy import deepcopy
    from rl.trajectory import make_samples
    g=group()
    first=g['episodes'][0]
    first['bootstrap']=0.
    first['actions'][0].update(reward=1.,terminated=True)
    second=deepcopy(first);second['final_reward']=0.;second['actions'][0]['reward']=0.
    g['episodes'].append(second)
    path=tmp_path/'group.json';path.write_text(json.dumps(g))
    buffer=ReplayBuffer(max_uses=2)
    buffer.add(path,g,0)
    original=make_samples(g,'grpo')
    for version in [1,2]:
        replayed,_=buffer.sample(version,[TaskSpec('a','/a',identity='digest')])
        assert make_samples(replayed,'grpo')==original
        assert [s['advantage'] for s in original]==pytest.approx([1.,-1.])
        assert [s['old_logp'] for s in original]==[[-.5],[-.5]]
    assert buffer.sample(3,[TaskSpec('a','/a',identity='digest')]) is None


def test_importance_clip_fraction_uses_ppo_clip_interval():
    stats=importance_diagnostics([-.999,-1.001],[-1.,-1.])
    assert stats['importance_clip_fraction']==0
    stats=importance_diagnostics([math.log(.7),math.log(1.3),0.],[0.,0.,0.])
    assert stats['importance_clip_fraction']==pytest.approx(2/3)


def test_pending_batch_excludes_repeated_replay_without_consuming_use(tmp_path):
    tasks = [TaskSpec('a', '/a', identity='digest')]
    p = tmp_path / 'group.json'
    p.write_text(json.dumps(group()))
    buffer = ReplayBuffer(max_uses=2)
    buffer.add(p, group(), 0)
    assert buffer.sample(0, tasks, exclude_paths={str(p)}) is None
    assert buffer.entries[0]['uses'] == 0
    assert buffer.sample(0, tasks) is not None


def test_replay_respects_scheduled_sandbox_identity(tmp_path):
    tasks = [TaskSpec('a', '/a', identity='digest'), TaskSpec('b', '/b', identity='other')]
    p = tmp_path / 'a.json'
    p.write_text(json.dumps(group()))
    buffer = ReplayBuffer()
    buffer.add(p, group(), 0)
    assert buffer.sample(0, tasks, task_id='b') is None
    assert buffer.entries[0]['uses'] == 0
    assert buffer.sample(0, tasks, task_id='a')[0]['task_id'] == 'a'
