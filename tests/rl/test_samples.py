import pytest
from rl.train import make_samples


def action(reward, terminated):
    return {'prompt': [99], 'tokens': [1, 2], 'old_logp': [-.1, -.2],
            'old_values': [0., 0.], 'reward': reward, 'terminated': terminated}


def test_group_weighting_and_no_bootstrap_on_terminal():
    group = {'episodes': [
        {'seed': 2, 'actions': [action(0,False),action(1,True)], 'bootstrap': 999., 'final_reward': 1},
        {'seed': 2, 'actions': [action(0,True)], 'bootstrap': 999., 'final_reward': 0}]}
    samples = make_samples(group,'grpo')
    assert sum(s['weight'] for s in samples) == pytest.approx(1)
    assert [s['advantage'] for s in samples] == pytest.approx([1,1,-1])
    assert samples[1]['return'] == [0., 0.]  # GRPO does not train a critic
    assert samples[2]['return'][-1] == 0
    group['episodes'][1]['seed'] = 3
    with pytest.raises(ValueError,match='same initial'):
        make_samples(group,'grpo')


def test_ppo_token_rewards_and_terminal_bootstrap():
    group = {'episodes': [{'seed': 1, 'actions': [action(1, True)],
                          'bootstrap': 999., 'final_reward': 1.}]}
    sample = make_samples(group, 'ppo', gamma=.9, lam=1., discount_unit='token')[0]
    assert sample['advantage'] == pytest.approx([.9, 1.])
    assert sample['return'] == pytest.approx([.9, 1.])
    group['episodes'][0]['actions'][0]['terminated'] = False
    group['episodes'][0]['bootstrap'] = 2.
    assert make_samples(group, 'ppo', gamma=.9, lam=1., discount_unit='token')[0]['return'] == pytest.approx([2.52, 2.8])


def test_action_discount_does_not_penalize_longer_action_text():
    def group(n):
        first = {'prompt':[99], 'tokens':[1]*n, 'old_logp':[-.1]*n,
                 'old_values':[0.]*n, 'reward':0., 'terminated':False}
        return {'episodes':[{'seed':1,'actions':[first,action(1,True)],
                             'bootstrap':0.,'final_reward':1.}]}
    short=make_samples(group(1),'ppo',gamma=1.,lam=.95)
    long=make_samples(group(200),'ppo',gamma=1.,lam=.95)
    assert short[0]['advantage'][0] == pytest.approx(.95)
    assert long[0]['advantage'][0] == pytest.approx(short[0]['advantage'][0])


def test_group_cannot_mix_tasks_even_with_same_seed():
    group={'episodes':[{'task_id':'a','seed':1},{'task_id':'b','seed':1}]}
    with pytest.raises(ValueError,match='same initial'):
        make_samples(group,'grpo')
