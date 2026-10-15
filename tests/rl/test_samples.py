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


def test_token_aggregation_weighs_every_token_of_the_group_alike():
    long_loop = [action(0, False) for _ in range(3)] + [action(0, True)]  # 8 tokens
    group = {'episodes': [
        {'seed': 2, 'actions': long_loop, 'bootstrap': 0., 'final_reward': 0},
        {'seed': 2, 'actions': [action(1, True)], 'bootstrap': 0., 'final_reward': 1}]}  # 2 tokens
    sequence = make_samples(group, 'grpo')
    token = make_samples(group, 'grpo', aggregation='token')
    weight = lambda samples, index: sum(s['weight'] for s in samples if s['episode_index'] == index)
    assert weight(sequence, 0) == pytest.approx(0.5) and weight(sequence, 1) == pytest.approx(0.5)
    assert weight(token, 0) == pytest.approx(0.8) and weight(token, 1) == pytest.approx(0.2)  # the loop's 8 of 10 tokens
    assert sum(s['weight'] for s in token) == pytest.approx(1)


def test_low_probability_mask_puts_the_action_weight_on_uncertain_tokens():
    import math
    from rl.backends.mlx.actor import collate
    sure, unsure = math.log(0.99), math.log(0.4)
    sample = {'prompt': [9], 'tokens': [1, 2, 3, 4], 'old_logp': [sure, unsure, sure, unsure], 'weight': 1.}
    plain = collate([sample])['weights'].tolist()[0]
    masked = collate([sample], low_prob_mask=0.9)['weights'].tolist()[0]
    assert plain == pytest.approx([0.25] * 4)
    assert masked == pytest.approx([0., 0.5, 0., 0.5])
    confident = {**sample, 'old_logp': [sure] * 4}
    assert collate([confident], low_prob_mask=0.9)['weights'].tolist()[0] == pytest.approx([0.25] * 4)  # weight kept
