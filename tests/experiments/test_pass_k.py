import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/bench'))
from pass_k import estimates, summarize, task_rows
from serve_openai import sampling_seed, seeded_sampler
from run_pass8_control import select_tasks


def test_any_success_is_not_all_success():
    assert estimates(8, 1, 8) == {'pass_at_k': 1.0, 'pass_hat_k': 0.0}
    assert estimates(8, 8, 8) == {'pass_at_k': 1.0, 'pass_hat_k': 1.0}
    assert estimates(8, 0, 8) == {'pass_at_k': 0.0, 'pass_hat_k': 0.0}
    assert estimates(8, 3, 1) == {'pass_at_k': .375, 'pass_hat_k': .375}
    with pytest.raises(ValueError):
        estimates(1, 1, 8)
    rows = [{'trials': 8, 'successes': c} for c in (0, 1, 8)]
    result = summarize(rows)
    assert result['pass_at_8'] == 2/3
    assert result['pass_hat_8'] == 1/3
    assert result['mixed_group_share'] == 1/3


def fixture_data():
    return {'tasks': [{'id': '1', 'evaluation_criteria': {'actions': [{'name': 'cancel_pending_order'}]}}],
            'simulations': [{'task_id': '1', 'trial': i, 'seed': 100+i, 'termination_reason': 'user_stop',
                             'reward_info': {'reward': float(i == 0)}} for i in range(8)]}


def test_missing_duplicate_and_infrastructure_trials_are_not_scores():
    data = fixture_data()
    assert task_rows(data, 'retail', ['1'], 8)[0]['successes'] == 1
    assert task_rows(data, 'retail', ['1'], 8)[0]['write']
    with pytest.raises(ValueError):
        task_rows(data, 'retail', ['1', '2'], 8)
    for bad in ('missing', 'duplicate', 'infra', 'reward'):
        d = copy.deepcopy(data)
        if bad == 'missing': d['simulations'].pop()
        if bad == 'duplicate': d['simulations'][7]['trial'] = 0
        if bad == 'infra': d['simulations'][7]['termination_reason'] = 'infrastructure_error'
        if bad == 'reward': d['simulations'][7]['reward_info'] = None
        with pytest.raises(ValueError): task_rows(d, 'retail', ['1'], 8)


def test_request_seed_is_stable_but_changes_with_trial_and_context():
    assert sampling_seed(300, [1, 2]) == sampling_seed(300, [1, 2])
    assert sampling_seed(300, [1, 2]) != sampling_seed(301, [1, 2])
    assert sampling_seed(300, [1, 2]) != sampling_seed(300, [1, 2, 3])


def test_sampler_does_not_share_random_state():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    logits = mx.zeros((1, 32))
    first, second, other = seeded_sampler(1, 17), seeded_sampler(1, 17), seeded_sampler(1, 18)
    a, b, c = [], [], []
    for _ in range(32):
        a.append(first(logits).item()); c.append(other(logits).item()); b.append(second(logits).item())
    assert a == b
    assert a != c


def test_fixed_subset():
    if not Path('.bench/tau2-repo/data/simulations/v3bench/qA3m-retail/results.json').exists():
        pytest.skip('local benchmark task definitions not present')
    selected, meta = select_tasks()
    assert sum(map(len, selected.values())) == 32
    assert all(len(ids) == len(set(ids)) for ids in selected.values())
    assert all(n > 0 for n in meta['strata'].values())
    assert (selected, meta) == select_tasks()
