import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
import release_diverse_dataset as release


def population():
    episodes, records = [], []
    for group, count in release.TARGETS.items():
        domain, split = group.rsplit('_', 1)
        for i in range(count):
            episodes.append({'id': f'{group}:{i}', 'domain': domain, 'split': split, 'scenario_family': split})
            records.append({'group': group, 'split': split, 'key': f'{group}:{i}'})
    return episodes, records


def test_complete_population_and_partial_inspection_are_distinct():
    episodes, records = population()
    assert release.validate_population(episodes, records, complete=True) == release.TARGETS
    release.validate_population(episodes[:-1], records[:-1], complete=False)
    with pytest.raises(ValueError, match='1000 train'):
        release.validate_population(episodes[:-1], records[:-1], complete=True)


def test_structural_leakage_is_rejected_even_with_unique_task_ids():
    episodes, records = population()
    records[500]['key'] = records[0]['key']
    with pytest.raises(ValueError):
        release.validate_population(episodes, records, complete=True)


def test_duplicate_ids_and_mislabeled_splits_are_rejected():
    episodes, records = population()
    episodes[1]['id'] = episodes[0]['id']
    with pytest.raises(ValueError, match='duplicate'):
        release.validate_population(episodes, records, complete=True)
    episodes, records = population()
    records[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='metadata'):
        release.validate_population(episodes, records, complete=True)


def evidence():
    episode = {'source_task_id': 'task', 'success': True, 'policy_aware_success': True,
               'quality_audit_passed': True, 'messages': [{'role': 'assistant', 'content': 'Done'}]}
    candidate = {'accepted': True, 'required_actions_passed': True, 'episode': copy.deepcopy(episode),
                 'audit': {'passed': True, 'raw': {'writes': [], 'issues': [], 'script_events': []}}}
    loaded = {'gates': {'passed': True}, 'request': {'id': 'task'},
              'env': SimpleNamespace(make_tools=lambda: {})}
    return episode, candidate, {'passed': True}, loaded


def test_selected_transcript_must_match_original_evidence():
    args = evidence()
    release.validate_evidence(*args)
    args[0]['messages'][0]['content'] = 'Edited after admission'
    with pytest.raises(ValueError, match='differs from source'):
        release.validate_evidence(*args)


def test_saved_success_cannot_bypass_missing_script_evidence():
    args = evidence()
    args[-1]['request']['user_script'] = {'inquisitive': ['What is the price?']}
    with pytest.raises(ValueError, match='audit evidence'):
        release.validate_evidence(*args)


def test_stopped_generation_cannot_start_training(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/bench'))
    import continue_diverse_sft as continuation
    (tmp_path / 'progress.json').write_text(json.dumps({'phase': 'budget_exhausted'}))
    monkeypatch.setattr(continuation.subprocess, 'run', lambda *a, **k: pytest.fail('no process launch'))
    with pytest.raises(RuntimeError, match='without a complete dataset'):
        continuation.producer_state(tmp_path, 1)


def test_release_retains_scopes_for_fully_populated_baseline_groups():
    train={'name':'example_train','data_scope':{'owners':['t']}}
    dev={'name':'example_dev','data_scope':{'owners':['d']}}
    plan={'domains':[train,dev],'jobs':[{'domain':train}]}
    assert release.plan_groups(plan)['example_dev']==dev
    assert release.plan_groups({'jobs':[{'domain':train}]})=={'example_train':train}
