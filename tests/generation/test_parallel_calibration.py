import copy
import json
import threading
from unittest.mock import patch

import pytest

from env_factory.generation.execution_progress import ExecutionProgress
from env_factory.generation.semantic_reward import calibrate_semantic_outcomes, compile_semantic_outcome, SemanticCalibrationUnavailable
from env_factory.runtime_llm import RuntimeLLMError
from env_factory.sandbox_runtime import EpisodeStore, ValidationJudgmentCache
from .test_semantic_reward import fixture


def setup(tmp_path, monkeypatch):
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test')
    item, schema = fixture()
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    store = EpisodeStore(tmp_path / 'calibration.sqlite3')
    store.reset(episode_id='calibration', seed=17)
    return {'metrics': [metric]}, {metric['id']: cases}, cases, store


def test_parallel_cases_overlap_but_preserve_serial_verdicts_and_contexts(tmp_path, monkeypatch):
    contract, calibration, cases, store = setup(tmp_path, monkeypatch)
    for index, case in enumerate(cases):
        case['reference_id'] = str(index)
    contexts = {str(index): {'business_state': {'value': index}} for index in range(4)}
    originals = copy.deepcopy(contexts)
    barrier = threading.Barrier(2)
    active = peak = 0
    lock = threading.Lock()
    parallel = False

    def judge(messages, **kwargs):
        nonlocal active, peak
        evidence = json.loads(messages[1]['content'])['runtime_evidence']
        index = evidence['business_data']['value']
        assert json.loads(evidence['final_agent_response']) == cases[index]['answer']
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if parallel:
                barrier.wait(timeout=3)
            return {'label': cases[index]['expected']}
        finally:
            with lock:
                active -= 1

    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=judge) as model:
        serial = calibrate_semantic_outcomes(contract, calibration, context={}, store=store,
            reference_contexts=contexts, workers=1)
        store.reset(episode_id='parallel', seed=17)
        parallel = True
        concurrent = calibrate_semantic_outcomes(contract, calibration, context={}, store=store,
            reference_contexts=contexts, workers=2)
    assert serial == concurrent
    assert concurrent['passed'] and peak == 2 and model.call_count == 8
    assert contexts == originals
    calls = [e for e in store.replay()['events'] if e['event'] == 'evaluator_call']
    assert [e['payload']['calibration_case']['index'] for e in calls] == [0, 1, 2, 3]
    assert all(e['payload']['judgment_obtained'] and not e['payload']['cached'] for e in calls)


def test_parallel_calibration_never_coalesces_identical_fresh_judgments(tmp_path, monkeypatch):
    contract, calibration, cases, store = setup(tmp_path, monkeypatch)
    # Bypass source duplicate checks to directly exercise runtime fresh judgments
    # and conflict detection under simultaneous identical evidence.
    calibration['explanation'] = [cases[0], copy.deepcopy(cases[0])]
    barrier, lock = threading.Barrier(2), threading.Lock()
    count = 0

    def judge(*args, **kwargs):
        nonlocal count
        with lock:
            count += 1
            label = 'pass' if count == 1 else 'fail'
        barrier.wait(timeout=3)
        return {'label': label}

    progress = ExecutionProgress(tmp_path)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=judge):
        report = calibrate_semantic_outcomes(contract, calibration, context={}, store=store,
            workers=2, judgment_cache=ValidationJudgmentCache(), progress=progress)
    assert count == 2 and not report['passed'] and report['conflicting_judgments']
    feedback = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert any(e.get('code') == 'SEMANTIC_CALIBRATION_FAILED' for e in feedback)


def test_parallel_fallback_rejects_and_retains_each_case_evidence(tmp_path, monkeypatch):
    contract, calibration, cases, store = setup(tmp_path, monkeypatch)
    def judge(messages, **kwargs):
        answer = json.loads(messages[1]['content'])['runtime_evidence']['final_agent_response']
        if json.loads(answer) == cases[2]['answer']:
            raise RuntimeLLMError('offline')
        return {'label': next(c['expected'] for c in cases if c['answer'] == json.loads(answer))}

    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=judge):
        with pytest.raises(SemanticCalibrationUnavailable):
            calibrate_semantic_outcomes(contract, calibration, context={}, store=store, workers=2)
    calls = [e for e in store.replay()['events'] if e['event'] == 'evaluator_call']
    assert len(calls) == 4
    assert calls[2]['payload']['used_fallback']
    assert not calls[2]['payload']['judgment_obtained']


def test_failed_case_feedback_does_not_wait_for_slow_sibling(tmp_path, monkeypatch):
    contract, calibration, cases, store = setup(tmp_path, monkeypatch)
    calibration['explanation'] = cases[:2]
    feedback_seen = threading.Event()
    progress = ExecutionProgress(tmp_path)
    emit = progress.feedback

    def feedback(*args, **kwargs):
        emit(*args, **kwargs)
        feedback_seen.set()
    progress.feedback = feedback

    def judge(messages, **kwargs):
        answer = json.loads(json.loads(messages[1]['content'])['runtime_evidence']['final_agent_response'])
        if answer == cases[1]['answer']:
            assert feedback_seen.wait(timeout=3), 'fast defect was hidden behind slow case'
            return {'label': 'pass'}
        return {'label': 'fail'}

    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=judge):
        report = calibrate_semantic_outcomes(contract, calibration, context={}, store=store,
            workers=2, progress=progress)
    assert not report['passed']
    assert [row['passed'] for row in report['cases']] == [False, True]


@pytest.mark.parametrize('workers', [0, 3, True, 1.5])
def test_worker_limit_is_explicit(tmp_path, monkeypatch, workers):
    contract, calibration, _, store = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='workers must be 1 or 2'):
        calibrate_semantic_outcomes(contract, calibration, context={}, store=store, workers=workers)
