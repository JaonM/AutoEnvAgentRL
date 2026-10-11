import copy
import hashlib
import json
from unittest.mock import patch

from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.examples import direct
from env_factory_v2.expressions import canonical
from env_factory_v2.repair_evidence import compact_repair_feedback


def record(passed):
    run = {'outcome': {'success': passed, 'reward': int(passed)}, 'error': None,
           'answer': {'value': 3}, 'trace': [{'detail': 'actual evidence ' * 200}],
           'final_state': {'rows': list(range(100))}, 'time': {'tick': 1}}
    return {'case': {'id': 'case', 'steps_json': '[]'}, 'passed': passed,
            'first': copy.deepcopy(run), 'repeat': copy.deepcopy(run)}


def test_failed_evidence_and_original_source_are_never_compacted():
    feedback = {'original_request': {'data': record(True)}, 'previous_source': {'example': record(True)},
                'records': [record(True), record(False)]}
    before = copy.deepcopy(feedback)
    compact, receipt = compact_repair_feedback(feedback)
    assert feedback == before
    assert compact['original_request'] == feedback['original_request']
    assert compact['previous_source'] == feedback['previous_source']
    assert compact['records'][1] == feedback['records'][1]
    successful = compact['records'][0]
    assert successful['case'] == feedback['records'][0]['case']
    assert successful['first']['answer'] == {'value': 3}
    assert 'trace' not in successful['first'] and 'final_state' not in successful['first']
    assert successful['first']['archived_execution_identity'] == hashlib.sha256(
        canonical(feedback['records'][0]['first']).encode()).hexdigest()
    assert receipt['compacted_passing_records'] == 1 and receipt['compact_bytes'] < receipt['original_bytes']


def test_unknown_or_already_compact_feedback_is_preserved():
    for value in [None, {'error': 'failed compile'}, {'passed': True, 'first': {'outcome': {}}},
                  {'literal': record(True)}]:
        compact, receipt = compact_repair_feedback(value)
        assert compact == value and receipt['compacted_passing_records'] == 0


def test_generator_archives_full_evidence_before_sending_compact_feedback(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a'*24, evidence_root=tmp_path)
    feedback = {'previous_source': direct(), 'records': [record(True), record(False)]}
    with patch('env_factory_v2.author_session.generate_in_session',
               return_value=(direct(), {'source_identity': 'fixture'})) as run:
        agent.generate({'goal': 'unchanged request'}, feedback=feedback, author_tools=True)
    folders = list((tmp_path/'repair-feedback').iterdir())
    assert len(folders) == 1
    original = json.loads((folders[0]/'original.json').read_text())
    compact = json.loads((folders[0]/'compact.json').read_text())
    assert original == feedback
    assert run.call_args.kwargs['feedback']['records'] == compact['records']
    assert run.call_args.kwargs['previous_source'] == direct()
    assert run.call_args.kwargs['feedback']['records'][1] == feedback['records'][1]
