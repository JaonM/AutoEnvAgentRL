import json
import pytest
from rl import train as trainer
from rl.checkpoint import atomic_json


def test_failed_run_records_failure_without_overwriting_another_run(monkeypatch,tmp_path):
    config=trainer.Config(sandbox='task',output=str(tmp_path))
    def fail(config,run_id):
        atomic_json(tmp_path/'run_status.json',{'run_id':run_id,'state':'running'})
        raise FloatingPointError('numerical failure')
    monkeypatch.setattr(trainer,'_train',fail)
    with pytest.raises(FloatingPointError):trainer.train(config)
    status=json.loads((tmp_path/'run_status.json').read_text())
    assert status['state']=='failed' and status['error_type']=='FloatingPointError'
    def reject(config,run_id):
        raise ValueError('incompatible resume')
    monkeypatch.setattr(trainer,'_train',reject)
    with pytest.raises(ValueError):trainer.train(config)
    assert json.loads((tmp_path/'run_status.json').read_text())==status


@pytest.mark.parametrize('error,expected', [(RuntimeError('training failed'), 'failed'),
                                          (KeyboardInterrupt(), 'interrupted')])
def test_failure_logs_traceback_and_closes_tensorboard(monkeypatch, tmp_path, error, expected):
    from rl.telemetry import TrainingMonitor
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    config = trainer.Config(sandbox='task', output=str(tmp_path))
    monitors = []
    def fail(config, run_id):
        atomic_json(tmp_path / 'run_status.json', {'run_id': run_id, 'state': 'running'})
        monitor = config._monitor = TrainingMonitor(config, run_id)
        monitor.restore(0, 0)
        monitors.append(monitor)
        raise error
    monkeypatch.setattr(trainer, '_train', fail)
    with pytest.raises(type(error)):
        trainer.train(config)
    assert json.loads((tmp_path / 'run_status.json').read_text())['state'] == expected
    monitor = monitors[0]
    assert monitor.runtime.stream.closed and monitor.runtime.writer is None
    assert not hasattr(config, '_monitor')
    events = [json.loads(line) for line in next((tmp_path / 'logs/runtime').glob('*/events.jsonl')).read_text().splitlines()]
    assert events[-1]['state'] == expected
    assert type(error).__name__ in events[-1]['text']
    dashboard = EventAccumulator(str(tmp_path / 'tensorboard' / monitor.runtime.source)).Reload()
    assert dashboard.Tensors('run/error/text_summary')
