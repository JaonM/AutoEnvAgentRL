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
