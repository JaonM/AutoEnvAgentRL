import hashlib,json
import pytest
from rl.tasks import TaskSpec,load_tasks
from rl.dataset import DatasetSchedule


def qualified(root, task):
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(task))


@pytest.fixture(autouse=True)
def bundles(monkeypatch):
    """Stand-in sandbox bundles: a manifest.json whose content is the identity."""
    from rl.environments import sandbox
    monkeypatch.setattr(sandbox, 'is_sandbox_bundle', lambda root: (root/'manifest.json').is_file())
    monkeypatch.setattr(sandbox, 'bundle_identity',
                        lambda root: hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())


def test_manifest_checks_identity_and_separates_eval(tmp_path):
    qualified(tmp_path/'a',{'task':'A'})
    qualified(tmp_path/'b',{'task':'B'})
    path=tmp_path/'tasks.json'
    path.write_text(json.dumps({'tasks':[{'id':'a','sandbox':'a'}, {'id':'b','sandbox':'b','split':'eval'}]}))
    tasks=load_tasks(manifest=path)
    assert tasks[0].identity != tasks[1].identity
    assert all(job['task_id'] == 'a' for job in DatasetSchedule(tasks, epochs=20, batch_size=1, seed=42).jobs)
    path.write_text(json.dumps({'tasks':[{'id':'a','sandbox':'a'}, {'id':'b','sandbox':'a','split':'eval'}]}))
    with pytest.raises(ValueError,match='duplicate sandbox identity'):load_tasks(manifest=path)
