import hashlib,json,sys,types
from pathlib import Path
from rl.environment import SandboxEpisode


def sandbox(root,value):
    root.mkdir()
    (root/'task.json').write_text('{}')
    (root/'helper.py').write_text(f'VALUE={value!r}\n')
    (root/'app.py').write_text('import helper\ndef create_app(db_path):\n    return {"value": helper.VALUE}\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    (root/'status.json').write_text(json.dumps({'training_ready':True,'artifact_hashes':hashes}))
    (root/'pipeline_result.json').write_text('{"training_ready":true}')


def test_switching_sandboxes_does_not_reuse_local_imports(tmp_path,monkeypatch):
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY','test-only')
    original=types.ModuleType('helper');original.VALUE='original'
    monkeypatch.setitem(sys.modules,'helper',original)
    before=list(sys.path)
    for value in ['A','B']:
        root=tmp_path/value;sandbox(root,value)
        episode=SandboxEpisode(root)
        assert episode.app['value']==value
        episode.close();episode.close()
        assert sys.modules['helper'] is original
        assert sys.path==before


def test_application_value_error_is_not_policy_protocol_error():
    from rl.environment import SandboxEpisode
    import pytest
    episode=SandboxEpisode.__new__(SandboxEpisode)
    episode.messages=[];episode.names={'tool'};episode.conversation=[]
    episode.reward=None
    def broken(*args,**kwargs):
        raise ValueError('sandbox implementation failed')
    episode.request=broken
    with pytest.raises(ValueError,match='sandbox implementation'):
        episode.step('{"kind":"tool","name":"tool","arguments":{}}')
