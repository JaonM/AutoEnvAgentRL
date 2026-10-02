import json
from rl.retention import prune_checkpoints, prune_snapshots_after_shutdown


def test_retention_protects_pointer_even_if_newer_uncommitted_exists(tmp_path):
    for step in range(6):
        p=tmp_path/f'step-{step}';p.mkdir()
        (p/'state.json').write_text(json.dumps({'optimizer_step':step}))
    (tmp_path/'latest.json').write_text(json.dumps({'directory':'step-2'}))
    assert set(prune_checkpoints(tmp_path,keep=2))=={'step-0','step-1','step-3','step-4'}
    assert (tmp_path/'step-2').exists() and (tmp_path/'step-5').exists()


def test_snapshot_retention_keeps_baseline_and_recent_versions(tmp_path):
    for step in range(6):
        (tmp_path/f'policy-{step:06d}.safetensors').write_bytes(b'weights')
    prune_snapshots_after_shutdown(tmp_path,current=5,keep=2)
    assert {p.name for p in tmp_path.iterdir()}=={'policy-000000.safetensors','policy-000004.safetensors','policy-000005.safetensors'}


def test_snapshot_retention_preserves_committed_version_after_uncommitted_publish(tmp_path):
    for step in range(4):
        (tmp_path/f'policy-{step:06d}.safetensors').write_bytes(b'weights')
    # Version 3 was published, but checkpoint commit failed at version 2.
    prune_snapshots_after_shutdown(tmp_path, current=2, keep=1)
    assert {p.name for p in tmp_path.iterdir()} == {
        'policy-000000.safetensors', 'policy-000002.safetensors', 'policy-000003.safetensors'}


def test_snapshot_retention_keeps_version_pinned_by_unfinished_rollout(tmp_path):
    for step in range(5):
        (tmp_path/f'policy-{step:06d}.safetensors').write_bytes(b'weights')
    prune_snapshots_after_shutdown(tmp_path, current=4, keep=1, protected_versions=[1])
    assert {p.name for p in tmp_path.iterdir()} == {
        'policy-000000.safetensors', 'policy-000001.safetensors', 'policy-000004.safetensors'}
