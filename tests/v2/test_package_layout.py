"""Exported packages must work without imports from the checkout."""
import json
import os
import subprocess
import sys

import pytest

from env_factory_v2.package import package_files, package_fingerprints
from env_factory_v2.examples import simple
from env_factory_v2.validation.verification import qualify
from env_factory_v2.validation.gates import DELIVERY_GATES, delivery_gates_pass
from env_factory_v2.delivery.export import export_bundle, verify_bundle


def test_recursive_inventory_detects_nested_changes_and_same_basename(tmp_path):
    for name in ('one/module.py', 'two/module.py', 'one/__pycache__/skip.py', 'notes.txt'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('before')
    before = package_fingerprints(root=tmp_path)
    assert set(before) == {'one/module.py', 'two/module.py'}
    (tmp_path / 'one/module.py').write_text('after')
    after = package_fingerprints(root=tmp_path)
    assert before['one/module.py'] != after['one/module.py']
    assert before['two/module.py'] == after['two/module.py']


def test_exported_cli_runs_without_checkout_and_preserves_inventory(tmp_path):
    root = tmp_path / 'bundle'
    compiled, report = qualify(simple())
    export_bundle(compiled, report, root)
    assert set(package_files(root=root / 'env_factory_v2')) == set(package_files())
    manifest = verify_bundle(root)
    result = subprocess.run([sys.executable, '-S', '-B', '-m', 'env_factory_v2',
                             'demo', '--example', 'simple', '--output', str(tmp_path / 'demo')],
                            cwd=root, env={**os.environ, 'PYTHONPATH': ''}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['executable_checks_passed'] is True
    assert verify_bundle(root) == manifest


def test_new_reports_have_only_effective_gates():
    _, report = qualify(simple())
    assert 'structural_contract' not in report['gates']
    assert set(report['gates']) == DELIVERY_GATES - {'container_execution'}


@pytest.mark.parametrize('legacy', [False, True])
def test_delivery_gate_reader_keeps_required_checks_and_reads_historical_flag(legacy):
    gates = {name: True for name in DELIVERY_GATES}
    if legacy:
        gates['structural_contract'] = True
    assert delivery_gates_pass(gates)
    for name in gates:
        assert not delivery_gates_pass({**gates, name: False})
    for name in DELIVERY_GATES:
        assert not delivery_gates_pass({k: v for k, v in gates.items() if k != name})
    assert not delivery_gates_pass({**gates, 'unknown_gate': True})
