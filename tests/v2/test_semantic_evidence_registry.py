import copy
import pytest

from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.native.evidence import snapshot, restore, restore_bundle
from env_factory_v2.delivery.export import export_bundle
from env_factory_v2.validation.verification import qualify
from env_factory_v2.examples import direct
from test_independent_evidence_reuse import native_judge, explained


def payload_for(agent):
    _, report = qualify(explained(), judge=agent.judge)
    return copy.deepcopy(agent.ask.call_args.args[1])


def test_exact_score_survives_stage_boundary_but_changed_evidence_calls_model(tmp_path):
    first = native_judge(tmp_path / 'first')
    payload = payload_for(first)
    second = native_judge(tmp_path / 'second')
    restore(second, snapshot(first))
    assert second.judge(payload)['provenance']['reused']
    second.ask.assert_not_called()
    changed = copy.deepcopy(payload)
    changed['task'] = {**changed['task'], 'additional_requirement': 'changed'}
    second.judge(changed)
    assert second.ask.call_count == 1


def test_explicit_independent_judgment_remains_fresh_after_restore(tmp_path):
    first = native_judge(tmp_path / 'first')
    payload = payload_for(first)
    second = native_judge(tmp_path / 'second')
    restore(second, snapshot(first))
    second.judge(payload, fresh=True)
    assert second.ask.call_count == 1


def test_platform_change_rejects_registry_without_mutating_agent(tmp_path):
    agent = native_judge(tmp_path)
    payload_for(agent)
    original = snapshot(agent)
    changed = copy.deepcopy(original)
    changed['implementation'] = 'old'
    with pytest.raises(ContractError, match='dependencies changed'):
        restore(agent, changed)
    assert snapshot(agent) == original


def test_conflicting_registry_poison_cannot_be_used(tmp_path):
    agent = native_judge(tmp_path)
    payload = payload_for(agent)
    other = snapshot(agent)
    for result in other['judgments'].values():
        result['criteria']['fact'] = not result['criteria']['fact']
    restore(agent, other)
    with pytest.raises(RuntimeError, match='poisoned'):
        agent.judge(payload)


def test_exported_registry_is_manifest_bound_and_tampering_rejected(tmp_path):
    agent = native_judge(tmp_path / 'native')
    payload_for(agent)
    compiled, report = qualify(direct())
    root = tmp_path / 'bundle'
    export_bundle(compiled, report, root, semantic_evidence=snapshot(agent))
    other = native_judge(tmp_path / 'other')
    assert restore_bundle(other, root)
    assert other.judgments == agent.judgments
    (root / 'semantic_evidence.json').write_text('{}')
    with pytest.raises(ContractError):
        restore_bundle(other, root)
