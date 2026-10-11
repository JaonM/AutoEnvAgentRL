from scripts.experiments.v2_legacy_generation import generate_legacy
from unittest.mock import Mock

import pytest

from env_factory_v2.native.json_syntax import recover_json
from env_factory_v2.contracts.expressions import ContractError


def test_short_repair_preserves_values_and_avoids_full_regeneration():
    agent = Mock()
    agent.ask.return_value = ({'replacements': [{'old': '18 "count"', 'new': '18,"count"'}]}, {'native': 'test'})
    value, evidence = recover_json('{"amount":18 "count":2}', agent)
    assert value == {'amount': 18, 'count': 2}
    assert evidence['provenance'] == {'native': 'test'}
    assert agent.ask.call_count == 1


@pytest.mark.parametrize('replacement', [
    {'old': '18 "count"', 'new': '19,"count"'},
    {'old': '18 "count"', 'new': '18,"other"'},
    {'old': 'missing', 'new': ','},
])
def test_syntax_repair_cannot_change_business_tokens_or_use_missing_anchor(replacement):
    agent = Mock()
    agent.ask.return_value = ({'replacements': [replacement]}, {})
    with pytest.raises(ContractError):
        recover_json('{"amount":18 "count":2}', agent)


def test_valid_json_never_calls_recovery_model():
    agent = Mock()
    assert recover_json('{"amount":18}', agent) == ({'amount': 18}, None)
    agent.ask.assert_not_called()


def test_native_generation_returns_recovered_source_with_both_provenances(tmp_path):
    from env_factory_v2.native.code_agent import NativeCodeAgent
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    agent.ask = Mock(side_effect=[({'source_json': '{"amount":18 "count":2}'}, {'generation': True}),
        ({'replacements': [{'old': '18 "count"', 'new': '18,"count"'}]}, {'syntax': True})])
    source, provenance = generate_legacy(agent, {'goal': 'Return the original amounts.'})
    assert source == {'amount': 18, 'count': 2}
    assert provenance['generation']
    assert provenance['format_repair']['provenance']['syntax']
    assert agent.ask.call_count == 2
