import pytest

from env_factory_v2.expressions import ContractError
from env_factory_v2.pipeline import InfrastructureError, require_qualified_delivery


def test_upstream_transport_failure_retains_type_phase_and_error():
    with pytest.raises(InfrastructureError, match='HTTP Error 502') as info:
        require_qualified_delivery({'qualified': False, 'status': 'infrastructure_failed',
            'attempts': [{'failed_phase': 'generation', 'error': 'HTTPError: HTTP Error 502: Bad Gateway'}]})
    assert info.value.upstream_phase == 'generation'


def test_contract_failure_remains_contract_failure():
    with pytest.raises(ContractError, match='did not qualify'):
        require_qualified_delivery({'qualified': False, 'attempts': [
            {'status': 'failed', 'failed_phase': 'compile', 'error': 'invalid schema'}]})


def test_qualified_delivery_passes_without_new_check_or_model_call():
    assert require_qualified_delivery({'qualified': True}) is None
