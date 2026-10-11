import copy

import pytest

from env_factory_v2.api_verification import verify_api
from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.examples import direct, lit
from env_factory_v2.reward import grade
from env_factory_v2.runtime import Sandbox
from env_factory_v2.verification import qualify, run_scenario


def timed():
    source = direct()
    source['success'] = [{'id': 'observed', 'public': '读取当前虚拟时间。',
        'when': lit(True), 'required_tools': [], 'required_stages': [],
        'required_platform_observations': [{'capability': 'time', 'freshness': 'current'}]}]
    for case in source['scenarios']:
        case['steps'] = [{'kind': 'time'}]
    return source


def test_passive_metadata_cannot_satisfy_explicit_observation():
    world = Sandbox(timed())
    before = world.time()
    world.time()
    assert not grade(world, {'temperature': 18})['success']
    world = Sandbox(timed())
    assert world.observe_time() == before
    assert world.time() == before and world.business_attempts == 0
    assert grade(world, {'temperature': 18})['success']


def test_stale_clock_requires_new_read_only_when_freshness_is_current():
    world = Sandbox(timed())
    world.observe_time()
    world.wait(1)
    assert not grade(world, {'temperature': 18})['success']
    source = timed()
    source['success'][0]['required_platform_observations'][0]['freshness'] = 'any'
    world = Sandbox(source)
    world.observe_time()
    world.wait(1)
    assert grade(world, {'temperature': 18})['success']


def test_clock_reference_and_http_api_share_real_evidence():
    compiled, report = qualify(timed())
    assert report['gates']['joint_execution_and_counterexamples']
    assert report['gates']['deterministic_reset_and_replay']
    assert verify_api(compiled['source'], report)['passed']
    missing = copy.deepcopy(compiled['source']['scenarios'][0])
    missing['steps'] = []
    assert not run_scenario(compiled, missing)['outcome']['success']


def test_unknown_platform_capabilities_are_not_silently_accepted():
    source = timed()
    source['success'][0]['required_platform_observations'][0]['capability'] = 'imaginary'
    with pytest.raises(ContractError, match='capability'):
        compile_contract(source)
