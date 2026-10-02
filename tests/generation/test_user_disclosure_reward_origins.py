import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.reward_bindings import reject_private_identifier_constants
from .test_interaction_contract import interactive_source


def disclosed_key_source():
    source, request = interactive_source()
    source['description']['task'] = '查询用户指定库区的温度，只回答温度数字。'
    source['tables'][0]['primary_key'] = ['name']
    return source, request


def test_validated_user_disclosed_key_is_usable_in_reward(tmp_path):
    source, request = disclosed_key_source()
    result = compile_source(source, root=tmp_path, request=request)
    assert result['task_readiness']['ready']
    assert all(result['generation_pipeline']['verification']['checks'].values())


def test_unvalidated_disclosure_does_not_whitelist_private_key():
    source, _ = disclosed_key_source()
    with pytest.raises(ValueError, match='PRIVATE_REWARD_IDENTIFIER'):
        reject_private_identifier_constants(source)


def test_disclosed_key_is_not_valid_for_an_unrelated_table(tmp_path):
    source, request = disclosed_key_source()
    compile_source(source, root=tmp_path, request=request)
    scripts = json.loads((tmp_path / 'data/user_simulation/user_scripts.json').read_text())
    unrelated = copy.deepcopy(source['tables'][0])
    unrelated['table_name'] = 'private_other'
    source['tables'].append(unrelated)
    source['metric_implementations'][0]['expected']['targets'][0]['expression']['lookup']['table'] = 'private_other'
    with pytest.raises(ValueError, match='PRIVATE_REWARD_IDENTIFIER: private_other.name'):
        reject_private_identifier_constants(source, validated_scripts=scripts)


def test_invalid_disclosure_binding_is_rejected_before_origin_exemption(tmp_path):
    source, request = disclosed_key_source()
    source['interaction_contract']['variants'][0]['stages'][0]['bind_to']['argument'] = 'region'
    with pytest.raises(ValueError, match='does not match executable reference argument'):
        compile_source(source, root=tmp_path, request=request)
