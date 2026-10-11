import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments'))
from env_factory_v2.contract import ContractError
from measure_v2_natural import verify_assets, summarize, digest, runner_identity
from compare_v2_measurements import compare, MATCH_FIELDS, load_round


def fixture(count=40):
    strata = ['direct']*(count//5)+['simple']*(count//5)+['ordinary']*(count//5)+['complex']*(count*2//5)
    samples = [{'id': i+1, 'stratum': group, 'raw_request': f'original task {i+1}',
        'status': 'passed', 'qualified': True, 'sandbox_qualified': True,
        'external_business_qualified': True, 'training_ready': True, 'policy_success': True,
        'seconds': 10, 'business': 'business.json', 'user': 'user.json', 'acceptance': 'acceptance.json'}
        for i,group in enumerate(strata)]
    protocol = {key:'fixed' for key in MATCH_FIELDS}
    protocol.update(planned=count, mix=dict(Counter(strata)), scope='unit-test-only', concurrency=1)
    result = summarize(protocol,samples,100)
    return protocol,result


def write_assets(root,count=40,omit=None):
    protocol,result=fixture(count)
    (root/'manifest.json').write_text(json.dumps({'samples':result['samples'],'mix':protocol['mix']}))
    for name in ['business.json','user.json','acceptance.json']:
        (root/name).write_text('{}')
    files={p.name:digest(p) for p in root.iterdir() if p.name!=omit}
    (root/'frozen.json').write_text(json.dumps({'files':files}))


@pytest.mark.parametrize('count',[20,40])
def test_scaled_batches_keep_ratio_and_all_failures_in_denominator(tmp_path,count):
    write_assets(tmp_path,count)
    assert len(verify_assets(tmp_path)['samples'])==count
    protocol,result=fixture(count)
    samples=result['samples']
    samples[0].update(status='timeout',qualified=False)
    samples[1].update(status='policy_validation_inconclusive',qualified=False)
    summary=summarize(protocol,samples,100)
    assert summary['planned']==count and summary['verified_end_to_end_yield']==(count-2)/count
    samples[2].update(status='pending',qualified=False)
    assert summarize(protocol,samples,100)['verified_end_to_end_yield'] is None


@pytest.mark.parametrize('omit',['manifest.json','business.json','user.json','acceptance.json'])
def test_core_asset_omission_cannot_bypass_freeze(tmp_path,omit):
    write_assets(tmp_path,omit=omit)
    with pytest.raises(ContractError,match='frozen'):
        verify_assets(tmp_path)


def test_runner_freeze_includes_transitive_mapping_implementations():
    identities=runner_identity()
    assert {'v2_mapping_validation.py','v2_record_semantics.py','v2_natural_semantics.py'} <= identities.keys()


@pytest.mark.parametrize('change',['assets_identity','concurrency','candidate_repairs','sample_timeout_seconds','author_tools'])
def test_different_workload_or_budget_is_not_a_speed_comparison(change):
    before,old=fixture();after,new=copy.deepcopy(before),copy.deepcopy(old)
    after[change]='changed'
    assert not compare(before,old,after,new)['comparable']


def test_partial_round_and_renamed_task_cannot_claim_speed():
    before,old=fixture();after,new=copy.deepcopy(before),copy.deepcopy(old)
    new['terminal']-=1
    assert not compare(before,old,after,new)['comparable']
    new=copy.deepcopy(old);new['samples'][0]['raw_request']='different task'
    assert not compare(before,old,after,new)['comparable']


def test_matched_speed_requires_no_yield_regression_in_any_stratum():
    protocol,old=fixture();new=copy.deepcopy(old);new['wall_seconds']=60
    report=compare(protocol,old,protocol,new)
    assert report['comparable'] and report['wall_reduction']==0.4
    assert report['speed_target_met_without_stratum_yield_regression']
    new['samples'][-1].update(status='timeout',qualified=False)
    report=compare(protocol,old,protocol,new)
    assert not report['speed_target_met_without_stratum_yield_regression']
    assert report['yield_change_percentage_points']==pytest.approx(-2.5)


def test_qualified_flag_without_real_stage_success_is_rejected():
    protocol,result=fixture();result['samples'][0]['training_ready']=False
    assert not compare(protocol,result,protocol,result)['comparable']


def test_comparison_checks_seal_before_reading_outcomes(tmp_path):
    protocol,result=fixture()
    for name,value in [('protocol.json',protocol),('results.json',result)]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'sealed_manifest.json').write_text(json.dumps({'files':{name:digest(tmp_path/name)
        for name in ['protocol.json','results.json']}}))
    assert load_round(tmp_path)==(protocol,result)
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(ContractError,match='changed'):
        load_round(tmp_path)
