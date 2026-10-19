import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from prepare_sop_experiment import adapt, assignment


def test_replanning_preserves_baseline_but_replaces_arbitrary_combinations(tmp_path):
    schema={p:{'approval':'approve_'+p} for p in ('renew','return','hold')}
    methods={'return_then_hold':{'intent':'Return a copy then place an eligible hold.',
        'processes':['return','hold'],'statuses':{'0':['completed'],'1':['completed']},'edges':[]}}
    (tmp_path/'request.json').write_text(json.dumps({'goal_schema':schema,'sop_blueprint':{'methods':methods}}))
    domain={'name':'library_train','bundle':str(tmp_path)}
    original={'index':10,'slot':10,'processes':['renew','hold'],'requirements':{},'quota_group':'multi_write','script':[]}
    plan={'domains':[domain],'initial_counts':{'library_train':14},'jobs':[
        {'domain':domain,'assignment':copy.deepcopy(original),'alternatives':[copy.deepcopy(original)]} for _ in range(6)]}
    result=adapt(plan)
    assert result['initial_counts']=={'library_train':14}
    assert len({j['assignment']['index'] for j in result['jobs']})==6
    for job in result['jobs']:
        for a in [job['assignment'],*job['alternatives']]:
            assert a['processes']==['return','hold']
            assert a['requirements']['sop_method']=='return_then_hold'
            assert a['requirements']['min_writes']==2
    assert result['sop_required_for_new_tasks']


def test_refusal_branch_must_be_the_declared_business_reason():
    schema={p:{'approval':'yes'} for p in ('delivered_exchange_request','delivered_return_request')}
    method={'intent':'Unavailable exchange then return','processes':list(schema),
            'statuses':{'0':['refused'],'1':['requested']}}
    a=assignment({},'unavailable_exchange_then_return',method,schema)
    assert a['requirements']['min_writes']==1
    assert {'path':'observed.reason','equals':'unavailable'} in a['coverage_predicates']
    assert a['coverage_obstacle']=='fallbacks'


def test_frozen_runtime_is_independent_and_has_pinned_code(tmp_path,monkeypatch):
    module_path=Path(__file__).resolve().parents[2]/'experiments/bench/advance_sop_sft.py'
    spec=importlib.util.spec_from_file_location('advance_sop_test',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';(source/'src').mkdir(parents=True);(source/'experiments').mkdir()
    (source/'src/main.py').write_text('version=1\n');(source/'experiments/run.sh').write_text('true\n')
    (source/'output').mkdir();(source/'output/record').write_text('preserved')
    monkeypatch.setattr(module,'ROOT',source)
    frozen=tmp_path/'frozen';identity=module.freeze(frozen)
    (source/'src/main.py').write_text('version=2\n')
    assert (frozen/'src/main.py').read_text()=='version=1\n'
    assert (frozen/'output/record').read_text()=='preserved'
    assert all(module.bundle.sha256(path)==sha for path,sha in identity['code'].items())
