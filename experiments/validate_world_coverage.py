"""Independent public-policy regressions and measured branch coverage for two worlds."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from env_factory.certify import bundle
from env_factory.coverage import report
from env_factory.production.world import check_world
from recheck_world_tasks import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upgrade', type=Path, required=True)
    args=parser.parse_args(); root=args.upgrade
    source=Path('output/world_tau2_task_dialogue_v4/retail/world/bundle')
    loaded=bundle.load_bundle(source,require_current_platform=False)
    issues,notes=check_world(source)
    if issues: raise ValueError(issues)
    quality=bundle.qualify_loaded(loaded)
    save(root/'retail/qualification.json',quality)
    if not quality['passed']:raise ValueError('retail requalification failed')
    bundle.export(source,source/'request.json',quality,root/'retail/bundle',status='qualified')
    jobs=[('retail','reference_qa_probe.py','policy.json'),('retail','conformance_probe.py','api.json'),
          ('airline','replay_policy_flows.py','policy'),('airline','replay_conformance.py','api')]
    runs=[]
    for domain,script,target in jobs:
        output=root/domain/target
        run=subprocess.run([sys.executable,str(Path('output/world_tau2_aligned_v2')/domain/script),
                            '--bundle',str(root/domain/'bundle'),'--output',str(output)],
                           capture_output=True,text=True,env=os.environ.copy())
        save(root/domain/(script+'.log.json'),dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        if run.returncode:raise ValueError(f'{domain}/{script}: {run.stdout[-1000:]} {run.stderr[-1000:]}')
        runs.append({'domain':domain,'script':script,'passed':True})
    retail_cases=json.loads(Path('output/world_tau2_aligned_v2/retail/reference_qa_cases.json').read_text())['cases']
    retail_results={r['id']:r for r in json.loads((root/'retail/policy.json').read_text())['cases']}
    airline_cases=json.loads(Path('output/world_tau2_aligned_v2/airline/policy_flow_cases.json').read_text())
    airline_results={r['id']:r for r in json.loads((root/'airline/policy/policy_flow_results.json').read_text())['results']}
    for domain,cases,results in [('retail',retail_cases,retail_results),('airline',airline_cases,airline_results)]:
        records=[]; cells={}
        for case in cases:
            facts=case.get('facts',case.get('user_facts')); actual=results[case['id']]
            answer=actual.get('answer',actual.get('reference',{}).get('answer'))
            for i,goal in enumerate(facts['goals'],1):
                outcome=answer[f'goal_{i}']
                # Obligations are defined by the frozen independent policy cases,
                # never by a generated task's claimed difficulty/category label.
                independent=(case['outcomes'][i-1] if domain=='retail' else
                             dict(status=case['expected_status'],reason=case['expected_reason']) if i==1 else
                             dict(status='completed',reason=None))
                key=':'.join([goal['process'],independent['status'],str(independent.get('reason'))])
                cells[key]={'id':key,'predicates':[{'path':'process','equals':goal['process']},
                     {'path':'observed.status','equals':independent['status']},
                     {'path':'observed.reason','equals':independent.get('reason')}]}
                records.append(dict(id=f"{case['id']}:{i}",process=goal['process'],params=goal['params'],
                    observed=outcome,reference_passed=actual['passed'],natural_passed=False))
        if domain=='airline':
            extension=json.loads((root/domain/'coverage_contract.json').read_text())
            for cell in extension:cells['extension:'+cell['id']]={**cell,'id':'extension:'+cell['id']}
            records += [json.loads(p.read_text()) for p in sorted((root/domain/'probes').glob('*.json'))]
        save(root/domain/'branch_contract.json',list(cells.values()))
        save(root/domain/'branch_records.json',records)
        save(root/domain/'branch_coverage.json',report(list(cells.values()),records))
    save(root/'regression_summary.json',{'passed':True,'runs':runs,
         'limits':'Independent API and policy fixtures only; natural dialogue coverage is reported separately.'})
    print(json.dumps({'passed':True,'regressions':len(runs)}))


if __name__=='__main__':main()
