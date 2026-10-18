"""Join executed branch evidence and independently admitted natural demonstrations."""
import argparse
import json
from pathlib import Path

from env_factory.certify import bundle
from env_factory.coverage import report, require
from env_factory.task_contract import approval_problems
from recheck_world_tasks import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upgrade',type=Path,required=True)
    args=parser.parse_args();root=args.upgrade
    regression=json.loads((root/'regression_summary.json').read_text())
    if not regression['passed']:raise ValueError('regressions must pass')
    natural=json.loads((root/'natural_dialogues/summary.json').read_text())
    if natural['completed']!=natural['total']:raise ValueError('natural probes are still running')
    manifest=bundle.load_bundle(root/'airline/bundle')['manifest']
    records=[];evidence=[]
    for selection in natural['results']:
        name=selection['id'];folder=root/'natural_dialogues'/name
        if not selection['passed']:continue
        loaded=bundle.load_bundle(folder/'bundle')
        if any(loaded['manifest']['files'][n]!=manifest['files'][n] for n in ('env.py','verifier.py')):
            raise ValueError('natural probe used different world implementations')
        if approval_problems(loaded['request']):raise ValueError('unrelated approval facts in probe')
        if not json.loads((folder/'sufficiency.json').read_text())['passed']:raise ValueError('missing sufficiency')
        attempt=next(r['index'] for r in selection['attempts'] if r['passed'])
        path=folder/f'attempt-{attempt:02d}.json';row=json.loads(path.read_text())
        if not (row['passed'] and row['episode']['policy_aware_success'] and row['audit']['passed'] and row['required_tools_passed']):
            raise ValueError('selected dialogue not admitted')
        tools=[]
        for message in row['episode']['messages']:
            if message['role']=='tool':
                try:
                    if json.loads(message['content']).get('ok') is True:tools.append(message['name'])
                except (ValueError,AttributeError):pass
        goal=loaded['request']['user_facts']['goals'][0]
        records.append(dict(id=name,process=goal['process'],params=goal['params'],
            observed=row['episode']['answer']['goal_1'],tools=tools,reference_passed=True,natural_passed=True))
        evidence.append(dict(id=name,attempt=str(path),sha256=bundle.sha256(path),
                             bundle_manifest_sha256=bundle.sha256(folder/'bundle/manifest.json')))
    cells=json.loads((root/'airline/coverage_contract.json').read_text())
    ids={r['id'] for r in natural['results']}
    required=[c for c in cells if c['id'] in ids]
    if len(required)!=natural['total']:raise ValueError('natural case/contract mismatch')
    # A failed/missing natural probe stops admission. Reference-only passes do not
    # fill these obligations, and extra labels do not count as execution evidence.
    measured=report(required,records)
    save(root/'natural_coverage.json',measured)
    require(required,records)
    branch_cells=json.loads((root/'airline/branch_contract.json').read_text())
    branch_records=json.loads((root/'airline/branch_records.json').read_text())
    save(root/'airline/branch_coverage_with_natural.json',report(branch_cells,branch_records+records))
    save(root/'natural_evidence.json',evidence)
    counts={}
    for domain in ('retail','airline'):
        value=json.loads((root/domain/'branch_coverage.json').read_text())
        counts[domain]={'reference_covered':value['reference_covered'],'declared_obligations':value['total']}
    save(root/'summary.json',dict(worlds_qualified=True,independent_regressions_passed=True,
         new_probes=json.loads((root/'airline/probe_summary.json').read_text()),
         declared_branch_coverage=counts,natural_probes={'passed':natural['passed'],'total':natural['total']},
         training_ready=False,frozen_training_dataset_changed=False,
         limits=['Coverage is for declared cases, not proof of every policy combination or benchmark task.',
                 'Eight representative natural probes; remaining branch cells are reference/API evidence only.',
                 'These are acceptance fixtures, not released SFT samples or automatic world-authoring evidence.']))
    print(json.dumps({'qualified':True,'natural':measured['natural_covered'],'total':len(required)}))


if __name__=='__main__':main()
