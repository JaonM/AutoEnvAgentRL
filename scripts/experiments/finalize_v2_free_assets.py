"""Derive an independently reviewed final freeze from untouched preparation evidence."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from env_factory_v2.code_agent import NativeCodeAgent, object_schema
from env_factory_v2.demand import layered_demand
from prepare_v2_natural_pilot import write
from measure_v2_natural import verify_assets


def validate_review(review, identities):
    checks = review['checks']
    if (len(checks) != len(identities)
            or any(type(check['id']) is not int for check in checks)
            or {check['id'] for check in checks} != identities):
        raise ValueError('Opening review must cover every sample exactly once')
    if not all(check['passed'] is True for check in checks):
        raise ValueError('final naturalness review failed; no freeze')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--repair-ids', default='', help='Explicit comma-separated sample IDs to regenerate from their unchanged frozen scenes')
    parser.add_argument('--business-fidelity', action='store_true', help='Also audit openings against public business policy')
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = verify_assets(args.draft)
    repair_ids = [int(value) for value in args.repair_ids.split(',') if value]
    identities = {row['id'] for row in manifest['samples']}
    if len(set(repair_ids)) != len(repair_ids) or not set(repair_ids) <= identities:
        raise ValueError('Repair IDs must be unique members of the frozen sample set')
    frozen = json.loads((args.draft / 'frozen.json').read_text())['files']
    scene_paths = {f"{row['id']:02d}/user_scene.json" for row in manifest['samples']}
    if not scene_paths <= set(frozen):
        raise ValueError('Every user scene must be frozen before opening review')
    shutil.copytree(args.draft, args.output)
    (args.output / 'frozen.json').unlink()
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'final-review-native')
    for identity in repair_ids:
        row = next(r for r in manifest['samples'] if r['id'] == identity)
        scene_path = args.output / f'{identity:02d}/user_scene.json'
        scene = json.loads(scene_path.read_text())
        value, provenance = agent.ask('Write a natural Chinese user opening from this scene, preserving all inputs and desired outputs. '
            'Preserve the meaning and units of all supplied facts. '
            'No tools, implementation steps, invented conditions or computed answers.', scene,
            object_schema({'user_request':{'type':'string'}}))
        original = copy.deepcopy(row)
        row['raw_request'] = value['user_request']
        row['demand'] = layered_demand(value['user_request'], row['demand']['agent_context']['business_policy'],
            document_id=row['business'], graph_evidence=row['demand']['task_spec']['knowledge_graph'])
        # Re-audit all final openings below; replacing a citation's text cannot
        # establish that a changed utterance preserves the same answer scope.
        write(args.output / f'{identity:02d}/final_opening.json', {'previous':original,'response':value,'provenance':provenance})
    scenes = [{'id':r['id'],'scene':json.loads((args.output/f"{r['id']:02d}/user_scene.json").read_text()),
               'user_request':r['raw_request']} for r in manifest['samples']]
    if args.business_fidelity:
        for sample, row in zip(scenes, manifest['samples'], strict=True):
            sample['public_business_policy'] = row['demand']['agent_context']['business_policy']
    review, provenance = agent.ask('Independently review every supplied user opening for naturalness and fidelity to the supplied scenes. '
        'Preserve desired outputs and known inputs. No unsupported additions, technical artifacts, computed answers or '
        'meaning changes. When public_business_policy is supplied, check that stated conditions '
        'apply to the correct objects and do not contradict policy. Users may omit policy details '
        'and ask the agent to apply them. '
        'Reasonable information gaps are allowed. No tools.',
        {'samples':scenes}, object_schema({'checks':{'type':'array','items':object_schema({'id':{'type':'integer'},
            'passed':{'type':'boolean'},'reason':{'type':'string'}})}}))
    write(args.output / 'final_opening_review.json', {'response':review,'provenance':provenance})
    validate_review(review, identities)
    from audit_v2_natural import canonical_truth
    from v2_answer_obligations import review_answer_obligations
    for row in manifest['samples']:
        acceptance_path = args.output / row['acceptance']
        acceptance = json.loads(acceptance_path.read_text())
        business = json.loads((args.output / row['business']).read_text())
        user = json.loads((args.output / row['user']).read_text())
        scope = review_answer_obligations(agent, user_request=row['raw_request'], business=business,
                                         truth=canonical_truth(acceptance, business, user))
        write(acceptance_path.parent / 'answer_scope_review.json', scope)
        if scope['passed'] is not True:
            raise ValueError('answer obligation scope unresolved; no freeze')
        acceptance['answer_obligations'] = scope['answer_obligations']
        write(acceptance_path, acceptance)
    manifest['preparation_source_sha256'] = hashlib.sha256((args.draft/'frozen.json').read_bytes()).hexdigest()
    write(args.output / 'manifest.json', manifest)
    files = {str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob('*')) if p.is_file()}
    write(args.output / 'frozen.json', {'files':files})
    print(json.dumps({'frozen':True,'count':len(manifest['samples']),'output':str(args.output)}))


if __name__ == '__main__':
    main()
