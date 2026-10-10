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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--repair-ids', default='3,16', help='Comma-separated openings to regenerate before batch review')
    parser.add_argument('--business-fidelity', action='store_true', help='Also audit openings against public business policy')
    args = parser.parse_args()
    assert not args.output.exists()
    frozen = json.loads((args.draft / 'frozen.json').read_text())
    assert all(hashlib.sha256((args.draft / p).read_bytes()).hexdigest() == h for p,h in frozen['files'].items())
    shutil.copytree(args.draft, args.output)
    (args.output / 'frozen.json').unlink()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'final-review-native')
    for identity in [int(value) for value in args.repair_ids.split(',') if value]:
        row = next(r for r in manifest['samples'] if r['id'] == identity)
        scene_path = args.output / f'{identity:02d}/user_scene.json'
        scene = json.loads(scene_path.read_text())
        if identity == 3:
            spec = json.loads((args.output / row['acceptance']).read_text())['params']
            scene['known_information'] = {
                'start_time': f"{spec['start_minutes']//60:02d}:{spec['start_minutes']%60:02d}",
                'end_time': f"{spec['end_minutes']//60:02d}:{spec['end_minutes']%60:02d}",
                'hourly_rate': spec['hourly_rate']}
        elif identity == 2:
            spec = json.loads((args.output / row['acceptance']).read_text())['params']
            scene['known_information'] = {'item_prices': spec['prices'], 'membership': '我是会员',
                'member_order_discount': {'minimum_order_total': spec['threshold'], 'rate': spec['rate']},
                'coupon': {'value': spec['coupon'], 'minimum_order_total': 0}}
        elif identity == 16:
            scene['desired_outcome'] = '选择兼容且有空余容量的设备替换，并调整维护安排'
        elif identity == 19:
            scene['desired_outcome'] = '预订提交前可能更改区域偏好；按最终区域偏好挑选并预订场地'
        value, provenance = agent.ask('Write a natural Chinese user opening from this scene, preserving all inputs and desired outputs. '
            'Use ordinary clock times; capacity refers to available device capacity, not performance. '
            'No tools, implementation steps, invented conditions or computed answers.', scene,
            object_schema({'user_request':{'type':'string'}}))
        original = copy.deepcopy(row)
        row['raw_request'] = value['user_request']
        row['demand'] = layered_demand(value['user_request'], row['demand']['agent_context']['business_policy'],
            document_id=row['business'], graph_evidence=row['demand']['task_spec']['knowledge_graph'])
        write(scene_path, scene)
        acceptance_path = args.output / row['acceptance']
        acceptance = json.loads(acceptance_path.read_text())
        for obligation in acceptance.get('answer_obligations', []):
            if obligation['source'] == 'user_request':
                obligation['text'] = value['user_request']
        write(acceptance_path, acceptance)
        write(args.output / f'{identity:02d}/final_opening.json', {'previous':original,'response':value,'provenance':provenance})
    scenes = [{'id':r['id'],'scene':json.loads((args.output/f"{r['id']:02d}/user_scene.json").read_text()),
               'user_request':r['raw_request']} for r in manifest['samples']]
    if args.business_fidelity:
        for sample, row in zip(scenes, manifest['samples'], strict=True):
            sample['public_business_policy'] = row['demand']['agent_context']['business_policy']
    review, provenance = agent.ask('Independently review all 20 user openings for naturalness and fidelity to the supplied scenes. '
        'Preserve desired outputs and known inputs. No unsupported additions, technical artifacts, computed answers or '
        'meaning changes (available capacity is not performance). When public_business_policy is supplied, '
        'check that stated eligibility conditions attach to the correct discount or coupon and do not '
        'contradict policy. Users may omit policy details and ask the agent to apply them. '
        'Reasonable information gaps are allowed. No tools.',
        {'samples':scenes}, object_schema({'checks':{'type':'array','items':object_schema({'id':{'type':'integer'},
            'passed':{'type':'boolean'},'reason':{'type':'string'}})}}))
    write(args.output / 'final_opening_review.json', {'response':review,'provenance':provenance})
    assert len(review['checks']) == 20 and {c['id'] for c in review['checks']} == set(range(1,21))
    assert all(c['passed'] for c in review['checks']), 'final naturalness review failed; no freeze'
    manifest['preparation_source_sha256'] = hashlib.sha256((args.draft/'frozen.json').read_bytes()).hexdigest()
    write(args.output / 'manifest.json', manifest)
    files = {str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob('*')) if p.is_file()}
    write(args.output / 'frozen.json', {'files':files})
    print(json.dumps({'frozen':True,'count':20,'output':str(args.output)}))


if __name__ == '__main__':
    main()
