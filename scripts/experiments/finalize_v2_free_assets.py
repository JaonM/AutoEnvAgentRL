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
    args = parser.parse_args()
    assert not args.output.exists()
    frozen = json.loads((args.draft / 'frozen.json').read_text())
    assert all(hashlib.sha256((args.draft / p).read_bytes()).hexdigest() == h for p,h in frozen['files'].items())
    shutil.copytree(args.draft, args.output)
    (args.output / 'frozen.json').unlink()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'final-review-native')
    for identity in [3,16]:
        row = next(r for r in manifest['samples'] if r['id'] == identity)
        scene_path = args.output / f'{identity:02d}/user_scene.json'
        scene = json.loads(scene_path.read_text())
        if identity == 3:
            scene['known_information'] = {'start_time':'09:20','end_time':'12:05','hourly_rate':24}
        else:
            scene['desired_outcome'] = '选择兼容且有空余容量的设备替换，并调整维护安排'
        value, provenance = agent.ask('Write a natural Chinese user opening from this scene, preserving all inputs and desired outputs. '
            'Use ordinary clock times; capacity refers to available device capacity, not performance. '
            'No tools, implementation steps, invented conditions or computed answers.', scene,
            object_schema({'user_request':{'type':'string'}}))
        original = copy.deepcopy(row)
        row['raw_request'] = value['user_request']
        row['demand'] = layered_demand(value['user_request'], row['demand']['agent_context']['business_policy'],
            document_id=row['business'], graph_evidence=row['demand']['task_spec']['knowledge_graph'])
        write(scene_path, scene)
        write(args.output / f'{identity:02d}/final_opening.json', {'previous':original,'response':value,'provenance':provenance})
    scenes = [{'id':r['id'],'scene':json.loads((args.output/f"{r['id']:02d}/user_scene.json").read_text()),
               'user_request':r['raw_request']} for r in manifest['samples']]
    review, provenance = agent.ask('Independently review all 20 user openings for naturalness and fidelity to the supplied scenes. '
        'Preserve desired outputs and known inputs. No unsupported additions, technical artifacts, computed answers or '
        'meaning changes (available capacity is not performance). Reasonable information gaps are allowed. No tools.',
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
