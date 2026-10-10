"""Freeze model-written natural user openings without author/scoring contracts."""
import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
from env_factory_v2.business_graph import install, retrieve
from env_factory_v2.code_agent import NativeCodeAgent, object_schema
from env_factory_v2.demand import layered_demand
from prepare_v2_natural_pilot import cases, write

SCENES = [
 ('办公室设备开销', '估算一天耗电与电费'), ('购物结算', '知道实付金额和节省金额'),
 ('会议室使用结束', '知道使用时长和费用'), ('筹备课程挑选讲师', '得到前三名候选与总报价'),
 ('送修后想安排取回', '知道维修进度和预计完成日期'), ('准备使用会员卡', '知道积分和可用状态'),
 ('准备买鞋', '知道合脚且有货的款号'), ('准备整理报销', '知道未报销票据编号与总额'),
 ('等待收货', '知道配送进度和预计到达时间'), ('设备故障准备送修', '知道是否保修以及检修费'),
 ('准备部门人员统计', '得到在职员工编号和人数'), ('临时无法参加健身课程', '取消当前预约'),
 ('追踪申请进度', '知道审批环节顺序、累计耗时和最终负责人'), ('核算项目报销', '统计已批准且超过4900元的全部报销笔数和总额'),
 ('盘点可调配库存', '统计至少可拿出6件的SKU数量和合计件数'), ('当前设备不合适', '选择兼容且有空余容量的设备替换，并调整维护安排'),
 ('目标仓库存不足', '从源仓调拨12件指定SKU并更新调拨单'), ('活动尚未开放但想参加', '开放后有名额时登记参加'),
 ('寻找合适活动场地', '提交预订前可能更改区域偏好，按最终确定的区域挑选并预订场地'),
 ('准备使用两个服务', '依次开启两个服务，如果第二个额度不足保留第一个')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--heldout', action='store_true', help='Use new synthetic business instances of existing task families')
    parser.add_argument('--heldout-variant', type=int, default=1)
    parser.add_argument('--business-relations', action='store_true', help='Declare and validate explicit object relationships in a new corpus')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = cases()
    if args.heldout:
        from v2_heldout_cases import heldout_cases
        entries = heldout_cases(args.heldout_variant)
    if args.business_relations:
        from v2_business_relations import with_business_relations
        entries = with_business_relations(entries)
    load_dotenv('.env')
    driver = GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password')))
    try:
        documents = [{'document_id': f'{i:02d}/business.json', 'title': c['title'], 'business': c['business']}
            for i, c in enumerate(entries, 1)]
        install(driver, args.namespace, documents, database=os.getenv('NEO4J_DATABASE', 'neo4j'))
        graph = retrieve(driver, args.namespace, database=os.getenv('NEO4J_DATABASE', 'neo4j'))
    finally:
        driver.close()
    write(args.output / 'knowledge_graph.json', graph)
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'utterance-native')
    manifest = {'version': 2, 'scope': 'native_natural_free_dialogue_baseline',
        'sampling': 'synthetic scene-designed; independent model opening, no production population claim',
        'mix': {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}, 'samples': []}
    if args.heldout:
        manifest.update(scope='native_natural_free_dialogue_heldout_instances', heldout_variant=args.heldout_variant,
            sampling='new synthetic business instances of known task families; independent model openings; not unseen domains or production population',
            preparation_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ['prepare_v2_free_baseline.py', 'v2_heldout_cases.py', 'prepare_v2_natural_pilot.py'] +
                    (['v2_business_relations.py'] if args.business_relations else [])})
    for index, (case, (situation, outcome)) in enumerate(zip(entries, SCENES, strict=True), 1):
        known = case['acceptance']['params'] if case['stratum'] == 'direct' else {}
        if case['acceptance']['kind'] == 'discount':
            known = {'item_prices': known['prices'], 'membership': '会员',
                     'member_order_discount': {'minimum_order_total': known['threshold'], 'rate': known['rate']},
                     'coupon': {'amount': known['coupon'], 'minimum_order_total': 0}}
        if case['acceptance']['kind'] == 'duration':
            p = known
            known = {'start_time': f"{p['start_minutes']//60:02d}:{p['start_minutes']%60:02d}",
                     'end_time': f"{p['end_minutes']//60:02d}:{p['end_minutes']%60:02d}", 'hourly_rate': p['hourly_rate']}
        # The utterance role receives neither acceptance specifications, business
        # policies, tool names, graph vocabulary nor reference execution steps.
        payload = {'situation': situation, 'desired_outcome': outcome, 'known_information': known,
            'communication': '自然、简洁；可有合理省略；不要主动编写平台执行步骤'}
        opening, provenance = agent.ask('Act as a real user in the described situation. Express your need '
            'naturally in Chinese. Preserve the desired outcome and stated numerical inputs. Do not add '
            'business rules, tool names, implementation steps or invented facts. Return the opening only. No tools.',
            payload, object_schema({'user_request': {'type': 'string'}}))
        check, check_provenance = agent.ask('Independently check this user opening against its scene. '
            'All desired outputs and stated input values must be preserved, with no invented facts, '
            'extra obligations or implementation vocabulary. Check naturalness: clock times must use ordinary '
            'hour/minute notation, not minutes since midnight; capacity must not be changed to performance. Reasonable clarification gaps are allowed. No tools.',
            {'scene': payload, 'opening': opening['user_request']},
            object_schema({'passed': {'type': 'boolean'}, 'reason': {'type': 'string'}}))
        if not check['passed']:
            write(args.output / 'preparation_failure.json', {'id': index, 'check': check})
            raise RuntimeError('opening fidelity review failed; assets not frozen')
        folder = args.output / f'{index:02d}'
        folder.mkdir()
        if args.heldout:
            from audit_v2_natural import canonical_truth
            truth = canonical_truth(case['acceptance'], case['business'], case['user'])
            writes = {'cancel', 'replace', 'transfer', 'enroll', 'correction', 'sequential'}
            case['acceptance']['answer_obligations'] = [] if case['acceptance']['kind'] in writes else [
                {'role': role, 'source': 'user_request', 'text': opening['user_request']}
                for role in truth['answer']]
        for name, value in [('business', case['business']), ('user', case['user']), ('acceptance', case['acceptance'])]:
            write(folder / (name + '.json'), value)
        write(folder / 'user_scene.json', payload)
        write(folder / 'opening.json', {'response': opening, 'provenance': provenance,
            'review': check, 'review_provenance': check_provenance})
        document = next(d for d in graph['documents'] if d['document_id'] == f'{index:02d}/business.json')
        demand = layered_demand(opening['user_request'], case['business']['policy'],
            document_id=document['document_id'], graph_evidence={'namespace': args.namespace,
                'document_sha256': document['sha256'], 'backend': 'neo4j'})
        manifest['samples'].append({'id': index, 'stratum': case['stratum'], 'title': case['title'],
            'category': 'direct_response' if case['stratum'] == 'direct' else 'simple_agentic' if case['stratum'] == 'simple' else 'multi_step_agentic',
            'raw_request': opening['user_request'], 'demand': demand,
            **{name: f'{index:02d}/{name}.json' for name in ('business', 'user', 'acceptance')}})
        write(args.output / 'manifest.json', manifest)
        print(json.dumps({'id': index, 'user_request': opening['user_request']}, ensure_ascii=False), flush=True)
    files = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*')) if p.is_file()}
    write(args.output / 'frozen.json', {'files': files})


if __name__ == '__main__':
    main()
