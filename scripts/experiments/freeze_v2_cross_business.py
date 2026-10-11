"""Freeze reviewed cross-business assets after scoped Neo4j readback verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from env_factory_v2.business_graph import install, retrieve
from env_factory_v2.demand import layered_demand, validate_demand
from env_factory_v2.expressions import canonical
from prepare_v2_cross_business import cases
from env_factory_v2.code_agent import NativeCodeAgent
from v2_answer_obligations import review_answer_obligations
from audit_v2_natural import canonical_truth


def reviewed_cases(openings_root, opening_review_root, business_review_root):
    def read(root, name): return json.loads((root / name).read_text())
    generated = read(openings_root, 'generated_openings.json')
    opening_review = read(opening_review_root, 'independent_review.json')
    business_review = read(business_review_root, 'business_review.json')
    for result, collection, flags in [
        (opening_review, 'reviews', ['natural', 'faithful', 'no_invented_facts_or_steps']),
        (business_review, 'reviews', ['genuine_rule_or_composition_novelty', 'public_demand_consistent', 'public_policy_sufficient'])]:
        rows = result['response'][collection]
        if sorted(row['id'] for row in rows) != list(range(1, 21)) or not all(
                all(row[flag] is True for flag in flags) for row in rows):
            raise ValueError('All twenty independent preparation reviews must pass')
    openings = {row['id']: row['user_request'] for row in generated['response']['openings']}
    if sorted(openings) != list(range(1, 21)):
        raise ValueError('Twenty native openings are required')
    if read(opening_review_root, 'review_input.json')['openings'] != generated['response']['openings']:
        raise ValueError('Opening review refers to different generation evidence')
    evidence = {row['id']: row for row in read(business_review_root, 'review_input.json')['new_cases']}
    rows = cases()
    for row in rows:
        audited = evidence[row['id']]
        if (audited['user_request'] != openings[row['id']] or audited['public_policy'] != row['business']['policy']
                or audited['business_tables'] != row['business']['tables']):
            raise ValueError('Current public business assets differ from independent review')
        row['raw_request'] = openings[row['id']]
    return rows, {'native_openings': generated, 'opening_review': opening_review, 'business_review': business_review}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('openings', 'opening-review', 'business-review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--connection', type=Path, required=True,
                        help='Native Code Agent connection for pre-generation answer scope review')
    args = parser.parse_args()
    rows, evidence = reviewed_cases(args.openings, args.opening_review, args.business_review)
    args.output.mkdir(parents=True, exist_ok=False)
    def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'answer-scope-native')
    answer_scopes = {}
    for row in rows:
        scope = review_answer_obligations(agent, user_request=row['raw_request'], business=row['business'],
            truth=canonical_truth(row['acceptance'], row['business'], row['user']))
        answer_scopes[str(row['id'])] = scope
        save(args.output / 'answer_scope_reviews.json', answer_scopes)
        if not scope['passed']:
            raise ValueError('Independent answer scope unresolved; assets not frozen')
        row['acceptance']['answer_obligations'] = scope['answer_obligations']
    documents = [dict(document_id=f"{r['id']:02d}/business.json", title=r['title'], business=r['business']) for r in rows]
    load_dotenv('.env')
    with GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))) as driver:
        install(driver, args.namespace, documents, database=os.getenv('NEO4J_DATABASE', 'neo4j'))
        graph = retrieve(driver, args.namespace, database=os.getenv('NEO4J_DATABASE', 'neo4j'))
    indexed = {r['document_id']: r for r in graph['documents']}
    for document in documents:
        actual = indexed[document['document_id']]
        if (actual['sha256'] != hashlib.sha256(canonical(document['business']).encode()).hexdigest()
                or actual['policy_summary'] != document['business']['policy']
                or sorted(actual['relations'], key=lambda r: r['id']) != sorted(document['business']['relations'], key=lambda r: r['id'])):
            raise ValueError('Scoped graph readback does not match reviewed public business assets')
    manifest = {'version': 1, 'scope': 'native_natural_free_dialogue_cross_business_heldout',
        'sampling': 'twenty synthetic new business-rule/composition cases; native openings independently reviewed; not production-user sampling',
        'mix': {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}, 'samples': []}
    for row in rows:
        folder = args.output / f"{row['id']:02d}"
        folder.mkdir()
        for name in ('business', 'user', 'acceptance'): save(folder / (name + '.json'), row[name])
        identity = f"{row['id']:02d}/business.json"
        demand = layered_demand(row['raw_request'], row['business']['policy'], document_id=identity,
            graph_evidence={'backend': 'neo4j', 'namespace': args.namespace, 'document_sha256': indexed[identity]['sha256']})
        validate_demand(demand, row['raw_request'])
        manifest['samples'].append({k: row[k] for k in ('id', 'stratum', 'title', 'business_family', 'raw_request')} |
            {'category': 'direct_response' if row['stratum'] == 'direct' else 'simple_agentic' if row['stratum'] == 'simple' else 'multi_step_agentic',
             'demand': demand, **{name: f"{row['id']:02d}/{name}.json" for name in ('business', 'user', 'acceptance')}})
    save(args.output / 'manifest.json', manifest)
    save(args.output / 'knowledge_graph.json', graph)
    save(args.output / 'preparation_review.json', evidence)
    save(args.output / 'frozen.json', {'files': {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in sorted(args.output.rglob('*.json'))}})
    print('Frozen twenty reviewed cross-business cases with verified Neo4j evidence; not measured yield', flush=True)


if __name__ == '__main__': main()
