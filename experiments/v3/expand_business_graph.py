"""Grow the V1 scene graph toward business and service scenes, for V3 domain sampling.

  .venv/bin/python experiments/v3/expand_business_graph.py [--rounds 2] [--max-scene-nodes 1200]

Reads experiments/v3/business_seeds.txt and runs V1's SeedGraphExpander: a local Wikipedia dump
(WIKIPEDIA_DUMP_DB) supplies the search results, LLM_* (deepseek) extracts terms, merges synonyms
and extracts relations, and Neo4j (NEO4J_*) receives the result through V1's checkpointing.
Unlike examples/build_graph.py it never appends to data/scene_seeds.txt. Before expanding, it
backs up the whole graph (every node and relationship) to output/v3_graph_backup_<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory.graph.graph_builder import Neo4jGraphStore  # noqa: E402
from env_factory.graph.graph_expander import GraphExpansionConfig, SeedGraphExpander  # noqa: E402
from env_factory.graph.wikipedia_dump import LocalWikipediaClient  # noqa: E402
from env_factory.llm import LLMClient  # noqa: E402
from env_factory_v3.llm import load_env_file  # noqa: E402


def backup(store, path):
    with store.driver.session(database=store.database) as session:
        nodes = [{'id': r['id'], 'labels': r['labels'], 'props': r['props']} for r in session.run(
            'MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props')]
        rels = [{'start': r['s'], 'end': r['e'], 'type': r['t'], 'props': r['props']} for r in session.run(
            'MATCH (a)-[r]->(b) RETURN elementId(a) AS s, elementId(b) AS e, type(r) AS t, properties(r) AS props')]
    path.write_text(json.dumps({'nodes': nodes, 'relationships': rels}, ensure_ascii=False, default=str))
    return len(nodes), len(rels)


def counts(store):
    with store.driver.session(database=store.database) as session:
        return {'scenes': session.run('MATCH (n:Scene) RETURN count(n) AS c').single()['c'],
                'same_event': session.run('MATCH ()-[r:SAME_EVENT_ELEMENT]->() RETURN count(r) AS c').single()['c'],
                'hierarchy': session.run('MATCH ()-[r:HIERARCHY]->() RETURN count(r) AS c').single()['c']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--max-scene-nodes', type=int, default=1200)
    parser.add_argument('--max-search-requests', type=int, default=1500)
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    load_env_file(ROOT / '.env')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    seeds = tuple(line.strip() for line in (ROOT / 'experiments/v3/business_seeds.txt').read_text().splitlines()
                  if line.strip() and not line.startswith('#'))
    config = GraphExpansionConfig(max_scene_nodes=args.max_scene_nodes, max_search_requests=args.max_search_requests,
                                  max_rounds=args.rounds)
    llm = LLMClient.from_env('LLM', timeout=float(os.getenv('LLM_TIMEOUT', '60')))
    expander = SeedGraphExpander(LocalWikipediaClient(os.environ['WIKIPEDIA_DUMP_DB']), llm,
                                 max_workers=args.max_workers, config=config)
    with Neo4jGraphStore(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as store:
        store.verify_connectivity()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        nodes, rels = backup(store, ROOT / f'output/v3_graph_backup_{stamp}.json')
        before = counts(store)
        print(json.dumps({'backup': f'output/v3_graph_backup_{stamp}.json', 'nodes': nodes, 'relationships': rels,
                          'before': before, 'seeds': len(seeds)}, ensure_ascii=False), flush=True)
        builder, groups = expander.expand_and_build(store, seeds, rounds=args.rounds)
        after = counts(store)
    print(json.dumps({'expanded_groups': len(groups), 'edges': len(builder.edges()), 'before': before,
                      'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
