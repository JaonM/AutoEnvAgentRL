"""Domain sampling from the scene graph x business grid. No Neo4j or model calls."""
from __future__ import annotations

import json
import random
import unittest

from env_factory.production import domains

NODES = [('涤纶', '涤纶'), ('服装', '服装'), ('尺寸', '尺寸'), ('退换货', '退换货'),
         ('粤菜', '粤菜'), ('闽菜', '闽菜'), ('外卖', '外卖'), ('赤狐', '赤狐')]
EVENT = [('涤纶', '服装'), ('服装', '尺寸'), ('尺寸', '退换货'), ('粤菜', '闽菜'), ('闽菜', '外卖')]


class Proposer:
    """Accepts clothing paths, rejects the fox, and repeats one name to exercise deduplication."""

    model = 'scripted-proposer'

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **parameters):
        self.calls += 1
        prompt = messages[-1]['content']
        if '赤狐' in prompt:
            return {'content': json.dumps({'accept': False, 'reason': '没有合理的客服业务'})}
        name = '服装电商退换货' if self.calls % 2 else f'粤菜外卖订餐{self.calls}'
        return {'content': json.dumps({'accept': True, 'name': name, 'detail': f'业务细节{self.calls}' * 3})}


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.graph = domains.SceneGraph(NODES, EVENT, [])

    def test_components_stay_in_one_split_and_walks_stay_inside(self):
        clothing = {self.graph.split_of(n) for n in ('涤纶', '服装', '尺寸', '退换货')}
        self.assertEqual(len(clothing), 1)
        split = clothing.pop()
        rng = random.Random(1)
        for _ in range(20):
            walk = self.graph.path(rng, split, 2)
            self.assertTrue(all(self.graph.split_of(n) == split for n in walk))
            self.assertEqual(len(walk), len(set(walk)))
            self.assertLessEqual(len(walk), 3)

    def test_splits_are_balanced_and_persist_as_the_graph_grows(self):
        nodes = [(f'n{i}', f'名{i}') for i in range(60)]
        edges = [(f'n{i}', f'n{i + 1}') for i in range(0, 60, 2)]  # 30 components of 2
        graph = domains.SceneGraph(nodes, edges)
        sizes = [len(graph.nodes_in(s)) for s in domains.SPLITS]
        self.assertLessEqual(max(sizes) - min(sizes), 2)
        known = dict(graph.split)
        grown = domains.SceneGraph(nodes + [('new', '新')], edges + [('n0', 'new')], splits=known)
        self.assertTrue(all(grown.split[n] == s for n, s in known.items()))  # nobody moves
        self.assertEqual(grown.split['new'], known['n0'])                     # newcomers follow their component

    def test_propose_accepts_rejects_and_deduplicates_with_provenance(self):
        split = self.graph.split_of('服装')
        report = domains.propose(self.graph, Proposer(), 3, split=split, seed=3, max_attempts=12)
        names = [d['name'] for d in report['accepted']]
        self.assertEqual(len(names), len(set(names)))
        for domain in report['accepted']:
            self.assertTrue(domain['provenance']['scene_path'])
            self.assertIn(domain['process'], domains.PROCESSES)
            self.assertTrue(set(domain['mechanisms']) <= set(domains.MECHANISMS))
            self.assertTrue(domain['family'].startswith(split))
        self.assertTrue(any('near-duplicate' in r['reason'] for r in report['rejected'])
                        or len(report['accepted']) < 2)

    def test_non_mainstream_services_are_rejected(self):
        class Occult(Proposer):
            def chat(self, messages, tools=None, **parameters):
                return {'content': json.dumps({'accept': True, 'name': '通灵仪式预约结算', 'detail': '按分钟计费'})}
        report = domains.propose(self.graph, Occult(), 1, split=self.graph.split_of('服装'), max_attempts=2)
        self.assertEqual(report['accepted'], [])
        self.assertTrue(all('mainstream' in r['reason'] for r in report['rejected']))

    def test_benchmark_domains_are_reserved_for_evaluation(self):
        class Airline(Proposer):
            def chat(self, messages, tools=None, **parameters):
                return {'content': json.dumps({'accept': True, 'name': '航空公司机票改签', 'detail': '改签与退票'})}
        report = domains.propose(self.graph, Airline(), 1, split=self.graph.split_of('服装'), max_attempts=2)
        self.assertEqual(report['accepted'], [])
        self.assertTrue(all('reserved for evaluation' in r['reason'] for r in report['rejected']))

    def test_development_families_are_never_proposed_again(self):
        class Copycat(Proposer):
            def chat(self, messages, tools=None, **parameters):
                return {'content': json.dumps({'accept': True, 'name': '包裹计费与承运', 'detail': 'x'})}
        report = domains.propose(self.graph, Copycat(), 2, split=self.graph.split_of('服装'), max_attempts=4)
        self.assertEqual(report['accepted'], [])


if __name__ == '__main__':
    unittest.main()
