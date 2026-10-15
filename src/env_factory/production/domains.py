"""Domain sampling for large-scale generation: scene graph × business grid.

The scene graph (V1's Neo4j Scene nodes with SAME_EVENT_ELEMENT and HIERARCHY edges, grown
toward business scenes by experiments/expand_business_graph.py) supplies concrete, long-tail
scene material; on its own it says nothing about business structure. The business grid supplies
that structure: a business process and one or two mechanisms (budgets, capacities,
independent approvals...). A proposer model (DOMAIN_*, falling back to LLM_*) combines a
sampled scene path, a process and mechanisms into one concrete customer-facing domain, or
rejects the combination when no plausible business fits it.

Family isolation uses the graph's structure: same-event components are balanced across splits
(dev, holdout-1, holdout-2, ...) and the node -> split map is persisted (experiments/
graph_splits.json), so assignments survive graph growth. A walk never leaves its split, so two
splits never share a scene node. Proposed domains are also deduplicated by
3-gram similarity against the development families, any excluded names and each other.

    python -m env_factory domains --count 12 --split holdout-4 --output domains.json
"""
from __future__ import annotations

import json
import random
import re

PROCESSES = ('预约排期', '计费结算', '资格核验', '退款售后', '库存配额', '权限授权', '理赔审批', '会员积分',
             '改签变更', '投诉处理')
MECHANISMS = ('预算上限', '容量名额', '多笔独立授权', '部分成功', '阶梯计价', '时间边界', '优先级排序', '互斥条件',
              '累计额度', '押金冻结与释放')
DEVELOPMENT_FAMILIES = ('图书借阅续借与预约', '供应商采购与分批履约', '订阅按天计费与退款', '文档权限继承与授权',
                        '包裹计费与承运', '停车预约与充电车位', '仓储批次先到期先出', '文件版本保留与清理')
SPLITS = ('dev', 'holdout-1', 'holdout-2', 'holdout-3', 'holdout-4', 'holdout-5')
SIMILAR = 0.5
# Deterministic backstop for the proposer: not mainstream, legitimate customer services.
BLOCKED = ('通灵', '灵媒', '占卜', '算命', '风水', '法事', '招魂', '驱魔', '巫术', '神秘主义', '玄学', '赌', '博彩',
           '色情', '成人用品', '邪教', '传销')
# Domains of the external benchmarks we evaluate on (τ²-bench airline and telecom) are never
# generated, so a gain there is transfer, not practice on the same business.
RESERVED_FOR_EVAL = ('航空', '机票', '航班', '登机', '值机', '托运', '电信', '话费', '手机套餐', '流量', '运营商',
                     '宽带', '携号转网', 'SIM')

PROPOSER = """你在为训练业务助理 agent 设计业务领域：agent 按一套政策、通过工具替一位用户办事。agent 的服务对象可以是消费者（客服）、企业员工（HR、IT、财务、采购等内部助理）、专业人士（律所、诊所、研究机构的事务助理）、合作企业（物流调度、供应商对账等 B2B 业务）或个人（日程、家庭事务助理），不限于面向消费者的客服。请把下面的素材组合成一个具体、真实存在的业务领域。

场景素材（来自知识图谱的一条相关概念路径）：{path}
业务环节：{process}
需要体现的机制：{mechanisms}

要求：
- 领域必须是现实中会有人请助理或客服代为办理的业务，场景素材要实质用上（至少一个概念决定了业务内容），不是贴标签；
- detail 用一句中文写清这个领域里有哪些业务对象、规则要点，以及上面的机制怎么出现；
- 只接受常见、合法的正规业务（如出行、住宿、零售、金融保险、公共事业、教育、医疗挂号、政务、物流仓储、文旅票务，
  以及企业内部的人事、IT、财务、采购，专业机构的事务管理，B2B 的调度与对账）；玄学迷信、宗教法事、博彩、成人、违法或灰色业务一律拒绝；
- 如果这些素材实在组合不出合理的业务（比如纯自然或纯历史概念配理赔），就拒绝，不要硬凑。

只输出 JSON：{{"accept": true/false, "name": "<领域名，10 字以内>", "detail": "<一句话>", "reason": "<拒绝原因，接受时为空>"}}"""


WORLD_PROPOSER = """你在为训练业务助理 agent 设计一个完整的业务世界：一个数据库加一本政策手册，之后会在上面出几十道不同的任务。
agent 按政策、通过工具替一位用户办事；agent 的服务对象可以是消费者（客服）、企业员工（HR、IT、财务、采购等内部助理）、专业人士（律所、诊所、研究机构的事务助理）、合作企业（物流调度、供应商对账等 B2B 业务）或个人（日程、家庭事务助理），不限于面向消费者的客服。
请把下面的素材组合成一个具体、真实存在的业务（一家机构或一类服务）。

场景素材（来自知识图谱同一事件分量的一条概念路径）：{path}
可参考的业务环节：{processes}
需要体现的机制：{mechanisms}

要求：
- 业务必须是现实中会有人请助理或客服代为办理的正规业务，场景素材要实质用上；
- 列出 6 到 10 个用户会来办理的业务流程（查询、预订、变更、取消、退款、补偿、升级、投诉……），其中至少一个是只需要按政策回答的咨询，至少一个是需要查账户后给建议的咨询；
- detail 用一段中文写清：业务对象有哪些，身份核验方式，主要规则（资格、时限、费用与取整、限额、需要用户同意的操作），上面的机制如何出现，哪些请求超出范围；
- 只接受常见、合法的正规业务；玄学迷信、宗教法事、博彩、成人、违法或灰色业务一律拒绝；组合不出合理业务就拒绝。

只输出 JSON：{{"accept": true/false, "name": "<业务名，12 字以内>", "detail": "<一段话>", "processes": ["<流程>", ...], "reason": "<拒绝原因，接受时为空>"}}"""


def grams(text):
    text = re.sub(r'\s+', '', text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))} or {text}


def similarity(a, b):
    left, right = grams(a), grams(b)
    return len(left & right) / max(1, len(left | right))


class SceneGraph:
    """Scene nodes, same-event edges (the edges walks follow) and a persisted node -> split map.

    Components are taken over same-event edges, so no single component dominates. Splits are
    balanced greedily by node count, largest components first. The assignment is persisted per
    node, so a node never changes split when the graph grows: new nodes follow the majority of
    their component's assigned nodes, or go to the smallest split. Walks only step to neighbours
    in the same split, so two splits never share a scene node.
    """

    def __init__(self, nodes, event_edges, hierarchy_edges=(), splits=None):
        self.names = dict(nodes)  # id -> name
        self.event = {node: set() for node in self.names}
        for a, b in event_edges:
            if a in self.event and b in self.event and a != b:
                self.event[a].add(b)
                self.event[b].add(a)
        self.component = {}
        for start in sorted(self.names):
            if start in self.component:
                continue
            stack = [start]
            self.component[start] = start
            while stack:
                node = stack.pop()
                for near in self.event[node]:
                    if near not in self.component:
                        self.component[near] = start
                        stack.append(near)
        self.split = {}
        self.assign(splits)

    def assign(self, known=None):
        """Keep `known` ({node: split}); place the rest by component, balancing split sizes."""
        self.split = {node: split for node, split in (known or {}).items() if node in self.names and split in SPLITS}
        members = {}
        for node, root in self.component.items():
            members.setdefault(root, []).append(node)
        size = {split: sum(1 for s in self.split.values() if s == split) for split in SPLITS}
        for root, group in sorted(members.items(), key=lambda item: (-len(item[1]), item[0])):
            placed = [self.split[n] for n in group if n in self.split]
            target = max(set(placed), key=placed.count) if placed else min(SPLITS, key=lambda s: (size[s], s))
            for node in group:
                if node not in self.split:
                    self.split[node] = target
                    size[target] += 1
        return self.split

    @classmethod
    def from_neo4j(cls, uri, user, password, database='neo4j', splits_path=None):
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session(database=database) as session:
                nodes = [(r['id'], r['name']) for r in session.run(
                    'MATCH (n:Scene) RETURN n.id AS id, n.name AS name')]
                event = [(r['a'], r['b']) for r in session.run(
                    'MATCH (a:Scene)-[:SAME_EVENT_ELEMENT]-(b:Scene) RETURN a.id AS a, b.id AS b')]
        finally:
            driver.close()
        known = json.loads(open(splits_path, encoding='utf-8').read()) if splits_path and _exists(splits_path) else None
        graph = cls(nodes, event, splits=known)
        if splits_path:
            with open(splits_path, 'w', encoding='utf-8') as handle:
                json.dump(dict(sorted(graph.split.items())), handle, ensure_ascii=False, indent=0)
        return graph

    def split_of(self, node):
        return self.split[node]

    def nodes_in(self, split):
        return sorted(node for node in self.names if self.split[node] == split)

    def path(self, rng, split, hops):
        """A random walk of up to `hops` same-event edges without revisits, inside one split."""
        pool = self.nodes_in(split)
        if not pool:
            raise ValueError(f'no scene nodes in split {split}')
        walk = [rng.choice(pool)]
        for _ in range(hops):
            options = sorted(n for n in self.event[walk[-1]] - set(walk) if self.split[n] == split)
            if not options:
                break
            walk.append(rng.choice(options))
        return walk


def _exists(path):
    import os
    return os.path.exists(path)


def screen(verdict, taken, accepted):
    """Why a proposal is rejected (None when it is kept): not accepted, a grey or occult business, a benchmark's
    domain, or a near-duplicate of a development family, an excluded domain or an accepted one."""
    if not verdict.get('accept') or not verdict.get('name') or not verdict.get('detail'):
        return verdict.get('reason') or 'rejected'
    blocked = [term for term in BLOCKED if term in verdict['name'] + verdict['detail']]
    if blocked:
        return f'not a mainstream legitimate service: {blocked}'
    reserved = [term for term in RESERVED_FOR_EVAL if term in verdict['name'] + verdict['detail']]
    if reserved:
        return f'reserved for evaluation: {reserved}'
    twin = next((name for name in taken if similarity(verdict['name'], name) >= SIMILAR), None) or next(
        (d['name'] for d in accepted if similarity(verdict['detail'], d['detail']) >= SIMILAR), None)
    if twin:
        return f'near-duplicate of {twin}'
    return None


def propose_worlds(graph, client, count, *, split, seed=0, hops=4, exclude=(), max_attempts=None):
    """Up to `count` domain worlds (production/world.py) for `split`: a longer walk inside one same-event component
    and several mechanisms become one whole business where an agent acts for a user with 6 to 10 processes. A world, like a
    family, never crosses splits, so a held-out split stays unseen as a whole."""
    rng = random.Random(seed)
    accepted, rejected = [], []
    taken = [*DEVELOPMENT_FAMILIES, *exclude]
    for attempt in range(max_attempts or count * 4):
        if len(accepted) >= count:
            break
        walk = graph.path(rng, split, rng.randint(2, hops))
        processes = rng.sample(PROCESSES, 4)
        mechanisms = rng.sample(MECHANISMS, rng.randint(3, 4))
        path = [graph.names[node] for node in walk]
        prompt = WORLD_PROPOSER.format(path=' → '.join(path), processes='、'.join(processes),
                                       mechanisms='、'.join(mechanisms))
        try:
            text = client.chat([{'role': 'user', 'content': prompt}], temperature=0.7).get('content') or ''
            verdict = json.loads(text[text.index('{'):text.rindex('}') + 1])
        except (ValueError, KeyError):
            rejected.append({'path': path, 'reason': 'unparseable proposal'})
            continue
        provenance = {'scene_path': path, 'mechanisms': mechanisms, 'component': str(graph.component[walk[0]]),
                      'split': split}
        reason = screen(verdict, taken, accepted)
        listed = verdict.get('processes') if isinstance(verdict.get('processes'), list) else []
        if reason is None and not 6 <= len(listed) <= 10:
            reason = f'a world needs 6 to 10 processes, got {len(listed)}'
        if reason:
            rejected.append({**provenance, 'reason': reason})
            continue
        taken.append(verdict['name'])
        detail = f"{verdict['detail']} 业务流程：{'、'.join(map(str, listed))}。"
        accepted.append({'name': verdict['name'], 'detail': detail, 'processes': listed, 'mechanisms': mechanisms,
                         'family': f"{split}:world:{provenance['component']}", 'world': True,
                         'provenance': provenance})
    return {'split': split, 'accepted': accepted, 'rejected': rejected, 'proposer': getattr(client, 'model', None)}


def propose(graph, client, count, *, split, seed=0, hops=2, exclude=(), max_attempts=None):
    """Up to `count` accepted domains for `split`, with provenance and every rejection."""
    rng = random.Random(seed)
    accepted, rejected = [], []
    taken = [*DEVELOPMENT_FAMILIES, *exclude]
    for attempt in range(max_attempts or count * 4):
        if len(accepted) >= count:
            break
        walk = graph.path(rng, split, rng.randint(0, hops))
        process = PROCESSES[attempt % len(PROCESSES)]
        mechanisms = rng.sample(MECHANISMS, rng.randint(1, 2))
        path = [graph.names[node] for node in walk]
        prompt = PROPOSER.format(path=' → '.join(path), process=process, mechanisms='、'.join(mechanisms))
        try:
            text = client.chat([{'role': 'user', 'content': prompt}], temperature=0.7).get('content') or ''
            verdict = json.loads(text[text.index('{'):text.rindex('}') + 1])
        except (ValueError, KeyError):
            rejected.append({'path': path, 'process': process, 'reason': 'unparseable proposal'})
            continue
        provenance = {'scene_path': path, 'process': process, 'mechanisms': mechanisms,
                      'component': str(graph.component[walk[0]]), 'split': split}
        reason = screen(verdict, taken, accepted)
        if reason:
            rejected.append({**provenance, 'reason': reason})
            continue
        taken.append(verdict['name'])
        accepted.append({'name': verdict['name'], 'detail': verdict['detail'], 'mechanisms': mechanisms,
                         'process': process, 'family': f"{split}:{provenance['component']}:{process}",
                         'provenance': provenance})
    return {'split': split, 'accepted': accepted, 'rejected': rejected, 'proposer': getattr(client, 'model', None)}
