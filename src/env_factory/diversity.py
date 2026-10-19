"""Domain-independent, inspectable diversity features; never a business scorer.

Identity and monetary values are masked, while categorical values, counts,
negation, order and value equality within a task are preserved. A shared tool
path is a concentration signal, not by itself a reason to reject a sample.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict


def packed(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


IDENTITY = re.compile(r'(^id$|_ids?$|_numbers?$|^email$|^phone$|^zip$|^last_four$|^first_name$|^last_name$|^address[12]$)')
MONEY = re.compile(r'(amount|price|fare|balance|cost|fee|budget|refund|charge|compensation)', re.I)
SCRIPT_KEYS = ('wrong_first', 'ambiguous', 'clarify', 'fallbacks', 'change_of_mind', 'pushback')


class Normalizer:
    def __init__(self, request):
        self.pool, self.aliases, self.counts = {}, {}, Counter()
        self._discover(request.get('tables', {}))
        self._discover(request.get('user_facts', {}))
        # Only identity-bearing script fields; arbitrary prose must remain prose.
        self._discover((request.get('user_script') or {}).get('wrong_first', {}))

    def _discover(self, value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                self._discover(v, k)
        elif isinstance(value, list):
            for v in value:
                self._discover(v, key.removesuffix('s') if key.endswith('_ids') else key)
        elif isinstance(value, str) and value and IDENTITY.search(key):
            kind = 'person' if key in ('first_name', 'last_name') else 'email' if key == 'email' else 'address' if key.startswith('address') or key == 'zip' else 'id'
            self.pool[value] = kind

    def alias(self, value):
        if value not in self.aliases:
            kind = self.pool[value]
            self.counts[kind] += 1
            self.aliases[value] = f'<{kind}:{self.counts[kind]}>'
        return self.aliases[value]

    def text(self, value):
        for raw in sorted(self.pool, key=lambda s: (-len(s), s)):
            if raw in value:
                value = re.sub(r'(?<![\w])' + re.escape(raw) + r'(?![\w])', lambda _: self.alias(raw), value)
        value = re.sub(r'[$€£]\s*\d[\d,]*(?:\.\d+)?', '<money>', value)
        value = re.sub(r'\b\d[\d,]*(?:\.\d+)?\s*(?:USD|dollars?|euros?)\b', '<money>', value, flags=re.I)
        return ' '.join(value.casefold().split())

    def value(self, value, key=''):
        if isinstance(value, dict):
            return {self.text(k): self.value(v, k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [self.value(v, key) for v in value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)) and MONEY.search(key):
            return '<money>'
        if isinstance(value, str):
            if value in self.pool:
                return self.alias(value)
            if MONEY.search(key) and re.fullmatch(r'-?\d+(?:\.\d+)?', value):
                return '<money>'
            return self.text(value)
        return value


def script_features(request):
    script = request.get('user_script') or {}
    result = [k for k in SCRIPT_KEYS if script.get(k)]
    if any(v is False for k, v in request.get('user_facts', {}).items() if k.startswith('approve_')):
        result.append('declined_approval')
    return result or ['plain']


def observed_obstacles(request, episode):
    """Conservative literal witnesses. Unknown is not falsely counted as absent.

    These are structural witnesses, not proof of every conversational nuance.
    Model-based audit may supplement them, but does not overwrite this evidence.
    """
    script = request.get('user_script') or {}
    messages = episode.get('messages') or []
    users = [(i, str(m.get('content') or '').casefold().replace('’', "'")) for i, m in enumerate(messages) if m.get('role') == 'user']
    assistants = [(i, str(m.get('content') or '').casefold()) for i, m in enumerate(messages) if m.get('role') == 'assistant']
    evidence = {}
    for key, wrong in (script.get('wrong_first') or {}).items():
        right = request.get('user_facts', {}).get(key)
        first = next((i for i, t in users if str(wrong).casefold() in t), None)
        second = next((i for i, t in users if right is not None and str(right).casefold() in t and first is not None and i > first), None)
        if second is not None and wrong != right:
            evidence['wrong_first'] = [first, second]
    if script.get('ambiguous') or script.get('clarify'):
        hits = [i for i, t in assistants if re.search(r'\b(which (?:one|order|reservation|booking)|could you (?:clarify|specify)|please (?:clarify|specify))\b', t)]
        if hits:
            evidence['ambiguous' if script.get('ambiguous') else 'clarify'] = hits
    declined = [i for i, t in users if re.search(r"\bno(?=[,.!?](?:\s|$))|\b(?:do not|don't|decline|not (?:approve|authorize|proceed)|won't|rather not)\b", t)]
    if 'declined_approval' in script_features(request) and declined:
        evidence['declined_approval'] = declined
    if script.get('pushback'):
        hits = [i for i, t in users if re.search(r'\b(still|exception|insist|anyway|really need|surely)\b', t)]
        if hits:
            evidence['pushback'] = hits
    if script.get('change_of_mind'):
        hits = [i for i, t in users if re.search(r'\b(instead|changed my mind|rather|in that case)\b', t)]
        if hits:
            evidence['change_of_mind'] = hits
    if script.get('fallbacks'):
        hits = [i for i, t in users if re.search(r'\b(instead|alternative|other (?:one|order|option)|in that case|then (?:cancel|change|book))\b', t)]
        if hits:
            evidence['fallbacks'] = hits
    if script_features(request) == ['plain']:
        evidence['plain'] = []
    return evidence


def interaction_preconditions(request, obstacle):
    """Structural preconditions for a caller-requested obstacle, not a scorer.

    A wrong ID held until an optional question can be bypassed by record lookup.
    Make the wrong value visible initially and keep the right value private.
    """
    if obstacle != 'wrong_first':return []
    wrong=(request.get('user_script') or {}).get('wrong_first') or {}
    opening=str(request.get('user_request') or '').casefold()
    issues=[]
    if not wrong:issues.append('missing wrong_first mapping')
    for key,value in wrong.items():
        right=request.get('user_facts',{}).get(key)
        if str(value).casefold() not in opening:issues.append('wrong identifier must appear in the opening')
        if right is None or right==value:issues.append('wrong identifier needs a distinct private correct value')
        elif str(right).casefold() in opening:issues.append('correct identifier leaked in opening')
    return issues


def features(request, reference, episode):
    normal = Normalizer(request)
    goals = normal.value(request.get('user_facts', {}).get('goals', []))
    outcomes = [{k: v.get(k) for k in ('status', 'reason')} for v in reference.get('reference_answer', {}).values() if isinstance(v, dict)]
    ref = [{'tool': e['tool'], 'args': normal.value({k: v for k, v in (e.get('args') or {}).items() if k != 'consent'}), 'status': e.get('status')}
           for e in reference.get('reference_trace', []) if e.get('kind') == 'tool']
    calls, batches = [], []
    for m in episode.get('messages') or []:
        batch = []
        for c in m.get('tool_calls') or []:
            fn = c['function']
            args = fn.get('arguments', {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except ValueError: pass
            call = {'tool': fn['name'], 'args': normal.value(args)}
            calls.append(call); batch.append(call)
        if batch: batches.append(batch)
    task = {'goals': goals, 'branches': outcomes, 'script': script_features(request), 'reference': ref}
    method = (request.get('sop_blueprint') or {}).get('methods', {}).get(request.get('sop_method'))
    graph = ({'processes':method['processes'], 'edges':method.get('edges',[]),
              'bindings':method.get('bindings',[]), 'branches':outcomes} if method and not request.get('sop_legacy') else None)
    if graph:
        task['sop_graph'] = graph
    texts = [request.get('user_request', ''), packed(request.get('user_scene') or {})]
    return {'task': task, 'task_key': digest(task), 'reference_path': [e['tool'] for e in ref],
            'tool_path': [c['tool'] for c in calls], 'normalized_calls': calls, 'batches': batches,
            'task_text': normal.text(' '.join(texts)), 'declared_obstacles': script_features(request),
            'observed_obstacles': observed_obstacles(request, episode),
            'sop_graph': graph, 'sop_graph_key': digest(graph) if graph else None}


def repetition(records, key):
    groups = defaultdict(list)
    for r in records: groups[digest(key(r))].append(r['id'])
    duplicates = sorted((v for v in groups.values() if len(v) > 1), key=lambda x: (-len(x), x[0]))
    n = len(records)
    return {'n': n, 'unique': len(groups), 'excess': n - len(groups), 'excess_rate': (n-len(groups))/n if n else 0,
            'in_repeated_clusters': sum(map(len, duplicates)), 'clusters': duplicates}


class AdmissionLedger:
    """Host-thread reservation ledger: failed demonstrations consume no quota.

    Reservations prevent races between workers. Only accepted demonstrations
    count toward the per-structure cap or permanent split ownership.
    """
    def __init__(self, records, caps):
        self.caps=caps;self.counts=Counter();self.splits=defaultdict(set);self.pending={}
        for r in records:
            key=r.get('contextual_task_key',r.get('task_key'))
            self.counts[r['split'],key]+=1;self.splits[key].add(r['split'])
        if any(len(s)>1 for s in self.splits.values()):raise ValueError('baseline has cross-split structures')
        if any(n>caps[s] for (s,k),n in self.counts.items()):raise ValueError('baseline exceeds structure cap')

    def reserve(self, owner, split, key):
        if owner in self.pending:raise ValueError('owner already reserved')
        if self.splits[key]-{split}:return 'cross_split_structure'
        active=[s for s,k in self.pending.values() if k==key]
        if set(active)-{split}:return 'cross_split_in_flight'
        if self.counts[split,key]+len(active)>=self.caps[split]:return 'structure_cap'
        self.pending[owner]=(split,key)
        return None

    def finish(self, owner, *, accepted):
        split,key=self.pending.pop(owner)
        if accepted:
            self.counts[split,key]+=1;self.splits[key].add(split)
