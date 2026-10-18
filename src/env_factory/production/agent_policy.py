"""Agent-facing policy: the task's business rules rewritten in business language.

business_rules is an implementation spec, written for the env author, the verifier and the auditors:
it names tables, fields, status codes and the virtual clock, and sometimes the order of steps. An
agent should learn from what a real assistant would be given, a policy manual. This module writes
that manual (request["agent_policy"]) with a model, rejects one that still exposes internals
(deterministic checks), and has a second pass confirm that every outcome-affecting rule survived
(coverage check), with up to `attempts` rewrites. Grading never changes: env, verifier and gates keep
following business_rules; the policy gate then shows the task is solvable from the manual alone.
"""
from __future__ import annotations

import json
import re

from env_factory.authoring import prompts

STEPS = re.compile(r'首先|第一步|第二步|然后调用|随后调用|先调用|\bstep \d\b|\bfirst,? (call|query|look up|check)\b'
                   r'|\bthen call\b', re.I)
LANGUAGE_RULES = {
    'zh': '手册用中文写。保留规格规定的币种、金额精度和报价单位，不因语言改变币种，不默认换成人民币或美元。'
          '只有规格允许且换算关系明确时，才把最小货币单位换成常用单位展示；不得改变费用或舍入规则。',
    'en': 'Write the manual in English. Preserve the currency, monetary precision and quotation units specified '
          'in the rules; do not assume dollars or change the currency to match the language. Convert minor '
          'units for display only when the rules permit it and the conversion is defined. Preserve all fees '
          'and rounding rules exactly.',
}


def internals(request, policy, tool_names=()):
    """Deterministic problems: implementation details or private data in the agent-facing policy."""
    problems = []
    identifiers = sorted(set(re.findall(r'[A-Za-z]+_[A-Za-z0-9_]+', policy)))
    if identifiers:
        problems.append(f'mentions identifiers {identifiers[:6]}: say them in business language')
    english = request.get('language', 'zh') == 'en'
    # In English a table named "orders" is also an ordinary word; only identifier-like names count there.
    tables = [t for t in request['tables'] if re.search(rf'(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])', policy)
              and ('_' in t or not english)]
    if tables:
        problems.append(f'mentions tables {tables}')
    if 'time.start' in policy or '虚拟' in policy or re.search(r'\bvirtual\b', policy, re.I):
        problems.append('mentions the virtual clock: use relative times such as “现在”')
    # Likewise a one-word tool name ("refund") is an ordinary English word.
    named = [t for t in tool_names if t in policy and ('_' in t or not english)]
    if named:
        problems.append(f'names tools {named}')
    if STEPS.search(policy):
        problems.append('prescribes solution steps: state policies, not a procedure')
    private = {k: v for k, v in request['user_facts'].items() if not k.startswith('approve')}
    values = [str(x) for v in private.values() for x in (v if isinstance(v, list) else [v])
              if isinstance(x, (str, int)) and not isinstance(x, bool) and len(str(x)) >= 3]
    leaked = [v for v in values if v in policy and v not in request['user_request'] and v not in request['business_rules']]
    if leaked:
        problems.append(f'reveals private user facts {leaked[:3]}')
    return problems


def coverage(client, rules, policy):
    """Outcome-affecting rules the policy drops or changes, according to a checking model."""
    content = client.chat([{'role': 'user', 'content': prompts.POLICY_CHECKER.format(rules=rules, policy=policy)}],
                          temperature=0).get('content') or ''
    try:
        found = json.loads(content[content.index('{'):content.rindex('}') + 1]).get('problems')
    except (ValueError, AttributeError):
        return ['the coverage check returned no verdict']
    return [str(p) for p in found] if isinstance(found, list) else ['the coverage check returned no verdict']


def write(client, request, tool_names=(), attempts=3):
    """(policy, problems): problems is empty when the policy passed both checks."""
    feedback, problems = '', []
    for _ in range(attempts):
        policy = (client.chat([{'role': 'user', 'content': prompts.POLICY_WRITER.format(
            rules=request['business_rules'], feedback=feedback,
            language_rule=LANGUAGE_RULES[request.get('language', 'zh')])}], temperature=0.3).get('content') or '').strip()
        problems = internals(request, policy, tool_names) or coverage(client, request['business_rules'], policy)
        if policy and not problems:
            return policy, []
        feedback = '\n上一版的问题，请改正：\n- ' + '\n- '.join(problems[:8]) + '\n上一版：\n' + policy + '\n'
    return None, problems
