"""Traceable separation of user wording, internal obligations and public context."""
from __future__ import annotations

import copy

from .contract import require


def layered_demand(user_request, business_policy, *, document_id, graph_evidence=None):
    require(isinstance(user_request, str) and user_request.strip(), 'natural user request required')
    require(isinstance(business_policy, str) and business_policy.strip(), 'public business policy required')
    return {'user_request': user_request,
        'task_spec': {'requirements': [
            {'id': 'user_intent', 'text': user_request, 'source': 'user_request', 'visibility': 'public'},
            {'id': 'business_rules', 'text': business_policy, 'source': 'business_policy', 'visibility': 'public'}],
            'document_id': document_id, 'knowledge_graph': copy.deepcopy(graph_evidence or {})},
        'agent_context': {'user_request': user_request, 'business_policy': business_policy}}


def validate_demand(demand, task):
    require(isinstance(demand, dict) and set(demand) == {'user_request', 'task_spec', 'agent_context'}, 'invalid layered demand')
    require(demand['user_request'] == task, 'user wording changed')
    context = demand['agent_context']
    require(isinstance(context, dict) and set(context) == {'user_request', 'business_policy'}
            and context['user_request'] == task and isinstance(context['business_policy'], str), 'invalid public demand context')
    spec = demand['task_spec']
    require(isinstance(spec, dict) and set(spec) == {'requirements', 'document_id', 'knowledge_graph'}, 'invalid internal task spec')
    require(isinstance(spec['requirements'], list) and spec['requirements'], 'task requirements required')
    for requirement in spec['requirements']:
        require(set(requirement) == {'id', 'text', 'source', 'visibility'}, 'requirement provenance required')
        require(requirement['source'] in context and requirement['text'] == context[requirement['source']]
                and requirement['visibility'] == 'public', 'hidden or ungrounded requirement')
    require({r['source'] for r in spec['requirements']} == set(context), 'demand sources not fully covered')
    return copy.deepcopy(context)
