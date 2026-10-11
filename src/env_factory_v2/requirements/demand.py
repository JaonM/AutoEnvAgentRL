"""Traceable separation of user wording, internal obligations and public context."""
from __future__ import annotations

import copy
import re

from env_factory_v2.contracts.contract import require


def cited_requirements(context):
    """Split public clauses without inventing or paraphrasing requirements.

    A citation proves textual origin, not semantic completeness of the contract.
    Independent business review still checks the executable interpretation.
    """
    requirements = []
    for source, text in context.items():
        for index, match in enumerate(re.finditer(r'[^。！？；;\n]+[。！？；;\n]*', text), 1):
            if not match.group().strip():
                continue
            requirements.append({'id': f'{source}_{index}', 'text': match.group(),
                'source': source, 'visibility': 'public',
                'citation': {'start': match.start(), 'end': match.end()}})
    return requirements


def layered_demand(user_request, business_policy, *, document_id, graph_evidence=None):
    require(isinstance(user_request, str) and user_request.strip(), 'natural user request required')
    require(isinstance(business_policy, str) and business_policy.strip(), 'public business policy required')
    context = {'user_request': user_request, 'business_policy': business_policy}
    return {'user_request': user_request,
        'task_spec': {'requirements': cited_requirements(context),
            'document_id': document_id, 'knowledge_graph': copy.deepcopy(graph_evidence or {})},
        'agent_context': context}


def validate_demand(demand, task):
    require(isinstance(demand, dict) and set(demand) == {'user_request', 'task_spec', 'agent_context'}, 'invalid layered demand')
    require(demand['user_request'] == task, 'user wording changed')
    context = demand['agent_context']
    require(isinstance(context, dict) and set(context) == {'user_request', 'business_policy'}
            and context['user_request'] == task and isinstance(context['business_policy'], str), 'invalid public demand context')
    spec = demand['task_spec']
    require(isinstance(spec, dict) and set(spec) == {'requirements', 'document_id', 'knowledge_graph'}, 'invalid internal task spec')
    require(isinstance(spec['requirements'], list) and spec['requirements'], 'task requirements required')
    ids, coverage = set(), {source: set() for source in context}
    for requirement in spec['requirements']:
        require(isinstance(requirement, dict) and set(requirement) in (
            {'id', 'text', 'source', 'visibility'},
            {'id', 'text', 'source', 'visibility', 'citation'}), 'requirement provenance required')
        require(isinstance(requirement['id'], str) and requirement['id'] and requirement['id'] not in ids,
                'requirement IDs must be unique')
        ids.add(requirement['id'])
        source = requirement['source']
        require(isinstance(source, str) and source in context and requirement['visibility'] == 'public',
                'hidden or ungrounded requirement')
        text = context[source]
        citation = requirement.get('citation', {'start': 0, 'end': len(text)})
        require(isinstance(citation, dict) and set(citation) == {'start', 'end'}, 'invalid requirement citation')
        start, end = citation['start'], citation['end']
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text),
                'invalid requirement citation range')
        require(isinstance(requirement['text'], str) and requirement['text'].strip()
                and requirement['text'] == text[start:end], 'hidden or ungrounded requirement')
        coverage[source].update(range(start, end))
    require({r['source'] for r in spec['requirements']} == set(context), 'demand sources not fully covered')
    for source, text in context.items():
        require(all(index in coverage[source] for index, char in enumerate(text)
                    if not char.isspace() and char not in '。！？；;'), 'public demand clauses not fully covered')
    return copy.deepcopy(context)
