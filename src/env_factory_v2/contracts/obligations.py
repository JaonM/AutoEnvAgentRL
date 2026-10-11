"""Lower task obligations into existing executable primitives, once per definition.

Citations establish origin only. Independent source review owns interpretation.
"""
import copy

from env_factory_v2.contracts.contract import fields, named, require

SECTIONS = {'state': 'goals', 'transaction': 'goals', 'answer': 'answers',
            'report': 'reports', 'communication': 'messages', 'process': 'success'}


def lower_obligations(task):
    if 'obligations' not in task:
        return copy.deepcopy(task), []
    require(not (set(task) & {'goals', 'answers', 'reports', 'messages', 'explanations', 'success'}),
            'typed obligations cannot coexist with separate task scoring definitions')
    lowered = copy.deepcopy(task)
    entries = named(lowered.pop('obligations'), 'task obligation')
    require(bool(entries), 'task obligations cannot be empty')
    context = task.get('demand', {}).get('agent_context', {'user_request': task['request']})
    bindings = []
    for identity, item in entries.items():
        fields(item, ['id', 'kind', 'citations', 'definition'])
        kind = item['kind']
        require(isinstance(kind, str) and kind in SECTIONS, 'unknown task obligation kind')
        citations = item['citations']
        require(isinstance(citations, list) and bool(citations), 'task obligation needs original-source citations')
        for citation in citations:
            fields(citation, ['source', 'quote'])
            require(isinstance(citation['source'], str) and citation['source'] in context
                    and isinstance(citation['quote'], str) and bool(citation['quote'].strip())
                    and citation['quote'] in context[citation['source']], 'task obligation citation not in original source')
        definition = copy.deepcopy(item['definition'])
        require(isinstance(definition, dict) and 'id' not in definition, 'obligation id has one owner')
        if kind == 'state':
            fields(definition, ['rule_ids', 'predicate'])
        elif kind == 'transaction':
            fields(definition, ['rule_ids', 'mode', 'intentions'])
            mode = definition.pop('mode')
            require(isinstance(mode, str) and mode in {'static', 'timeline'}, 'unknown transaction obligation mode')
            definition['transaction_plan' if mode == 'static' else 'transaction_timeline'] = definition.pop('intentions')
        section = SECTIONS[kind]
        target = lowered.setdefault(section, [])
        target.append({'id': identity, **definition})
        bindings.append({'id': identity, 'kind': kind, 'citations': copy.deepcopy(citations),
                         'definition_section': section, 'definition_index': len(target) - 1})
    return lowered, bindings
