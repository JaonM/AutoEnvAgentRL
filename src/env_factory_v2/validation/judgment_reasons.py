"""Validate rationale references without treating a model explanation as proof."""
import copy
import re

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import canonical


def validate_reasons(reasons, payload):
    criteria = {item['id']: item for item in payload['field']['criteria']}
    require(isinstance(reasons, dict) and set(reasons) == set(criteria), 'judgment reason identities mismatch')
    for identity, item in reasons.items():
        require(isinstance(item, dict) and set(item) == {'reason', 'obligation_quote', 'evidence'},
                'invalid judgment reason')
        require(isinstance(item['reason'], str) and 0 < len(item['reason'].strip()) <= 1600,
                'bounded judgment reason required')
        quote = item['obligation_quote']
        require(isinstance(quote, str) and quote.strip() and quote in criteria[identity]['obligation'],
                'judgment reason must cite its declared obligation')
        require(isinstance(item['evidence'], list) and 1 <= len(item['evidence']) <= 8,
                'judgment reason requires bounded evidence citations')
        for citation in item['evidence']:
            require(isinstance(citation, dict) and set(citation) == {'path', 'quote'}, 'invalid judgment citation')
            pointer, quote = citation['path'], citation['quote']
            require(isinstance(pointer, str) and pointer.startswith('/') and len(pointer) <= 1024
                    and not re.search(r'~(?![01])', pointer), 'invalid judgment evidence pointer')
            node = payload
            for token in pointer[1:].split('/'):
                key = token.replace('~1', '/').replace('~0', '~')
                if isinstance(node, list):
                    require(key.isdecimal() and str(int(key)) == key and int(key) < len(node),
                            'judgment evidence index missing')
                    node = node[int(key)]
                else:
                    require(isinstance(node, dict) and key in node, 'judgment evidence path missing')
                    node = node[key]
            text = node if isinstance(node, str) else canonical(node)
            require(isinstance(quote, str) and len(quote) <= 1600 and
                    ((quote.strip() and quote in text) or quote == text == ''),
                    'judgment evidence quote must match supplied data')
    return copy.deepcopy(reasons)
