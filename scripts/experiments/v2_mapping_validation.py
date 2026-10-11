"""Validate evaluator projection structure without comparing to oracle values."""
import copy

from env_factory_v2.code_agent import object_schema
from env_factory_v2.contract import ContractError, require
from v2_natural_semantics import COLLECTION_MAPPING_INSTRUCTIONS, path_segments, project


class MappingValidationError(RuntimeError):
    """Evaluator could not express a valid mapping; not a proven sandbox defect."""


def check_mapping(entries, role_types, fields, answers):
    require(isinstance(entries, list) and len(entries) == len(role_types)
            and {e['role'] for e in entries} == set(role_types), 'incomplete or duplicate semantic mapping')
    for entry in entries:
        selectors = entry['selectors']
        require(selectors and all(s['field'] in fields for s in selectors), 'unknown or empty answer selectors')
        require(entry['packing'] in {'single', 'list'}, 'invalid packing')
        require(entry['packing'] != 'single' or len(selectors) == 1, 'single packing requires one selector')
        for selector in selectors + entry.get('exclude_selectors', []):
            require(selector['field'] in fields, 'unknown answer selector')
            path_segments(selector['path'])
        for column in entry.get('columns', []):
            path_segments(column['path'])
        for answer in answers:
            value = project(answer, entry)
            representation = entry.get('representation', 'native')
            if representation in {'records', 'set_membership'}:
                require(isinstance(value, list), 'record or membership projection requires a collection')
                continue  # Full independent content checks still run in acceptance.
            require(representation in {'native', 'total_quantity'}, 'unknown answer representation')
            expected_type = role_types[entry['role']]
            if expected_type is not list and isinstance(value, list):
                require(len(value) == 1, 'scalar semantic role requires exactly one observed value')
                value = value[0]
            require(type(value) is expected_type or
                    (type(value) in {int, float} and expected_type in {int, float}),
                    'projected type does not match semantic role')


def validated_mapping(review, *, agent, mapping_schema, role_types, fields, answers,
                      user_request, public_context):
    """One evaluator-only correction; never changes the business review verdict."""
    entries = copy.deepcopy(review['mapping'])
    try:
        check_mapping(entries, role_types, fields, answers)
        return entries, None
    except (ContractError, KeyError, TypeError) as error:
        diagnostic = str(error)
    corrected, provenance = agent.ask(
        'Correct only this evaluator answer mapping. Use the declared answer meanings and public '
        'task semantics, not an expected answer. Paths are closed property/index/wildcard selectors. '
        'Preserve every required role exactly once. Do not invent computed selectors or change the '
        'contract, observations, business verdict or task. If no faithful mapping exists, return an '
        'empty mapping; do not guess a matching value. ' + COLLECTION_MAPPING_INSTRUCTIONS + 'No tools.',
        {'user_request': user_request, 'public_context': public_context,
         'answer_fields': fields, 'observed_answers': answers,
         'required_roles': {key: value.__name__ for key, value in role_types.items()},
         'previous_mapping': entries, 'structural_error': diagnostic},
        object_schema({'mapping': mapping_schema}))
    entries = corrected['mapping']
    try:
        check_mapping(entries, role_types, fields, answers)
    except (ContractError, KeyError, TypeError) as error:
        raise MappingValidationError('independent evaluator mapping invalid after bounded correction: ' + str(error)) from error
    return entries, {'original_mapping': review['mapping'], 'structural_error': diagnostic,
                     'provenance': provenance, 'scope': 'mapping_structure_only_business_verdict_unchanged'}
