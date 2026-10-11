"""Protocol-complete mocked review records; never independent business evidence."""
import copy

from env_factory_v2.contract import compile_contract
from env_factory_v2.requirement_review import obligations, mandatory_obligations, implementation_paths


def with_requirement_reviews(source, review, original_request=None):
    result = copy.deepcopy(review)
    compiled = compile_contract(source)
    source = compiled['source']
    requirements = obligations(compiled, original_request)
    cases = [c['id'] for c in result['cases']]
    reason = 'Unit-test protocol fixture only; this does not prove business correctness.'
    result['requirement_reviews'] = [{'requirement_id': r['id'], 'consistent': True,
        'reason': reason, 'applicability': 'applicable',
        'implementation_paths': [implementation_paths(source)[0]], 'case_ids': cases}
        for r in requirements]
    result['obligation_reviews'] = [{'paths': [r['path'] for r in mandatory_obligations(source)],
        'consistent': True, 'reason': reason, 'requirement_ids': [r['id'] for r in requirements],
        'case_ids': cases}]
    return result
