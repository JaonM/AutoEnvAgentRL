"""Observed conditional branches, without inventing reachability requirements."""


def communication_coverage(compiled, records):
    entries = []
    for explanation in compiled['source'].get('explanations', []):
        for criterion in explanation['criteria']:
            if 'when' not in criterion:
                continue
            active, inactive, unknown, conflicts = [], [], [], []
            for record in records:
                values = []
                for execution in ('first', 'repeat'):
                    run = record[execution]
                    value = run.get('outcome', {}).get('communication_applicability', {}).get(
                        explanation['id'], {}).get(criterion['id'])
                    values.append(value if run.get('error') is None and type(value) is bool else None)
                identity = record['case']['id']
                if None in values:
                    unknown.append(identity)
                elif values[0] != values[1]:
                    conflicts.append(identity)
                elif values[0]:
                    active.append(identity)
                else:
                    inactive.append(identity)
            status = ('replay_disagreement' if conflicts else 'both_observed' if active and inactive
                      else 'applicable_only' if active else 'inapplicable_only' if inactive else 'not_observed')
            entries.append({'field': explanation['id'], 'criterion': criterion['id'],
                'rule_id': criterion['rule_id'], 'status': status,
                'applicable_case_ids': active, 'inapplicable_case_ids': inactive,
                'unobserved_case_ids': unknown, 'replay_disagreement_case_ids': conflicts})
    return {'scope': 'executed_branch_observations_not_reachability_or_business_correctness',
            'conditions': entries,
            'policy': 'A missing observed branch requires original-requirement and reachability analysis; '
                      'it is not automatically a business failure or permission to remove the condition.'}
