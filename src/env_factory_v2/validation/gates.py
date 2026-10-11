"""Acceptance gate vocabulary shared by delivery and historical round readers."""
EXECUTION_GATES = frozenset({
    'joint_execution_and_counterexamples',
    'deterministic_reset_and_replay',
    'independent_business_challenge',
})
DELIVERY_GATES = EXECUTION_GATES | {'container_execution'}


def delivery_gates_pass(gates):
    """Read the former constant structural flag without requiring new reports to emit it."""
    return (isinstance(gates, dict)
            and set(gates) in (DELIVERY_GATES, DELIVERY_GATES | {'structural_contract'})
            and all(value is True for value in gates.values()))
