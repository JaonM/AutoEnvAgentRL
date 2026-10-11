"""Transport already assessed exact evidence through a manifest-bound bundle.

This ledger never replaces scenario execution or an explicitly fresh judgment.
"""
import copy
import hashlib
import json
from pathlib import Path

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.package import package_fingerprints


def implementation():
    return hashlib.sha256(canonical(package_fingerprints(suffixes=('.py',))).encode()).hexdigest()


def snapshot(agent):
    from env_factory_v2.native.code_agent import MODEL
    return {'version': 1, 'model': MODEL, 'implementation': implementation(),
            'judgments': copy.deepcopy(agent.judgments), 'poisoned': sorted(agent.poisoned)}


def restore(agent, ledger):
    from env_factory_v2.native.code_agent import MODEL
    require(isinstance(ledger, dict) and ledger.get('version') == 1
            and ledger.get('model') == MODEL and ledger.get('implementation') == implementation(),
            'semantic evidence dependencies changed')
    judgments, poisoned = ledger.get('judgments'), ledger.get('poisoned')
    require(isinstance(judgments, dict) and len(judgments) <= 128
            and isinstance(poisoned, list) and all(isinstance(key, str) for key in poisoned),
            'invalid semantic evidence registry')
    for key, result in judgments.items():
        require(isinstance(key, str) and isinstance(result, dict)
                and isinstance(result.get('criteria'), dict)
                and all(type(value) is bool for value in result['criteria'].values())
                and isinstance(result.get('provenance'), dict)
                and result['provenance'].get('dependency_identity') == key,
                'semantic evidence dependency mismatch')
    # Validate everything before changing a live session. Contradictory cached
    # assessments poison the dependency rather than silently picking one.
    combined = copy.deepcopy(agent.judgments)
    conflicts = {key for key in judgments if key in combined
                 and judgments[key]['criteria'] != combined[key]['criteria']}
    poison = agent.poisoned | set(poisoned) | conflicts
    combined.update(copy.deepcopy(judgments))
    for key in poison:
        combined.pop(key, None)
    agent.judgments = dict(list(combined.items())[-128:])
    agent.poisoned = poison


def restore_bundle(agent, root):
    """Only a verified bundle may seed a new host/container scoring session."""
    from env_factory_v2.delivery.export import verify_bundle
    root = Path(root)
    path = root / 'semantic_evidence.json'
    if not path.exists():
        return False
    verify_bundle(root)
    restore(agent, json.loads(path.read_text()))
    return True
