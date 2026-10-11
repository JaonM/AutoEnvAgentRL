"""Adapter to V2's handwritten independent oracles (evaluation only, never shown to models)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

import v2_access_oracle as access  # noqa: E402
import v2_fresh_business_oracle as fresh  # noqa: E402
import v2_library_oracle as library  # noqa: E402
import v2_procurement_oracle as procurement  # noqa: E402
import v2_subscription_oracle as subscription  # noqa: E402
from v2_natural_oracle import expand  # noqa: E402


def truth(spec, data, facts, now):
    data, facts = copy.deepcopy(data), copy.deepcopy(facts)
    kind = spec['kind']
    if kind.startswith('library_'):
        return library.expected(spec, data, facts)
    if kind.startswith('subscription_'):
        return subscription.expected(spec, data, facts)
    if kind == 'procurement_partial':
        return procurement.expected(spec, data, facts, now=[now, now])
    if kind == 'procurement_large':
        facts.setdefault('proposal_time', now)
        # Frozen time: the authorized snapshot is the state at proposal time, i.e. the initial state.
        snapshot = {'quotes': copy.deepcopy(data['quotes']), 'budget_cents': data['budgets'][0]['remaining_cents']}
        return procurement.expected(spec, data, facts, now=now, authorized_snapshot=snapshot)
    if kind.startswith('procurement_'):
        return procurement.expected(spec, data, facts, now=now)
    if kind.startswith('fresh_'):
        return fresh.expected(spec, data, facts)
    if kind.startswith('access_'):
        return access.expected(spec, data, facts, now=now)
    raise ValueError(f'no truth adapter for {kind}')
