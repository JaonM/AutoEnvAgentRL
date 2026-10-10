"""Held-out synthetic instances of established business families, not unseen domains."""
import copy
from prepare_v2_natural_pilot import cases
from v2_natural_oracle import expand

def heldout_cases(variant=1):
    if type(variant) is not int or not 1 <= variant <= 100:
        raise ValueError("heldout variant must be an integer in 1..100")
    result = copy.deepcopy(cases())
    for index, case in enumerate(result, 1):
        prefix = ("" if variant == 1 else f"V{variant}_") + f"H{index:02d}_"
        identities = {row['id']: prefix + row['id'] for rows in expand(case['business']).values()
                      for row in rows if row['id'] not in {'F1', 'F2'}}
        def rewrite(value):
            if isinstance(value, str): return identities.get(value, value)
            if isinstance(value, list): return [rewrite(v) for v in value]
            if isinstance(value, dict): return {k: rewrite(v) for k,v in value.items()}
            return value
        case.update(rewrite(case))
        for table in case['business']['tables']:
            if 'generator' in table:
                plan = table['generator']
                plan['seed'] = index + 7 + (variant - 1) * 13
                plan['columns']['id']['prefix'] = prefix + plan['columns']['id']['prefix']
            for row in table.get('rows', []):
                for key in ['points', 'amount', 'duration', 'capacity', 'stock', 'remaining', 'inspection_fee']:
                    if type(row.get(key)) is int:
                        row[key] += index % 5 + 1 + (variant - 1) * 3
        p = case['acceptance']['params']
        kind = case['acceptance']['kind']
        if kind == 'energy': p.update(watts=[920 + (variant - 1) * 40, 85 + (variant - 1) * 5], hours=[4, 6], price=0.82)
        elif kind == 'discount': p['prices'] = [147 + (variant - 1) * 17, 109 + (variant - 1) * 11]
        elif kind == 'duration': p.update(start_minutes=615 + (variant - 1) * 5, end_minutes=790 + (variant - 1) * 9)
        elif kind == 'ranking':
            for i, candidate in enumerate(p['candidates']):
                candidate['price'] += (i + 1) * (31 + (variant - 1) * 7)
        case['business']['provenance'] = 'held-out synthetic instance; frozen before contract generation'
    return result
