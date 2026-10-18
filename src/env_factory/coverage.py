"""Caller-defined coverage obligations, grounded in actual requests and execution.

Independent of industry names, tools and policy rules. Labels on generated tasks
are never evidence. A cell matches only its predicates and execution requirements.
"""


def values(value, path):
    if not path: return [value]
    part, *rest = path.split('.')
    if part == '*':
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
        return [v for item in children for v in values(item, '.'.join(rest))]
    if isinstance(value, dict) and part in value:
        return values(value[part], '.'.join(rest))
    if isinstance(value, list) and part.isdigit() and int(part)<len(value):
        return values(value[int(part)], '.'.join(rest))
    return []


def matches(record, predicate):
    if set(predicate)-{'path','equals','min','max','contains'} or 'path' not in predicate:
        raise ValueError('invalid coverage predicate')
    if not isinstance(predicate['path'],str):raise ValueError('coverage path must be a string')
    def check(v):
        if 'equals' in predicate and v != predicate['equals']: return False
        if 'contains' in predicate and (not isinstance(v,(list,str,dict)) or predicate['contains'] not in v): return False
        if 'min' in predicate and (type(v) not in (int,float) or v<predicate['min']): return False
        if 'max' in predicate and (type(v) not in (int,float) or v>predicate['max']): return False
        return True
    return any(check(v) for v in values(record, predicate['path']))


def report(cells, records):
    ids=[c['id'] for c in cells]
    if len(set(ids))!=len(ids): raise ValueError('duplicate coverage cell IDs')
    result=[]
    for cell in cells:
        if not cell.get('predicates'): raise ValueError('coverage cells need observable predicates')
        for predicate in cell['predicates']:
            matches({},predicate)  # Validate even when no records have arrived yet.
        matched=[]
        for r in records:
            if all(matches(r,p) for p in cell['predicates']):
                matched.append(r)
        reference={r['id'] for r in matched if r.get('reference_passed') is True}
        natural={r['id'] for r in matched if r.get('reference_passed') is True and r.get('natural_passed') is True}
        minimum=cell.get('minimum',1)
        if type(minimum) is not int or minimum<1: raise ValueError('invalid coverage minimum')
        result.append({'id':cell['id'],'minimum':minimum,'reference_ids':sorted(reference),
                       'natural_ids':sorted(natural),'reference_covered':len(reference)>=minimum,
                       'natural_covered':len(natural)>=minimum})
    return {'cells':result,'total':len(result),'reference_covered':sum(r['reference_covered'] for r in result),
            'natural_covered':sum(r['natural_covered'] for r in result),
            'reference_gaps':[r['id'] for r in result if not r['reference_covered']],
            'natural_gaps':[r['id'] for r in result if not r['natural_covered']],
            'limits':'Coverage of declared obligations only; not exhaustive proof of all possible tasks.'}


def require(cells, records, *, stage='natural'):
    """Fail-closed release check; declarations never substitute for executed evidence."""
    if stage not in ('reference','natural'):raise ValueError('invalid coverage stage')
    result=report(cells,records)
    if result[stage+'_gaps']:
        raise ValueError(f'{stage} coverage incomplete: {result[stage+"_gaps"]}')
    return result
