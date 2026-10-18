"""Cross-domain consistency checks between declared goals and private decisions."""
import re


def approval_problems(request):
    schema=request.get('goal_schema')
    facts=request.get('user_facts') or {}
    if not isinstance(schema,dict) or not isinstance(facts.get('goals'),list):return []
    # Legacy catalogues have no approval declarations at all. Their existing
    # executable approval gates remain authoritative; do not invent metadata.
    if not any(isinstance(p,dict) and p.get('approval') for p in schema.values()):return []
    templates=[schema.get(g.get('process'),{}).get('approval') for g in facts['goals'] if isinstance(g,dict)]
    patterns=[]
    for template in templates:
        if not isinstance(template,str):continue
        # Escape literals; only declared <parameter> slots are wildcards.
        pieces=re.split(r'(<[^<>]+>)',template)
        patterns.append(''.join('.+' if p.startswith('<') and p.endswith('>') else re.escape(p) for p in pieces))
    return [f'approval fact {key} has no corresponding declared goal approval'
            for key in facts if key.startswith('approve_') and not any(re.fullmatch(p,key) for p in patterns)]
