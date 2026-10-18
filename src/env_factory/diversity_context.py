"""Caller-declared policy context for conservative structural diversity selection.

No industry-specific keys or rules live here. A policy condition must not be
erased merely because two trajectories call the same tools and end successfully.
"""
from datetime import datetime


def values(value,path):
    if not path:return [value]
    part,*rest=path.split('.')
    if part=='*':
        children=value.values() if isinstance(value,dict) else value if isinstance(value,list) else []
        return [x for child in children for x in values(child,'.'.join(rest))]
    if isinstance(value,dict) and part in value:return values(value[part],'.'.join(rest))
    return []


def signature(request,reference,feature,normalizer,rules,digest):
    context=[]
    for event in reference.get('reference_trace',[]):
        if event.get('kind')!='tool':continue
        selected=[]
        for rule in rules.get(event['tool'],[]):
            raw=values(event.get('result'),rule['path'])
            if rule.get('transform')=='length':raw=[len(v) for v in raw if isinstance(v,(list,dict))]
            if rule.get('transform')=='age_hours_bucket':
                converted=[]
                for value in raw:
                    age=(datetime.fromisoformat(request['time']['start'])-datetime.fromisoformat(value)).total_seconds()/3600
                    threshold=rule['threshold']
                    converted.append('below' if age<threshold else 'equal' if age==threshold else 'above')
                raw=converted
            selected.append({'field':rule['path'],'values':normalizer.value(raw)})
        if selected:context.append({'tool':event['tool'],'fields':selected})
    return digest({'structure':feature['task'],'policy_context':context}),context
