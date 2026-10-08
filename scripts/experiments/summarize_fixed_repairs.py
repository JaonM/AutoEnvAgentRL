#!/usr/bin/env python3
"""Summarize paired fixed-source repair trials without treating pending as failed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics



def network_interrupted(path):
    """A network failure after the last work event, without a completed turn."""
    if not path.exists():return False
    pending=False
    markers=('workspace routing discovery failed','failed to lookup address information',
             'waiting for network','stream disconnected before completion','connection failed: error sending request')
    for line in path.read_text().splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('type') in {'item.completed','turn.completed'}:pending=False
        if event.get('type') in {'error','turn.failed'}:
            message=str(event.get('message') or event.get('error',{}).get('message','')).lower()
            if any(marker in message for marker in markers):pending=True
    return pending


def usage_from_events(paths):
    usage={};incomplete=[]
    for path in sorted(paths):
        completed=[]
        for line in path.read_text().splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue  # A live final line can be partial.
            if event.get('type')=='turn.completed':completed.append(event)
        if not completed:incomplete.append(str(path))
        for event in completed:
            for key,value in event.get('usage',{}).items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):usage[key]=usage.get(key,0)+value
    return {'reported_tokens':usage,'unreported_or_pending_event_files':incomplete}


def summarize(root):
    if (root/'invalidated.json').exists():
        raise ValueError('Invalidated experiment must not be counted: '+str(root))
    plan=json.loads((root/'plan.json').read_text())
    cases=[case for case in plan['cases'] if not plan.get('selected_cases') or case['id'] in plan['selected_cases']]
    rows=[];groups={model:[] for model in plan['models']}
    for case in cases:
        for model in plan['models']:
            trial=root/case['id']/model;path=trial/'result.json'
            row={'case':case['id'],'model':model,'status':'pending'}
            if path.exists():
                result=json.loads(path.read_text())
                if result.get('case')!=case['id'] or result.get('model')!=model:
                    raise ValueError('Result identity mismatch: '+str(path))
                attempts=result.get('attempts',[])
                final=attempts[-1] if attempts else {}
                infrastructure=(final.get('status')=='infrastructure_error'
                    or final.get('error_type')=='SemanticReviewUnavailable'
                    or (final.get('status')=='author_timeout' and network_interrupted(
                        trial/f"attempt-{final.get('index')}"/'events.jsonl')))
                row.update(status='passed' if result.get('passed') else 'infrastructure_error' if infrastructure else 'failed',
                    first_attempt_passed=bool(attempts and attempts[0].get('status')=='passed'),
                    baseline_execution_passed=result.get('execution_passed') is True,
                    review_passed=result.get('review_passed') is True,docker_passed=result.get('docker_passed') is True,
                    author_seconds=result.get('author_seconds'),total_seconds=result.get('seconds'),
                    author_turns=len(attempts),platform_unchanged=result.get('platform_unchanged') is True,
                    failure_phase=final.get('status') if not result.get('passed') else None,
                    failure_reason=final.get('error') if not result.get('passed') else None,
                    author_usage=usage_from_events(trial.glob('attempt-*/events.jsonl')),
                    review_usage=usage_from_events(trial.glob('attempt-*/review/**/events.jsonl')))
                if not row['platform_unchanged']:row['status']='invalid_platform_changed'
                groups[model].append(row)
            rows.append(row)
    summaries={}
    for model,finished in groups.items():
        evaluable=[r for r in finished if r['status'] in {'passed','failed'}]
        passed=sum(r['status']=='passed' for r in evaluable)
        times=[r['total_seconds'] for r in evaluable if isinstance(r['total_seconds'],(int,float))]
        summaries[model]={'planned':len(cases),'terminal':len(finished),
            'pending':len(cases)-len(finished),'evaluable':len(evaluable),
            'infrastructure_errors':sum(r['status']=='infrastructure_error' for r in finished),
            'invalidated_trials':sum(r['status']=='invalid_platform_changed' for r in finished),
            'passed':passed,'pass_rate_among_evaluable':passed/len(evaluable) if evaluable else None,
            'first_attempt_passed':sum(r['first_attempt_passed'] for r in evaluable),
            'median_total_seconds':statistics.median(times) if times else None}
    paired=[]
    for case in cases:
        pair=[r for r in rows if r['case']==case['id']]
        if all(r['status'] in {'passed','failed'} for r in pair):
            winners=[r['model'] for r in pair if r['status']=='passed']
            paired.append({'case':case['id'],'outcome':'both_passed' if len(winners)==2 else 'both_failed' if not winners else 'one_passed',
                           'passed_models':winners})
    return {'scope':plan['scope'],'complete':all(r['status'] in {'passed','failed'} for r in rows),
        'all_trials_terminal':all(r['status']!='pending' for r in rows),
        'models':summaries,'paired_cases':paired,'trials':rows,
        'limits':['Four fixed failed sources, not fresh-generation yield or training readiness.',
                  'Same Astra reviewer for both authors; paths retain model labels, so not formally blinded.',
                  'Token usage covers reported author/reviewer turns; semantic judge billing is not instrumented.',
                  'Infrastructure errors and invalidated trials are excluded from model pass-rate denominators.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=summarize(args.root)
    if args.output:args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'complete':report['complete'],'models':report['models'],'paired_cases':report['paired_cases']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
