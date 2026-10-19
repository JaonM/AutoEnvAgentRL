"""Version coherent worlds with frozen behavior contracts and independently computed outcomes."""
import argparse
import copy
import json
from pathlib import Path
from env_factory.core import spec as request_spec
import shutil

from env_factory.certify import bundle, behavior
from env_factory.core.sandbox import load_module
from env_factory.production import contracts
from env_factory.production.world import task_schema
from world_extensions.semantic_cases import make

SOURCE = Path(__file__).resolve().parent / 'world_extensions/coherent'

COMMON = '''Coherent business simulation v1. The public agent policy and public tool signatures remain
authoritative. Historical tau2 implementation snapshots are provenance, not behavioral authority for this
version. No hardcoded fixture IDs, canned success for an unperformed business operation, hidden creation
quota, or shared last-item values are allowed. Public queries return actual current records. A failed
operation changes no state. Read calls change no state. User confirmation and agent-side policy decisions
remain separate from low-level execution. No new fee, deadline, benefit, identity API or public parameter
is introduced. Behavior contracts exercise independent synthetic data, sequential operations and scale.
Scope: return/exchange service ends at a queryable request stored on the order; it does not perform
shipping, receipt inspection or final settlement. Handover ends at the actual transfer tool invocation and
its durable execution-trace receipt; no human response or human ticketing service is claimed. Existing
fields and trace are the records of these scope boundaries. Product availability remains a boolean.
Refunds previously routed to a different method offset positive outstanding funding in first-seen
method order after per-method netting; total refunded cannot exceed the signed outstanding total.
A negative overall funding total is invalid.
The runtime-provided clock determines creation time and elapsed-time eligibility; baseline time.start
is unchanged. Domain calculations and expected outcomes must be implemented independently.
'''
NOTES = {
 'retail': '''Refund only outstanding net funding to each payment method, after accounting for all prior
payments and refunds. Do not refund a refund entry. Gift-card balance changes by the actual newly
refunded/charged amount. A cancelled order cannot be cancelled again. Positional old/new item pairs
consume distinct original item occurrences, including duplicate IDs; each replacement takes its own
catalogue price and options. Resolve all target occurrences before mutation, including swaps. Keep the
existing once-only edit policy. Payment replacement refunds the original and charges the new funding.
Address writes affect only the chosen profile/order. Return/exchange records remain requests without
immediate settlement; preserve the public refund-destination, availability and price-difference rules.
''',
 'airline': '''Booking consumes seats for each distinct dated segment/cabin and writes a linked reservation.
Changing flights/cabin releases old occupancy and consumes new occupancy atomically; unchanged segments
have zero net change. Cancellation releases occupancy once, refunds outstanding signed funding by method,
and changes status. Gift balances receive refunds; consumed certificates regain only their refunded paid
portion under the original certificate ID, not forfeited unused face value; credit-card refunds are recorded
in payment history (no external bank service is simulated). Repeated cancellation is a no-op. Modification
of a cancelled reservation is rejected. Empty/duplicate segments, negative payments and duplicate payment
methods are invalid. New reservation IDs are NR plus the smallest unused positive eight-digit number.
Certificate IDs are certificate_coherent_ plus the smallest unused positive eight-digit number across all
saved methods AND historical payments; IDs have no fixed three-use cap. No extra public idempotency argument
is required. Prices, fees, eligibility, membership and insurance rules remain those in the agent policy.
Connection dates use calendar arithmetic, including month rollover, and compare the next day's local time.
'''}


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def amend(request, domain, behavioral):
    request=copy.deepcopy(request)
    old=request['public_contract'];contract={k:copy.deepcopy(v) for k,v in old.items()
        if k in ('policy','tools','constraints','constraint_sources')}
    contract.update(implementation_notes=COMMON+NOTES[domain],semantic_profile='coherent-v1',
                    predecessor_contract_sha256=contracts.digest(old))
    # Remove obsolete embedded source listings, rather than append contradictory overrides.
    request['business_rules']='Business process outcomes are defined by goal_schema.\n'+COMMON+NOTES[domain]
    contracts.attach(request,contract)
    request['behavior_contract']=copy.deepcopy(behavioral)
    request['behavior_contract_sha256']=behavior.digest(behavioral)
    request['source']={**request.get('source',{}),'semantic_profile':'coherent-v1',
                       'predecessor_request_id':request['id']}
    return request


def prepare(source, output, domain, behavioral, *, sop=False):
    loaded=bundle.load_bundle(source,require_current_platform=False)
    request=amend(loaded['request'],domain,behavioral)
    if sop:
        from world_extensions.sop_specs import blueprint, executable_contract
        request['sop_blueprint']=blueprint(domain,request)
        request['sop_contract']=executable_contract(behavioral)
        request['sop_contract_sha256']=behavior.digest({'blueprint':request['sop_blueprint'],'contract':request['sop_contract']})
        # Historical tasks retain their original goals; they are never relabelled
        # as sampled SOP tasks. compose() removes this marker for every new task.
        request['sop_legacy']=True
        request['source']['semantic_profile']='sop-v1'
    output=Path(output);stage=output/'source';stage.mkdir(parents=True,exist_ok=False)
    for name in ('env','verifier'):shutil.copyfile(SOURCE/f'{domain}_{name}.py',stage/f'{name}.py')
    request_spec.write(request, stage / 'request.json');save(stage/'answer_schema.json',loaded['schema'])
    candidate={**loaded,'request':request,'env':load_module(stage/'env.py',role='env'),
               'verifier':load_module(stage/'verifier.py',role='verifier')}
    errors=contracts.interface_problems(candidate['env'].make_tools(),request)
    if errors:raise ValueError(errors)
    report=bundle.qualify_loaded(candidate)
    save(output/'qualification.json',{**{k:v for k,v in report.items() if k!='worlds'},
        'worlds':[{k:w.get(k) for k in ('world','status','reward')} for w in report.get('worlds',[])]})
    if not report['passed']:
        raise ValueError({k:v for k,v in report.items() if isinstance(v,dict) and v.get('passed') is False})
    bundle.export(stage,stage/'request.json',report,output/'bundle',status='qualified')
    save(output/'lineage.json',{'source':str(Path(source).resolve()),'source_manifest':loaded['manifest'],
        'behavior_contract_sha256':behavior.digest(behavioral),'profile':'sop-v1' if sop else 'coherent-v1'})
    return str((output/'bundle').resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('output/world_scale_v3'))
    p.add_argument('--sop',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((args.source/'summary.json').read_text());summary['source']=str(args.source.resolve())
    summary.update(passed=False,semantic_profile='sop-v1' if args.sop else 'coherent-v1',training_ready=False)
    for d,info in summary['domains'].items():
        full=bundle.load_bundle(args.source/d/'world/bundle',require_current_platform=False)
        if args.sop:
            from world_extensions.sop_specs import make_behavior
            contract=make_behavior(d,full['request']['tables'])
        else:contract=make(d,full['request']['tables'])
        save(args.output/d/'behavior_contract.json',contract)
        old_report=behavior.run(full['env'],contract);save(args.output/d/'old_behavior.json',old_report)
        info['bundle']=prepare(args.source/d/'world/bundle',args.output/d/'world',d,contract,sop=args.sop)
        for g in info['groups']:
            g['bundle']=prepare(Path(g['bundle']),args.output/d/g['split'],d,contract,sop=args.sop)
        print(json.dumps({'domain':d,'qualified':True,'behavior_cases':len(contract['cases'])}),flush=True)
    summary['passed']=True;save(args.output/'summary.json',summary)


if __name__=='__main__':main()
