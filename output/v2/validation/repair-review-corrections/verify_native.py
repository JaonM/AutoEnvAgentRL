import copy
import json
from pathlib import Path
from env_factory_v2.authoring.author_session import generate_in_session
from env_factory_v2.examples import direct
from env_factory_v2.contracts.expressions import canonical
root = Path('output/v2/validation/repair-review-corrections')
source = direct()
source['answers'][0]['expected']['literal'] = 19
source['scenarios'][0]['answer']['temperature']['literal'] = 19
source['scenarios'][1]['answer']['temperature']['literal'] = 18
request = {'goal': source['task'], 'requirement': 'Preserve the stated public fact and task. Correct acceptance and rejection for the actual temperature.'}
feedback = {'requirement_review': {'consistent': False, 'reason': 'The original request and public rule state 18, but the answer contract and self-authored cases accept 19 and reject 18.'}, 'execution_archive': {'trace': 'archived duplicate observation; ' * 100000}}
result, provenance = generate_in_session(request, evidence_root=root/'sessions', previous_source=copy.deepcopy(source), feedback=feedback, timeout=300, idle_timeout=180)
assert result['task'] == source['task']
assert result['rules'] == source['rules']
assert result['answers'][0]['expected'] == {'literal': 18}
assert result['scenarios'][0]['answer']['temperature'] == {'literal': 18}
assert result['scenarios'][1]['answer']['temperature'] != {'literal': 18}
folder = root/'sessions'/provenance['response_id']
accounting = json.loads((folder/'input-accounting.json').read_text())
assert accounting['characters'] < accounting['limit']
assert accounting['context_characters'] > 2500000
operations = [json.loads(line)['tool'] for line in (folder/'workbench/operations.jsonl').read_text().splitlines()]
assert 'address_repair' not in operations and 'submit_contract' in operations
(root/'native-result.json').write_text(canonical({'passed': True, 'scope': 'independent deterministic fixture; not batch yield or proof of cross-business generalization', 'provenance': provenance, 'input_accounting': accounting, 'operations': operations})+'\n')
print(json.dumps({'passed': True, 'seconds': provenance['seconds'], 'input_accounting': accounting, 'operations': operations}))

# Simulate interruption after a corrected draft was saved but before submission.
from env_factory_v2.authoring.author_session import incomplete_draft
interrupted = root/'interrupted'
(interrupted/'workbench').mkdir(parents=True, exist_ok=True)
(interrupted/'context.json').write_text((folder/'context.json').read_text())
(interrupted/'workbench/draft.json').write_text(canonical(result))
recovery = incomplete_draft(interrupted, 'simulated interruption').recovery
recovered = recovery.pop('previous_source')
resume_feedback = {**feedback, 'author_recovery': recovery, 'failed_phase': 'generation', 'error': 'interrupted'}
resumed, resume_provenance = generate_in_session(request, evidence_root=root/'resume-sessions', previous_source=recovered, feedback=resume_feedback, timeout=300, idle_timeout=180)
assert resumed == result
(root/'resume-result.json').write_text(canonical({'passed': True, 'unchanged_corrected_draft': resumed == result, 'provenance': resume_provenance})+'\n')
print(json.dumps({'resume_passed': True, 'seconds': resume_provenance['seconds']}))
