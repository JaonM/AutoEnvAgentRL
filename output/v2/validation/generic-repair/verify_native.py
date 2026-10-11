import copy
import json
from pathlib import Path
from env_factory_v2.authoring.author_session import generate_in_session
from env_factory_v2.examples import direct
from env_factory_v2.contracts.expressions import canonical
root = Path('output/v2/validation/generic-repair')
source = direct()
source['answers'][0]['expected']['literal'] = 19
source['scenarios'][0]['answer']['temperature']['literal'] = 19
source['scenarios'][1]['answer']['temperature']['literal'] = 18
request = {'goal': source['task'], 'requirement': 'Preserve the stated public fact and task. Correct acceptance and rejection for the actual temperature.'}
feedback = {'review': {'consistent': False, 'reason': 'The original request and public rule state 18, but the answer contract and self-authored cases accept 19 and reject 18.'}, 'execution_archive': {'trace': 'archived duplicate observation; ' * 100000}}
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
assert 'address_repair' in operations and 'submit_contract' in operations
(root/'native-result.json').write_text(canonical({'passed': True, 'scope': 'independent deterministic fixture; not batch yield or proof of cross-business generalization', 'provenance': provenance, 'input_accounting': accounting, 'operations': operations})+'\n')
print(json.dumps({'passed': True, 'seconds': provenance['seconds'], 'input_accounting': accounting, 'operations': operations}))
