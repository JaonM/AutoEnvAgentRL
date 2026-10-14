"""Truth proxy for generated tasks: a third, independently written oracle from another provider.

  .venv/bin/python experiments/third_oracle.py BATCH_DIR [--prefix LLM]

Generated requests have no handwritten truth. For each training-ready build, a model from a
different provider (default LLM_*: not the codex model that wrote env.py and oracle.py) writes
oracle.py from brief.md alone (rules, tables, facts, answer schema). The script runs it and
feeds errors back (3 rounds), then compares it with the delivered env.py: final state on the
original world and every approval flip, and answer numbers/ids on the same worlds plus perturbed
ones. Agreement is evidence, not proof: two implementations can still share a misreading.
Evaluation only; never shown to the pipeline. Writes third_oracle.json into the batch folder.
"""
from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments' / 'v3'))

from env_factory.certify import bundle
from env_factory.core import spec  # noqa: E402
from env_factory.core.llm import ChatClient, load_env_file  # noqa: E402
from env_factory.core.sandbox import load_module  # noqa: E402
from env_factory.core.worlds import state_diff, variants  # noqa: E402
from env_factory.runtime import InvalidWorld, run_reference  # noqa: E402
from evaluate_truth import missing_leaves  # noqa: E402

PROMPT = """You are an independent auditor. From the business task below, write a Python module
defining exactly one function:

    def expected(data, facts, now):
        return {{"answer": ..., "final_state": ...}}

- data: dict of table name -> list of row dicts (the initial state; copy before changing).
- facts: the user's private facts (approve_* booleans say whether the user approves that write).
- final_state: every table after all operations a correct handling performs. An operation happens
  only if the rules allow it at that point and the matching approval fact is true.
- answer: exactly the answer schema's shape.
Standard library only, no I/O. Output only the code in one ```python block.

{brief}

Initial data.json:
{data}

facts.json:
{facts}
"""


def code_of(text):
    if '```' in text:
        text = text.split('```python', 1)[-1] if '```python' in text else text.split('```', 1)[1]
        text = text.split('```', 1)[0]
    return text


def write_oracle(client, folder, request, schema):
    path = folder / 'third_oracle.py'
    messages = [{'role': 'user', 'content': PROMPT.format(
        brief=spec.brief(request, schema), data=json.dumps(request['tables'], ensure_ascii=False),
        facts=json.dumps(request['user_facts'], ensure_ascii=False))}]
    for _ in range(3):
        reply = client.chat(messages, temperature=0).get('content') or ''
        path.write_text(code_of(reply))
        try:
            module = load_module(path, role='oracle')
            for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=0):
                result = module.expected(copy.deepcopy(data), copy.deepcopy(facts), request['time']['start'])
                assert set(result) == {'answer', 'final_state'}, 'must return answer and final_state'
            return module
        except Exception:
            messages += [{'role': 'assistant', 'content': reply},
                         {'role': 'user', 'content': 'Running it failed:\n' + traceback.format_exc(limit=3)
                          + '\nFix it and output the whole module again.'}]
    return None


def compare(env, third, request, perturbed=10):
    core, extra, rows = [], [], []
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=perturbed):
        try:
            want = third.expected(copy.deepcopy(data), copy.deepcopy(facts), request['time']['start'])
            got = run_reference(env, data, facts, start=request['time']['start'])
        except InvalidWorld:
            continue
        except Exception as exc:
            rows.append({'world': name, 'error': f'{type(exc).__name__}: {exc}'[:160]})
            (core if not name.startswith('perturb') else extra).append(name)
            continue
        wrong_state = not name.startswith('perturb') and bool(
            state_diff(want['final_state'], got['final_state'], sorted(want['final_state'])))
        missing = missing_leaves(want['answer'], got['answer'])
        if wrong_state or missing:
            (core if not name.startswith('perturb') else extra).append(name)
            rows.append({'world': name, 'state_differs': wrong_state, 'answer_missing': missing[:6]})
    return core, extra, rows[:6]


def main(batch, prefix='LLM'):
    load_env_file(ROOT / '.env')
    client = ChatClient(prefix)
    results = {}
    for build in sorted(Path(batch).glob('*/bundle')):
        loaded = bundle.load_bundle(build, require_current_platform=False)
        if loaded['rubric'] is not None:
            results[build.parent.name] = {'agree': None, 'note': 'rubric task: no expected outcome to compare'}
            continue
        third = write_oracle(client, build.parent, loaded['request'], loaded['schema'])
        if third is None:
            results[build.parent.name] = {'agree': None, 'note': 'third oracle never ran'}
        else:
            core, extra, rows = compare(loaded['env'], third, loaded['request'])
            results[build.parent.name] = {'agree': not core, 'core_disagreements': core,
                                          'perturbed_disagreements': extra, 'details': rows}
        print(build.parent.name, json.dumps(results[build.parent.name], ensure_ascii=False)[:300], flush=True)
    ran = [r for r in results.values() if r['agree'] is not None]
    summary = {'model': client.model, 'builds': len(results), 'ran': len(ran),
               'agree': sum(r['agree'] for r in ran), 'results': results}
    (Path(batch) / 'third_oracle.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}))


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[3:4] if '--prefix' in sys.argv else []))
