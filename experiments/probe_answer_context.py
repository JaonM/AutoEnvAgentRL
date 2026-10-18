"""Paired extraction diagnostic on fixed transcripts; never overwrites rewards or admits data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from env_factory.certify import bundle
from env_factory.core import answers
from env_factory.core.extract import Extractor
from env_factory.core.llm import ChatClient, load_env_file
from recheck_world_tasks import save


class ContextReader:
    def __init__(self, client, dialogue):
        self.client, self.dialogue = client, dialogue

    def chat(self, messages, **parameters):
        context = ('\n\nAdditional conversational context, for resolving the meaning of the agent\'s explicit '
                   'reported result only:\n' + json.dumps(self.dialogue, ensure_ascii=False) +
                   '\nUse this context only to resolve references or distinguish the reason for an explicitly '
                   'reported result (e.g. an agent acknowledges leaving a record unchanged in response to a '
                   'user refusal). Never treat a customer\'s statement, request, number or identifier as an '
                   'agent-reported result. Do not fill any outcome the assistant did not report. Private facts, '
                   'reference answers and database state are not provided. Keep every other extraction rule.')
        return self.client.chat([{**messages[0], 'content': messages[0]['content'] + context}, *messages[1:]], **parameters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--tasks', nargs='+', required=True, help='domain:tNNN')
    args = parser.parse_args()
    load_env_file()
    jobs = []
    for task in args.tasks:
        domain, index = task.split(':')
        folder = args.source / domain / 'tasks' / index
        record = json.loads((folder / 'natural.json').read_text())
        for i, episode in enumerate(record['raw_episodes']):
            jobs.append((task, i, episode, folder))

    def one(job):
        task, index, episode, folder = job
        target = args.output / task.replace(':', '-') / f'raw-{index}.json'
        if target.exists():
            return json.loads(target.read_text())
        loaded = bundle.load_bundle(folder / 'bundle')
        r = loaded['request']
        dialogue = [{'role': m['role'], 'content': m['content']} for m in episode['messages']
                    if m['role'] in ('user', 'assistant') and m.get('content')]
        speech = [m['content'] for m in dialogue if m['role'] == 'assistant']
        expected = loaded['verifier'].expected(r['tables'], r['user_facts'], r['time']['start'])['answer']
        variants = {}
        for name in ('speech_only', 'with_context'):
            reader = ChatClient('JUDGE')
            if name == 'with_context':
                reader = ContextReader(reader, dialogue)
            extracted = Extractor(reader).extract(loaded['schema'], speech)
            fields = answers.compare(loaded['schema'], expected, extracted, answers.known_strings(r))
            variants[name] = {'answer': extracted, 'fields': fields, 'fields_passed': all(fields.values())}
        row = {'task': task, 'episode': index, 'original_answer': episode.get('answer'),
               'original_fields': episode.get('fields'), 'variants': variants,
               'diagnostic_only': True, 'training_ready': False,
               'limits': 'Targeted failure/success controls; no change to saved business scores.'}
        save(target, row)
        return row

    rows = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            rows.append(future.result())
            save(args.output / 'summary.json', {'results': rows, 'completed': len(rows), 'total': len(jobs),
                                               'training_ready': False, 'diagnostic_only': True})
            print(json.dumps({'task': rows[-1]['task'], 'episode': rows[-1]['episode'],
                              **{k: v['fields_passed'] for k, v in rows[-1]['variants'].items()}}), flush=True)


if __name__ == '__main__':
    main()
