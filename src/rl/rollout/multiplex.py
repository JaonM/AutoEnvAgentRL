"""Bounded cross-group scheduling on one policy and one pool of isolated environments.

Groups share execution slots and decode forwards, never reward baselines. A wave
pins one policy version and drains before loading new weights.
"""
import json
import time
from pathlib import Path

from rl.errors import EnvironmentProtocolError
from .scheduler import rollout_steps
from rl.checkpointing.checkpoint import atomic_json


class TaskConnection:
    def __init__(self, connection, sandbox, factory, defer_grading=False):
        self.connection, self.sandbox, self.factory = connection, sandbox, factory
        self.defer_grading = defer_grading

    def send(self, value):
        command, argument = value
        self.connection.send(('task', {'sandbox': self.sandbox, 'factory': self.factory,
                                       'command': command, 'argument': argument, 'defer_grading': self.defer_grading}))

    def poll(self):
        return self.connection.poll()

    def recv(self):
        return self.connection.recv()


class GroupPool:
    def __init__(self, pool, owners, key, sandbox, factory, defer_grading=False):
        self.owners, self.key = owners, key
        self.workers = [(TaskConnection(c, sandbox, factory, defer_grading), p) for c, p in pool.workers]

    def acquire(self, slot):
        if slot in self.owners and self.owners[slot] != self.key:
            return False
        self.owners[slot] = self.key
        return True

    def release(self, slot):
        assert self.owners.pop(slot) == self.key


class GroupDecoder:
    def __init__(self, decoder, key):
        self.decoder, self.key, self.finished = decoder, key, {}

    @property
    def requests(self):
        return {**{slot: value for (group, slot), value in self.decoder.requests.items() if group == self.key},
                **self.finished}

    def add(self, slot, *args, **kwargs):
        self.decoder.add((self.key, slot), *args, **kwargs)

    def tick(self):
        results, self.finished = self.finished, {}
        return results


def sample_groups(policy, pool, groups, *, stop=None, notify=None):
    from .scoring import ScoringPool
    config = groups[0]['config'] if groups else {}
    workers = config.get('scoring_workers', 0)
    scoring = ScoringPool(workers, config.get('scoring_queue_size', 8)) if workers else None
    try:
        yield from _sample_groups(policy, pool, groups, stop=stop, notify=notify, scoring=scoring)
    finally:
        if scoring is not None:
            scoring.close()


def _sample_groups(policy, pool, groups, *, stop=None, notify=None, scoring=None):
    """Yield (group key, episodes) as each complete group is ready; preserve journals on failure.

    Each spec supplies key, sandbox, config, seed, continuation and optional observe.
    Caller must pin the common policy version for the entire iterator lifetime.
    """
    owners, iterators, decoders = {}, {}, {}
    decoder = policy.batch_decoder() if hasattr(policy, 'batch_decoder') else None
    # Episode journals are keyed by episode identity, not physical slot. Groups may
    # have last run on different workers with overlapping local slot numbers.
    journals = []
    for spec in groups:
        path = Path(spec['continuation']) / 'progress.json' if spec.get('continuation') else None
        if path and path.exists():
            journals.append((path, json.loads(path.read_text())))
    required = sum(len(saved['active']) for _, saved in journals)
    if required > len(pool.workers):
        raise ValueError('continuations exceed execution slot capacity; admit fewer groups')
    next_slot = 0
    for path, saved in journals:
        remapped = {}
        for old_slot in sorted(saved['active'], key=int):
            remapped[str(next_slot)] = saved['active'][old_slot]
            next_slot += 1
        saved['active'] = remapped
        atomic_json(path, saved)
    for spec in groups:
        key = spec['key']
        if key in iterators:
            raise ValueError('duplicate group identity')
        continuation = spec.get('continuation')
        journal = Path(continuation) / 'progress.json' if continuation else None
        if journal and journal.exists():
            for slot in json.loads(journal.read_text())['active']:
                slot = int(slot)
                if slot in owners or slot >= len(pool.workers):
                    raise ValueError('conflicting continuation slot ownership')
                owners[slot] = key
        view = GroupPool(pool, owners, key, spec['sandbox'],
                         spec['config'].get('environment_factory', 'rl.environment:SandboxEpisode'), scoring is not None)
        local_decoder = GroupDecoder(decoder, key) if decoder else None
        decoders[key] = local_decoder
        iterators[key] = rollout_steps(policy, view, spec['config'], spec['seed'], stop=stop, notify=notify,
                                       continuation=continuation, observe=spec.get('observe'), decoder=local_decoder,
                                       scoring=scoring, sandbox=spec['sandbox'])
    order = list(iterators)
    try:
        while iterators:
            progressed = False
            for key in order:
                if key not in iterators:
                    continue
                try:
                    progressed = next(iterators[key]) or progressed
                except StopIteration as done:
                    del iterators[key]
                    yield key, done.value
                except EnvironmentProtocolError as error:
                    error.group_key = key
                    raise
            if decoder is not None and decoder.requests:
                progressed = True
                if notify:
                    notify('cross-group-policy-decode')
                for (key, slot), sample in decoder.tick().items():
                    decoders[key].finished[slot] = sample
            order = order[1:] + order[:1]  # round-robin admission of newly freed slots
            if not progressed:
                time.sleep(.01)
    finally:
        for iterator in iterators.values():
            iterator.close()
