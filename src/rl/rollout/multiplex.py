"""Bounded cross-group scheduling on one policy and one pool of isolated environments.

Groups share execution slots and decode forwards, never reward baselines. A wave
pins one policy version and drains before loading new weights.
"""
import json
import time
from pathlib import Path

from rl.errors import EnvironmentProtocolError
from .scheduler import rollout_steps


class TaskConnection:
    def __init__(self, connection, sandbox, factory):
        self.connection, self.sandbox, self.factory = connection, sandbox, factory

    def send(self, value):
        command, argument = value
        self.connection.send(('task', {'sandbox': self.sandbox, 'factory': self.factory,
                                       'command': command, 'argument': argument}))

    def poll(self):
        return self.connection.poll()

    def recv(self):
        return self.connection.recv()


class GroupPool:
    def __init__(self, pool, owners, key, sandbox, factory):
        self.owners, self.key = owners, key
        self.workers = [(TaskConnection(c, sandbox, factory), p) for c, p in pool.workers]

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
    """Yield (group key, episodes) as each complete group is ready; preserve journals on failure.

    Each spec supplies key, sandbox, config, seed, continuation and optional observe.
    Caller must pin the common policy version for the entire iterator lifetime.
    """
    owners, iterators, decoders = {}, {}, {}
    decoder = policy.batch_decoder() if hasattr(policy, 'batch_decoder') else None
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
                         spec['config'].get('environment_factory', 'rl.environment:SandboxEpisode'))
        local_decoder = GroupDecoder(decoder, key) if decoder else None
        decoders[key] = local_decoder
        iterators[key] = rollout_steps(policy, view, spec['config'], spec['seed'], stop=stop, notify=notify,
                                       continuation=continuation, observe=spec.get('observe'), decoder=local_decoder)
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
