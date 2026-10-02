"""Bounded, checksummed replay of whole trajectory groups with real behavior LPs."""
from copy import deepcopy
from pathlib import Path
import json
import math
import random

from .checkpoint import sha256


def validate_group(group, tasks):
    if group.get('schema_version') not in {2, 3}:
        raise ValueError('replay requires trajectory schema version 2')
    if group.get('schema_version') == 3:
        from .contracts import validate_contract
        validate_contract(group)
    task = next((task for task in tasks if task.id == group.get('task_id')), None)
    if task is None or task.split != 'train' or group.get('task_identity') != task.identity:
        raise ValueError('replay task identity is absent, changed, or belongs to eval')
    if not isinstance(group.get('policy_version'), int) or group['policy_version'] < 0:
        raise ValueError('invalid behavior policy version')
    episodes = group.get('episodes')
    if not episodes or len({e['seed'] for e in episodes}) != 1:
        raise ValueError('replay group must share initial task/seed')
    for episode in episodes:
        if episode.get('task_id') != task.id or not episode.get('actions'):
            raise ValueError('mixed-task or empty replay episode')
        if not math.isfinite(episode['final_reward']):
            raise ValueError('non-finite replay reward')
        if not episode['terminated'] and not episode.get('bootstrap_prompt'):
            raise ValueError('truncated replay requires its next policy context')
        for action in episode['actions']:
            tokens, prompt, logp = action.get('tokens'), action.get('prompt'), action.get('old_logp')
            if not tokens or not prompt or not isinstance(logp, list) or len(tokens) != len(logp):
                raise ValueError('missing/alignment error in behavior token log-probabilities')
            if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in prompt + tokens):
                raise ValueError('invalid replay token IDs')
            if not all(math.isfinite(x) and x <= 1e-5 for x in logp):
                raise ValueError('invalid behavior log-probabilities')
    return group


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


class ReplayBuffer:
    def __init__(self, *, capacity=32, max_uses=4, max_age=32, seed=42, state=None):
        if min(capacity, max_uses, max_age) < 1:
            raise ValueError('positive replay capacity/use/age limits required')
        self.capacity, self.max_uses, self.max_age = capacity, max_uses, max_age
        self.entries = []
        self.rng = random.Random(seed)
        if state:
            self.entries = deepcopy(state['entries'])
            self.rng.setstate(_tuple_tree(state['rng']))
            if len(self.entries) > capacity:
                raise ValueError('checkpoint replay exceeds configured capacity')

    def add(self, path, group, current_version, *, imported=False):
        path = Path(path).resolve()
        digest = sha256(path)
        if any(e['sha256'] == digest for e in self.entries):
            return False
        self.entries.append({'path':str(path), 'sha256':digest, 'task_id':group['task_id'],
                             'behavior_version':group['policy_version'],
                             'age_origin':current_version if imported else group['policy_version'],
                             'imported':imported, 'uses':0})
        self.entries = self.entries[-self.capacity:]
        return True

    def eligible(self, current_version):
        return [e for e in self.entries if e['uses'] < self.max_uses and
                0 <= current_version - e['age_origin'] <= self.max_age]

    def sample(self, current_version, tasks, *, exclude_paths=(), task_id=None):
        candidates = [e for e in self.eligible(current_version) if e['path'] not in exclude_paths and (task_id is None or e['task_id'] == task_id)]
        if not candidates:
            return None
        # Choose a task first, so faster environments do not dominate replay.
        weights = {t.id:t.weight for t in tasks if t.split == 'train'}
        available = sorted({e['task_id'] for e in candidates})
        task_id = self.rng.choices(available, weights=[weights[x] for x in available], k=1)[0]
        entry = self.rng.choice([e for e in candidates if e['task_id'] == task_id])
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('replay artifact changed after ingestion')
        group = validate_group(json.loads(Path(entry['path']).read_text()), tasks)
        entry['uses'] += 1  # Rejected numerical/drift batches also consume a bounded attempt.
        return group, deepcopy(entry)

    def state(self):
        return {'entries':deepcopy(self.entries), 'rng':self.rng.getstate()}


def importance_diagnostics(target_logp, behavior_logp, epsilon=.2):
    if not 0 < epsilon < 1:
        raise ValueError('epsilon must be in (0,1)')
    if not target_logp or len(target_logp) != len(behavior_logp):
        raise ValueError('importance diagnostics require aligned probabilities')
    delta = [t-b for t,b in zip(target_logp,behavior_logp)]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError('non-finite importance ratios')
    maximum = max(delta)
    scaled = [math.exp(x-maximum) for x in delta]
    ess = sum(scaled)**2 / sum(x*x for x in scaled) / len(scaled)
    return {'effective_sample_fraction':ess, 'max_abs_log_ratio':max(abs(x) for x in delta),
            'importance_clip_fraction':sum(x < math.log1p(-epsilon) or x > math.log1p(epsilon) for x in delta)/len(delta)}


def import_replay(source, buffer, tasks, policy_identity, *, current_version=0):
    """Import only identified schema-v2 groups; source artifacts must remain available."""
    source = Path(source).resolve()
    provenance = json.loads((source / 'execution_provenance.json').read_text())
    if provenance.get('policy_identity') != policy_identity:
        raise ValueError('replay source model/tokenizer/tuning/temperature identity mismatch')
    count = 0
    for path in sorted(source.glob('group-*.json')):
        group = validate_group(json.loads(path.read_text()), tasks)
        if group.get('policy_identity') != policy_identity or group.get('temperature') != policy_identity['temperature']:
            raise ValueError('replay group policy identity mismatch')
        count += buffer.add(path, group, current_version, imported=True)
    if not count:
        raise ValueError('replay source contains no usable identified groups')
    return count


def validate_ppo_behavior(group):
    """PPO replay requires genuine stored critic values, never synthesized zeros."""
    for episode in group['episodes']:
        bootstrap = episode.get('bootstrap')
        if not isinstance(bootstrap, (int, float)) or not math.isfinite(bootstrap):
            raise ValueError('PPO replay requires finite behavior bootstrap')
        for action in episode['actions']:
            values = action.get('old_values')
            if not isinstance(values, list) or len(values) != len(action['tokens']):
                raise ValueError('PPO replay requires aligned behavior critic values')
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError('PPO replay requires finite behavior critic values')
    return group
