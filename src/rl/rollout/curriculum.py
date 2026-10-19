"""Reverse curriculum (R2): hard tasks start from the late states of recorded successful episodes.

A task the policy almost never solves from the beginning gives GRPO all-zero groups and no gradient. With a start
library (experiments/start_states.py: exact snapshots after each turn of successful episodes, replayed with their
recorded user replies), such a task's groups start from a state with only the last part of a successful episode left;
once two consecutive groups succeed at least `threshold` of the time there, the start moves earlier, until the task
plays from the beginning like any other. All episodes of a group share one start, so the group comparison stays fair;
the prefix is context only (its actions are the recording's, never trained).

Levels give the share of the recorded episode left to play: 0.25, 0.5, 0.75, then 1.0 (the normal reset). State is a
JSON file shared by the rollout workers under a file lock.
"""
from __future__ import annotations

import fcntl
import json
import random
from pathlib import Path

LEVELS = (0.25, 0.5, 0.75, 1.0)


class Curriculum:
    def __init__(self, library, state, *, threshold=0.6, window=2, seed=0, minimum=8, minimum_group=2):
        self.library = json.loads(Path(library).read_text()) if library else {}
        self.state_path = Path(state)
        if not 0 <= threshold <= 1 or min(window, minimum, minimum_group) < 1:
            raise ValueError("invalid curriculum evidence thresholds")
        self.threshold, self.window = threshold, window
        self.minimum, self.minimum_group = minimum, minimum_group
        self.rng = random.Random(seed)

    def _locked(self, update=None):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path.with_suffix('.lock'), 'a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            if update is not None:
                update(state)
                tmp = self.state_path.with_suffix('.tmp')
                tmp.write_text(json.dumps(state, indent=1))
                tmp.replace(self.state_path)
            return state

    def start(self, task_id):
        """{'snapshot', 'prefix_steps', 'level'} for a curriculum task below the last level, else None."""
        recordings = self.library.get(task_id)
        if not recordings:
            return None
        level = self._locked().get(task_id, {}).get('level', 0)
        share = LEVELS[min(level, len(LEVELS) - 1)]
        if share >= 1.0:
            return None
        recording = self.rng.choice(recordings)
        turns = len(recording['snapshots']) + 1  # a snapshot after every turn but the last
        prefix = min(len(recording['snapshots']), max(1, round(turns * (1 - share))))
        return {'snapshot': recording['snapshots'][prefix - 1], 'prefix_steps': prefix, 'level': level}

    def record(self, task_id, level, episodes, *, group_id=None):
        """Update the task's level from a finished group started at `level`."""
        trainable = [e for e in episodes if e and e.get('trainable', True) is not False]
        if len(trainable) < self.minimum_group:
            return None
        rate = sum(bool((e.get('score') or {}).get('success')) for e in trainable) / len(trainable)

        def update(state):
            entry = state.setdefault(task_id, {'level': 0, 'rates': []})
            if entry['level'] != level:
                return  # another worker already moved the task on
            evidence = entry.setdefault('evidence', [])
            if group_id is not None and any(item.get('group_id') == group_id for item in evidence):
                return  # a replayed completed journal must not advance the curriculum twice
            evidence.append({'rate': rate, 'valid_episodes': len(trainable), 'group_id': group_id})
            # Retain enough groups to satisfy both the window and sample-count threshold.
            limit = max(self.window, (self.minimum + self.minimum_group - 1) // self.minimum_group)
            del evidence[:-limit]
            while (len(evidence) > self.window
                   and sum(item['valid_episodes'] for item in evidence[1:]) >= self.minimum):
                evidence.pop(0)
            entry['rates'] = [item['rate'] for item in evidence]
            if (level < len(LEVELS) - 1 and len(evidence) >= self.window
                    and sum(item['valid_episodes'] for item in evidence) >= self.minimum
                    and min(item['rate'] for item in evidence) >= self.threshold):
                entry['level'] += 1
                entry['rates'], entry['evidence'] = [], []
        return self._locked(update)[task_id]['level']
