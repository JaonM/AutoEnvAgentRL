"""Backend-independent agent segments and environment runtime contract."""
from dataclasses import dataclass
import importlib
from typing import Protocol


class AgentRuntime(Protocol):
    messages: list
    terminated: bool
    def reset(self, seed: int): ...
    def step(self, action: str): ...
    def finish(self) -> dict: ...
    def close(self): ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict): ...


def create_runtime(factory: str, sandbox: str) -> AgentRuntime:
    module, name = factory.split(':', 1)
    runtime = getattr(importlib.import_module(module), name)(sandbox)
    if any(not callable(getattr(runtime, method, None)) for method in ('reset', 'step', 'finish', 'close')):
        raise TypeError('runtime factory must implement reset/step/finish/close')
    return runtime


@dataclass(frozen=True)
class TrainingSegment:
    sandbox_id: str
    group_id: str
    rollout_id: str
    segment_id: str
    policy_version: int
    prompt: list[int]
    tokens: list[int]
    behavior_logprobs: list[float]
    loss_mask: list[int]

    @property
    def input_ids(self):
        return self.prompt + self.tokens

    @property
    def response_length(self):
        return len(self.tokens)

    @classmethod
    def from_action(cls, action):
        value = cls(*(action[key] for key in ('sandbox_id', 'group_id', 'rollout_id', 'segment_id', 'policy_version')),
                    action['prompt'], action['tokens'], action['old_logp'], action['loss_mask'])
        if not value.prompt or not value.tokens or len(value.tokens) != len(value.behavior_logprobs):
            raise ValueError('segment token/logprob alignment error')
        if len(value.loss_mask) != len(value.tokens) or any(x not in (0, 1) for x in value.loss_mask) or not any(value.loss_mask):
            raise ValueError('segment requires an aligned, nonempty assistant loss mask')
        if any(not isinstance(x, str) or not x for x in (value.sandbox_id, value.group_id, value.rollout_id, value.segment_id)):
            raise ValueError('segment identities must be nonempty strings')
        return value


def attach_contract(group):
    """Assign stable identities without re-tokenizing any sampled output."""
    group['schema_version'] = 3
    group_id = group.setdefault('group_id', f"{group['task_id']}:{group.get('dataset_index', group.get('group_index', 0))}:{group['policy_version']}")
    for index, episode in enumerate(group['episodes']):
        rollout_id = f'{group_id}/rollout-{index}'
        episode.update(sandbox_id=group['task_id'], group_id=group_id, rollout_id=rollout_id)
        for offset, action in enumerate(episode['actions']):
            action.update(sandbox_id=group['task_id'], group_id=group_id, rollout_id=rollout_id,
                          segment_id=f'{rollout_id}/segment-{offset}', policy_version=group['policy_version'])
            action.setdefault('loss_mask', [1] * len(action['tokens']))
            TrainingSegment.from_action(action)
    return group


def validate_contract(group):
    seen = set()
    for episode in group['episodes']:
        for action in episode['actions']:
            segment = TrainingSegment.from_action(action)
            if (segment.sandbox_id != group['task_id'] or segment.group_id != group['group_id']
                    or segment.rollout_id != episode['rollout_id'] or segment.policy_version != group['policy_version']
                    or segment.segment_id in seen):
                raise ValueError('segment identity does not match its rollout group')
            seen.add(segment.segment_id)
