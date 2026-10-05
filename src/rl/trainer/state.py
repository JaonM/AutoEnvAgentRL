"""Mutable progress owned by one trainer; publication and durable commit are distinct."""

from dataclasses import dataclass, field


@dataclass
class TrainingState:
    updates: int = 0
    groups: int = 0
    dropped: int = 0
    optimizer_step: int = 0
    dataset_cursor: int = 0
    committed_version: int = 0
    checkpoint_step: int = 0
    published_step: int = 0
    inflight: dict = field(default_factory=dict)
    skipped_jobs: dict = field(default_factory=dict)
    eval_state: dict = field(default_factory=dict)
    rollout_positions: dict = field(default_factory=dict)
    rollout_attempts: dict = field(default_factory=dict)
    updated_jobs: set = field(default_factory=set)
    consumed_jobs: set = field(default_factory=set)
    initial_digest: str | None = None
    initial_policy_digest: str | None = None
    initial_effective_digest: str | None = None
    previous_seconds: float = 0.0
