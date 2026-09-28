"""Task model definitions."""

from dataclasses import dataclass
from typing import Any

from env_factory.graph.knowledge_graph import TaskType


@dataclass
class Task:
    """A task description, its environment, and its observation metrics."""

    desc: str
    env: list[Any]
    metrics: list[Any]
    task_type: TaskType = TaskType.EVENT
    task_intent: str = "query"
    complexity: str = "standard"
    artifacts: dict[str, Any] | None = None
