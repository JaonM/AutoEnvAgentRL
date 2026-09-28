"""Schema for the task knowledge graph."""

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata


class NodeType(str, Enum):
    """The node types supported by the knowledge graph."""

    SCENE = "scene"
    TASK_TYPE = "task_type"


class SceneRelation(str, Enum):
    """Relations allowed between two scene nodes."""

    HIERARCHY = "hierarchy"
    SAME_EVENT_ELEMENT = "same_event_element"


class TaskType(str, Enum):
    """Task types represented by task type nodes."""

    QA = "QA"
    EVENT = "Event"
    CODING = "Coding"
    CHAT = "Chat"
    RESEARCH = "Research"


@dataclass(frozen=True)
class SceneNode:
    """A node representing a task scenario."""

    name: str
    words: tuple[str, ...] = ()
    expanded: bool = False
    expanded_words: tuple[str, ...] = ()

    @property
    def node_type(self) -> NodeType:
        return NodeType.SCENE


@dataclass(frozen=True)
class TaskTypeNode:
    """A node representing a task type."""

    task_type: TaskType

    @property
    def node_type(self) -> NodeType:
        return NodeType.TASK_TYPE


def normalize_scene_name(name: str) -> str:
    """Return a stable key used to merge equivalent scene names."""

    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


@dataclass(frozen=True)
class KnowledgeGraphSchema:
    """The node and relation vocabulary of the task knowledge graph."""

    node_types: tuple[NodeType, ...] = (NodeType.SCENE, NodeType.TASK_TYPE)
    scene_relations: tuple[SceneRelation, ...] = (
        SceneRelation.HIERARCHY,
        SceneRelation.SAME_EVENT_ELEMENT,
    )
    task_types: tuple[TaskType, ...] = tuple(TaskType)


DEFAULT_KNOWLEDGE_GRAPH_SCHEMA = KnowledgeGraphSchema()
