from .graph.knowledge_graph import (
    DEFAULT_KNOWLEDGE_GRAPH_SCHEMA,
    KnowledgeGraphSchema,
    NodeType,
    SceneNode,
    SceneRelation,
    TaskType,
    TaskTypeNode,
    normalize_scene_name,
)
from .graph.graph_builder import KnowledgeGraphBuilder, Neo4jGraphStore, SceneEdge
from .graph.graph_expander import (
    DEFAULT_SCENE_SEEDS,
    GraphExpansionError,
    GraphExpansionConfig,
    SceneWordGroup,
    SearchExpansion,
    SeedGraphExpander,
)
from .graph.scene_relation import (
    LLMSceneRelationExtractor,
    SceneRelationExtraction,
    SceneRelationExtractionError,
)
from .llm import LLMClient, LLMError, LLMResponse
from .graph.wikipedia import WikipediaClient, WikipediaError, WikipediaResponse, WikipediaResult
from .graph.wikipedia_dump import LocalWikipediaClient, WikipediaDumpIndexer
from .tasks.task import Task
from .tasks.task_spec import TaskSpecError, compile_task_spec, validate_task_spec
from .generation.task_generator import TaskGenerationError, TaskGenerator
from .task_pipeline import PipelineGenerationError, TaskGenerationPipeline

__all__ = [
    "DEFAULT_KNOWLEDGE_GRAPH_SCHEMA",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "KnowledgeGraphSchema",
    "KnowledgeGraphBuilder",
    "Neo4jGraphStore",
    "NodeType",
    "SceneNode",
    "SceneRelation",
    "SceneEdge",
    "DEFAULT_SCENE_SEEDS",
    "GraphExpansionError",
    "GraphExpansionConfig",
    "SceneWordGroup",
    "SearchExpansion",
    "SeedGraphExpander",
    "LLMSceneRelationExtractor",
    "SceneRelationExtraction",
    "SceneRelationExtractionError",
    "normalize_scene_name",
    "WikipediaClient",
    "WikipediaError",
    "WikipediaResponse",
    "WikipediaResult",
    "LocalWikipediaClient",
    "WikipediaDumpIndexer",
    "Task",
    "TaskSpecError",
    "compile_task_spec",
    "validate_task_spec",
    "TaskGenerator",
    "TaskGenerationError",
    "PipelineGenerationError",
    "TaskGenerationPipeline",
    "TaskType",
    "TaskTypeNode",
]

# Preserve the original module import paths for existing integrations while
# the implementation lives in domain packages.
from importlib import import_module as _import_module
from sys import modules as _modules

_legacy_modules = {
    "graph": (
        "knowledge_graph", "graph_builder", "graph_expander", "scene_relation",
        "wikipedia", "wikipedia_dump",
    ),
    "tasks": (
        "task", "task_spec", "task_quality", "task_routing",
        "task_similarity", "task_portability",
    ),
    "evidence": (
        "certification_policy", "container_provenance", "data_governance",
        "execution_provenance", "experiment_contract", "generation_provenance",
        "material_artifacts", "material_attestation", "material_consumer",
        "material_privacy", "model_response_provenance", "model_roles",
        "portable_metadata", "production_preflight", "runtime_provenance",
        "trajectory_schema",
    ),
    "generation": ("task_generator",),
}
for _package, _names in _legacy_modules.items():
    for _name in _names:
        _module = _import_module(f".{_package}.{_name}", __name__)
        _modules[f"{__name__}.{_name}"] = _module
        globals()[_name] = _module
del _import_module, _modules, _legacy_modules, _package, _names, _name, _module
