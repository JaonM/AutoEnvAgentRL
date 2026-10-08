"""Generate long-horizon task descriptions from Scene graph paths."""

import logging
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from env_factory.graph.graph_builder import Neo4jGraphStore
from env_factory.graph.knowledge_graph import SceneNode, TaskType
from env_factory.tasks.task import Task
from env_factory.generation.contract_compiler import (
    HIGH_STAKES_CHEMICAL_MARKERS,
    HIGH_STAKES_MARKERS,
)
from env_factory.tasks.task_routing import training_contract

logger = logging.getLogger(__name__)


class TaskGenerationError(ValueError):
    """Raised when a graph path or LLM task response is invalid."""



class TaskGenerator:
    """Sample event-element paths and turn their keywords into tasks."""

    INTENTS = (
        "query",
        "explain",
        "compare",
        "recommend",
        "diagnose",
        "modify",
        "execute",
        "plan",
        "summarize",
        "create",
        "extract",
        "classify",
        "validate",
        "audit",
        "calculate",
        "estimate",
        "schedule",
        "monitor",
        "troubleshoot",
        "transform",
        "decide",
        "simulate",
    )

    STYLES = (
        "直接请求",
        "带背景说明",
        "带约束条件",
        "遇到问题寻求建议",
        "多步骤委托",
        "临时起意的生活请求",
        "专业人士委托",
        "日常对话",
    )
    LOW_INFORMATION_KEYWORDS = frozenset({
        "内容", "数据", "信息", "资料", "问题", "情况", "事项", "其他",
        "标准", "材质", "材料", "产品", "服务", "项目", "活动", "对象",
        "方法", "方式", "类型", "类别", "名称", "描述", "状态", "结果",
    })
    # Graph paths currently carry names and keywords, without source URLs.
    # These topics inevitably fail the later sourced-domain gate and should be
    # resampled before any model calls are spent on a task candidate.
    UNSOURCED_DOMAIN_KEYWORDS = (
        "宪法", "抗酸剂", "胃酸", "氢氧化镁",
    )
    MAX_TASK_KEYWORDS = 3
    def __init__(
        self,
        store: Neo4jGraphStore,
        *,
        user_script_count: int = 3,
        available_environment_modes: tuple[str, ...] | None = None,
        code_agent_timeout: float = 600,
        code_agent: str = "codex",
        code_agent_model: str = "gpt-6.1-sol",
        language: str = "zh-CN",
    ) -> None:
        self.store = store
        self.available_environment_modes = available_environment_modes
        if user_script_count <= 0:
            raise ValueError("user_script_count must be positive")
        if not 0 < code_agent_timeout < float("inf"):
            raise ValueError("code_agent_timeout must be positive and finite")
        if not code_agent_model.strip() or not language.strip():
            raise ValueError("code_agent_model and language must not be empty")
        if code_agent not in {"codex", "claude", "opencode"}:
            raise ValueError("Unsupported code agent")
        self.code_agent = code_agent
        self.code_agent_model = code_agent_model
        self.language = language
        self.code_agent_timeout = code_agent_timeout
        self.script_count = user_script_count

    def generate(
        self,
        hops: int = 3,
        task_type: TaskType | str | None = None,
        task_style: str | None = None,
        artifact_dir: str | Path | None = None,
        task_intent: str | None = None,
        training_category: str = "multi_step_agentic",
        seed: int | None = None,
        realism_requirements: dict | None = None,
        difficulty_profile: str = 'standard',
    ) -> Task:
        """Generate one task from a graph path with a uniformly sampled 0..hops length."""

        rng = random.Random(seed) if seed is not None else random
        from .code_agent import generate as generate_with_agent
        import tempfile
        if not 0 <= hops <= 20:
            raise TaskGenerationError("CODE_AGENT_GRAPH: hops upper bound must be between 0 and 20")
        max_hops = hops
        hops = rng.randint(0, max_hops)
        path, keywords = (), ()
        for _ in range(4):
            path = self.store.random_scene_event_path(hops, attempts=3, rng=rng)
            if len(path) == hops + 1:
                keywords = self._keywords(path, rng=rng)
                if keywords:
                    break
        if len(path) != hops + 1 or not keywords:
            raise TaskGenerationError("CODE_AGENT_GRAPH: no usable path of the sampled length")
        selected_type = self._select_task_type(task_type, rng=rng)
        request = {"version": "1.0", "language": self.language, "code_agent_model": self.code_agent_model, "code_agent": self.code_agent, "training_category": training_category,
            "realism_requirements": realism_requirements or {"version": "1.0", "mechanisms": []},
            "difficulty_profile": difficulty_profile,
            "training_contract": training_contract(training_category),
            "task_type": selected_type.value, "task_style": task_style,
            "task_intent": task_intent, "seed": seed,
            "available_environment_modes": list(self.available_environment_modes or
                ("stateless", "reference_data", "stateful")),
            "graph_context": {"source": "neo4j_scene_path", "hops": hops, "max_hops": max_hops,
                "nodes": [node.name for node in path], "keywords": list(keywords),
                "relation": "SAME_EVENT_ELEMENT"}}
        from .difficulty_profiles import augment_realism, requirements as difficulty_requirements
        request['difficulty_requirements'] = difficulty_requirements(difficulty_profile)
        request['realism_requirements'] = augment_realism(request)
        artifacts = generate_with_agent(request=request,
            artifact_dir=Path(artifact_dir) if artifact_dir else Path(tempfile.mkdtemp(prefix="envfactory-agent-")),
            script_count=self.script_count, timeout=self.code_agent_timeout)
        return Task(desc=artifacts["task"], env=artifacts["environment"], metrics=artifacts["metrics"],
            task_type=selected_type, task_intent=artifacts["task_intent"],
            complexity=artifacts["complexity"], artifacts=artifacts)

    def resume(self, checkpoint: str | Path, *, artifact_dir: str | Path) -> Task:
        """Continue a fixed design without sampling another graph path or task type."""
        from .authoring_checkpoint import load_checkpoint
        from .code_agent import generate as generate_with_agent
        saved = load_checkpoint(Path(checkpoint))
        request = saved['request']
        if (request.get('code_agent', 'codex') != self.code_agent
                or request.get('code_agent_model') != self.code_agent_model
                or request.get('language', 'zh-CN') != self.language
                or request.get('user_script_count') != self.script_count):
            raise TaskGenerationError('AUTHOR_CHECKPOINT: generator configuration changed')
        artifacts = generate_with_agent(request=request, artifact_dir=Path(artifact_dir),
            script_count=self.script_count, timeout=self.code_agent_timeout, resume_from=Path(checkpoint))
        return Task(desc=artifacts['task'], env=artifacts['environment'], metrics=artifacts['metrics'],
            task_type=TaskType(request['task_type']), task_intent=artifacts['task_intent'],
            complexity=artifacts['complexity'], artifacts=artifacts)

    @staticmethod
    def _select_task_type(
        task_type: TaskType | str | None, *, rng: random.Random | None = None
    ) -> TaskType:
        random_source = rng or random
        if task_type is None:
            return random_source.choice(tuple(TaskType))
        if isinstance(task_type, TaskType):
            return task_type
        try:
            if not isinstance(task_type, str):
                raise ValueError
            values = tuple(value.strip() for value in task_type.split(","))
            if not values or any(not value for value in values):
                raise ValueError
            selected_types = tuple(dict.fromkeys(TaskType(value) for value in values))
            return random_source.choice(selected_types)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in TaskType)
            raise ValueError(f"unsupported task_type {task_type!r}; expected comma-separated values from: {allowed}") from exc

    @staticmethod
    def _keywords(
        path: tuple[SceneNode, ...], *, rng: random.Random | None = None
    ) -> tuple[str, ...]:
        random_source = rng or random
        keywords: list[str] = []
        for scene in path:
            candidates = [
                normalized for word in (scene.name, *scene.words)
                if (normalized := TaskGenerator._normalize_keyword(word)) is not None
            ]
            if candidates:
                keywords.append(random_source.choice(candidates))
            else:
                logger.info(
                    "过滤低质量 Scene 关键词：scene=%s candidates=%s",
                    scene.name, (scene.name, *scene.words),
                )
        return tuple(dict.fromkeys(keywords))[:TaskGenerator.MAX_TASK_KEYWORDS]

    @staticmethod
    def _normalize_keyword(value: Any) -> str | None:
        """Return a clean, task-worthy keyword or ``None`` for noisy nodes."""
        if not isinstance(value, str):
            return None
        keyword = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
        folded = keyword.casefold()
        if not 2 <= len(keyword) <= 24:
            return None
        if folded in TaskGenerator.LOW_INFORMATION_KEYWORDS:
            return None
        if any(marker in folded for marker in TaskGenerator.UNSOURCED_DOMAIN_KEYWORDS):
            return None
        if any(
            (
                re.search(rf"\b{re.escape(marker.casefold())}\b", folded) is not None
                if marker.isascii() else marker.casefold() in folded
            )
            for marker in (*HIGH_STAKES_MARKERS, *HIGH_STAKES_CHEMICAL_MARKERS)
        ):
            return None
        if re.search(r"[\x00-\x1f\x7f\ufffd]", keyword):
            return None
        if re.search(r"https?://|www\.", folded):
            return None
        alphanumeric = sum(char.isalnum() for char in keyword)
        if alphanumeric / len(keyword) < 0.7:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", keyword):
            return None
        return keyword
