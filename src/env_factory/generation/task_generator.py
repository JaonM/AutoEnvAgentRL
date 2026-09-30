"""Generate long-horizon task descriptions from Scene graph paths."""

import logging
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from env_factory.graph.graph_builder import Neo4jGraphStore
from env_factory.graph.knowledge_graph import SceneNode, TaskType
from env_factory.llm import LLMClient
from env_factory.tasks.task import Task
from env_factory.task_pipeline import (
    HIGH_STAKES_CHEMICAL_MARKERS,
    HIGH_STAKES_MARKERS,
    TaskGenerationPipeline,
)
from env_factory.tasks.task_routing import select_training_intent, training_contract

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
        llm: LLMClient,
        *,
        user_script_count: int = 3,
        noise_tool_max: int = 3,
        available_environment_modes: tuple[str, ...] | None = None,
        generation_backend: str = "code_agent",
        code_agent_timeout: float = 600,
    ) -> None:
        self.store = store
        self.llm = llm
        self.available_environment_modes = available_environment_modes
        if generation_backend not in {"legacy", "spec", "code_agent"}:
            raise ValueError("generation_backend must be legacy, spec or code_agent")
        self.generation_backend = generation_backend
        if code_agent_timeout <= 0:
            raise ValueError("code_agent_timeout must be positive")
        self.code_agent_timeout = code_agent_timeout
        self.script_count = user_script_count
        self.pipeline = TaskGenerationPipeline(
            llm,
            script_count=user_script_count,
            noise_tool_max=noise_tool_max,
        )

    def generate(
        self,
        hops: int = 3,
        task_type: TaskType | str | None = None,
        task_style: str | None = None,
        artifact_dir: str | Path | None = None,
        task_intent: str | None = None,
        training_category: str = "multi_step_agentic",
        seed: int | None = None,
        prototype: str | None = None,
    ) -> Task:
        """Generate one task using a random path length between 0 and ``hops``."""

        rng = random.Random(seed) if seed is not None else random
        if self.generation_backend == "code_agent":
            from .code_agent import generate as generate_with_agent
            import tempfile
            if not 2 <= hops <= 20:
                raise TaskGenerationError("CODE_AGENT_GRAPH: real multi-hop input requires hops between 2 and 20")
            path, keywords = (), ()
            for _ in range(4):
                path = self.store.random_scene_event_path(hops, attempts=3, rng=rng)
                if len(path) == hops + 1:
                    keywords = self._keywords(path, rng=rng)
                    if keywords:
                        break
            if len(path) != hops + 1 or not keywords:
                raise TaskGenerationError("CODE_AGENT_GRAPH: no usable full multi-hop path; refusing single-node fallback")
            selected_type = self._select_task_type(task_type, rng=rng)
            request = {"version": "1.0", "training_category": training_category,
                "training_contract": training_contract(training_category),
                "task_type": selected_type.value, "task_style": task_style,
                "task_intent": task_intent, "seed": seed,
                "available_environment_modes": list(self.available_environment_modes or
                    ("stateless", "reference_data", "stateful")),
                "graph_context": {"source": "neo4j_scene_path", "hops": hops,
                    "nodes": [node.name for node in path], "keywords": list(keywords),
                    "relation": "SAME_EVENT_ELEMENT"}}
            artifacts = generate_with_agent(request=request,
                artifact_dir=Path(artifact_dir) if artifact_dir else Path(tempfile.mkdtemp(prefix="envfactory-agent-")),
                script_count=self.script_count, timeout=self.code_agent_timeout)
            return Task(desc=artifacts["task"], env=artifacts["environment"], metrics=artifacts["metrics"],
                task_type=selected_type, task_intent=artifacts["task_intent"],
                complexity=artifacts["complexity"], artifacts=artifacts)
        if self.generation_backend == "spec" and training_category == "multi_step_agentic":
            intent_prototypes = {"calculate": "lookup_join_sum", "modify": "lookup_update", "execute": "constraint_create"}
            if task_intent is not None and task_intent not in intent_prototypes:
                raise TaskGenerationError("SPEC_UNSUPPORTED: supported intents are calculate, modify, execute")
            selected_prototype = prototype or intent_prototypes.get(task_intent) or rng.choice(tuple(intent_prototypes.values()))
            if selected_prototype not in intent_prototypes.values() or (task_intent and selected_prototype != intent_prototypes[task_intent]):
                raise TaskGenerationError("SPEC_UNSUPPORTED: prototype and intent do not match")
            mode = "reference_data" if selected_prototype == "lookup_join_sum" else "stateful"
            if self.available_environment_modes is not None and mode not in self.available_environment_modes:
                raise TaskGenerationError(f"SPEC_UNSUPPORTED: {mode} runtime is required")
            from .spec_pipeline import generate as generate_spec
            artifacts = generate_spec(seed=seed if seed is not None else rng.getrandbits(63),
                                      artifact_dir=Path(artifact_dir) if artifact_dir else None,
                                      script_count=self.script_count, prototype=selected_prototype,
                                      subject=rng.choice(("办公用品", "包装材料", "印刷耗材", "维修备件")))
            selected_type = self._select_task_type(task_type or TaskType.EVENT, rng=rng)
            artifacts["task_type"] = selected_type.value
            return Task(desc=artifacts["task"], env=artifacts["environment"], metrics=artifacts["metrics"],
                        task_type=selected_type, task_intent=artifacts["task_intent"], complexity="standard", artifacts=artifacts)
        path, selected_hops, keywords = self._sample_graph_path(hops, rng)
        selected_type = self._select_task_type(task_type, rng=rng)
        selected_style = task_style or rng.choice(self.STYLES)
        if selected_style not in self.STYLES:
            raise ValueError(f"unsupported task_style: {selected_style}")
        if task_intent is not None and task_intent not in self.INTENTS:
            raise ValueError(f"unsupported task_intent: {task_intent}")
        selected_intent = select_training_intent(training_category, task_intent, rng=rng)
        artifacts = self.pipeline.generate(
            keywords=list(keywords),
            task_type=selected_type.value,
            style=selected_style,
            task_intent=selected_intent,
            graph_context={
                "hops": selected_hops,
                "nodes": [node.name for node in path],
                "keywords": list(keywords),
            },
            artifact_dir=artifact_dir,
            training_category=training_category,
            rng=rng,
            available_environment_modes=self.available_environment_modes,
        )
        if artifacts.get("complexity") not in {"simple", "standard", "complex"}:
            raise TaskGenerationError("task description returned an invalid complexity")
        return Task(
            desc=artifacts["task"], env=artifacts["environment"], metrics=artifacts["metrics"],
            task_type=selected_type, task_intent=artifacts.get("task_intent", selected_intent), complexity=artifacts["complexity"],
            artifacts=artifacts,
        )

    def _sample_graph_path(self, hops: int, rng: random.Random):
        """Sample a Scene path and its filtered keywords."""
        if hops < 0:
            raise ValueError("hops must not be negative")
        started = time.perf_counter()
        selected_hops = rng.randint(0, hops)
        path = self.store.random_scene_event_path(selected_hops, rng=rng)
        if not path and selected_hops > 0:
            # Sparse graphs may not contain a path at the initially selected
            # depth. Try shorter paths so batch generation remains productive.
            for fallback_hops in range(selected_hops - 1, -1, -1):
                path = self.store.random_scene_event_path(fallback_hops, attempts=3, rng=rng)
                if path:
                    logger.info(
                        "路径跳数降级：请求=%d，实际=%d",
                        selected_hops, fallback_hops,
                    )
                    selected_hops = fallback_hops
                    break
        if not path:
            logger.warning("任务路径抽取失败：请求跳数=%d，实际跳数=%d", hops, selected_hops)
            raise TaskGenerationError(f"no Scene node or path found for {selected_hops} hops")
        def usable_keywords(candidate: tuple[SceneNode, ...]) -> tuple[str, ...]:
            if any(
                marker in node.name.casefold()
                for node in candidate
                for marker in self.UNSOURCED_DOMAIN_KEYWORDS
            ):
                return ()
            return self._keywords(candidate, rng=rng)

        keywords = usable_keywords(path)
        if not keywords:
            logger.warning(
                "路径关键词全部被质量过滤：实际跳数=%d 节点=%s",
                selected_hops, [node.name for node in path],
            )
            for keyword_retry in range(1, 4):
                path = self.store.random_scene_event_path(selected_hops, attempts=3, rng=rng)
                keywords = usable_keywords(path) if path else ()
                if keywords:
                    logger.info(
                        "低质量关键词路径重采样成功：attempt=%d keywords=%s",
                        keyword_retry, keywords,
                    )
                    break
            if not keywords:
                raise TaskGenerationError("scene paths contain no usable task keywords after resampling")
        logger.info(
            "任务路径抽取完成：请求跳数=%d，实际跳数=%d，节点数=%d，关键词数=%d，关键词=%s，耗时=%.2fs",
            hops,
            selected_hops,
            len(path),
            len(keywords),
            keywords,
            time.perf_counter() - started,
        )
        return path, selected_hops, keywords

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
