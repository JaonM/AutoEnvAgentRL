"""LLM-driven extraction of relations between Scene nodes only."""

from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from env_factory.graph.graph_builder import SceneEdge
from env_factory.graph.knowledge_graph import SceneNode, SceneRelation, normalize_scene_name
from env_factory.llm import LLMClient

logger = logging.getLogger(__name__)

class SceneRelationExtractionError(ValueError):
    """Raised when scene relation output cannot be validated."""


@dataclass(frozen=True)
class SceneRelationExtraction:
    """Validated relations between generated scene nodes."""

    edges: tuple[SceneEdge, ...]


class LLMSceneRelationExtractor:
    """Extract scene-to-scene relations after scene nodes have been generated."""

    SYSTEM_PROMPT = """你是任务知识图谱关系抽取助手。
输入是一组已经生成的 Scene 节点，只在这些 Scene 节点之间抽取关系。

关系类型只有：
- hierarchy：上位场景包含下位场景，方向必须是上位节点到下位节点
- same_event_element：两个场景共同属于同一事件的要素，该关系无方向，程序会自动保存正反两个方向

不要创建新节点，不要抽取 TaskType，也不要输出 Scene 与 TaskType 之间的关系。
不确定的关系不要输出。只返回 JSON，不要解释。
输出格式：{"relations":[{"source":"上位节点","target":"下位节点","relation":"hierarchy"}]}
"""

    def __init__(self, llm: LLMClient, *, max_tokens: int = 2_000) -> None:
        self.llm = llm
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.max_tokens = max_tokens

    def extract(
        self,
        scenes: Iterable[SceneNode],
        *,
        candidate_names: Iterable[str] | None = None,
    ) -> SceneRelationExtraction:
        """Extract relations involving only the incremental candidate scenes."""

        scene_list = tuple(scenes)
        if not scene_list:
            return SceneRelationExtraction(())
        node_names = tuple(dict.fromkeys(scene.name for scene in scene_list))
        candidates = tuple(
            dict.fromkeys(node_names if candidate_names is None else candidate_names)
        )
        if not candidates:
            return SceneRelationExtraction(())
        unknown_candidates = set(candidates) - set(node_names)
        if unknown_candidates:
            raise SceneRelationExtractionError("candidate references an unknown Scene node")
        prompt = json.dumps({"scene_nodes": list(node_names)}, ensure_ascii=False)
        prompt = (
            f"{prompt}\n增量节点：{json.dumps(list(candidates), ensure_ascii=False)}\n"
            "只输出至少涉及一个增量节点的关系，已有节点仅作为上下文。"
        )
        response = self.llm.complete(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            thinking=False,
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format="json_object",
        )
        try:
            return self._parse(response.content, node_names, candidates)
        except SceneRelationExtractionError as exc:
            logger.warning("关系抽取响应格式异常，重试一次：%s", exc)
            retry = self.llm.complete(
                prompt,
                system_prompt=self.SYSTEM_PROMPT + "\n上次响应无效，请只输出简短且完整的合法 JSON。",
                thinking=False,
                temperature=0.0,
                max_tokens=max(self.max_tokens, 4_000),
                response_format="json_object",
            )
            try:
                return self._parse(retry.content, node_names, candidates)
            except SceneRelationExtractionError as retry_error:
                logger.error("关系抽取重试失败，本轮跳过关系：%s", retry_error)
                return SceneRelationExtraction(())

    @staticmethod
    def _parse(
        content: str,
        node_names: tuple[str, ...],
        candidate_names: Iterable[str] | None = None,
    ) -> SceneRelationExtraction:
        try:
            payload = LLMSceneRelationExtractor._parse_json(content)
            raw_relations = payload["relations"]
            if not isinstance(raw_relations, list):
                raise TypeError
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SceneRelationExtractionError("invalid scene relation response") from exc

        known = {normalize_scene_name(name): name for name in node_names}
        candidate_keys = {
            normalize_scene_name(name) for name in (candidate_names or node_names)
        }
        edges: list[SceneEdge] = []
        relation_by_pair: dict[frozenset[str], SceneRelation] = {}
        aliases = {
            "hierarchy": SceneRelation.HIERARCHY,
            "上下位关系": SceneRelation.HIERARCHY,
            "same_event_element": SceneRelation.SAME_EVENT_ELEMENT,
            "同属事件要素关系": SceneRelation.SAME_EVENT_ELEMENT,
        }
        for item in raw_relations:
            if not isinstance(item, dict):
                raise SceneRelationExtractionError("each relation must be an object")
            source, target, relation_value = item.get("source"), item.get("target"), item.get("relation")
            if not all(isinstance(value, str) for value in (source, target, relation_value)):
                raise SceneRelationExtractionError("relation fields must be strings")
            relation = aliases.get(relation_value)
            if relation is None:
                raise SceneRelationExtractionError(f"unsupported scene relation: {relation_value}")
            source_name = known.get(normalize_scene_name(source))
            target_name = known.get(normalize_scene_name(target))
            if source_name is None or target_name is None:
                raise SceneRelationExtractionError("relation references an unknown Scene node")
            source_key = normalize_scene_name(source_name)
            target_key = normalize_scene_name(target_name)
            if source_key == target_key:
                continue
            if source_key not in candidate_keys and target_key not in candidate_keys:
                continue
            pair = frozenset((source_key, target_key))
            previous_relation = relation_by_pair.get(pair)
            if previous_relation is not None and previous_relation != relation:
                raise SceneRelationExtractionError(
                    "a Scene node pair cannot have both hierarchy and same_event_element relations"
                )
            if previous_relation is None:
                edges.append(SceneEdge(source_name, target_name, relation))
                relation_by_pair[pair] = relation
        return SceneRelationExtraction(tuple(edges))

    @staticmethod
    def _parse_json(content: str) -> Any:
        normalized = LLMSceneRelationExtractor._strip_code_fence(content)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as original_error:
            repaired = re.sub(r",\s*([}\]])", r"\1", normalized)
            if repaired != normalized:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            raise original_error

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        content = content.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else content
