"""Neo4j graph construction for the task knowledge graph."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging
import os
import random
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Query

from env_factory.graph.knowledge_graph import (
    DatasetNode,
    FieldNode,
    ResourceNode,
    SceneNode,
    SceneRelation,
    TaskType,
    TaskTypeNode,
    normalize_scene_name,
)


load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneEdge:
    """A directed relation between two scene nodes."""

    source: str
    target: str
    relation: SceneRelation


@dataclass(frozen=True)
class SceneDatasetLink:
    """A reviewed scene-to-source relation with an optional observed group value."""

    scene_name: str
    dataset_key: str
    source_sha256: str
    evidence: str
    group_field: str | None = None
    group_value: str | None = None


class Neo4jGraphStore:
    """Persist the task graph in Neo4j."""

    _RELATION_QUERIES = {
        SceneRelation.HIERARCHY: "HIERARCHY",
        SceneRelation.SAME_EVENT_ELEMENT: "SAME_EVENT_ELEMENT",
    }

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        database: str = "neo4j",
        path_query_timeout: float = 10.0,
        driver: Driver | None = None,
    ) -> None:
        if path_query_timeout <= 0:
            raise ValueError("path_query_timeout must be greater than zero")
        self.database = database
        self.path_query_timeout = path_query_timeout
        self._owns_driver = driver is None
        self.driver = driver or GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self) -> None:
        """Close the underlying driver when this store created it."""

        if self._owns_driver:
            self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        """Raise if Neo4j cannot be reached with the configured credentials."""

        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        """Create uniqueness constraints required by graph upserts."""

        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT scene_id IF NOT EXISTS "
                "FOR (node:Scene) REQUIRE node.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT task_type_id IF NOT EXISTS "
                "FOR (node:TaskType) REQUIRE node.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT dataset_key IF NOT EXISTS "
                "FOR (node:Dataset) REQUIRE node.key IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT resource_key IF NOT EXISTS "
                "FOR (node:Resource) REQUIRE node.key IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT field_key IF NOT EXISTS "
                "FOR (node:Field) REQUIRE node.key IS UNIQUE"
            ).consume()

    def upsert_dataset(self, node: DatasetNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (dataset:Dataset {key: $key}) "
                "SET dataset.title = $title, dataset.source_url = $source_url, "
                "dataset.source_sha256 = $source_sha256",
                key=node.key, title=node.title, source_url=node.source_url,
                source_sha256=node.source_sha256,
            ).consume()

    def upsert_resource(self, node: ResourceNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (dataset:Dataset {key: $dataset_key}) "
                "MERGE (resource:Resource {key: $key}) "
                "SET resource.source_sha256 = $source_sha256, "
                "resource.source_format = $source_format "
                "MERGE (dataset)-[:HAS_RESOURCE]->(resource)",
                dataset_key=node.dataset_key, key=node.key,
                source_sha256=node.source_sha256, source_format=node.source_format,
            ).consume()

    def upsert_field(self, node: FieldNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (resource:Resource {key: $resource_key}) "
                "MERGE (field:Field {key: $key}) "
                "SET field.name = $name, field.role = $role "
                "MERGE (resource)-[:HAS_FIELD]->(field)",
                resource_key=node.resource_key, key=node.key,
                name=node.name, role=node.role,
            ).consume()

    def link_scene_dataset(self, link: SceneDatasetLink) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (scene:Scene {id: $scene_id}), (dataset:Dataset {key: $dataset_key}) "
                "MERGE (scene)-[relation:SUPPORTED_BY]->(dataset) "
                "SET relation.source_sha256 = $source_sha256, relation.evidence = $evidence, "
                "relation.group_field = $group_field, relation.group_value = $group_value, "
                "relation.reviewed = true, relation.managed_by = 'graph_dataset_links'",
                scene_id=normalize_scene_name(link.scene_name), dataset_key=link.dataset_key,
                source_sha256=link.source_sha256, evidence=link.evidence,
                group_field=link.group_field, group_value=link.group_value,
            ).consume()

    def reconcile_reviewed_links(self, links: tuple[SceneDatasetLink, ...]) -> None:
        """Remove only registry-managed edges no longer in the reviewed registry."""
        pairs = [[normalize_scene_name(link.scene_name), link.dataset_key] for link in links]
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (scene:Scene)-[relation:SUPPORTED_BY]->(dataset:Dataset) "
                "WHERE relation.managed_by = 'graph_dataset_links' "
                "AND NOT [scene.id, dataset.key] IN $pairs "
                "DELETE relation",
                pairs=pairs,
            ).consume()

    def supported_dataset_links(self, platform: str) -> tuple[tuple[SceneNode, SceneDatasetLink], ...]:
        """Return only reviewed links for a requested source platform."""
        if platform not in {"kaggle", "data_gov_hk"}:
            raise ValueError(f"unsupported dataset platform: {platform}")
        with self.driver.session(database=self.database) as session:
            records = session.run(
                "MATCH (scene:Scene)-[relation:SUPPORTED_BY]->(dataset:Dataset) "
                "WHERE dataset.key STARTS WITH $prefix AND relation.reviewed = true "
                "RETURN scene.name AS scene_name, scene.words AS scene_words, "
                "dataset.key AS dataset_key, relation.source_sha256 AS source_sha256, "
                "relation.evidence AS evidence, relation.group_field AS group_field, "
                "relation.group_value AS group_value",
                prefix=platform + ":",
            )
            return tuple((
                SceneNode(str(record["scene_name"]), tuple(record["scene_words"] or ())),
                SceneDatasetLink(
                    str(record["scene_name"]), str(record["dataset_key"]),
                    str(record["source_sha256"]), str(record["evidence"]),
                    record["group_field"], record["group_value"],
                ),
            ) for record in records)

    def upsert_scene(self, node: SceneNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (scene:Scene {id: $id}) "
                "SET scene.name = $name, scene.words = $words, "
                "scene.expanded = coalesce(scene.expanded, false) OR $expanded, "
                "scene.expanded_words = CASE WHEN $expanded THEN "
                "reduce(result = coalesce(scene.expanded_words, []), word IN $expanded_words | "
                "CASE WHEN word IN result THEN result ELSE result + word END) "
                "ELSE coalesce(scene.expanded_words, []) END",
                id=normalize_scene_name(node.name),
                name=node.name,
                words=list(node.words),
                expanded=node.expanded,
                expanded_words=[normalize_scene_name(word) for word in node.expanded_words],
            ).consume()

    def get_expanded_scene_words(self) -> set[str]:
        """Return names and aliases of Scene nodes already expanded."""

        return {
            normalize_scene_name(word)
            for node in self.get_scene_nodes()
            if node.expanded
            for word in (node.name, *node.words)
        }

    def get_scene_nodes(self) -> tuple[SceneNode, ...]:
        """Load persisted Scene nodes for cross-run de-duplication."""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (scene:Scene) "
                "RETURN scene.name AS name, scene.words AS words, "
                "coalesce(scene.expanded, false) AS expanded, "
                "coalesce(scene.expanded_words, []) AS expanded_words"
            )
            return tuple(
                SceneNode(
                    name=str(record["name"]),
                    words=tuple(str(word) for word in (record["words"] or [])),
                    expanded=bool(record["expanded"]),
                    expanded_words=tuple(
                        str(word) for word in (record["expanded_words"] or [])
                    ),
                )
                for record in result
            )

    def get_hierarchy_children(self, word: str, *, limit: int = 10) -> tuple[SceneNode, ...]:
        """Return direct hierarchy children for a Scene name or alias."""

        if not word.strip():
            return ()
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        normalized = normalize_scene_name(word)
        with self.driver.session(database=self.database) as session:
            records = session.run(
                "MATCH (parent:Scene)-[:HIERARCHY]->(child:Scene) "
                "WHERE parent.id = $term OR parent.id = $normalized "
                "OR $term IN coalesce(parent.words, []) "
                "OR $normalized IN coalesce(parent.words, []) "
                "RETURN child.name AS name, child.words AS words "
                "LIMIT $limit",
                term=word.strip(),
                normalized=normalized,
                limit=limit,
            )
            children = tuple(
                SceneNode(
                    name=str(record["name"]),
                    words=tuple(str(value) for value in (record["words"] or [])),
                )
                for record in records
            )
        logger.debug("下位节点查询完成：关键词=%s，结果数=%d", word, len(children))
        return children
    def random_scene_event_path(
        self, hops: int, *, attempts: int = 8, rng: random.Random | None = None
    ) -> tuple[SceneNode, ...]:
        """Return a random Scene node or simple event-element path."""

        if hops < 0 or hops > 20:
            raise ValueError("hops must be between 0 and 20")
        if attempts <= 0:
            raise ValueError("attempts must be greater than zero")
        random_source = rng or random
        logger.debug("开始随机抽取 Scene 路径：跳数=%d，最大尝试次数=%d", hops, attempts)
        with self.driver.session(database=self.database) as session:
            for _ in range(attempts):
                total = session.run(
                    "MATCH (scene:Scene) RETURN count(scene) AS total"
                ).single()["total"]
                if not total:
                    logger.warning("随机抽取 Scene 路径失败：图谱中没有 Scene 节点")
                    return ()
                start = session.run(
                    "MATCH (selected:Scene) "
                    "RETURN selected.id AS id "
                    "SKIP $offset LIMIT 1",
                    offset=random_source.randrange(total),
                ).single()
                if start is None:
                    logger.warning("随机抽取 Scene 路径失败：随机节点偏移无结果")
                    return ()
                if hops == 0:
                    record = session.run(
                        "MATCH (scene:Scene {id: $start_id}) "
                        "RETURN scene.name AS name, scene.words AS words",
                        start_id=start["id"],
                    ).single()
                    if record is not None:
                        logger.debug("随机节点抽取完成：节点=%s", record["name"])
                        return (SceneNode(
                            name=str(record["name"]),
                            words=tuple(str(word) for word in (record["words"] or [])),
                        ),)
                    continue
                records = list(session.run(
                    Query(
                        f"MATCH p=(start:Scene {{id: $start_id}})-[:SAME_EVENT_ELEMENT*1..{hops}]-(end:Scene) "
                        f"WHERE length(p) = {hops} AND all(node IN nodes(p) "
                        "WHERE single(other IN nodes(p) WHERE other = node)) "
                        "RETURN [node IN nodes(p) | {name: node.name, words: node.words}] AS path "
                        "LIMIT 100",
                        timeout=self.path_query_timeout,
                    ),
                    start_id=start["id"],
                    hops=hops,
                ))
                if records:
                    logger.debug(
                        "随机路径候选抽取完成：跳数=%d，起点=%s，候选数=%d",
                        hops,
                        start["id"],
                        len(records),
                    )
                    return tuple(
                        SceneNode(
                            name=str(item["name"]),
                            words=tuple(str(word) for word in (item.get("words") or [])),
                        )
                        for item in random_source.choice(records)["path"]
                    )
        logger.info("未找到符合条件的 Scene 路径：跳数=%d，尝试次数=%d", hops, attempts)
        return ()
    def upsert_task_type(self, node: TaskTypeNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (task_type:TaskType {id: $id}) "
                "SET task_type.name = $name, task_type.value = $value",
                id=node.task_type.value,
                name=node.task_type.value,
                value=node.task_type.value,
            ).consume()

    def add_scene_edge(self, edge: SceneEdge) -> None:
        relation = self._RELATION_QUERIES[edge.relation]
        query = (
            f"MATCH (source:Scene {{id: $source}}), (target:Scene {{id: $target}}) "
            f"MERGE (source)-[:{relation}]->(target)"
        )
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                source=normalize_scene_name(edge.source),
                target=normalize_scene_name(edge.target),
            ).consume()


class KnowledgeGraphBuilder:
    """Build and persist a graph while merging aliases into canonical scenes."""

    def __init__(
        self,
        *,
        canonicalizer: Callable[[str], str] = normalize_scene_name,
    ) -> None:
        self.canonicalizer = canonicalizer
        self._scenes: dict[str, SceneNode] = {}
        self._aliases: dict[str, str] = {}
        self._edges: set[SceneEdge] = set()
        self._relation_by_scene_pair: dict[frozenset[str], SceneRelation] = {}

    def add_scene(
        self,
        name: str,
        *,
        words: Iterable[str] = (),
        expanded: bool = False,
        expanded_words: Iterable[str] = (),
    ) -> SceneNode:
        """Add a scene or merge its words into an existing scene."""

        if not name.strip():
            raise ValueError("scene name must not be empty")
        words = tuple(words)
        expanded_words = tuple(expanded_words)
        raw_id = self.canonicalizer(name)
        scene_id = self._aliases.get(raw_id, raw_id)
        if scene_id not in self._scenes:
            word_scene_ids = {
                self._aliases[self.canonicalizer(word)]
                for word in words
                if self.canonicalizer(word) in self._aliases
            }
            if len(word_scene_ids) > 1:
                raise ValueError("scene words map to multiple existing scene nodes")
            if word_scene_ids:
                scene_id = word_scene_ids.pop()
        existing = self._scenes.get(scene_id)
        if existing is None:
            node = SceneNode(
                name=name.strip(), words=tuple(words), expanded=expanded,
                expanded_words=tuple(expanded_words)
            )
        else:
            node = SceneNode(
                name=existing.name,
                words=tuple(dict.fromkeys((*existing.words, *words))),
                expanded=existing.expanded or expanded,
                expanded_words=tuple(dict.fromkeys((*existing.expanded_words, *expanded_words))),
            )
        self._scenes[scene_id] = node
        self._aliases[raw_id] = scene_id
        for word in node.words:
            self._aliases[self.canonicalizer(word)] = scene_id
        return node

    def add_relation(self, source: str, target: str, relation: SceneRelation) -> None:
        """Add a relation, enforcing direction and one type per scene pair.

        Hierarchy is stored in the supplied direction. Same-event-element is
        symmetric and is therefore stored in both directions.
        """

        if not isinstance(relation, SceneRelation):
            raise ValueError(f"unsupported scene relation: {relation!r}")
        self.add_scene(source)
        self.add_scene(target)
        source_id = self._aliases[self.canonicalizer(source)]
        target_id = self._aliases[self.canonicalizer(target)]
        if source_id == target_id:
            return
        pair = frozenset((source_id, target_id))
        previous_relation = self._relation_by_scene_pair.get(pair)
        if previous_relation is not None and previous_relation != relation:
            raise ValueError(
                "a Scene node pair cannot have both hierarchy and same_event_element relations"
            )
        self._relation_by_scene_pair[pair] = relation
        self._edges.add(SceneEdge(source_id, target_id, relation))
        if relation is SceneRelation.SAME_EVENT_ELEMENT:
            self._edges.add(SceneEdge(target_id, source_id, relation))

    def scenes(self) -> tuple[SceneNode, ...]:
        return tuple(self._scenes.values())

    def edges(self) -> tuple[SceneEdge, ...]:
        return tuple(self._edges)

    def build(
        self,
        store: Neo4jGraphStore,
        *,
        task_types: Iterable[TaskType] = tuple(TaskType),
    ) -> None:
        """Write all collected nodes and relationships to Neo4j."""

        store.ensure_schema()
        for scene in self.scenes():
            store.upsert_scene(scene)
        for task_type in task_types:
            store.upsert_task_type(TaskTypeNode(task_type))
        for edge in self.edges():
            store.add_scene_edge(edge)
