import unittest
import random

from env_factory import KnowledgeGraphBuilder, Neo4jGraphStore, SceneRelation, TaskType


class RecordingStore:
    def __init__(self) -> None:
        self.schema_created = False
        self.scenes = []
        self.task_types = []
        self.reconciled_task_types = None
        self.edges = []

    def ensure_schema(self) -> None:
        self.schema_created = True

    def upsert_scene(self, node) -> None:
        self.scenes.append(node)

    def upsert_task_type(self, node) -> None:
        self.task_types.append(node)

    def reconcile_task_types(self, task_types) -> None:
        self.reconciled_task_types = tuple(task_types)

    def add_scene_edge(self, edge) -> None:
        self.edges.append(edge)


class KnowledgeGraphBuilderTest(unittest.TestCase):
    def test_scene_sampler_uses_all_scene_nodes(self) -> None:
        class Result:
            def __init__(self, rows):
                self.rows = rows

            def single(self):
                return self.rows[0] if self.rows else None

            def __iter__(self):
                return iter(self.rows)

        class Session:
            queries = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def run(self, query, **kwargs):
                statement = query.text if hasattr(query, "text") else query
                self.queries.append(statement)
                if "count(scene)" in statement:
                    return Result([{"total": 1}])
                if "RETURN selected.id" in statement:
                    return Result([{"id": "订单"}])
                if "RETURN scene.name" in statement:
                    return Result([{"name": "订单", "words": []}])
                return Result([{"path": [{"name": "订单", "words": []},
                                         {"name": "消费核对", "words": []}]}])

        class Driver:
            def __init__(self):
                self.active = Session()

            def session(self, **kwargs):
                return self.active

        driver = Driver()
        store = Neo4jGraphStore(driver=driver)
        self.assertEqual(store.random_scene_event_path(0, rng=random.Random(1))[0].name, "订单")
        self.assertEqual(len(store.random_scene_event_path(1, rng=random.Random(1))), 2)
        self.assertFalse(any("SUPPORTED_BY" in query for query in driver.active.queries))
        self.assertIn("single(other IN nodes(p)", driver.active.queries[-1])

    def test_aliases_are_merged(self) -> None:
        builder = KnowledgeGraphBuilder()
        builder.add_scene("外卖", words=["点外卖", "叫外卖"])
        builder.add_scene(" 点外卖 ", words=["外卖订购"])

        self.assertEqual(len(builder.scenes()), 1)
        self.assertEqual(
            builder.scenes()[0].words,
            ("点外卖", "叫外卖", "外卖订购"),
        )

    def test_build_writes_schema_nodes_and_edges(self) -> None:
        builder = KnowledgeGraphBuilder()
        builder.add_relation("酒店", "点外卖", SceneRelation.HIERARCHY)
        builder.add_relation("点外卖", "房间配送", SceneRelation.SAME_EVENT_ELEMENT)
        store = RecordingStore()

        builder.build(store)

        self.assertTrue(store.schema_created)
        self.assertEqual(len(store.scenes), 3)
        self.assertEqual(len(store.edges), 3)
        self.assertEqual(
            sum(edge.relation is SceneRelation.SAME_EVENT_ELEMENT for edge in store.edges),
            2,
        )
        self.assertEqual(
            [node.task_type for node in store.task_types],
            list(TaskType),
        )
        self.assertEqual(store.reconciled_task_types, tuple(TaskType))

    def test_scene_pair_relation_types_are_mutually_exclusive(self) -> None:
        builder = KnowledgeGraphBuilder()
        builder.add_relation("衣", "男装", SceneRelation.HIERARCHY)

        with self.assertRaisesRegex(ValueError, "cannot have both"):
            builder.add_relation("男装", "衣", SceneRelation.SAME_EVENT_ELEMENT)

    def test_existing_word_alias_prevents_duplicate_scene(self) -> None:
        builder = KnowledgeGraphBuilder()
        builder.add_scene("服装", words=["衣"])
        builder.add_scene("衣物", words=["衣物", "衣"])

        self.assertEqual(len(builder.scenes()), 1)
        self.assertEqual(builder.scenes()[0].name, "服装")


if __name__ == "__main__":
    unittest.main()
