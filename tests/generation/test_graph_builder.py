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


def sparse_store(paths, *, total=10):
    from unittest.mock import MagicMock
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    queries = []
    def run(query, **parameters):
        text = query.text if hasattr(query, 'text') else query
        queries.append((query, parameters))
        result = MagicMock()
        if 'count(scene)' in text:
            result.single.return_value = {'total': total}
        elif 'RETURN selected.id' in text:
            result.single.return_value = {'id': 'isolated'}
        elif '$start_id' in text:
            result.__iter__.return_value = iter([])
        else:
            result.__iter__.return_value = iter(paths)
        return result
    session.run.side_effect = run
    return Neo4jGraphStore(driver=driver, path_query_timeout=2.5), queries


def test_sparse_graph_falls_back_to_same_length_without_replacing_task():
    path = [{'name': name, 'words': []} for name in ['甲', '乙', '丙']]
    store, queries = sparse_store([{'path': path}])
    selected = store.random_scene_event_path(2, attempts=3, rng=random.Random(7))
    assert [node.name for node in selected] == ['甲', '乙', '丙']
    fallback, parameters = queries[-1]
    assert '*2]' in fallback.text and 'LIMIT 100' in fallback.text
    assert 'single(other IN nodes(p)' in fallback.text
    assert fallback.timeout == 2.5
    assert parameters == {}
    assert sum('RETURN selected.id' in (q.text if hasattr(q, 'text') else q) for q, _ in queries) == 3


def test_absent_exact_path_stays_absent_after_bounded_fallback():
    store, queries = sparse_store([])
    assert store.random_scene_event_path(2, attempts=1, rng=random.Random(7)) == ()
    assert len(queries) == 4


def test_empty_graph_does_not_attempt_path_fallback():
    store, queries = sparse_store([], total=0)
    assert store.random_scene_event_path(2) == ()
    assert len(queries) == 1
