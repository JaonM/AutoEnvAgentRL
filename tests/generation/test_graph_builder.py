import unittest

from env_factory import KnowledgeGraphBuilder, SceneRelation, TaskType


class RecordingStore:
    def __init__(self) -> None:
        self.schema_created = False
        self.scenes = []
        self.task_types = []
        self.edges = []

    def ensure_schema(self) -> None:
        self.schema_created = True

    def upsert_scene(self, node) -> None:
        self.scenes.append(node)

    def upsert_task_type(self, node) -> None:
        self.task_types.append(node)

    def add_scene_edge(self, edge) -> None:
        self.edges.append(edge)


class KnowledgeGraphBuilderTest(unittest.TestCase):
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
