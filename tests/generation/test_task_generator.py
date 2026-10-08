import json
import unittest
from unittest.mock import patch

from env_factory import (
    SceneNode,
    TaskGenerator,
    TaskType,
)


class TaskGeneratorTest(unittest.TestCase):
    def test_task_type_accepts_comma_separated_values(self):
        with patch("env_factory.generation.task_generator.random.choice", return_value=TaskType.RESEARCH):
            self.assertEqual(
                TaskGenerator._select_task_type("QA, Event, Research"),
                TaskType.RESEARCH,
            )

    def test_coding_task_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported task_type"):
            TaskGenerator._select_task_type("Coding")





    def test_keywords_filter_generic_unsafe_and_malformed_candidates(self):
        path = (
            SceneNode("材质", ("衣物材质",)),
            SceneNode("癌症", ("医疗诊断",)),
            SceneNode("有效主题", ("https://invalid.example", "\ufffd乱码")),
        )
        with patch("env_factory.generation.task_generator.random.choice", side_effect=lambda items: items[0]):
            self.assertEqual(TaskGenerator._keywords(path), ("衣物材质", "有效主题"))

    def test_keywords_limit_forced_topic_combinations(self):
        path = tuple(SceneNode(f"主题{index}") for index in range(5))
        with patch("env_factory.generation.task_generator.random.choice", side_effect=lambda items: items[0]):
            self.assertEqual(
                TaskGenerator._keywords(path),
                ("主题0", "主题1", "主题2"),
            )

    def test_keyword_filter_does_not_match_tax_inside_syntax(self):
        self.assertEqual(TaskGenerator._normalize_keyword("syntax"), "syntax")

    def test_keyword_filter_rejects_high_risk_solvent_topics(self):
        self.assertIsNone(TaskGenerator._normalize_keyword("甲基叔丁基醚"))
        self.assertIsNone(TaskGenerator._normalize_keyword("格氏试剂"))



if __name__ == "__main__":
    unittest.main()


def test_generator_calls_code_agent_with_graph_request(tmp_path):
    from unittest.mock import Mock
    from env_factory.generation import code_agent
    store = Mock()
    store.random_scene_event_path.side_effect = lambda hops, **_: tuple(SceneNode('仓储', ('仓储',)) for _ in range(hops + 1))
    artifacts = {'task': '核对库存', 'environment': [], 'metrics': [],
                 'task_intent': 'query', 'complexity': 'standard'}
    with patch.object(code_agent, 'generate', return_value=artifacts) as author:
        result = TaskGenerator(store, code_agent_model='author-model', language='en').generate(
            2, TaskType.EVENT, artifact_dir=tmp_path, seed=12)
    assert result.desc == '核对库存'
    request = author.call_args.kwargs['request']
    assert request['graph_context']['hops'] == __import__('random').Random(12).randint(0, 2)
    assert request['graph_context']['max_hops'] == 2
    assert len(request['graph_context']['nodes']) == request['graph_context']['hops'] + 1
    assert request['code_agent_model'] == 'author-model'
    assert request['language'] == 'en'
    assert request['seed'] == 12
    assert author.call_count == 1


def test_generator_does_not_accept_retired_backends():
    import pytest
    for backend in ('legacy', 'spec'):
        with pytest.raises(TypeError):
            TaskGenerator(None, generation_backend=backend)


def test_random_hops_covers_zero_and_upper_bound(tmp_path):
    from unittest.mock import Mock
    from env_factory.generation import code_agent
    store = Mock()
    store.random_scene_event_path.side_effect = lambda hops, **_: tuple(
        SceneNode('仓储', ('仓储',)) for _ in range(hops + 1))
    artifacts = {'task': '查询仓储', 'environment': [], 'metrics': [],
                 'task_intent': 'query', 'complexity': 'standard'}
    with patch.object(code_agent, 'generate', return_value=artifacts) as author:
        generator = TaskGenerator(store)
        for seed in range(30):
            generator.generate(2, artifact_dir=tmp_path, seed=seed)
        assert {call.kwargs['request']['graph_context']['hops'] for call in author.call_args_list} == {0, 1, 2}
        generator.generate(0, artifact_dir=tmp_path, seed=1)
        assert author.call_args.kwargs['request']['graph_context']['hops'] == 0
        assert len(author.call_args.kwargs['request']['graph_context']['nodes']) == 1
