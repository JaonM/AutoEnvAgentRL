import json
import unittest

from env_factory import (
    GraphExpansionConfig,
    SceneRelation,
    WikipediaResponse,
    WikipediaResult,
    SeedGraphExpander,
)


class FakeSearch:
    def __init__(self):
        self.requests = 0

    def search(self, query, **kwargs):
        self.requests += 1
        return WikipediaResponse(
            query=query,
            results=(
                WikipediaResult(
                    title=f"{query} 相关指南",
                    url=f"https://example.test/{query}",
                    content=f"{query} 服装 外卖 相关内容",
                ),
            ),
        )


class FakeExpansionLLM:
    def complete(self, prompt, *, system_prompt, **kwargs):
        class Response:
            content = ""

        response = Response()
        if "提取可作为任务场景" in system_prompt:
            payload = json.loads(prompt)
            response.content = json.dumps(
                {"items": [{"seed": item["seed"], "terms": ["服装"]} for item in payload["items"]]},
                ensure_ascii=False,
            )
        elif "语义相同" in system_prompt:
            response.content = json.dumps(
                {
                    "groups": [
                        {"name": "服装", "words": ["衣", "男装", "服装"]},
                        {"name": "点外卖", "words": ["点外卖"]},
                    ]
                },
                ensure_ascii=False,
            )
        else:
            response.content = json.dumps(
                {"relations": [{"source": "服装", "target": "点外卖", "relation": "hierarchy"}]},
                ensure_ascii=False,
            )
        return response


class SeedGraphExpanderTest(unittest.TestCase):
    def test_expand_initializes_seeds_and_merges_search_terms(self) -> None:
        expander = SeedGraphExpander(FakeSearch(), FakeExpansionLLM(), max_workers=2)
        builder, groups = expander.expand(["衣", "男装", "点外卖"])

        self.assertEqual(groups[0].name, "服装")
        self.assertIn("衣", groups[0].words)
        self.assertEqual(len(builder.scenes()), 2)
        self.assertEqual(len(builder.scenes()), 2)
        self.assertEqual(len(builder.edges()), 1)
        self.assertEqual(builder.edges()[0].relation, SceneRelation.HIERARCHY)

    def test_expansion_limits_rounds_requests_and_scene_nodes(self) -> None:
        search = FakeSearch()
        expander = SeedGraphExpander(
            search,
            FakeExpansionLLM(),
            config=GraphExpansionConfig(
                max_scene_nodes=1,
                max_search_requests=2,
                max_rounds=3,
            ),
        )
        expander.relation_extractor = type(
            "EmptyRelationExtractor",
            (),
            {"extract": lambda self, scenes, **kwargs: type("Result", (), {"edges": ()})()},
        )()
        builder, groups = expander.expand(["衣", "男装", "点外卖"])

        self.assertEqual(search.requests, 2)
        self.assertLessEqual(len(builder.scenes()), 1)
        self.assertLessEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
