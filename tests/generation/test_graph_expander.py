import json
import unittest

from env_factory import (
    GraphExpansionConfig,
    SceneRelation,
    WikipediaResponse,
    WikipediaResult,
    SeedGraphExpander,
)
from env_factory.llm import LLMError


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
    def test_content_rejection_isolates_one_seed_and_keeps_other_results(self) -> None:
        class RejectingLLM:
            calls = 0

            def complete(self, prompt, **kwargs):
                self.calls += 1
                seeds = [item["seed"] for item in json.loads(prompt)["items"]]
                if "角色扮演 (心理学)" in seeds:
                    raise LLMError("LLM returned HTTP 400: Content Exists Risk")
                return type("Response", (), {"content": json.dumps({
                    "items": [{"seed": seed, "terms": [seed + "场景"]} for seed in seeds]
                })})()

        llm = RejectingLLM()
        expander = SeedGraphExpander(FakeSearch(), llm)
        items = tuple((seed, FakeSearch().search(seed).results) for seed in
                      ("买衣服", "角色扮演 (心理学)", "点外卖"))
        result = expander._extract_terms_batch(items)
        self.assertEqual(result["买衣服"], ("买衣服场景",))
        self.assertEqual(result["角色扮演 (心理学)"], ())
        self.assertEqual(result["点外卖"], ("点外卖场景",))
        calls = llm.calls
        self.assertEqual(expander._extract_terms_batch(items), result)
        self.assertEqual(llm.calls, calls)

    def test_transient_batch_llm_failure_remains_visible(self) -> None:
        class UnavailableLLM:
            def complete(self, prompt, **kwargs):
                raise LLMError("LLM returned HTTP 503: unavailable")

        expander = SeedGraphExpander(FakeSearch(), UnavailableLLM())
        with self.assertRaisesRegex(LLMError, "503"):
            expander._extract_terms_batch((("买衣服", FakeSearch().search("买衣服").results),))

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
