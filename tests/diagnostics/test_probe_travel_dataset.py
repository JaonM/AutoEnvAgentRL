"""Public task wording must remain answerable from the selected source rows."""

from __future__ import annotations

import pytest

from scripts.diagnostics.probe_travel_dataset import (
    _description,
    _natural_description,
    _oracle_coverage,
    _validate_user_message,
)


ROWS = [
    {"Booking_ID": "5", "Destination_City": "Rishikesh",
     "Booking_Status": "Completed", "Total_Trip_Cost": "15600"},
    {"Booking_ID": "130", "Destination_City": "Rishikesh",
     "Booking_Status": "Completed", "Total_Trip_Cost": "11500"},
    {"Booking_ID": "7", "Destination_City": "Leh",
     "Booking_Status": "Pending", "Total_Trip_Cost": "72000"},
]


@pytest.mark.parametrize(("category", "message"), [
    ("direct_response", "我把 5 号和 130 号预订的费用放在附件里了，帮我看看哪笔更便宜，差了多少？"),
    ("simple_agentic", "帮我查一下 5 号预订吧，它要去哪个城市、现在是什么状态，总费用是多少？"),
    ("simple_agentic", "我这边有个预订编号 5，想确认一下它的目的地、当前预订状态，还有总共花了多少钱。麻烦帮我查一下内部记录。"),
    ("multi_step_agentic", "想核对一下 5 号预订去哪里，再看看那个目的地已经完成的订单，哪笔总费用最低？把订单号也告诉我。"),
])
def test_natural_requests_keep_required_information(category: str, message: str) -> None:
    _validate_user_message(category, ROWS, message)


@pytest.mark.parametrize(("category", "message"), [
    ("direct_response", "附件里有 5 号和 130 号预订的费用，直接告诉我哪笔便宜了 4100。"),
    ("simple_agentic", "帮我查 5 号预订去 Rishikesh 的状态和总费用。"),
    ("multi_step_agentic", "5 号预订要去 Rishikesh，帮我找已完成订单中 130 号的最低费用 11500。"),
])
def test_natural_requests_cannot_reveal_source_answers(category: str, message: str) -> None:
    with pytest.raises(ValueError, match="hidden answer"):
        _validate_user_message(category, ROWS, message)


def test_natural_request_cannot_expose_tool_contract() -> None:
    with pytest.raises(ValueError, match="implementation"):
        _validate_user_message(
            "simple_agentic", ROWS,
            "请对预订 5 调用 lookup_booking，返回目的地、状态和总费用。",
        )


def test_rejected_wording_is_retried_without_changing_the_fixed_contract() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.payloads = []

        def _call(self, _stage: str, _prompt: str, payload: dict) -> dict:
            self.payloads.append(payload)
            messages = [
                "帮我查 5 号预订去 Rishikesh 的状态和总费用。",
                "帮我查一下 5 号预订吧，它要去哪个城市、现在是什么状态，总费用是多少？",
            ]
            return {"user_message": messages[len(self.payloads) - 1]}

    pipeline = FakePipeline()
    base = _description("simple_agentic", ROWS)
    natural = _natural_description(pipeline, "simple_agentic", ROWS, base, "日常对话")
    assert natural["task"] == natural["public_input"]["initial_user_message"]
    assert natural["expected_result"] == base["expected_result"]
    assert natural["route_plan"] == base["route_plan"]
    assert len(pipeline.payloads) == 2
    assert "hidden answer" in pipeline.payloads[1]["previous_validation_error"]


def test_dataset_oracle_accepts_renamed_tools_only_with_captured_dependency() -> None:
    task = {
        "metrics": [{"category": "outcome", "rubric":
                     "Rishikesh booking 130 has minimum total trip cost 11500",
                     "evaluator": {"kind": "external_llm_judge"}}],
        "task_spec": {"capability_dag": {"edges": [{
            "from_tool": "get_destination", "to_tool": "find_minimum",
            "result_path": "$.records[0].destination_city", "argument_path": "$.destination_city",
        }]}},
        "acceptance_contract": {"executable_scenarios": [{
            "kind": "goal_success", "steps": [
                {"tool_name": "get_destination", "capture": {"city": "$.records[0].destination_city"}},
                {"tool_name": "find_minimum", "arguments": {"destination_city": {"$ref": "city"}}},
            ],
        }]},
    }
    assert _oracle_coverage("multi_step_agentic", ROWS, task)["passed"]
    task["acceptance_contract"]["executable_scenarios"][0]["steps"][1]["arguments"] = {
        "destination_city": "Rishikesh",
    }
    assert not _oracle_coverage("multi_step_agentic", ROWS, task)["passed"]
