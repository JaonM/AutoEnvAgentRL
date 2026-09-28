#!/usr/bin/env python3
"""Local, non-publishing trial of dataset-first task generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import secrets
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (PROJECT / "data" / "sources" / "kaggle" / "madhavw"
                  / "travel-and-tourism" / "v1" / "raw" / "Travel And Tourism.csv")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from env_factory import LLMClient, TaskType  # noqa: E402
from env_factory.task_pipeline import TaskGenerationPipeline  # noqa: E402
from env_factory.generation.task_generator import TaskGenerator  # noqa: E402
from env_factory.tasks.task import Task  # noqa: E402
from examples.generate_task import _write_task_artifact, _validate_generated_candidate  # noqa: E402


def _records(source: Path) -> list[dict[str, str]]:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Booking_ID", "Destination_City", "Booking_Status", "Total_Trip_Cost"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"dataset lacks booking fields: {sorted(required)}")
    if len({row["Booking_ID"] for row in rows}) != len(rows):
        raise ValueError("Booking_ID is not unique")
    return rows


def _table(rows: list[dict[str, str]]) -> dict:
    return {
        "table_name": "travel_bookings",
        "description": "Original booking facts projected from Travel & Tourism v1; no synthetic rows",
        "columns": [
            {"name": "booking_id", "type": "INTEGER", "nullable": False,
             "description": "Original Booking_ID"},
            {"name": "destination_city", "type": "TEXT", "nullable": False,
             "description": "Original Destination_City"},
            {"name": "booking_status", "type": "TEXT", "nullable": False,
             "description": "Original Booking_Status"},
            {"name": "total_trip_cost", "type": "REAL", "nullable": False,
             "description": "Original Total_Trip_Cost"},
        ],
        "primary_key": ["booking_id"],
        "foreign_keys": [], "indexes": [], "constraints": [],
        "rows": [
            {
                "booking_id": int(row["Booking_ID"]),
                "destination_city": row["Destination_City"],
                "booking_status": row["Booking_Status"],
                "total_trip_cost": float(row["Total_Trip_Cost"]),
            }
            for row in rows
        ],
    }


def _description(category: str, rows: list[dict[str, str]]) -> dict:
    common = {"context": ["Use only the Travel & Tourism v1 booking records supplied for this task."],
              "requirements": {"input_modalities": ["text"], "output_format": "text"}}
    if category == "direct_response":
        first, second = rows[0], rows[1]
        cheaper = min((first, second), key=lambda row: float(row["Total_Trip_Cost"]))
        difference = abs(float(first["Total_Trip_Cost"]) - float(second["Total_Trip_Cost"]))
        public = [
            {"booking_id": int(row["Booking_ID"]),
             "total_trip_cost": float(row["Total_Trip_Cost"])}
            for row in (first, second)
        ]
        request = (
            f"Compare the total trip costs for bookings {first['Booking_ID']} and "
            f"{second['Booking_ID']} in the supplied records. State which booking "
            "is cheaper and the exact cost difference. Do not call a tool."
        )
        return {**common, "task": request, "task_intent": "compare",
                "goal": "Compare two supplied booking costs and compute their difference.",
                "public_input": {"initial_user_message": request, "materials": [{
                    "name": "booking_costs.json", "mime_type": "application/json",
                    "content": json.dumps(public),
                }]},
                "route_plan": {"environment_operations": []},
                "expected_result": (
                    f"Booking {cheaper['Booking_ID']} is cheaper by {difference:g} "
                    "in the dataset's cost units. Both facts must be correct."
                ),
                "complexity": "simple"}
    if category == "simple_agentic":
        booking_id = rows[0]["Booking_ID"]
        request = (
            f"Look up booking {booking_id} in the internal travel booking records. "
            "Report its destination city, booking status, and total trip cost."
        )
        return {**common, "task": request, "task_intent": "query",
                "goal": "Retrieve the exact booking facts for the specified ID.",
                "public_input": {"initial_user_message": request, "materials": []},
                "route_plan": {"environment_operations": [{
                    "action_name": "lookup_booking", "purpose": "Read a booking by its ID",
                    "dependencies": [],
                }]},
                "expected_result": (
                    f"Booking {booking_id}: {rows[0]['Destination_City']}, "
                    f"{rows[0]['Booking_Status']}, cost {float(rows[0]['Total_Trip_Cost']):g}. "
                    "All three facts must be correct."
                ),
                "complexity": "simple"}
    booking_id = next(row["Booking_ID"] for row in rows if row["Destination_City"] == "Rishikesh")
    city = next(row["Destination_City"] for row in rows if row["Booking_ID"] == booking_id)
    eligible = [row for row in rows if row["Destination_City"] == city and row["Booking_Status"] == "Completed"]
    cheapest = min(eligible, key=lambda row: float(row["Total_Trip_Cost"]))
    request = (
        f"Look up booking {booking_id} to learn its destination city. Then find "
        "completed bookings for that destination in the internal records. "
        "Report the lowest total trip cost and its booking ID. The destination "
        "filter for the second lookup must come from the first lookup result."
    )
    return {**common, "task": request, "task_intent": "query",
            "goal": "Use a booking lookup result to search completed bookings in the same city and find the minimum cost.",
            "public_input": {"initial_user_message": request, "materials": []},
            "route_plan": {"environment_operations": [
                {"action_name": "lookup_booking", "purpose": "Read the destination city for the given booking", "dependencies": []},
                {"action_name": "list_completed_destination_bookings",
                 "purpose": "Find completed bookings in the city returned by lookup_booking",
                 "dependencies": ["lookup_booking"]},
            ]},
            "expected_result": (
                f"The first lookup yields {city}; among completed {city} bookings "
                f"the cheapest is booking {cheapest['Booking_ID']} at cost "
                f"{float(cheapest['Total_Trip_Cost']):g}. The second lookup must use "
                "the captured city from the first result."
            ),
            "complexity": "standard"}


def _validate_user_message(category: str, rows: list[dict[str, str]], message: str) -> None:
    """Keep natural phrasing inside the source-backed task's public boundary."""
    if not isinstance(message, str) or not 15 <= len(message.strip()) <= 260:
        raise ValueError("user message must be 15–260 characters")
    message = message.strip()
    folded = message.casefold()
    if re.search(r"lookup_booking|list_completed_destination_bookings|\$ref|capture|不要调用工具|不调用工具", folded):
        raise ValueError("user message contains implementation or evaluation instructions")
    required_ids = [] if category == "direct_response" else [rows[0]["Booking_ID"]]
    for booking_id in required_ids:
        if not re.search(rf"(?<!\d){re.escape(booking_id)}(?!\d)", message):
            raise ValueError(f"user message omits booking {booking_id}")
    if category == "direct_response":
        mentioned_ids = [bool(re.search(rf"(?<!\d){re.escape(row['Booking_ID'])}(?!\d)", message))
                         for row in rows[:2]]
        if any(mentioned_ids) and not all(mentioned_ids):
            raise ValueError("user message names only one of the two bookings")
        if not any(mentioned_ids) and not re.search(r"两笔|两条|这两|两个|both|two", folded):
            raise ValueError("user message does not identify the two supplied bookings")
        required_concepts = (
            ("费用比较", r"费用|花费|价格|成本|cost|price"),
            ("差额", r"差|相差|difference"),
            ("随附记录", r"附件|附上|表|资料|数据|记录|attached|provided"),
        )
        difference = abs(float(rows[0]["Total_Trip_Cost"]) - float(rows[1]["Total_Trip_Cost"]))
        hidden_values = [f"{difference:g}"]
    elif category == "simple_agentic":
        required_concepts = (
            ("目的地", r"目的地|去哪里|去哪|去的是哪里|去的是哪|去的地方|城市|destination"),
            ("预订状态", r"状态|是否完成|进展|status"),
            ("费用", r"费用|花费|价格|成本|多少钱|花了多少|cost|price"),
        )
        hidden_values = [rows[0]["Destination_City"], f"{float(rows[0]['Total_Trip_Cost']):g}"]
    else:
        starting = next(row for row in rows if row["Destination_City"] == "Rishikesh")
        required_concepts = (
            ("目的地", r"目的地|去哪里|去哪|去的是哪里|去的是哪|去的地方|城市|destination"),
            ("已完成", r"已完成|已经完成|完成的|成交|completed"),
            ("最低费用", r"最低|最便宜|最省|最少|花钱最少|cheapest|lowest|minimum"),
            ("预订号", r"预订号|订单号|编号|哪笔|booking id"),
        )
        eligible = [row for row in rows if row["Destination_City"] == starting["Destination_City"]
                    and row["Booking_Status"] == "Completed"]
        cheapest = min(eligible, key=lambda row: float(row["Total_Trip_Cost"]))
        hidden_values = [starting["Destination_City"], cheapest["Booking_ID"],
                         f"{float(cheapest['Total_Trip_Cost']):g}"]
    for value in hidden_values:
        if value.isdigit():
            leaked = re.search(rf"(?<!\d){re.escape(value)}(?!\d)", message)
        else:
            leaked = value.casefold() in folded
        if leaked:
            raise ValueError(f"user message reveals hidden answer: {value}")
    for label, pattern in required_concepts:
        if not re.search(pattern, folded, flags=re.I):
            raise ValueError(f"user message omits {label}")
    if category != "direct_response" and re.search(
        r"https?://|上网|联网|搜索引擎|天气|实时|自己编|猜测|web search", folded, re.I
    ):
        raise ValueError("user message requires an unavailable external source")


def _natural_description(
    pipeline: TaskGenerationPipeline,
    category: str,
    rows: list[dict[str, str]],
    description: dict,
    style: str,
) -> dict:
    """Use the original style vocabulary while retaining a fixed data oracle."""
    requirements = {
        "direct_response": "比较随附两笔预订的费用，指出更便宜的一笔和准确差额；用户可见数据仅在附件中。",
        "simple_agentic": "询问指定预订的目的地、预订状态和总费用；这些事实只能从内部预订记录查询。",
        "multi_step_agentic": (
            "先查指定预订的目的地，再查该目的地已完成的预订，找最低总费用和对应预订号。"
            "第二次查询的目的地必须来自第一次查询结果。"
        ),
    }
    previous_error = ""
    for attempt in range(1, 4):
        result = pipeline._call(
            "dataset_task_voice",
            "你正在为 Agentic RL 环境撰写真实用户会说的一句话或两句话。沿用原任务生成器的风格，"
            "让请求像日常工作或生活中的自然询问，可以有简短且不引入新事实的背景。"
            "不要像测试规范一样列步骤、解释工具、使用 lookup/capture/$ref 等技术术语，"
            "也不要说‘不要调用工具’。只改写用户话术，不改任务目标、数据边界或必要操作。"
            "不要写出隐藏答案、猜测城市/状态/费用、添加日期/币种/预算/新筛选条件，"
            "不要要求网络查询或文件交付。只返回 JSON 对象 user_message。",
            {
                "style": style,
                "training_category": category,
                "required_user_goal": requirements[category],
                "starting_booking_id": rows[0]["Booking_ID"],
                "second_booking_id": rows[1]["Booking_ID"] if category == "direct_response" else None,
                "public_materials": description["public_input"]["materials"],
                "previous_validation_error": previous_error,
                "output": {"user_message": "string"},
            },
        )
        message = result["user_message"].strip()
        try:
            _validate_user_message(category, rows, message)
            if message == description["task"]:
                raise ValueError("user message repeats the rigid task template")
        except ValueError as exc:
            previous_error = str(exc)
            logging.warning("natural task wording rejected: attempt=%d reason=%s", attempt, exc)
            continue
        natural = dict(description)
        natural["task"] = message
        natural["public_input"] = dict(description["public_input"])
        natural["public_input"]["initial_user_message"] = message
        return natural
    raise ValueError(f"could not generate a source-grounded natural task: {previous_error}")


def _oracle_coverage(category: str, rows: list[dict[str, str]], task: dict) -> dict:
    """Check whether rewards cover every source-derived outcome fact."""
    if category == "direct_response":
        cheaper = min(rows[:2], key=lambda row: float(row["Total_Trip_Cost"]))
        difference = abs(float(rows[0]["Total_Trip_Cost"]) - float(rows[1]["Total_Trip_Cost"]))
        expected = {"cheaper_booking_id": cheaper["Booking_ID"], "cost_difference": difference}
        markers = [f"booking {cheaper['Booking_ID']}", f"{difference:g}"]
    elif category == "simple_agentic":
        row = rows[0]
        expected = {"booking_id": row["Booking_ID"],
                    "destination_city": row["Destination_City"],
                    "booking_status": row["Booking_Status"],
                    "total_trip_cost": float(row["Total_Trip_Cost"])}
        markers = [row["Destination_City"], row["Booking_Status"],
                   f"{float(row['Total_Trip_Cost']):g}"]
    else:
        starting_row = next(row for row in rows if row["Destination_City"] == "Rishikesh")
        city = starting_row["Destination_City"]
        eligible = [row for row in rows if row["Destination_City"] == city and row["Booking_Status"] == "Completed"]
        cheapest = min(eligible, key=lambda row: float(row["Total_Trip_Cost"]))
        expected = {"starting_booking_id": starting_row["Booking_ID"],
                    "destination_city": city, "cheapest_booking_id": cheapest["Booking_ID"],
                    "minimum_total_trip_cost": float(cheapest["Total_Trip_Cost"])}
        markers = [city, f"booking {cheapest['Booking_ID']}",
                   f"{float(cheapest['Total_Trip_Cost']):g}"]
    semantic_outcomes = [
        metric for metric in task.get("metrics", [])
        if isinstance(metric, dict) and metric.get("category") == "outcome"
        and isinstance(metric.get("evaluator"), dict)
        and metric["evaluator"].get("kind") in {"external_llm_judge", "hybrid_outcome"}
    ]
    reward_text = json.dumps(semantic_outcomes, ensure_ascii=False).casefold()
    missing = [marker for marker in markers if marker.casefold() not in reward_text]
    edges = task.get("task_spec", {}).get("capability_dag", {}).get("edges", [])
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    success = next((item for item in scenarios if item.get("kind") == "goal_success"), {})
    steps = success.get("steps", [])
    multi_dependency = False
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or edge.get("result_path") != "$.records[0].destination_city"
            or edge.get("argument_path") != "$.destination_city"
        ):
            continue
        for source_step in steps:
            if not isinstance(source_step, dict) or source_step.get("tool_name") != edge.get("from_tool"):
                continue
            capture = source_step.get("capture")
            if not isinstance(capture, dict):
                continue
            for variable, path in capture.items():
                if path != edge["result_path"]:
                    continue
                multi_dependency = any(
                    isinstance(target_step, dict)
                    and target_step.get("tool_name") == edge.get("to_tool")
                    and isinstance(target_step.get("arguments"), dict)
                    and target_step["arguments"].get("destination_city") == {"$ref": variable}
                    for target_step in steps
                )
                if multi_dependency:
                    break
            if multi_dependency:
                break
        if multi_dependency:
            break
    return {"expected": expected, "missing_reward_facts": missing,
            "multi_step_dependency_present": multi_dependency if category == "multi_step_agentic" else None,
            "passed": not missing and (category != "multi_step_agentic" or multi_dependency)}


def _oracle_criterion(category: str, expected: dict) -> str:
    if category == "direct_response":
        return (
            f"The answer must identify booking {expected['cheaper_booking_id']} as cheaper "
            f"and state the exact cost difference {expected['cost_difference']:g}."
        )
    if category == "simple_agentic":
        return (
            f"The answer must state booking {expected['booking_id']} has destination "
            f"{expected['destination_city']}, "
            f"status {expected['booking_status']}, and total trip cost "
            f"{expected['total_trip_cost']:g}."
        )
    return (
        f"The first lookup must establish destination {expected['destination_city']}; "
        f"the answer must identify booking {expected['cheapest_booking_id']} as the "
        f"cheapest completed booking there, with total trip cost "
        f"{expected['minimum_total_trip_cost']:g}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("direct_response", "simple_agentic", "multi_step_agentic"), required=True)
    parser.add_argument("--style", choices=TaskGenerator.STYLES,
                        help="沿用原任务生成器的风格；默认按 seed 抽样")
    parser.add_argument("--seed", type=int, help="口语化风格抽样 seed；默认随机并写入来源记录")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.source == DEFAULT_SOURCE and not args.source.exists():
        subprocess.run(
            [sys.executable, str(PROJECT / "scripts" / "diagnostics" / "download_kaggle_dataset.py"),
             "madhavw/travel-and-tourism"], check=True,
        )
    rows = _records(args.source)
    if args.category == "direct_response":
        selected = rows[:2]
    elif args.category == "simple_agentic":
        selected = rows[:10]
    else:
        selected = [row for row in rows if row["Destination_City"] == "Rishikesh"][:20]
        selected += [row for row in rows if row["Destination_City"] != "Rishikesh"][:10]
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    style = args.style or random.Random(seed).choice(TaskGenerator.STYLES)
    llm = LLMClient.from_env("LLM", timeout=90.0)
    pipeline = TaskGenerationPipeline(llm, retries=3, noise_tool_max=1)
    description = _natural_description(
        pipeline, args.category, selected, _description(args.category, selected), style
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    (args.output / "source_selection.json").write_text(json.dumps({
        "source": "madhavw/travel-and-tourism", "version": 1,
        "license": "CC BY-NC-ND 4.0", "source_sha256": source_sha256,
        "selected_booking_ids": [row["Booking_ID"] for row in selected],
        "projected_fields": ["Booking_ID", "Destination_City", "Booking_Status", "Total_Trip_Cost"],
        "category": args.category, "task_description": description,
        "style": style, "seed": seed,
    }, indent=2), encoding="utf-8")
    verified_facts = _oracle_coverage(args.category, selected, {})["expected"]
    source_data = {"verified_reward_facts": verified_facts,
                   "verified_reward_criterion": _oracle_criterion(args.category, verified_facts)}
    if args.category != "direct_response":
        source_data.update({
            "entities": [{"entity_id": "travel_booking", "name": "Travel booking",
                          "description": "One source booking record", "required_facts": [
                              "booking_id", "destination_city", "booking_status", "total_trip_cost",
                          ], "relationships": []}],
            "data_tables": [_table(selected)],
            "data_document": (
                "# Travel & Tourism v1 booking data\n\n"
                "These are selected original rows projected to four fields. "
                "No records or values were invented. Booking ID is the primary key. "
                "The data is reset before each episode. Source: "
                "https://www.kaggle.com/datasets/madhavw/travel-and-tourism/data. "
                "License: CC BY-NC-ND 4.0. Local noncommercial trial only."
            ),
        })
    artifacts = pipeline.generate(
        keywords=["travel bookings", "destination", "trip cost"],
        task_type=TaskType.QA.value, style=style, task_intent=description["task_intent"],
        graph_context={"dataset": "Travel & Tourism v1", "source_url": "https://www.kaggle.com/datasets/madhavw/travel-and-tourism/data"},
        artifact_dir=args.output, training_category=args.category,
        available_environment_modes=("stateless", "reference_data"),
        description_override=description, source_data=source_data,
    )
    task = Task(artifacts["task"], artifacts["environment"], artifacts["metrics"],
                task_type=TaskType.QA, task_intent=artifacts["task_intent"],
                complexity=artifacts["complexity"], artifacts=artifacts)
    _write_task_artifact(args.output, task, args.category)
    _validate_generated_candidate(args.output)
    oracle = _oracle_coverage(args.category, selected, json.loads(
        (args.output / "task.json").read_text(encoding="utf-8")
    ))
    (args.output / "dataset_oracle.json").write_text(
        json.dumps(oracle, indent=2) + "\n", encoding="utf-8"
    )
    print(f"buildability accepted: {args.output / 'task.json'}; oracle passed={oracle['passed']}")
    return 0 if oracle["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
