#!/usr/bin/env python3
"""Independent deterministic reward oracle used by outer conformance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(task: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    metrics = task.get("metrics", [])
    positive = 0.0
    negative = 0.0
    components: dict[str, float] = {}
    for metric in metrics:
        metric_id = metric["id"]
        value = float(scores.get(metric_id, 0.0))
        low, high = metric.get("score_range", [0, 1])
        if value < low or value > high:
            raise ValueError(f"score for {metric_id} outside {low, high}")
        components[metric_id] = value
        if metric.get("category") in {"process", "outcome"}:
            positive += float(metric["weight"]) * value
        elif metric.get("category") == "penalty":
            negative += float(metric["weight"]) * value
        else:
            raise ValueError(f"unknown metric category: {metric.get('category')}")
    reward = max(-1.0, min(1.0, positive + negative))
    return {"reward": reward, "positive": positive, "negative": negative, "components": components}


def main() -> int:
    parser = argparse.ArgumentParser(description="根据 task.json 独立计算参考奖励")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="metric_id 到 score 的 JSON object")
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(task, scores), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
