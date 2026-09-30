"""Shared task artifact projection for CLI and Code Agent preflight."""

import json
from pathlib import Path


def write_task_artifact(task_dir: Path, task: object, training_category: str) -> Path:
    """Materialize the candidate before its buildability gate runs."""
    pipeline_artifacts = task.artifacts or {}
    artifact_manifest = {
        key: pipeline_artifacts[key]
        for key in (
            "data_manifest", "user_simulation_manifest", "tools_manifest",
            "media_generation", "generation_pipeline", "graph_context",
        ) if key in pipeline_artifacts
    }
    task_path = task_dir / "task.json"
    task_path.write_text(
        json.dumps({
            "task": task.desc,
            "task_type": task.task_type.value,
            "task_intent": task.task_intent,
            "training_category": pipeline_artifacts.get("training_category", training_category),
            "training_contract": pipeline_artifacts.get("training_contract", {}),
            "runtime_capabilities": pipeline_artifacts.get("runtime_capabilities", {}),
            "user_simulation_policy": pipeline_artifacts.get("user_simulation_policy", {"mode": "interactive"}),
            "task_spec": pipeline_artifacts.get("task_spec", {}),
            "complexity": task.complexity,
            "requirements": pipeline_artifacts.get("requirements", {}),
            "public_input": pipeline_artifacts.get("public_input", {
                "initial_user_message": task.desc, "materials": []
            }),
            "environment_plan": pipeline_artifacts.get("environment_plan", {}),
            "environment": task.env,
            "actions": pipeline_artifacts.get("actions", []),
            "capability_plan": pipeline_artifacts.get("capability_plan", []),
            "tools": pipeline_artifacts.get("tools", []),
            "tool_bindings": pipeline_artifacts.get("tool_bindings", []),
            "tool_implementations": pipeline_artifacts.get("tool_implementations", []),
            "noise_tools": pipeline_artifacts.get("noise_tools", []),
            "observation_schema": pipeline_artifacts.get("observation_schema", {}),
            "reward_key_steps": pipeline_artifacts.get("reward_key_steps", []),
            "metrics": task.metrics,
            "metric_implementations": pipeline_artifacts.get("metric_implementations", []),
            "reward_formula": pipeline_artifacts.get("reward_formula", {}),
            "acceptance_contract": pipeline_artifacts.get("acceptance_contract", {}),
            "task_readiness": pipeline_artifacts.get("task_readiness", {}),
            "artifacts": artifact_manifest,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return task_path
