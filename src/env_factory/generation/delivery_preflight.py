"""Run the existing final deterministic gate before spending on a sandbox agent."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from env_factory.generation.artifacts import write_task_artifact
from env_factory.graph.knowledge_graph import TaskType
from env_factory.tasks.task import Task

PROJECT = Path(__file__).resolve().parents[3]


def verify_delivery(artifacts: dict, root: Path) -> dict:
    # Only runner-generated scaffolding is executed; author preview files are
    # never trusted. The final builder still receives independent qualification.
    with tempfile.TemporaryDirectory(prefix="envfactory-delivery-preflight-") as directory:
        check_root = Path(directory)
        if (root / "data").is_dir():
            shutil.copytree(root / "data", check_root / "data")
        task_path = write_task_artifact(check_root, Task(artifacts["task"], artifacts["environment"],
            artifacts["metrics"], task_type=TaskType(artifacts["task_type"]),
            task_intent=artifacts["task_intent"], complexity=artifacts["complexity"], artifacts=artifacts),
            artifacts["training_category"])
        contract = json.loads(task_path.read_text())
        contract.pop("actions", None)
        (check_root / "BUILD_CONTRACT.json").write_text(json.dumps(contract, ensure_ascii=False))
        spec = importlib.util.spec_from_file_location("delivery_preflight_scaffold",
            PROJECT / "scripts/sandbox/generate_sandbox_scaffold.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.generate(check_root)
        for name in ("sandbox_runtime.py", "runtime_llm.py"):
            shutil.copy2(PROJECT / "src/env_factory" / name, check_root / name)
        check_spec = importlib.util.spec_from_file_location("delivery_contract_gate",
            PROJECT / "scripts/sandbox/validate_contract_and_tools.py")
        check_module = importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(check_module)
        try:
            check_module.validate(task_path, check_root / "BUILD_CONTRACT.json", check_root / "tools.json")
        except SystemExit as exc:
            raise ValueError(f"AGENT_CONTRACT_PREFLIGHT_FAILED: {exc}") from exc
        report_path = root / "prebuild_agentic_value.json"
        log_path = root / "prebuild_agentic_value.log"
        env = {**os.environ, "PYTHONPATH": str(PROJECT / "src"),
               "SANDBOX_TRAINER_API_KEY": "preflight-local-key", "SANDBOX_EVALUATOR_MOCK": "1"}
        with log_path.open("w") as log:
            try:
                result = subprocess.run([sys.executable,
                    str(PROJECT / "scripts/sandbox/validate_agentic_training_value.py"),
                    "--root", str(check_root), "--output", str(report_path)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, timeout=90)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("AGENT_DELIVERY_PREFLIGHT_TIMEOUT: final deterministic gate exceeded 90 seconds") from exc
        if not report_path.is_file():
            raise ValueError("AGENT_DELIVERY_PREFLIGHT_ERROR: " + log_path.read_text()[-4000:])
        report = json.loads(report_path.read_text())
        if result.returncode or not report.get("curriculum_training_ready"):
            raise ValueError("AGENT_DELIVERY_PREFLIGHT_FAILED: " + json.dumps(
                report.get("failures", []), ensure_ascii=False))
        evidence = {"passed": True, "scope": "fresh_scaffold_original_agentic_gate",
                    "contract_and_tools": True, "report": report_path.name}
        artifacts["generation_pipeline"]["verification"]["delivery_preflight"] = evidence
        return evidence
