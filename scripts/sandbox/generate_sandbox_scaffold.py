#!/usr/bin/env python3
"""Generate the deterministic composition layer for one task sandbox."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from env_factory.evidence.material_artifacts import DOCKERIGNORE_SOURCE

APP_SOURCE = '''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from wsgiref.simple_server import make_server
from sandbox_runtime import ContractRewardAggregator, ContractRewardGate, ContractModelMetricEvaluator, ContractToolRegistry, ContractUserSimulator, DeclarativeMetricEvaluator, DeclarativeToolCompiler, EpisodeStore, ManifestDataStore, SandboxApplication
from task_impl import TaskHooks
ROOT = Path(__file__).resolve().parent
def _json(path): return json.loads(path.read_text(encoding="utf-8"))
def create_app(*, db_path=None):
    contract = _json(ROOT / "BUILD_CONTRACT.json"); artifacts = contract.get("artifacts", {})
    manifest = artifacts["data_manifest"]; store = EpisodeStore(db_path or ROOT / ".runtime" / "episodes.sqlite3")
    data = ManifestDataStore(manifest, ROOT / manifest["root"], store); hooks = TaskHooks(contract, data, store)
    handlers = DeclarativeToolCompiler(data).compile_all(contract.get("tool_implementations", []))
    extensions = hooks.business_handlers()
    if handlers.keys() & extensions.keys(): raise ValueError("custom handlers cannot override compiled tools")
    handlers.update(extensions)
    registry = ContractToolRegistry(contract.get("tools", []), handlers, noise_tools=contract.get("noise_tools", []), event_recorder=store.event, tool_contracts=contract.get("task_spec", {}).get("tool_contracts", []))
    user_manifest = artifacts.get("user_simulation_manifest", {}); user_root = ROOT / user_manifest.get("root", "data/user_simulation")
    profiles = _json(user_root / user_manifest["profiles_file"]) if user_manifest.get("profiles_file") else []
    scripts = _json(user_root / user_manifest["scripts_file"]) if user_manifest.get("scripts_file") else []
    # Runtime user behavior defaults to the shared external-LLM boundary.
    # Task-specific hooks must not silently replace semantic FSM reasoning.
    user = ContractUserSimulator(store, profiles=profiles, scripts=scripts)
    metric_evaluator = DeclarativeMetricEvaluator(); aggregator = ContractRewardAggregator(contract.get("metrics", [])); reward_gate = ContractRewardGate(contract.get("task_spec", {}), contract.get("metrics", []))
    model_metrics = ContractModelMetricEvaluator(contract, store)
    def reward():
        context = dict(hooks.metric_context())
        context.update(business_state={name: data.table(name) for name in data.baseline}, initial_business_state=data.baseline, trajectory=store.replay(), final_agent_response=store.get_state("final_agent_response", ""))
        scores = metric_evaluator.evaluate_all(contract.get("metric_implementations", []), context)
        scores.update(model_metrics.evaluate_all(context, scores))
        extensions = hooks.custom_metric_scores(context, dict(scores))
        if scores.keys() & extensions.keys(): raise ValueError("custom metrics cannot override compiled scores")
        scores.update(extensions); return aggregator.aggregate(reward_gate.apply(scores, context))
    def reset(episode): data.reset(episode); user.reset(episode); hooks.reset(episode)
    app = SandboxApplication(episode_store=store, tool_registry=registry, observation=hooks.observation, reward=reward, user_turn=user.turn, business_snapshot=lambda: {name: data.table(name) for name in data.baseline}, reset_hook=reset, data_hash=data.data_hash)
    app.mutate_business_state = lambda mutation: data.update(mutation["table"], mutation.get("selector", {}), mutation.get("changes", {}))
    return app
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="0.0.0.0"); parser.add_argument("--port",type=int,required=True); args=parser.parse_args()
    with make_server(args.host,args.port,create_app()) as server: server.serve_forever()
if __name__ == "__main__": main()
'''

TASK_IMPL_SOURCE = '''"""Task-specific extension points; platform behavior stays in sandbox_runtime.py."""
from typing import Any, Callable, Mapping
from sandbox_runtime import ContractEvaluatorRuntime, ExternalCapabilityClient
class TaskHooks:
    def __init__(self, contract, data, episode_store):
        self.contract, self.data, self.episode_store = contract, data, episode_store
        self.evaluator = ContractEvaluatorRuntime(episode_store)
        self.external = ExternalCapabilityClient()
    def business_handlers(self) -> dict[str, Callable[[dict[str, Any]], Any]]: return {}
    def observation(self) -> dict[str, Any]:
        replay = self.episode_store.replay(); events = replay.get("events", [])
        conversation = []
        for event in events:
            if event.get("event") == "user_turn" and isinstance(event.get("result"), Mapping):
                conversation.append({"role": "user", "content": str(event["result"].get("user_query", ""))})
            elif event.get("event") == "agent_response":
                conversation.append({"role": "assistant", "content": str(event.get("payload", {}).get("content", ""))})
        return {
            "episode_id": self.episode_store.current().episode_id,
            "conversation": conversation,
            "available_tools": self.contract.get("tools", []),
            "tool_results": [event.get("result") for event in events if event.get("event") == "tool_call"],
            "public_observation": {"task_input": self.contract.get("public_input", {})},
            "final_agent_response": self.episode_store.get_state("final_agent_response", ""),
        }
    def metric_context(self) -> dict[str, Any]:
        observation = self.observation()
        return {"business_state": {name: self.data.table(name) for name in self.data.baseline}, "trajectory": self.episode_store.replay(), "final_agent_response": self.episode_store.get_state("final_agent_response", ""), "conversation": observation.get("conversation", []), "public_observation": observation.get("public_observation", {}), "observation": observation}
    def custom_metric_scores(self, context: Mapping[str, Any], existing: Mapping[str, float]) -> dict[str, float]: return {}
    def reset(self, episode) -> None: return None
'''

ACCEPTANCE_RUNNER_SOURCE = '''#!/usr/bin/env python3
"""EnvFactory-owned contract acceptance; no task-specific code generation."""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
from app import create_app
from sandbox_runtime import AcceptanceScenarioRunner, validate_json_schema

ROOT = Path(__file__).resolve().parent

def main() -> int:
    contract = json.loads((ROOT / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
    result_path = ROOT / "acceptance_result.json"
    failure_path = ROOT / "acceptance_failure.log"
    failure_path.unlink(missing_ok=True)
    evidence = []
    try:
        db_path = Path(tempfile.mkdtemp(prefix="envfactory-acceptance-")) / "episodes.sqlite3"
        app = create_app(db_path=db_path)
        # This probe is part of baseline acceptance and kills bypass_trainer_auth.
        status, body, _ = app.handle("GET", "/v1/observation", None, {})
        if status != 401:
            raise AssertionError(f"unauthorized observation expected 401, got {status}: {body}")
        trainer_headers = {"Authorization": f"Bearer {os.environ['SANDBOX_TRAINER_API_KEY']}"}
        status, observation, _ = app.handle("GET", "/v1/observation", None, trainer_headers)
        if status != 200:
            raise AssertionError(f"authorized observation expected 200, got {status}: {observation}")
        validate_json_schema(contract.get("observation_schema", {}), observation, "observation")
        runner = AcceptanceScenarioRunner(
            app.handle,
            trainer_headers=trainer_headers,
            business_snapshot=app.business_snapshot,
            mutate_business_state=app.mutate_business_state,
        )
        scenarios = contract.get("acceptance_contract", {}).get("executable_scenarios", [])
        if not isinstance(scenarios, list) or not scenarios:
            raise AssertionError("contract has no executable scenarios")
        for scenario in scenarios:
            outcome = runner.run(scenario)
            evidence.append({
                "scenario_id": scenario.get("scenario_id"),
                "status": "passed",
                "history": outcome.get("history", []),
            })
        result = {
            "version": "1.0", "business_acceptance": "passed",
            "http_conformance": "skipped",
            "http_skip_reason": "validated through the in-process public application boundary",
            "scenarios": evidence,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        return 0
    except Exception as exc:
        failure = {
            "version": "1.0", "business_acceptance": "failed",
            "http_conformance": "skipped", "http_skip_reason": "baseline acceptance failed",
            "scenarios": evidence,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        text = json.dumps(failure, ensure_ascii=False, indent=2) + "\\n"
        result_path.write_text(text, encoding="utf-8")
        failure_path.write_text(text, encoding="utf-8")
        print(text, file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
'''

ACCEPTANCE_SH_SOURCE = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SANDBOX_TRAINER_API_KEY="${SANDBOX_TRAINER_API_KEY:-acceptance-trainer-key}"
export SANDBOX_EVALUATOR_MOCK="${SANDBOX_EVALUATOR_MOCK:-1}"
exec python3 ./acceptance_runner.py
'''

IMPLEMENTATION_REPORT_SOURCE = '''# EnvFactory deterministic sandbox delivery

The shared scaffold, declarative tools, compiled metrics, persistence layer,
user simulator, and executable acceptance scenarios are generated from
`BUILD_CONTRACT.json`. `acceptance.sh` runs the EnvFactory-owned in-process
contract runner. Task-specific model work is limited to extensions explicitly
listed in `development_plan.json`; this delivery layer contains no task-specific
reward shortcut or hard-coded business answer.
'''

DOCKERFILE_SOURCE = '''FROM docker.m.daocloud.io/library/python:3.14-slim
WORKDIR /app
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt \\
    && useradd --create-home --uid 10001 sandbox
COPY . .
RUN chown -R sandbox:sandbox /app
USER sandbox
EXPOSE 8000
CMD ["python3", "app.py", "--port", "8000"]
'''

DOCKER_BUILD_SOURCE = '''#!/usr/bin/env bash
set -euo pipefail
docker build -t "${SANDBOX_IMAGE:-envfactory-sandbox}" .
'''

DOCKER_RUN_SOURCE = '''#!/usr/bin/env bash
set -euo pipefail
docker run --rm --read-only \\
  --cap-drop ALL \\
  --security-opt no-new-privileges:true \\
  --cpus "${SANDBOX_CPUS:-1.0}" \\
  --memory "${SANDBOX_MEMORY:-512m}" \\
  --pids-limit "${SANDBOX_PIDS_LIMIT:-128}" \\
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \\
  --tmpfs /app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700 \\
  -p "${SANDBOX_PORT:-8000}:8000" \\
  -e SANDBOX_TRAINER_API_KEY \\
  -e SANDBOX_LLM_API_KEY \\
  -e SANDBOX_LLM_BASE_URL \\
  -e SANDBOX_LLM_MODEL \\
  "${SANDBOX_IMAGE:-envfactory-sandbox}"
'''

SCAFFOLD_TEST_SOURCE = '''import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_tool_projection_matches_contract():
    contract = json.loads((ROOT / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
    tools = json.loads((ROOT / "tools.json").read_text(encoding="utf-8"))
    assert tools == contract.get("tools", [])
    assert all(item.get("type") == "function" for item in tools)


def test_sandbox_profile_matches_training_contract():
    contract = json.loads((ROOT / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "sandbox_profile.json").read_text(encoding="utf-8"))
    expected = contract.get("training_contract")
    if expected is not None:
        if expected.get("category") == "noise_resistance":
            assert profile.get("legacy_category") == "noise_resistance"
            assert profile.get("category") == "simple_agentic"
        else:
            assert profile == expected
    category = profile.get("category")
    noise_names = {
        item.get("name") or item.get("tool_name")
        for item in contract.get("noise_tools", []) if isinstance(item, dict)
    }
    business_tools = [
        item for item in contract.get("tools", [])
        if item.get("function", {}).get("name") not in noise_names
    ]
    if category == "direct_response":
        assert business_tools == []
    elif category == "simple_agentic":
        assert business_tools
    elif category == "multi_step_agentic":
        assert business_tools
'''

def generate(root: Path, *, preserve_implementation: bool = False) -> None:
    contract = json.loads((root / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
    if not isinstance(contract, dict): raise ValueError("BUILD_CONTRACT.json must be an object")
    tools = contract.get("tools", [])
    if not isinstance(tools, list): raise ValueError("BUILD_CONTRACT.json.tools must be a list")
    training = contract.get("training_contract")
    if not isinstance(training, dict):
        # Backward-compatible profile for artifacts generated before typed skeletons.
        category = contract.get("training_category", "multi_step_agentic")
        training = {"version": "legacy", "category": category, "sandbox_profile": category}
    if training.get("category") == "noise_resistance":
        training = {
            **training,
            "legacy_category": "noise_resistance",
            "category": "simple_agentic",
            "sandbox_profile": "single_tool",
        }
    profile = training.get("sandbox_profile")
    if profile not in {"direct_response", "single_tool", "dependent_tool_chain", "multi_step_agentic", "simple_agentic"}:
        raise ValueError(f"unsupported sandbox profile: {profile}")
    # Resuming preserves business extensions, never stale platform wiring.
    (root / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    if not preserve_implementation:
        (root / "task_impl.py").write_text(TASK_IMPL_SOURCE, encoding="utf-8")
    # tools.json is an immutable projection, not model-authored business code.
    # Generate it deterministically so weaker builders cannot omit or alter the
    # externally visible tool contract during delivery assembly.
    (root / "tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "sandbox_profile.json").write_text(
        json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "requirements-dev.txt").write_text(
        "iniconfig==2.3.0\n"
        "packaging==26.3\n"
        "pluggy==1.6.0\n"
        "Pygments==2.21.0\n"
        "pytest==9.1.1\n",
        encoding="utf-8",
    )
    (root / "acceptance_runner.py").write_text(ACCEPTANCE_RUNNER_SOURCE, encoding="utf-8")
    (root / "acceptance.sh").write_text(ACCEPTANCE_SH_SOURCE, encoding="utf-8")
    (root / "acceptance.sh").chmod(0o755)
    (root / "IMPLEMENTATION_REPORT.md").write_text(
        IMPLEMENTATION_REPORT_SOURCE, encoding="utf-8"
    )
    tests_dir = root / "tests"
    tests_dir.mkdir(exist_ok=True)
    scaffold_test = tests_dir / "test_scaffold_contract.py"
    if not scaffold_test.exists():
        scaffold_test.write_text(SCAFFOLD_TEST_SOURCE, encoding="utf-8")
    (root / "Dockerfile").write_text(DOCKERFILE_SOURCE, encoding="utf-8")
    (root / ".dockerignore").write_text(DOCKERIGNORE_SOURCE, encoding="utf-8")
    for name, source in (("docker_build.sh", DOCKER_BUILD_SOURCE), ("docker_run.sh", DOCKER_RUN_SOURCE)):
        path = root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--preserve-implementation", action="store_true")
    args=parser.parse_args(); generate(args.root.resolve(), preserve_implementation=args.preserve_implementation); return 0
if __name__ == "__main__": raise SystemExit(main())
