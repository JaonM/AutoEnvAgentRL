#!/usr/bin/env python3
"""Run executable mutation tests against one generated sandbox.

The sandbox owns its business acceptance tests, while this script owns the
mutation verdict.  A mutant is *killed* only when at least one acceptance
layer fails under that mutant.  A mutant that survives is a delivery failure
and its exact output is suitable for Code Agent repair feedback.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_factory.evidence.material_artifacts import docker_build_context_digest


def load(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(base_url: str, method: str, path: str, body: object | None = None, *, key: str | None = None) -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(base_url + path, data=data, headers=headers, method=method), timeout=3) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        return exc.code, None
    except (URLError, TimeoutError, OSError):
        return 0, None


def mutate_argument(value: object) -> object:
    """Create a materially different value while preserving its JSON shape."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "__mutation_probe__"
    if isinstance(value, list):
        if not value:
            return ["__mutation_probe__"]
        changed = list(value)
        changed[0] = mutate_argument(changed[0])
        return changed
    if isinstance(value, dict):
        changed = dict(value)
        if changed:
            key = next(iter(changed))
            changed[key] = mutate_argument(changed[key])
        else:
            changed["__mutation_probe__"] = True
        return changed
    return "__mutation_probe__"


def argument_probe_cases(task: dict) -> list[tuple[str, dict, dict]]:
    """Extract concrete tool calls from goal-critical scenarios.

    The task generator records these calls as prose so they remain readable to
    a Code Agent.  literal_eval lets the outer mutation gate reuse the
    concrete arguments without inventing task-specific tool names.
    """
    cases: list[tuple[str, dict, dict]] = []
    structured = task.get("acceptance_contract", {}).get("argument_probes", [])
    if isinstance(structured, list):
        for probe in structured:
            if not isinstance(probe, dict):
                continue
            name, args = probe.get("tool_name"), probe.get("arguments")
            if isinstance(name, str) and isinstance(args, dict) and args:
                changed = dict(args)
                field = next(iter(changed))
                changed[field] = mutate_argument(changed[field])
                cases.append((name, args, changed))
        if cases:
            return cases
    # Backward compatibility for tasks generated before argument_probes was
    # introduced. New tasks never require parsing prose.
    for scenario in task.get("acceptance_contract", {}).get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        for step in scenario.get("steps", []):
            if not isinstance(step, str) or "call_tool " not in step or " with " not in step:
                continue
            prefix, raw = step.split(" with ", 1)
            name = prefix.split("call_tool ", 1)[1].strip()
            try:
                # Scenario prose uses a compact ``field=value`` notation
                # rather than JSON.  Ignore chained values such as
                # ``from step-2`` because they are not concrete probes.
                if " from " in raw:
                    continue
                normalized = raw.replace("=true", "=True").replace("=false", "=False").replace("=null", "=None")
                expression = "{" + re.sub(r"([A-Za-z_][A-Za-z0-9_]*)=", r"'\1':", normalized) + "}"
                args = ast.literal_eval(expression)
            except (SyntaxError, ValueError):
                continue
            if isinstance(args, dict) and args:
                changed = dict(args)
                field = next(iter(changed))
                changed[field] = mutate_argument(changed[field])
                cases.append((name, args, changed))
    return cases


def argument_sensitivity(task: dict, base_url: str, key: str) -> bool:
    """Return whether at least one declared call changes under changed input."""
    found = False
    for index, (name, original, changed) in enumerate(argument_probe_cases(task)):
        request(base_url, "POST", "/v1/reset", {"episode_id": f"mutation-probe-{index}-a", "seed": 17}, key=key)
        first = request(base_url, "POST", f"/v1/tools/{name}", original)
        request(base_url, "POST", "/v1/reset", {"episode_id": f"mutation-probe-{index}-b", "seed": 17}, key=key)
        second = request(base_url, "POST", f"/v1/tools/{name}", changed)
        if first != second:
            found = True
    return found


def direct_mutation_probe(task: dict, base_url: str, key: str, mode: str) -> bool:
    """Return True when the outer workflow directly observes the mutant.

    Generated acceptance remains responsible for business invariants. These
    probes cover platform-owned mutation seams so a model cannot accidentally
    let a mutant survive merely by omitting a redundant assertion.
    """
    if mode == "bypass_trainer_auth":
        status, _ = request(base_url, "GET", "/v1/observation")
        return 200 <= status < 300
    probes = argument_probe_cases(task)
    valid_call = (probes[0][0], probes[0][1]) if probes else None
    if valid_call is None:
        scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
        for scenario in sorted(scenarios, key=lambda item: item.get("kind") != "goal_success" if isinstance(item, dict) else True):
            if not isinstance(scenario, dict) or scenario.get("kind") not in {None, "goal_success"}:
                continue
            for step in scenario.get("steps", []) if isinstance(scenario, dict) else []:
                if isinstance(step, dict) and step.get("operation") == "tool_call":
                    name, arguments = step.get("tool_name"), step.get("arguments", {})
                    if isinstance(name, str) and isinstance(arguments, dict):
                        valid_call = (name, arguments)
                        break
            if valid_call:
                break
    if mode in {"constant_tool_result", "skip_business_write"} and valid_call:
        name, arguments = valid_call
        request(base_url, "POST", "/v1/reset", {"episode_id": "direct-mutation-probe", "seed": 17}, key=key)
        status, body = request(base_url, "POST", f"/v1/tools/{name}", arguments)
        return status == 200 and isinstance(body, dict) and body.get("mutation") == mode
    if mode == "ignore_tool_arguments":
        tools = task.get("tools", [])
        if tools:
            name = tools[0].get("function", {}).get("name")
            if isinstance(name, str):
                status, _ = request(base_url, "POST", f"/v1/tools/{name}", {"__unexpected__": True})
                return 200 <= status < 300
    if mode == "constant_reward":
        request(base_url, "POST", "/v1/reset", {"episode_id": "direct-reward-probe", "seed": 17}, key=key)
        status, body = request(base_url, "GET", "/v1/reward", key=key)
        return status == 200 and isinstance(body, dict) and "__mutation__" in body.get("components", {})
    return False


def in_process_constant_tool_probe(task: dict, root: Path, env: dict[str, str], key: str) -> bool:
    """Observe the platform mutant through its public handler when TCP is unavailable."""
    probes = argument_probe_cases(task)
    if probes:
        name, arguments, _ = probes[0]
    else:
        # Parameterless tools still have a successful reference call. Do not
        # substitute an earlier scaffold scenario with deliberately bad args.
        scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
        calls = [step for scenario in sorted(scenarios, key=lambda item: item.get("kind") != "goal_success" if isinstance(item, dict) else True)
                 if isinstance(scenario, dict) and scenario.get("kind") in {None, "goal_success"}
                 for step in scenario.get("steps", [])
                 if isinstance(step, dict) and step.get("operation") == "tool_call"]
        if not calls:
            return False
        name, arguments = calls[0]["tool_name"], calls[0].get("arguments", {})
    program = """import json,sys
from pathlib import Path
from app import create_app
app=create_app(db_path=Path(sys.argv[4]))
headers={"Authorization":"Bearer "+sys.argv[3]}
status,_,_=app.handle("POST","/v1/reset",{"episode_id":"direct-mutation-probe","seed":17},headers)
if status != 200: raise SystemExit(2)
status,body,_=app.handle("POST","/v1/tools/"+sys.argv[1],json.loads(sys.argv[2]))
print(json.dumps({"status":status,"mutation":body.get("mutation") if isinstance(body,dict) else None}))
"""
    with tempfile.TemporaryDirectory(prefix="envfactory-in-process-mutant-") as directory:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", program, name, json.dumps(arguments), key,
                 str(Path(directory) / "episodes.sqlite3")],
                cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=30,
            )
            if completed.returncode != 0:
                return False
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            return result.get("status") == 200 and result.get("mutation") == "constant_tool_result"
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
            return False


def wait_health(base_url: str, process: subprocess.Popen[bytes]) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        status, _ = request(base_url, "GET", "/health")
        if 200 <= status < 300:
            return True
        time.sleep(0.1)
    return False


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_acceptance(root: Path, env: dict[str, str]) -> tuple[int, str]:
    # Generated acceptance scripts may honor PYTHON.  Pin it to the same
    # interpreter that launches the mutation runner so an inherited `PYTHON`
    # value cannot select a missing or incompatible executable.
    acceptance_env = dict(env)
    acceptance_env["PYTHON"] = sys.executable
    try:
        completed = subprocess.run(
            ["bash", "./acceptance.sh"], cwd=root, env=acceptance_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180,
        )
        return completed.returncode, completed.stdout[-12000:]
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return 124, str(output)[-12000:] + "\nacceptance.sh timeout"


def reusable_baseline(root: Path, expected_digest: str | None) -> bool:
    """Reuse an outer acceptance pass only for the exact validated context."""
    if not expected_digest or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        return False
    try:
        result = load(root, "acceptance_result.json")
        return (
            isinstance(result, dict)
            and result.get("business_acceptance") == "passed"
            and result.get("http_conformance") in {"passed", "skipped"}
            and docker_build_context_digest(root) == expected_digest
        )
    except (OSError, ValueError, TypeError):
        return False


def run_outer(root: Path, project_dir: Path, base_url: str, env: dict[str, str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="envfactory-mutant-outer-") as directory:
        completed = subprocess.run(
            [sys.executable, str(project_dir / "scripts/sandbox/generate_outer_conformance.py"),
             "--root", str(root), "--output", directory, "--check", "--base-url", base_url],
            cwd=project_dir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=60,
        )
        return completed.returncode, completed.stdout[-12000:]


def main() -> int:
    parser = argparse.ArgumentParser(description="执行沙箱自动 mutation testing")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-context-digest", default=None,
                        help="reuse a passed outer acceptance only when its build context is unchanged")
    args = parser.parse_args()
    root = args.root.resolve()
    project_dir = Path(__file__).resolve().parents[2]
    task = load(root, "task.json")
    interface = task.get("requirements", {}).get("runtime_interface", {})
    mutation = interface.get("mutation_testing", {})
    modes = mutation.get("modes")
    if not isinstance(modes, list) or not modes:
        raise SystemExit("runtime_interface.mutation_testing.modes 不能为空")
    if not (root / "app.py").is_file():
        raise SystemExit("mutation testing requires app.py")

    # A suppressed business write is meaningful only for a stateful contract.
    # Direct-response, read-only reference and external-capability sandboxes
    # have no write whose absence could be observed; requiring them to kill
    # this mutant creates an impossible and misleading acceptance gate.
    environment_plan = task.get("environment_plan", {})
    applicable_modes = []
    for mode in modes:
        if (mode in {"constant_tool_result", "ignore_tool_arguments"}
                and task.get("training_category") == "direct_response"
                and task.get("tools") == []):
            print(f"mutation not applicable: {mode} (direct response has no tools)")
            continue
        if mode == "skip_business_write" and not (
            isinstance(environment_plan, dict)
            and environment_plan.get("mode") == "stateful"
            and environment_plan.get("requires_persistence") is True
            and task.get("tool_bindings")
        ):
            print("mutation not applicable: skip_business_write (no stateful business write)")
            continue
        applicable_modes.append(mode)
    modes = applicable_modes

    trainer_key = os.environ.get("SANDBOX_TRAINER_API_KEY", "envfactory-mutation-test-key")
    base_env = os.environ.copy()
    base_env["SANDBOX_TRAINER_API_KEY"] = trainer_key
    base_env.setdefault("SANDBOX_EVALUATOR_MOCK", "true")
    base_env["SANDBOX_MUTATION_MODE"] = "disabled"

    if reusable_baseline(root, args.baseline_context_digest):
        print("mutation baseline acceptance reused from outer workflow")
    else:
        baseline_code, baseline_output = run_acceptance(root, base_env)
        if baseline_code != 0:
            print("mutation baseline acceptance failed", file=sys.stderr)
            print(baseline_output, file=sys.stderr)
            return 1

    # HTTP is an independent protocol gate, but some managed/macOS runners
    # prohibit local TCP bind. In that case acceptance.sh remains authoritative
    # for business and mutation behavior, while HTTP conformance is explicitly
    # reported as skipped instead of being misclassified as a business failure.
    http_available = True
    baseline_argument_sensitive = False
    try:
        port = free_port()
    except (PermissionError, OSError) as exc:
        http_available = False
        print(f"independent HTTP mutation checks skipped: local TCP bind unavailable: {exc}", file=sys.stderr)
    if http_available:
        baseline_log = tempfile.NamedTemporaryFile(prefix="envfactory-baseline-", suffix=".log", delete=False)
        baseline_log.close()
        baseline_process = subprocess.Popen(
            [sys.executable, "app.py", "--port", str(port)], cwd=root, env=base_env,
            stdout=open(baseline_log.name, "wb"), stderr=subprocess.STDOUT,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            if not wait_health(base_url, baseline_process):
                print(f"baseline runtime did not become healthy; log={baseline_log.name}", file=sys.stderr)
                return 1
            code, output = run_outer(root, project_dir, base_url, base_env)
            if code != 0:
                print("independent baseline conformance failed", file=sys.stderr)
                print(output, file=sys.stderr)
                return 1
            baseline_argument_sensitive = argument_sensitivity(task, base_url, trainer_key)
        finally:
            stop(baseline_process)

    survivors: list[str] = []
    records = []
    def finish(code: int) -> int:
        (root / "mutation_report.json").write_text(json.dumps({
            "version": "2.0", "passed": code == 0,
            "source_context_digest": docker_build_context_digest(root),
            "results": records,
        }, indent=2) + "\n")
        return code
    for mode in modes:
        mode = str(mode)
        env = dict(base_env)
        env["SANDBOX_MUTATION_MODE"] = mode
        acceptance_code, acceptance_output = run_acceptance(root, env)
        if acceptance_code in {124, 125, 126, 127} or acceptance_code < 0:
            print(f"mutation infrastructure_error: {mode} acceptance exit={acceptance_code}", file=sys.stderr)
            records.append({"mode": mode, "status": "infrastructure_error", "exit_code": acceptance_code})
            return finish(75)
        # Generated acceptance failure alone is not a kill witness. Require
        # an independent outer assertion or an observed mutation seam.
        outer_code, outer_output, argument_probe_failed = 0, "HTTP conformance skipped: local TCP bind unavailable", False
        direct_probe_failed = False
        log_file = None
        if http_available:
            port = free_port()
            log_file = tempfile.NamedTemporaryFile(prefix=f"envfactory-{mode}-", suffix=".log", delete=False)
            log_file.close()
            process = subprocess.Popen(
                [sys.executable, "app.py", "--port", str(port)], cwd=root, env=env,
                stdout=open(log_file.name, "wb"), stderr=subprocess.STDOUT,
            )
            try:
                base_url = f"http://127.0.0.1:{port}"
                healthy = wait_health(base_url, process)
                try:
                    outer_code, outer_output = (run_outer(root, project_dir, base_url, env)
                                                if healthy else (75, "mutant runtime did not become healthy"))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    outer_code, outer_output = 75, type(exc).__name__
                if healthy and mode == "ignore_tool_arguments" and baseline_argument_sensitive:
                    argument_probe_failed = not argument_sensitivity(task, base_url, trainer_key)
                if healthy:
                    direct_probe_failed = direct_mutation_probe(task, base_url, trainer_key, mode)
            finally:
                stop(process)
        elif mode == "constant_tool_result":
            direct_probe_failed = in_process_constant_tool_probe(task, root, env, trainer_key)

        if outer_code not in {0, 1}:
            print(f"mutation infrastructure_error: {mode}: {outer_output}", file=sys.stderr)
            records.append({"mode": mode, "status": "infrastructure_error", "exit_code": outer_code})
            return finish(75)
        killed = outer_code == 1 or argument_probe_failed or direct_probe_failed
        if killed:
            reason = ("outer conformance" if outer_code != 0 else
                      "argument sensitivity probe" if argument_probe_failed else
                      "outer direct mutation probe")
            records.append({"mode": mode, "status": "killed", "witness": reason})
            print(f"mutation killed: {mode} ({reason})")
        else:
            survivors.append(mode)
            records.append({"mode": mode, "status": "survived"})
            print(f"mutation survived: {mode}", file=sys.stderr)
            print("--- acceptance output ---", file=sys.stderr)
            print(acceptance_output[-4000:], file=sys.stderr)
            print("--- outer output ---", file=sys.stderr)
            print(outer_output[-4000:], file=sys.stderr)
            if log_file is not None:
                print(f"--- runtime log: {log_file.name} ---", file=sys.stderr)

    if survivors:
        finish(1)
        raise SystemExit("mutation testing failed; surviving mutants: " + ", ".join(survivors))
    print(f"mutation testing: ok ({len(modes)} mutants killed)")
    return finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
