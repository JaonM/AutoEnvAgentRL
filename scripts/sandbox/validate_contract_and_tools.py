#!/usr/bin/env python3
"""Shared contract/tool/interface gate used by authoring and final construction."""
import json
import sys
from pathlib import Path


def validate(task_path: Path, contract_path: Path, tools_path: Path) -> None:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tools = json.loads(tools_path.read_text(encoding="utf-8"))

    expected_contract = {key: value for key, value in task.items() if key != "actions"}
    if contract != expected_contract:
        raise SystemExit("BUILD_CONTRACT.json 与去除 actions 后的 task.json 深度不相等")
    if not isinstance(tools, list):
        raise SystemExit("tools.json 必须是顶层 Function Tool 数组")
    if not tools and task.get("training_category") != "direct_response":
        raise SystemExit("需要工具的训练类别不允许 tools.json 为空")

    declared_tools = task.get("tools")
    if not isinstance(declared_tools, list) or tools != declared_tools:
        raise SystemExit("tools.json 与 task.json.tools 深度不相等")

    names = set()
    def check_schema(schema, location):
        if not isinstance(schema, dict):
            raise SystemExit(f"工具参数 {location} schema 必须是 object")
        if not isinstance(schema.get("description"), str) or not schema["description"].strip():
            raise SystemExit(f"工具参数 {location} 缺少 description")
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                raise SystemExit(f"工具参数 {location}.properties 必须是 object")
            required = schema.get("required", [])
            if not isinstance(required, list) or any(key not in properties for key in required):
                raise SystemExit(f"工具参数 {location}.required 无效")
            for key, value in properties.items():
                check_schema(value, f"{location}.{key}")
        elif schema.get("type") == "array" and "items" in schema:
            check_schema(schema["items"], f"{location}[]")

    for index, tool in enumerate(tools, 1):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise SystemExit(f"tools.json 第 {index} 项不是标准 function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise SystemExit(f"tools.json 第 {index} 项缺少 function")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise SystemExit(f"tools.json 工具名无效或重复：{name!r}")
        if not isinstance(function.get("description"), str) or not function["description"].strip():
            raise SystemExit(f"工具 {name} 缺少 function.description")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise SystemExit(f"工具 {name} 的 parameters 必须是 object schema")
        if not isinstance(parameters.get("properties"), dict):
            raise SystemExit(f"工具 {name} 缺少 parameters.properties")
        for key, value in parameters["properties"].items():
            check_schema(value, f"{name}.{key}")
        names.add(name)

    noise_tools = contract.get("noise_tools", [])
    if not isinstance(noise_tools, list):
        raise SystemExit("BUILD_CONTRACT.noise_tools 必须是数组")
    noise_by_name = {}
    for index, item in enumerate(noise_tools):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise SystemExit(f"noise_tools[{index}] 缺少 name")
        category = item.get("category")
        if category not in {"unrelated", "related_irrelevant"}:
            raise SystemExit(f"noise_tools[{index}] category 无效")
        if item["name"] in noise_by_name:
            raise SystemExit(f"noise_tools 工具名重复：{item['name']}")
        noise_by_name[item["name"]] = category
    if not set(noise_by_name) <= names:
        raise SystemExit("BUILD_CONTRACT.noise_tools 中存在未暴露的工具")

    interface = contract.get("requirements", {}).get("runtime_interface")
    if not isinstance(interface, dict) or interface.get("protocol") != "http":
        raise SystemExit("BUILD_CONTRACT.requirements.runtime_interface 缺少 http 约定")
    endpoints = interface.get("endpoints")
    if not isinstance(endpoints, list):
        raise SystemExit("runtime_interface.endpoints 必须是数组")
    from env_factory.contracts.runtime_contract import missing_system_endpoints
    if missing_system_endpoints(interface):
        raise SystemExit("runtime_interface 缺少系统或 reward endpoint")
    declared_tool_endpoints = [
        item for item in endpoints
        if isinstance(item, dict) and item.get("kind") == "llm_tool"
    ]
    if {item.get("name") for item in declared_tool_endpoints} != names:
        raise SystemExit("runtime_interface 未逐一声明所有 LLM tool")
    if "ask_user" in names:
        raise SystemExit("ask_user 不再是 LLM Tool，必须使用 user_simulator endpoint")

    metrics = task.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise SystemExit("task.json.metrics 必须是非空数组")
    key_steps = task.get("reward_key_steps")
    if not isinstance(key_steps, list):
        raise SystemExit("task.json.reward_key_steps 必须是数组")
    action_names = {
        str(item.get("name") or item.get("action"))
        for item in task.get("actions", [])
        if isinstance(item, dict)
    }
    key_step_ids = set()
    key_action_names = set()
    for index, step in enumerate(key_steps):
        if not isinstance(step, dict) or not isinstance(step.get("step_id"), str) or not step["step_id"]:
            raise SystemExit(f"reward_key_steps[{index}] 缺少有效 step_id")
        if step["step_id"] in key_step_ids:
            raise SystemExit(f"reward_key_steps[{index}] step_id 重复")
        if step.get("action_name") not in action_names:
            raise SystemExit(f"reward_key_steps[{index}] 引用了未知 action")
        if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
            raise SystemExit(f"reward_key_steps[{index}] 缺少 rationale")
        if not isinstance(step.get("required_for_goal"), bool):
            raise SystemExit(f"reward_key_steps[{index}] required_for_goal 必须是 boolean")
        key_step_ids.add(step["step_id"])
        key_action_names.add(step["action_name"])
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            raise SystemExit(f"metrics[{index}] 必须是 object")
        evaluator = metric.get("evaluator")
        if not isinstance(evaluator, dict):
            raise SystemExit(f"metrics[{index}] 缺少 executable evaluator")
        if not isinstance(evaluator.get("kind"), str) or not evaluator["kind"]:
            raise SystemExit(f"metrics[{index}].evaluator.kind 无效")
        if evaluator.get("source") not in {"runtime_rule", "external_llm"}:
            raise SystemExit(f"metrics[{index}].evaluator.source 无效")
        if not isinstance(evaluator.get("score_mapping"), dict) or not evaluator["score_mapping"]:
            raise SystemExit(f"metrics[{index}].evaluator.score_mapping 缺失")
        score_range = [-1, 0] if metric.get("category") == "penalty" else [0, 1]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not score_range[0] <= value <= score_range[1]
               for value in evaluator["score_mapping"].values()):
            raise SystemExit(f"metrics[{index}].evaluator.score_mapping 超出 {score_range}")
        category = metric.get("category")
        metric_type = metric.get("type")
        if category == "process":
            compiled_rule = (
                metric_type == "rule-based"
                and evaluator.get("kind") == "trajectory_rule"
                and evaluator.get("source") == "runtime_rule"
                and set(evaluator["score_mapping"]) == {"pass", "fail"}
            )
            model_rule = (
                metric_type == "hybrid"
                and evaluator.get("kind") == "hybrid_tool_call"
                and evaluator.get("source") == "external_llm"
                and evaluator.get("comparison") == "exact_tool_name_and_canonical_arguments"
                and set(evaluator["score_mapping"]) == {"match", "mismatch"}
            )
            if not (compiled_rule or model_rule):
                raise SystemExit(f"metrics[{index}] process evaluator 与声明式或外部模型契约不匹配")
            if metric.get("target_action") not in key_action_names:
                raise SystemExit(f"metrics[{index}] process target_action 必须属于 reward_key_steps")
        elif metric_type == "rule-based":
            if evaluator.get("kind") not in {"business_state_rule", "document_rule", "trajectory_rule"}:
                raise SystemExit(f"metrics[{index}] rule-based evaluator.kind 无效")
            if evaluator.get("source") != "runtime_rule" or not isinstance(evaluator.get("assertion"), str) or not evaluator["assertion"].strip():
                raise SystemExit(f"metrics[{index}] rule-based evaluator 缺少可执行 assertion")
        elif metric_type == "model-based":
            if evaluator.get("kind") != "external_llm_judge" or evaluator.get("source") != "external_llm":
                raise SystemExit(f"metrics[{index}] model-based evaluator 必须使用 external_llm_judge")
        elif metric_type == "hybrid":
            if evaluator.get("kind") != "hybrid_outcome" or evaluator.get("source") != "external_llm":
                raise SystemExit(f"metrics[{index}] hybrid evaluator.kind 无效")
            if not isinstance(evaluator.get("rule"), dict) or not isinstance(evaluator.get("external_llm"), dict):
                raise SystemExit(f"metrics[{index}] hybrid evaluator 缺少 rule/external_llm")
    if interface.get("llm_tools") != [item.get("name") for item in declared_tool_endpoints]:
        raise SystemExit("runtime_interface.llm_tools 与 endpoint 顺序不一致")
    mutation = interface.get("mutation_testing")
    if (not isinstance(mutation, dict) or mutation.get("environment_variable") != "SANDBOX_MUTATION_MODE"
            or not isinstance(mutation.get("modes"), list) or not mutation["modes"]
            or mutation.get("production_default") != "disabled"):
        raise SystemExit("runtime_interface 缺少 mutation testing 约定")
    if interface.get("launcher", {}).get("command") != ["python", "app.py", "--port", "{port}"]:
        raise SystemExit("runtime_interface launcher 必须使用 python app.py --port {port}")
    reward_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "reward_function"]
    if len(reward_endpoints) != 1 or reward_endpoints[0].get("name") != "reward":
        raise SystemExit("runtime_interface 必须声明一个 reward function")
    if reward_endpoints[0].get("access") != "rl_trainer_only":
        raise SystemExit("reward endpoint 必须仅允许 RL Trainer 调用")
    user_simulator_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "user_simulator"]
    if len(user_simulator_endpoints) != 1 or user_simulator_endpoints[0].get("name") != "user_simulator":
        raise SystemExit("runtime_interface 必须声明一个 user_simulator endpoint")
    if user_simulator_endpoints[0].get("access") != "rl_trainer_only":
        raise SystemExit("user_simulator endpoint 必须仅允许 RL Trainer 调用")
    replay_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "replay"]
    if len(replay_endpoints) != 1 or replay_endpoints[0].get("name") != "replay" or replay_endpoints[0].get("access") != "rl_trainer_only":
        raise SystemExit("runtime_interface 必须声明 Trainer-only replay endpoint")
    security = interface.get("security")
    trainer_security = security.get("trainer") if isinstance(security, dict) else None
    if not isinstance(trainer_security, dict) or trainer_security.get("scheme") != "bearer" or trainer_security.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
        raise SystemExit("runtime_interface 缺少 Trainer Bearer 鉴权约定")
    episode = interface.get("episode")
    if not isinstance(episode, dict) or episode.get("isolation") != "per_episode" or episode.get("reset_accepts_seed") is not True or episode.get("deterministic_replay") is not True:
        raise SystemExit("runtime_interface 缺少 episode 隔离/seed/replay 约定")
    llm_runtime = interface.get("llm_runtime")
    if not isinstance(llm_runtime, dict) or llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
        raise SystemExit("runtime_interface 缺少外部 LLM 凭据边界")
    evaluator_runtime = interface.get("evaluator_runtime")
    if not isinstance(evaluator_runtime, dict) or evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
        raise SystemExit("runtime_interface 缺少 evaluator mock 约定")
    errors = interface.get("errors")
    if not isinstance(errors, dict) or errors.get("content_type") != "application/json" or not isinstance(errors.get("schema"), dict):
        raise SystemExit("runtime_interface 缺少统一错误协议")
    observability = interface.get("observability")
    if not isinstance(observability, dict) or observability.get("request_id_header") != "X-Request-ID" or observability.get("credential_redaction") is not True:
        raise SystemExit("runtime_interface 缺少可观测性/凭据脱敏约定")
    if reward_endpoints[0].get("formula") != contract.get("reward_formula"):
        raise SystemExit("runtime_interface reward formula 与 task.json.reward_formula 不一致")
    acceptance = task.get("acceptance_contract")
    if not isinstance(acceptance, dict) or acceptance.get("authority") != "env_factory_outer_workflow":
        raise SystemExit("task.json 缺少 EnvFactory-owned acceptance_contract")
    for key in ("fixtures", "scenarios", "tool_cases", "invariants", "mutations", "reward_cases", "mutation_tests"):
        if not isinstance(acceptance.get(key), (list, dict)):
            raise SystemExit(f"acceptance_contract.{key} 无效")
    if not acceptance.get("invariants") or not acceptance.get("mutation_tests"):
        raise SystemExit("acceptance_contract 缺少业务不变量或 mutation tests")
    print("BUILD_CONTRACT/task.json deep equality: ok")
    print("tool schema: ok")
    print("runtime HTTP interface: ok")


if __name__ == "__main__":
    validate(*map(Path, sys.argv[1:]))
