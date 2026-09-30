"""Versioned stateful prototypes with complete write contracts and executable goals."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

from .pipeline_errors import PipelineGenerationError
from .spec_pipeline import _metric, _tool, assemble_spec

PROTOTYPES = ("lookup_update", "constraint_create")


def table(name, columns, rows, *, foreign_keys=()):
    return {"table_name": name, "description": name, "primary_key": ["id"],
            "columns": [{"name": key, "type": kind, "nullable": False} for key, kind in columns.items()],
            "foreign_keys": list(foreign_keys), "constraints": [], "indexes": [], "rows": rows}


def instantiate(*, seed: int, prototype: str, subject: str) -> dict:
    rng = random.Random(seed)
    ids = rng.sample(range(100000, 999999), 6)
    selector = f"{subject}采购申请{rng.randint(1, 3)}"
    if prototype == "lookup_update":
        tables = [table("requests", {"id": "integer", "name": "text", "status": "text"},
                        [{"id": ids[i], "name": f"{subject}采购申请{i+1}", "status": "待复核"} for i in range(3)])]
        return {"version": "1.0", "prototype": prototype, "seed": seed, "subject": subject,
                "selector": selector, "tables": tables, "goal": {"status": "已复核"}}
    if prototype != "constraint_create":
        raise PipelineGenerationError("SPEC_UNSUPPORTED: unknown stateful prototype")
    quantity, price = rng.randint(10, 30), rng.randint(30, 90)
    requests = [{"id": ids[i], "name": f"{subject}采购申请{i+1}", "quantity": quantity+i, "max_price": price}
                for i in range(3)]
    # Exactly one supplier satisfies both constraints for every sampled request.
    suppliers = [{"id": ids[3], "capacity": quantity + 10, "unit_price": price - 5},
                 {"id": ids[4], "capacity": quantity - 1, "unit_price": price - 10},
                 {"id": ids[5], "capacity": quantity + 30, "unit_price": price + 1}]
    rng.shuffle(suppliers)
    tables = [table("requests", {"id": "integer", "name": "text", "quantity": "integer", "max_price": "integer"}, requests),
              table("suppliers", {"id": "integer", "capacity": "integer", "unit_price": "integer"}, suppliers),
              table("bookings", {"id": "integer", "request_id": "integer", "supplier_id": "integer", "quantity": "integer"}, [{"id": 1, "request_id": next(r["id"] for r in requests if r["name"] != selector), "supplier_id": ids[3], "quantity": quantity}],
                    foreign_keys=[{"column": "request_id", "ref_table": "requests", "ref_column": "id"},
                                  {"column": "supplier_id", "ref_table": "suppliers", "ref_column": "id"}])]
    next(column for column in tables[2]["columns"] if column["name"] == "request_id")["unique"] = True
    return {"version": "1.0", "prototype": prototype, "seed": seed, "subject": subject,
            "selector": selector, "tables": tables, "goal": {"constraint": "capacity_gte_quantity_and_price_lte_budget"}}


def compile_spec(spec: dict, *, artifact_dir: Path, script_count: int = 3) -> dict:
    from env_factory.task_pipeline import TaskGenerationPipeline as P
    started = time.perf_counter()
    prototype = spec.get("prototype")
    if spec.get("version") != "1.0" or prototype not in PROTOTYPES:
        raise PipelineGenerationError("SPEC_UNSUPPORTED: unknown stateful prototype")
    tables = copy.deepcopy(spec["tables"])
    P._validate_data_tables(tables)
    P._validate_relational_data(tables)
    rows = [r for r in tables[0]["rows"] if r["name"] == spec["selector"]]
    if len(rows) != 1:
        raise PipelineGenerationError("SPEC_UNSOLVABLE: request selector must identify exactly one row")
    request = rows[0]
    tools = [_tool("lookup_request", "按申请名称查询内部申请记录和编号，返回 records。", {"name": "string"})]
    implementations = [{"tool_name": "lookup_request", "operation": "select", "table": "requests", "result_field": "records",
                        "filters": [{"argument": "name", "column": "name", "operator": "eq"}],
                        "projection": list(request)}]
    calls = [{"step_id": "lookup", "operation": "tool_call", "tool_name": "lookup_request",
              "arguments": {"name": spec["selector"]}, "capture": {"request_id": "$.records[0].id"}, "expected_status": 200}]
    if prototype == "lookup_update":
        if spec.get("goal") != {"status": "已复核"} or request["status"] != "待复核":
            raise PipelineGenerationError("SPEC_UNSOLVABLE: update prototype requires its declared initial and target status")
        task = f"请把内部系统中“{spec['selector']}”的状态从“待复核”改为“已复核”。先查询申请编号和当前状态，再更新该申请，保持其他申请不变。"
        tools.append(_tool("update_request", "按申请编号更新该申请的状态，返回 updated_count。", {"id": "integer", "status": "string"}))
        implementations.append({"tool_name": "update_request", "operation": "update", "table": "requests",
                                "selector": {"id": "id"}, "changes": {"status": "status"}})
        calls.append({"step_id": "update", "operation": "tool_call", "tool_name": "update_request",
                      "arguments": {"id": {"$ref": "request_id"}, "status": "已复核"}, "expected_status": 200})
        predicate = {"table": "requests", "where": {"name": spec["selector"]}, "values": {"status": "已复核"}, "count": 1}
        path = '$.requests[?(@.name==' + json.dumps(spec["selector"], ensure_ascii=False) + ')][0].status'
        outcome = {"metric_id": "outcome_request_updated", "source": "business_state", "path": path,
                   "operator": "eq", "expected": "已复核", "score_mapping": {"pass": 1, "fail": 0}}
        names, intent = ["查询申请记录", "更新申请状态"], "modify"
    else:
        if spec.get("goal") != {"constraint": "capacity_gte_quantity_and_price_lte_budget"}:
            raise PipelineGenerationError("SPEC_UNSUPPORTED: create goal differs from its prototype")
        suppliers = [r for r in tables[1]["rows"] if r["capacity"] >= request["quantity"] and r["unit_price"] <= request["max_price"]]
        if len(suppliers) != 1:
            raise PipelineGenerationError("SPEC_UNSOLVABLE: supplier constraints must identify exactly one candidate")
        supplier = suppliers[0]
        task = (f"请为内部系统中“{spec['selector']}”创建一条采购预订。先查询申请的编号、数量和最高单价，"
                "再查找可供数量不少于申请数量、单价不高于申请最高单价的供应商。系统记录中只有一家同时符合条件；"
                "每个申请只允许一条预订。预订需记录该申请编号、供应商编号和申请数量，不要修改申请、供应商或其他预订。")
        tools.extend([_tool("query_suppliers", "按最低可供数量和最高单价查询符合条件的供应商，返回 records 中的 id、capacity、unit_price。",
                            {"min_capacity": "integer", "max_price": "integer"}),
                      _tool("create_booking", "创建采购预订，保存申请编号、供应商编号和数量，返回 record。每个申请只允许一条，重复申请会被拒绝且不改变已有记录。",
                            {"request_id": "integer", "supplier_id": "integer", "quantity": "integer"})])
        implementations.extend([
            {"tool_name": "query_suppliers", "operation": "select", "table": "suppliers", "result_field": "records",
             "filters": [{"argument": "min_capacity", "column": "capacity", "operator": "gte"},
                         {"argument": "max_price", "column": "unit_price", "operator": "lte"}],
             "projection": ["id", "capacity", "unit_price"]},
            {"tool_name": "create_booking", "operation": "insert", "table": "bookings",
             "values": {"request_id": "request_id", "supplier_id": "supplier_id", "quantity": "quantity"}},
        ])
        calls[0]["capture"].update(quantity="$.records[0].quantity", max_price="$.records[0].max_price")
        calls.extend([
            {"step_id": "select", "operation": "tool_call", "tool_name": "query_suppliers",
             "arguments": {"min_capacity": {"$ref": "quantity"}, "max_price": {"$ref": "max_price"}},
             "capture": {"supplier_id": "$.records[0].id"}, "expected_status": 200},
            {"step_id": "insert", "operation": "tool_call", "tool_name": "create_booking",
             "arguments": {"request_id": {"$ref": "request_id"}, "supplier_id": {"$ref": "supplier_id"}, "quantity": {"$ref": "quantity"}}, "expected_status": 200},
        ])
        predicate = {"table": "bookings", "where": {"request_id": request["id"]},
                     "values": {"supplier_id": supplier["id"], "quantity": request["quantity"]}, "count": 1}
        path = f'$.bookings[?(@.request_id=={request["id"]})][?(@.supplier_id=={supplier["id"]})][?(@.quantity=={request["quantity"]})]'
        outcome = {"metric_id": "outcome_booking_created", "source": "business_state", "path": path,
                   "operator": "count_eq", "expected": 1, "score_mapping": {"pass": 1, "fail": 0}}
        names, intent = ["查询申请约束", "筛选供应商", "创建采购预订"], "execute"
    description = {"task": task, "task_intent": intent, "goal": task, "expected_result": "目标记录满足请求，其他业务记录保持不变",
                   "public_input": {"initial_user_message": task, "materials": []}, "complexity": "standard",
                   "requirements": {"input_modalities": ["text"], "output_modalities": ["text"]}}
    actions = [{"name": name, "description": tool["function"]["description"], "atomicity_rationale": "单次业务读取或写入",
                "inputs": [], "outputs": [], "preconditions": ["输入来自题面或前序查询"],
                "effects": ["写入目标业务记录" if i == len(names)-1 else "获得私有业务信息"]} for i, (name, tool) in enumerate(zip(names, tools))]
    bindings = [{"tool_name": tool["function"]["name"], "action_name": name} for tool, name in zip(tools, names)]
    capabilities = [{"action_name": name, "kind": "environment_operation", "requires_tool": True,
                     "dependencies": names[:i], "reason": "读取私有记录或执行必要持久化写入"} for i, name in enumerate(names)]
    steps = [{"step_id": f"step-{i+1}", "action_name": name, "dependencies": [f"step-{j+1}" for j in range(i)],
              "required_for_goal": True, "rationale": "输入与前序私有数据具有真实依赖"} for i,name in enumerate(names)]
    metrics = [_metric("process_" + tool["function"]["name"], "process", .2/len(tools), name, name) for tool,name in zip(tools,names)]
    metric = _metric(outcome["metric_id"], "outcome", .8, "当前业务状态满足目标行谓词，并保持无关记录不变")
    metric["evaluator"]["kind"] = "business_state_rule"
    metric["evaluation_inputs"] = ["business_data", "tool_results"]
    metric["criteria"] = ["以持久化业务状态验证完成，口头声称完成不能获得奖励。"]
    metrics.append(metric)
    if len(actions) < 3:
        actions.append({"name": "确认更新结果", "description": "根据 updated_count 确认结果", "atomicity_rationale": "结果判断", "inputs": [], "outputs": [], "preconditions": [], "effects": []})
        capabilities.append({"action_name": "确认更新结果", "kind": "agent_reasoning", "requires_tool": False,
                             "dependencies": [names[-1]], "reason": "根据已有返回结果确认完成"})
    reset = {"operation": "reset", "body": {"episode_id": "goal-success", "seed": spec["seed"]}, "expected_status": 200}
    reward = {"step_id": "reward", "operation": "reward", "expected_status": 200}
    scenarios = [{"scenario_id": "goal_success", "kind": "goal_success", "steps": [reset, *calls,
                  {"operation": "agent_response", "content": "已完成请求的操作。", "expected_status": 200}, reward],
                  "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]},
                 {"scenario_id": "goal_failure", "kind": "goal_failure", "steps": [reset, reward],
                  "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "lte", "expected": 0}]}]
    semantic_goal = {"row_predicates": [predicate]}
    if prototype == "lookup_update":
        semantic_goal.update(
            initial_predicates=[{"kind": "row", "table": "requests", "where": {"name": spec["selector"]},
                                 "values": {"status": "待复核"}, "count": 1}],
            success_predicates=[copy.deepcopy(predicate)],
            expected_delta=[{"table": "requests", "where": {"name": spec["selector"]},
                             "field": "status", "before": "待复核", "after": "已复核"}],
        )
    P._preview_success_tool_results(scenarios=scenarios, data_tables=tables, tool_implementations=implementations,
                                    environment_mode="stateful", tools=tools, semantic_goal=semantic_goal)
    return assemble_spec(spec, artifact_dir=artifact_dir, script_count=script_count, started=started,
        tables=tables, description=description, tools=tools, implementations=implementations, actions=actions,
        bindings=bindings, capabilities=capabilities, key_steps=steps, metrics=metrics, outcome=outcome,
        scenarios=scenarios, plan={"mode": "stateful", "requires_business_data": True, "requires_persistence": True,
                                  "reason": "查询私有记录后修改目标状态"},
        semantic_goal=semantic_goal, verification={"passed": True, "scope": "state_goal_and_preservation"})
