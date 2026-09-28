"""External-LLM task generation pipeline.

Each stage has an independent prompt, JSON envelope, validation and retry
boundary.  The final task is assembled from the validated stage artifacts;
the LLM never writes the final task contract in one call.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import time
import uuid
import os
import hashlib
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .generation.pipeline_errors import PipelineGenerationError
from .generation.pipeline_prompts import REWARD_DESIGN_PROMPT, TOOL_DEFINITION_PROMPT
from .contracts.reward_contract import (
    declared_terminal_outcome_ids, has_literal_payload_argument,
    reward_contract_issues,
)
from .generation.pipeline_stage import (
    StageExecutor,
)
from .generation.stage_cache import StageCache
from .generation.user_simulation_contract import UserSimulationContractMixin

logger = logging.getLogger(__name__)

HIGH_STAKES_MARKERS = (
    "法律", "法规", "法条", "政策文件", "合规", "法律意见",
    "医疗", "诊断", "用药", "食物中毒", "腹痛", "呕吐", "腹泻",
    "肚子不舒服", "手脚颤抖", "震颤", "就医", "疾病", "症状", "癌症", "癌细胞", "器官移植", "移植排斥",
    "mhc", "冠状病毒", "sars", "投资", "证券", "税务", "投资建议",
    "legal", "regulation", "compliance", "legal advice", "medical",
    "diagnosis", "medication", "food poisoning", "investment",
    "investment advice", "securities", "tax",
)

HIGH_STAKES_CHEMICAL_MARKERS = (
    "乙醚", "丙酮", "甲醇", "乙醇", "甲基叔丁基醚", "有机镁化合物",
    "格氏试剂", "偏三甲苯", "溶剂", "萃取", "混合液体",
)
HIGH_STAKES_CHEMICAL_RISK_MARKERS = (
    "安全", "危险", "毒性", "暴露", "防护", "替代", "操作", "事故",
    "收缩", "膨胀", "hazard", "safety", "toxic", "exposure",
)



GENERIC_ACTION_NAMES = frozenset({
    "接收并解析用户输入", "识别用户输入", "检查信息完整性", "请求补充信息",
    "执行诊断处理", "执行处理", "格式化输出结果", "生成最终回答",
    "接收用户输入", "解析用户输入", "处理数据", "分析数据",
})

GENERIC_ACTION_MARKERS = (
    "接收并", "解析用户输入", "检查输入", "检查信息", "补充信息",
    "执行诊断处理", "执行处理", "格式化并输出", "格式化输出", "生成最终",
    "receive input", "parse user input", "check input", "process data", "format output",
)


class TaskGenerationPipeline(UserSimulationContractMixin):
    """Generate a complete task through independently validated stages."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        script_count: int = 3,
        retries: int = 3,
        noise_tool_max: int = 3,
    ) -> None:
        if (
            script_count <= 0
            or retries <= 0
            or noise_tool_max < 0
        ):
            raise ValueError("invalid pipeline counts or retries")
        self.llm = llm
        self.script_count = script_count
        self.retries = retries
        self.noise_tool_max = noise_tool_max
        revision = hashlib.sha256(b"".join(
            source.read_bytes() for source in sorted(Path(__file__).parent.rglob("*.py"))
        )).hexdigest()
        provider_hash = hashlib.sha256(str(getattr(llm, "base_url", "")).encode()).hexdigest()
        self.stage_cache = StageCache(os.getenv("ENVFACTORY_STAGE_CACHE_DIR"), revision=revision,
                                      model=str(getattr(llm, "model", "unknown")) + ":" + provider_hash)
        self.stage_executor = StageExecutor(
            llm,
            retries=retries,
            cache=self.stage_cache,
            logger=logger,
            error_type=PipelineGenerationError,
        )

    def _call(self, stage: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.stage_executor.call(
            stage,
            system,
            payload,
            jitter=random.uniform,
            sleep=time.sleep,
        )

    @staticmethod
    def _list(value: Any, name: str, *, minimum: int = 1) -> list[Any]:
        if not isinstance(value, list) or len(value) < minimum:
            raise PipelineGenerationError(f"{name} must be a list with at least {minimum} item(s)")
        return value

    def generate(
        self,
        *,
        keywords: list[str],
        task_type: str,
        style: str,
        task_intent: str = "query",
        graph_context: dict[str, Any],
        artifact_dir: str | Path | None = None,
        training_category: str = "multi_step_agentic",
        rng: random.Random | None = None,
        available_environment_modes: tuple[str, ...] | None = None,
        description_override: dict[str, Any] | None = None,
        source_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        random_source = rng or random
        supported_modes = set(available_environment_modes or (
            "stateless", "reference_data", "stateful", "external_capability"
        ))
        if not supported_modes or not supported_modes <= {
            "stateless", "reference_data", "stateful", "external_capability"
        }:
            raise PipelineGenerationError("available_environment_modes is invalid")
        pipeline_started = time.perf_counter()
        logger.info(
            "task pipeline started: task_type=%s style=%s keywords=%d scripts=%d",
            task_type,
            style,
            len(keywords),
            self.script_count,
        )
        intent_rules = {
            "query": "目标是查询、查找或整理已有信息，不生成策划方案",
            "explain": "目标是解释概念、原理或现象，不把任务改写为策划方案",
            "compare": "目标是比较两个或多个对象、方案或属性，必须体现比较维度",
            "recommend": "目标是基于条件筛选并推荐选项，必须体现选择依据",
            "diagnose": "目标是定位问题、分析原因或提出排查路径",
            "modify": "目标是修改、校正或更新已有业务数据或内容",
            "execute": "目标是执行明确操作并产生可验证结果",
            "plan": "目标是规划、安排或策划方案，允许生成执行计划",
            "summarize": "目标是汇总已有信息并形成简洁摘要，不扩展为策划任务",
            "create": "目标是从输入和业务数据创建新的内容、记录或交付物",
            "extract": "目标是从非结构化或多字段信息中提取指定事实或字段",
            "classify": "目标是按照明确标准对对象、记录或内容进行分类",
            "validate": "目标是验证内容、记录或结果是否满足明确规则",
            "audit": "目标是审查完整性、风险、合规性或异常，并输出问题清单",
            "calculate": "目标是基于业务数据执行计算并给出可复核结果",
            "estimate": "目标是基于已有信息进行估算、预测或区间判断，并说明依据",
            "schedule": "目标是根据时间、资源或约束安排具体日程，不生成泛化策划方案",
            "monitor": "目标是查看、跟踪或汇报对象的当前状态和变化",
            "troubleshoot": "目标是根据症状、日志或记录定位故障并提出处理步骤",
            "transform": "目标是将已有内容转换为指定格式、结构、语言或表达方式",
            "decide": "目标是基于约束和证据支持一个明确选择或决策",
            "simulate": "目标是基于业务数据模拟场景、过程或结果，并输出假设条件",
        }
        if task_intent not in intent_rules:
            raise PipelineGenerationError(f"unsupported task_intent: {task_intent}")
        category_rules = {
            "direct_response": "生成 simple 的直接回答任务。用户输入必须足以完成任务；不要虚构业务数据依赖或业务工具。正确策略是不调用工具，若提供噪声工具则调用它应受罚。",
            "simple_agentic": "生成 simple 的单步 Agentic 任务。必须存在一个模型常识无法替代的信息缺口或状态操作，并可由一个必要业务工具完成；工具结果必须决定最终回答。",
            "multi_step_agentic": "生成 standard 或 complex 的多步 Agentic 任务。至少需要两个业务工具，且至少一个后续工具参数或分支必须依赖前序工具的公开结果。",
        }
        if training_category not in category_rules:
            raise PipelineGenerationError(f"unsupported training_category: {training_category}")
        from .tasks.task_routing import select_training_intent
        try:
            select_training_intent(training_category, task_intent)
        except ValueError as exc:
            raise PipelineGenerationError(str(exc)) from exc
        # 1. Task description.
        has_source_urls = bool(re.search(
            r"https?://[^\s\"']+", json.dumps(graph_context, ensure_ascii=False), re.I
        ))
        source_safety_hint = ""
        if not has_source_urls:
            source_safety_hint = (
                " 当前 graph_context 不含可核验来源 URL，因此禁止把医疗诊断、用药、法律、合规、"
                "税务、证券或投资建议设为任务目标；即使关键词涉及这些领域，也必须改写成无需权威"
                "事实的低风险语言处理任务。"
            )
        description_output = {
            "task": "string", "task_intent": task_intent, "goal": "string", "context": [],
            "public_input": {"initial_user_message": "string", "materials": [{"name": "string", "mime_type": "text/plain|application/json", "content": "string"}]},
            "route_plan": {"environment_operations": [{"action_name": "string", "purpose": "string", "dependencies": []}]},
            "expected_result": "string", "complexity": "simple|standard|complex", "requirements": {},
        }
        description = description_override if description_override is not None else self._call(
            "task_description",
            f"根据主题和关键词生成真实用户任务。任务意图已经固定为 {task_intent}：{intent_rules[task_intent]}。必须严格遵循该意图，不得用其他意图替换它。训练路由已经固定为 {training_category}：{category_rules[training_category]} route_plan.environment_operations 是该任务真正必要的环境操作骨架：direct_response 必须为空；simple_agentic 必须恰好一个；multi_step_agentic 至少两个，且后续操作 dependencies 必须引用前序 action_name，表示参数、记录标识或分支条件的真实数据依赖。action_name 必须是具体业务动作，不得把分析、比较、总结或最终回答算作环境操作。题面必须让这些操作成为完成目标的必要条件。当前运行环境只支持 {sorted(supported_modes)}，不得生成依赖其他环境模式的任务；external_capability 不在列表时，禁止要求实时搜索、天气、行情、公共网络查询或未提供的外部计算服务。不得把简单任务机械拆成多个查询，也不得为满足工具数量虚构数据源。若当前关键词不足以形成该路由要求，可以忽略弱相关关键词并围绕最有信息量的关键词设计真实业务场景。必须明确目标、上下文、约束、预期结果和复杂度事实。任务必须能由用户运行时提供的信息、明确声明的业务资料或工具能力完成；不得要求 Agent 猜测价格、成分、属性或排名等未提供事实。public_input 是训练时真实交付给 Agent 的公开输入：initial_user_message 必须是完整请求；任务若提到“以下文本、用户提供的资料、给定数据、附件内容”等输入，必须把实际合成内容逐项放入 materials，不能只写“用户已提供”。业务数据库中的隐藏事实不得复制到 public_input。修改、执行或排程任务若涉及事实替换，必须在 public_input、context 或 requirements 中给出权威替换值或确定性规则；当前任务契约尚不能把未定义的新业务真值推迟到后续 User Simulator 回合，因此禁止“先问我、稍后提供、暂时没想好”等未决关键输入。当前 Agent 运行时只能提交自然语言最终回答，不能上传或返回 PDF、PNG、DOCX、XLSX、PPTX 等二进制文件；除非任务明确提供了可执行文件交付能力，否则 output_format 必须是文本、Markdown、JSON 或表格内容，禁止让 Agent 声称已经生成不可验证的文件。只有任务确实需要图片、音频、视频或文件输入时，requirements.input_modalities 才能包含对应媒体类型；否则只使用 text 或 structured_data。{source_safety_hint}",
            {"keywords": keywords, "task_type": task_type, "style": style, "task_intent": task_intent,
             "training_category": training_category,
             "graph_context": graph_context, "output": description_output},
        )
        route_error: PipelineGenerationError | None = None
        for route_contract_attempt in range(1, self.retries + 1):
            try:
                self._validate_route_plan(description.get("route_plan"), training_category)
                self._validate_route_input_boundary(description, training_category)
                self._validate_no_deferred_business_truth(description, task_intent)
                task_desc = description.get("task")
                if not isinstance(task_desc, str) or not task_desc.strip():
                    raise PipelineGenerationError("task_description.task must be non-empty")
                returned_intent = description.get("task_intent")
                if returned_intent is not None and returned_intent != task_intent:
                    raise PipelineGenerationError(
                        f"task_description.task_intent must be {task_intent!r}, got {returned_intent!r}"
                    )
                route_error = None
                break
            except PipelineGenerationError as exc:
                route_error = exc
                logger.warning(
                    "task route contract validation failed: attempt=%d/%d error=%s",
                    route_contract_attempt, self.retries, exc,
                )
                if route_contract_attempt >= self.retries:
                    break
                public_only_route = "fully solvable from public user input" in str(exc)
                repair_payload = {
                    "task_description": description,
                    "training_category": training_category,
                    "task_intent": task_intent,
                    "validation_error": str(exc),
                    "output": description_output,
                }
                repair_instruction = (
                    "只修复任务描述的训练路由契约并返回完整任务描述。"
                )
                if public_only_route:
                    # Rewriting the same fully supplied public task usually
                    # preserves the defect. Regenerate from the route and
                    # grounded context while keeping the requested intent.
                    repair_payload = {
                        "keywords": keywords,
                        "task_type": task_type,
                        "style": style,
                        "task_intent": task_intent,
                        "training_category": training_category,
                        "graph_context": graph_context,
                        "validation_error": str(exc),
                        "output": description_output,
                    }
                    repair_instruction = (
                        "重新生成完整任务描述，不复用上一版只依赖公开输入即可完成的目标。"
                        "至少一个必要操作必须读取沙箱内部业务记录，公开输入只给查询标识、"
                        "筛选条件或待处理请求；另一个操作必须真实依赖前序工具的公开结果。"
                        "后续阶段会按任务描述生成内部合成数据，不得把隐藏事实或最终答案写入 public_input。"
                    )
                repaired = self._call(
                    "task_description.route_repair",
                    repair_instruction + "必须逐字保持 task_intent；"
                    "direct_response 的 environment_operations 为空，simple_agentic 恰好一个，"
                    "multi_step_agentic 至少两个且后序 dependencies 只引用前序唯一 action_name。"
                    "每个 action_name 必须是非空、唯一、稳定的 snake_case 业务动作名；"
                    "多步任务至少一个后序动作必须真实依赖前序工具返回的私有业务字段。"
                    "不得用分析、总结、格式化或最终回答充当环境动作，也不得把私有业务真值复制到 public_input。"
                    + source_safety_hint,
                    repair_payload,
                )
                description = self._unwrap_task_description(repaired)
        if route_error is not None:
            raise route_error
        grounding_error: PipelineGenerationError | None = None
        for grounding_attempt in range(1, self.retries + 1):
            audit = self._call(
                "task_description_grounding_audit",
                "独立审查任务是否可完成且内部一致。self_contained 表示任务或明确约定的运行时用户输入提供了完成目标所需的信息；no_unprovided_facts 表示任务不要求 Agent 猜测价格、成分、属性、排名或其他事实；expected_result_derivable 表示预期结论可以从已声明输入、业务资料或工具能力推出；internally_consistent 表示 task、goal、context、expected_result、requirements 全部描述同一组对象、约束和交付物，不得混入其他任务的地点、产品、人物或字段。允许任务明确要求信息不足时向用户澄清。issues 必须具体。",
                {
                    "task_description": description,
                    "output": {
                        "self_contained": True,
                        "no_unprovided_facts": True,
                        "expected_result_derivable": True,
                        "internally_consistent": True,
                        "issues": [],
                    },
                },
            )
            audit = self._unwrap_structured_output(
                audit,
                expected_fields={
                    "self_contained", "no_unprovided_facts",
                    "expected_result_derivable", "internally_consistent", "issues",
                },
            )
            try:
                self._validate_task_grounding_audit(audit)
                self._validate_task_description_consistency(description)
                self._validate_task_generation_scope(
                    description, graph_context=graph_context
                )
                grounding_error = None
                break
            except PipelineGenerationError as exc:
                grounding_error = exc
                logger.warning(
                    "task description grounding failed: attempt=%d/%d error=%s",
                    grounding_attempt, self.retries, exc,
                )
                if grounding_attempt >= self.retries:
                    break
                repair_payload: dict[str, Any] = {
                    "task_description": description,
                    "grounding_issues": [str(exc)] + list(audit.get("issues", [])),
                    "task_intent": task_intent,
                    "output": description,
                }
                repair_instruction = ""
                if (
                    "high-stakes task requires authoritative source URLs" in str(exc)
                    and not has_source_urls
                ):
                    # Avoid anchoring a smaller model on the unsafe domain it
                    # repeatedly failed to remove.  Preserve the validated route
                    # shape, not the rejected task prose.
                    repair_payload = {
                        "keywords": keywords,
                        "task_type": task_type,
                        "style": style,
                        "task_intent": task_intent,
                        "training_category": training_category,
                        "required_route_plan": description.get("route_plan"),
                        "grounding_issues": [str(exc)],
                        "output": description_output,
                    }
                    repair_instruction = (
                        "这是重新生成而不是改写：不要复用上一版任务的领域、对象或高风险措辞；"
                        "仅保留 required_route_plan 的动作数量、依赖关系和固定 task_intent，"
                        "围绕安全关键词生成全新的低风险业务任务。"
                    )
                repaired_description = self._call(
                    "task_description.grounding_repair",
                    "修复任务描述，使任务能够由用户运行时输入、明确业务资料或工具能力完成。补充缺失的权威值或规则，或明确要求 Agent 在信息不足时向用户澄清；不得直接编造隐藏答案。必须逐条解决 grounding_issues 中的真实校验错误。若错误指出缺少权威来源，必须彻底移除医疗诊断、食物中毒判断、用药、法律、合规、税务、证券或投资建议等高风险目标，并在保持 task_intent 的前提下改写为无需权威事实的低风险主题；仅改措辞但保留高风险目标不算修复。不得要求输出思考过程、推理过程或隐藏思维链，只能要求简短结论依据。保持原 task_intent 不变，返回完整任务描述对象。" + repair_instruction,
                    repair_payload,
                )
                description = self._unwrap_task_description(repaired_description)
                if (
                    not isinstance(description.get("task"), str)
                    or not description["task"].strip()
                    or description.get("task_intent", task_intent) != task_intent
                    or description.get("complexity") not in {"simple", "standard", "complex"}
                ):
                    raise PipelineGenerationError(
                        "repaired task description is structurally invalid"
                    )
                self._validate_route_plan(
                    description.get("route_plan"), training_category
                )
                self._validate_route_input_boundary(description, training_category)
                self._validate_no_deferred_business_truth(description, task_intent)
        if grounding_error is not None:
            if "hidden chain-of-thought" in str(grounding_error):
                description = self._normalize_reasoning_request(description)
                self._validate_task_description_consistency(description)
                try:
                    self._validate_task_generation_scope(
                        description, graph_context=graph_context
                    )
                    grounding_error = None
                except PipelineGenerationError as exc:
                    grounding_error = exc
            if (
                grounding_error is not None
                and "high-stakes task requires authoritative source URLs" in str(grounding_error)
                and not has_source_urls
            ):
                description = self._normalize_unsourced_governance_task(description)
                try:
                    self._validate_task_description_consistency(description)
                    self._validate_task_generation_scope(
                        description, graph_context=graph_context
                    )
                    grounding_error = None
                    logger.warning(
                        "normalized unsourced governance wording to internal business rules"
                    )
                except PipelineGenerationError as exc:
                    grounding_error = exc
            if grounding_error is not None:
                raise grounding_error
        # Repairs must not silently replace failed tasks with unrelated templates.
        task_desc = description["task"]
        complexity = description["complexity"]
        requirements = description.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {"input_modalities": ["text"]}
        public_input = self._normalize_public_input(description)
        description["public_input"] = public_input
        input_modalities = requirements.get("input_modalities", ["text"])
        input_media_required = isinstance(input_modalities, list) and any(
            item in {"image", "audio", "video", "file"} for item in input_modalities
        )
        deliverable_required = self._requires_file_deliverable(description)
        media_required = input_media_required or deliverable_required

        environment_candidate = self._call(
            "environment_plan",
            "判断任务运行时真正需要的环境模式。stateless 表示只处理用户提供的文本或结构化输入，不需要预置业务数据或持久化；reference_data 表示需要只读业务资料；stateful 表示任务明确要求创建、修改、审批、排程、交易或持久化业务状态；external_capability 表示核心依赖搜索、天气、计算或其他外部能力。training_category=direct_response 时必须选择 stateless；其他训练类别必须选择能支持必要业务工具的非 stateless 模式。不要因为后续对话可能提出扩展请求而选择 stateful；只依据 task_description 中明确的任务目标和预期结果。",
            {
                "task_description": description, "training_category": training_category,
                "task_intent": task_intent,
                "output": {"mode": "stateless|reference_data|stateful|external_capability", "requires_business_data": False, "requires_persistence": False, "reason": "string"},
            },
        )
        environment_plan = self._resolve_environment_plan(
            environment_candidate, task_description=description, task_intent=task_intent
        )
        if training_category == "direct_response":
            environment_plan = {
                "mode": "stateless",
                "requires_business_data": False,
                "requires_persistence": False,
                "reason": "direct_response curriculum route",
            }
        else:
            environment_plan = self._align_environment_plan_with_route(
                environment_plan,
                route_plan=description["route_plan"],
                task_intent=task_intent,
                supported_modes=supported_modes,
            )
        if environment_plan["mode"] not in supported_modes:
            raise PipelineGenerationError(
                "task buildability: environment mode "
                f"{environment_plan['mode']!r} is unavailable; supported={sorted(supported_modes)}"
            )
        self._validate_public_input(
            task_description=description,
            public_input=public_input,
            environment_mode=environment_plan["mode"],
        )
        self._validate_authoritative_task_input(
            task_description=description,
            environment_mode=environment_plan["mode"],
            graph_context=graph_context,
        )

        materialized_dir = Path(artifact_dir) if artifact_dir is not None else Path("output/task_artifacts") / f"task-{uuid.uuid4().hex}"
        if environment_plan["mode"] in {"stateless", "external_capability"}:
            entity_plan = {"entities": []}
            table_definitions: list[dict[str, Any]] = []
            data_tables: list[dict[str, Any]] = []
            records: list[Any] = []
            data_document = "# 无业务数据环境\n\n该任务不预置业务实体、数据表或持久化业务状态。\n"
            data_manifest = self._materialize_business_data(
                [], data_document, materialized_dir / "data" / "business_data",
                environment_mode=environment_plan["mode"],
                manifest_root="data/business_data",
                data_governance=source_data.get("data_governance") if source_data else None,
            )
        elif source_data is not None:
            # Dataset trials keep the supplied business rows authoritative.
            # Downstream tools, rewards, scenarios and TaskSpec use these rows
            # through the same path as model-generated business data.
            entity_plan = {"entities": source_data.get("entities", [])}
            data_tables = source_data.get("data_tables")
            if not isinstance(data_tables, list) or not data_tables:
                raise PipelineGenerationError("source_data requires non-empty data_tables")
            self._validate_relational_data(data_tables)
            table_definitions = [
                {key: value for key, value in table.items() if key != "rows"}
                for table in data_tables
            ]
            records = []
            data_document = source_data.get("data_document")
            if not isinstance(data_document, str) or not data_document.strip():
                raise PipelineGenerationError("source_data requires data_document")
            data_manifest = self._materialize_business_data(
                data_tables,
                data_document,
                materialized_dir / "data" / "business_data",
                environment_mode=environment_plan["mode"],
                manifest_root="data/business_data",
                data_governance=source_data.get("data_governance"),
            )
        else:
            # 2a. Analyze business entities. This stage does not generate state or perform web search.
            entity_plan = self._call(
            "environment_entities",
            "根据任务描述分析完成任务所需的最小必要业务实体。只保留完成任务、支持 Agent 查询或修改、以及奖励评测真正需要持久化的业务事实；不需要独立查询、复用或更新的静态说明、标签和建议作为其他实体的字段或 JSON 保存。只有存在独立生命周期、独立查询/更新需求或明确业务关系时才拆分实体。输出实体、用途、必须保存的业务事实和实体关系，实体必须足够覆盖完整任务但遵循最小必要原则。",
            {"task_description": description, "task_type": task_type, "keywords": keywords,
             "output": {"entities": [{"entity_id": "string", "name": "string", "description": "string", "required_facts": [], "relationships": []}]}},
        )
            self._validate_entities(entity_plan.get("entities"))

        # 2b. Design atomic table schemas, without rows yet.
            table_output = {"tables": [{
                "table_name": "string", "description": "string",
                "columns": [{"name": "string", "type": "string", "description": "string", "nullable": False}],
                "primary_key": ["id"],
                "foreign_keys": [{"column": "parent_id", "ref_table": "parent", "ref_column": "id"}],
                "indexes": [], "constraints": ["status IN ('active','inactive')"],
            }]}
            table_definitions = None
            table_error: PipelineGenerationError | None = None
            for table_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if table_error is not None:
                    repair_hint = (
                        f" 上一版表结构未通过校验：{table_error}。"
                        "foreign_keys 每项必须使用 column、ref_table、ref_column；"
                        "唯一性必须使用 unique index，非空必须使用 nullable=false；"
                        "constraints 只能写可执行 SQL CHECK 表达式，不得写中文说明或‘必须为…之一’。"
                    )
                table_design = self._call(
                    "environment_table_design" if table_attempt == 1 else "environment_table_design.repair",
                    "根据任务描述和最小必要业务实体设计可持久化的原子数据库表。对需要持久化的关系数据遵循第三范式（3NF）：每个表表达一个清晰主题，字段依赖候选键、依赖整个键且不通过非键字段传递依赖；使用主键、外键和必要的关联表表达关系。外键严格使用 column、ref_table、ref_column；唯一性使用 unique index，非空使用 nullable=false；constraints 只允许 field IN (...) 或 field 与常量的比较表达式，可用 AND 连接，禁止自然语言约束。优先使用最少数量的表完整覆盖任务；只有确有独立生命周期、独立访问需求或必要的一对多/多对多业务关系时才拆表，否则将信息作为字段、枚举、JSON 或文本保存。如果任务要求推荐唯一最佳项、排序、判断是否合规或选择首选项，表结构必须包含足以确定该结论的优先级、适配分数、首选标记、规则结果或理由字段，不能只建立无方向的多对多关联。法律、法规、政策、医疗、投资、税务等权威参考资料必须带 source_url、retrieved_at 和 content_hash 字段，且来源必须可追溯；不得由模型凭空编写权威原文。每张表说明存在必要性，并声明主键、外键、字段类型、可见性、索引和约束。输出表定义。" + repair_hint,
                    {"task_description": description, "entities": entity_plan["entities"],
                     "previous_tables": table_definitions, "output": table_output},
                )
                try:
                    candidate_tables = self._normalize_structural_constraints(
                        table_design.get("tables")
                    )
                    self._validate_table_definitions(candidate_tables)
                    self._validate_authoritative_source_schema(
                        task_description=description,
                        table_definitions=candidate_tables,
                    )
                    table_definitions = candidate_tables
                    table_error = None
                    break
                except PipelineGenerationError as exc:
                    table_definitions = table_design.get("tables")
                    table_error = exc
                    logger.warning(
                        "environment table schema invalid: attempt=%d/%d error=%s",
                        table_attempt, self.retries, exc,
                    )
            if table_error is not None:
                raise table_error

        # 2c. Generate each table's rows independently so tables can be built
        # concurrently and retried without regenerating unrelated data.
            generated_tables: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(8, len(table_definitions))) as executor:
                futures = [executor.submit(
                    self._generate_table_rows, table,
                    task_description=description, entities=entity_plan["entities"],
                ) for table in table_definitions]
                for future in as_completed(futures):
                    generated_tables.append(future.result())
            generated_tables.sort(key=lambda table: str(table.get("table_name", "")))
            data_tables = generated_tables
            self._validate_data_tables(data_tables)

        # 2d. Check cross-table consistency and return actual business records.
            consistency = self._call(
            "environment_data_consistency",
            "检查并修正完整初始业务表数据，校验主键唯一、外键存在、字段类型、必填字段、业务关系和任务覆盖度。必须跨表复核冗余汇总字段：数量、总额、当前状态、有效对象数等若可由明细表计算，必须与明细状态以及日期/季度等时间边界一致；不能一张表声明对象已关闭或失效，另一张较晚快照仍把它计入当前总数。stateful 任务的 rows 是每个 episode reset 后的执行前基线，不是任务完成后的最终状态；任务明确要求从旧值改为新值时，基线必须包含旧值且不得提前包含目标终态。任务要求唯一推荐、排序、合规判断或首选结论时，数据必须提供唯一且可追溯的决定性证据；不得同时保留多个等价候选却在预期答案中武断指定其中一个。所有结论所引用的数值、属性和理由必须与 rows 精确一致。返回修正后的 data_tables 与 records。",
            {"task_description": description, "entities": entity_plan["entities"], "tables": data_tables,
             "output": {"data_tables": data_tables, "records": []}},
        )
            data_tables = consistency.get("data_tables")
            self._validate_data_tables(data_tables)
            records = consistency.get("records", [])
            if not isinstance(records, list):
                raise PipelineGenerationError("environment_data_consistency.records must be a list")

            grounding_error: PipelineGenerationError | None = None
            for grounding_attempt in range(1, self.retries + 1):
                audit = self._call(
                    "environment_data_grounding_audit",
                    "独立审查业务数据是否足以完成任务。task_supported 表示 rows 覆盖任务所需事实；decision_determinate 表示任务要求唯一推荐、排序、合规判断或首选结论时，数据存在唯一且可追溯的决定性证据，没有多个等价候选；facts_consistent 表示输入事实彼此不矛盾，且预期结论可由 rows 直接读取或通过题面明确规则确定性计算得到。逐表重算可验证的 count、total、current、active 等冗余汇总，并结合日期、季度、关闭/失效状态检查跨表时间一致性；明细与汇总不一致必须判 false。不得因为 rows 未预先存储汇总值、对比表、计算结果或最终答案而判 false；这些应由 Agent 调用工具后推导。只有原始事实矛盾、缺少计算所需输入或预期结论无法由数据推导时才判 false。若任务不要求唯一决策，decision_determinate 应为 true。issues 必须具体说明问题。",
                    {
                        "task_description": description,
                        "data_tables": data_tables,
                        "output": {
                            "task_supported": True,
                            "decision_determinate": True,
                            "facts_consistent": True,
                            "issues": [],
                        },
                    },
                )
                audit = self._unwrap_structured_output(
                    audit,
                    expected_fields={
                        "task_supported", "decision_determinate",
                        "facts_consistent", "issues",
                    },
                )
                try:
                    self._validate_data_grounding_audit(audit)
                    self._validate_relational_data(data_tables)
                    self._validate_data_keyword_alignment(
                        data_tables,
                        task_description=description,
                        keywords=keywords,
                    )
                    self._validate_stateful_preconditions(
                        data_tables,
                        task_description=description,
                        environment_mode=environment_plan["mode"],
                    )
                    grounding_error = None
                    break
                except PipelineGenerationError as exc:
                    grounding_error = exc
                    logger.warning(
                        "environment data grounding failed: attempt=%d/%d error=%s",
                        grounding_attempt, self.retries, exc,
                    )
                    if grounding_attempt >= self.retries:
                        break
                    repaired = self._call(
                        "environment_data_consistency.repair",
                        "根据独立 grounding 审计问题和确定性校验错误修正表结构与 rows。必须保持主键、外键和字段类型合法，数据内容必须直接覆盖 task_description 中的任务主题和 required_grounding_keywords。跨表冗余数量、总额、当前/有效状态必须与明细行及其时间边界一致；优先删除非必要冗余汇总，保留时必须能从明细确定性复算。stateful 任务必须保留执行前基线：从旧值改为新值时 rows 必须含旧值、不得提前含目标终态。并让推荐、排序、合规判断或首选结论具有唯一、可追溯且数值一致的证据。返回完整 data_tables 与 records。",
                        {
                            "task_description": description,
                            "data_tables": data_tables,
                            "grounding_issues": audit.get("issues", []),
                            "validation_error": str(exc),
                            "required_grounding_keywords": self._task_relevant_keywords(
                                description, keywords
                            ),
                            "output": {"data_tables": data_tables, "records": records},
                        },
                    )
                    data_tables = repaired.get("data_tables")
                    try:
                        self._validate_data_tables(data_tables)
                    except PipelineGenerationError as repair_error:
                        grounding_error = repair_error
                        logger.warning(
                            "environment data repair remained structurally invalid: attempt=%d/%d error=%s",
                            grounding_attempt, self.retries, repair_error,
                        )
                        continue
                    records = repaired.get("records", [])
                    if not isinstance(records, list):
                        raise PipelineGenerationError(
                            "environment_data_consistency.repair.records must be a list"
                        )
            if grounding_error is not None:
                raise grounding_error

        # 2e. Write the persistence handoff document from final tables.
            document_prompt = "根据已验证的初始业务表和完整 rows 编写给 Code Agent 的 data_document。说明这些 rows 是每个 episode reset 后、Agent 执行任务之前的基线状态，并说明每张表的用途、字段、类型、可见性、主键、外键、索引、约束、初始化顺序、关系和持久化要求。不得把目标终态描述成初始化数据。返回非空、完整、可执行的 Markdown 文档，至少包含每张表的初始化说明和字段说明。"
            document_payload = {"task_description": description, "entities": entity_plan["entities"], "data_tables": data_tables,
                            "output": {"data_document": "# 业务数据说明\n"}}
            data_document: str | None = None
            document_error: PipelineGenerationError | None = None
            for document_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if document_error is not None:
                    repair_hint = f"\n上一版文档未通过校验，必须修复以下错误后重新输出完整文档：{document_error}"
                document = self._call(
                "environment_data_document" if document_attempt == 1 else "environment_data_document.repair",
                document_prompt + repair_hint,
                document_payload,
            )
                candidate_document = document.get("data_document")
                if isinstance(candidate_document, str) and candidate_document.strip():
                    data_document = candidate_document
                    break
                document_error = PipelineGenerationError("environment_data_document requires a non-empty data_document")
                logger.warning(
                "environment data document validation failed: attempt=%d/%d error=%s",
                document_attempt,
                self.retries,
                document_error,
            )
            if data_document is None:
                raise document_error or PipelineGenerationError("environment_data_document requires a non-empty data_document")
            data_manifest = self._materialize_business_data(
                data_tables,
                data_document,
                materialized_dir / "data" / "business_data",
                environment_mode=environment_plan["mode"],
                manifest_root="data/business_data",
            )

        media_generation = {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""}
        if media_required:
            media_error: PipelineGenerationError | None = None
            for media_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if media_error is not None:
                    repair_hint = (
                        f"\n上一版未通过校验：{media_error}。请返回修复后的完整 media_generation；"
                        "不得返回说明文字或省略必填字段。"
                    )
                media_result = self._call(
                    "environment_media_generation" if media_attempt == 1 else "environment_media_generation.repair",
                    "根据任务描述和最终业务数据编写可执行的 Python 交付程序。任务要求输出 PDF、图片、音频、视频或其他文件时，程序必须真实生成该交付文件；任务依赖用户运行时提供的图片、文件或结构化数据时，程序必须通过命令行参数或明确输入路径读取它们，不得硬编码模拟业务数据、伪造用户照片或用占位内容冒充结果。允许生成明确标识为 acceptance fixture 的测试输入，但测试输入与运行时交付逻辑必须分离。代码必须包含 main 入口、创建输出目录并返回依赖、入口和输出目录。媒体识别和质量评测不属于本阶段。" + repair_hint,
                    {"task_description": description, "data_tables": data_tables, "data_document": data_document,
                     "output": {"media_generation": media_generation | {"required": True}}},
                )
                media_result = self._unwrap_structured_output(
                    media_result, expected_fields={"media_generation"}
                )
                candidate_media = media_result.get("media_generation") if isinstance(media_result, dict) else None
                try:
                    if (
                        not isinstance(candidate_media, dict)
                        or candidate_media.get("required") is not True
                        or candidate_media.get("language") != "python"
                        or not isinstance(candidate_media.get("code"), str)
                        or not candidate_media["code"].strip()
                        or not isinstance(candidate_media.get("dependencies", []), list)
                        or not isinstance(candidate_media.get("entrypoint"), str)
                        or not candidate_media.get("entrypoint", "").strip()
                        or not isinstance(candidate_media.get("output_dir"), str)
                    ):
                        raise PipelineGenerationError(
                            "media task requires executable Python media_generation code, dependencies, entrypoint and output_dir"
                        )
                    self._validate_media_generation(
                        candidate_media,
                        input_media_required=input_media_required,
                        deliverable_required=deliverable_required,
                        task_description=description,
                    )
                    media_generation = candidate_media
                    media_error = None
                    break
                except PipelineGenerationError as exc:
                    media_error = exc
                    logger.warning(
                        "media generation validation failed: attempt=%d/%d error=%s",
                        media_attempt, self.retries, exc,
                    )
            if media_error is not None:
                raise media_error
        if not media_required:
            media_generation = {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""}
        environment = {"records": records, "data_tables": data_tables, "data_document": data_document,
                       "media_generation": media_generation}
        # Do not embed the generated rows (or the consistency-stage record
        # summary) in task.json. Downstream stages already received the full
        # in-memory data; the persisted task only needs the manifest.
        environment_summary = {"records": [], "data_manifest": data_manifest,
                               "media_generation": media_generation}

        # 3. User profiles and task-specific scripts are deliberately generated
        # by separate prompts. Profiles are free-form persona variation and do
        # not need to match the task; scripts are task-specific behavior plans
        # grounded in the generated business environment.
        profiles_result = self._call(
            "user_profiles",
            "自由设计多样、真实且彼此有明显差异的用户画像。画像独立于当前任务和业务环境，不要把任务关键词、业务实体或隐藏真值写入画像。每个画像使用结构化字段：profile_id、identity_summary、age_range、occupation_or_life_stage、location_context、education_background、domain_knowledge、goals_and_motivations、communication_style、language_habits、decision_style、risk_tolerance、patience_level、trust_level、information_disclosure_style、questioning_style、feedback_style、budget_or_resource_sensitivity、time_sensitivity、accessibility_needs、frustration_triggers、misconceptions_or_biases、known_facts、unknown_facts、behavior_tendencies。数组字段至少提供 2 项，枚举或等级字段要给出清晰值和简短解释；画像应能指导 User LLM 在对话中表现出不同的措辞、节奏、追问、犹豫、接受和拒绝行为。只输出画像对象。",
            {"count": self.script_count, "output": {"user_profiles": [{
                "profile_id": "profile-1",
                "identity_summary": "...",
                "age_range": "...",
                "occupation_or_life_stage": "...",
                "location_context": "...",
                "education_background": "...",
                "domain_knowledge": {"level": "...", "areas": [], "evidence": "..."},
                "goals_and_motivations": [],
                "communication_style": {"tone": "...", "verbosity": "...", "directness": "..."},
                "language_habits": [],
                "decision_style": {"pattern": "...", "needs": []},
                "risk_tolerance": "...",
                "patience_level": "...",
                "trust_level": "...",
                "information_disclosure_style": "...",
                "questioning_style": "...",
                "feedback_style": "...",
                "budget_or_resource_sensitivity": "...",
                "time_sensitivity": "...",
                "accessibility_needs": [],
                "frustration_triggers": [],
                "misconceptions_or_biases": [],
                "known_facts": [],
                "unknown_facts": [],
                "behavior_tendencies": []
            }]}},
        )
        profiles = self._list(profiles_result.get("user_profiles"), "user_profiles", minimum=self.script_count)
        normalized_profiles: list[dict[str, Any]] = []
        seen_profile_ids: set[str] = set()
        for index, raw_profile in enumerate(profiles[:self.script_count], start=1):
            if not isinstance(raw_profile, dict):
                raise PipelineGenerationError(f"user_profiles[{index - 1}] must be an object")
            profile = dict(raw_profile)
            profile_id = profile.get("profile_id")
            profile_id = profile_id.strip() if isinstance(profile_id, str) else ""
            if not profile_id or profile_id in seen_profile_ids:
                profile_id = f"profile-{index}"
                while profile_id in seen_profile_ids:
                    profile_id = f"profile-{index + len(seen_profile_ids)}"
            profile["profile_id"] = profile_id
            seen_profile_ids.add(profile_id)
            normalized_profiles.append(profile)
        profiles = normalized_profiles
        # FSM topology is an executable protocol owned by EnvFactory, not
        # creative task content.  Asking a medium model to regenerate the same
        # graph caused avoidable cycles, missing outcome classes and invalid
        # terminal edges.  Keep task-specific wording in the goal/profile, but
        # construct the complete eight-outcome protocol deterministically.
        user_scripts = self._deterministic_user_scripts(
            description=description,
            count=self.script_count,
        )

        # Runtime conversations are generated by ContractUserSimulator. The
        # task artifact contains only its reusable persona and FSM inputs;
        # pre-generated transcripts would be unused, duplicate truth sources.
        user_simulation_manifest = self._materialize_user_simulation(
            profiles,
            user_scripts,
            materialized_dir / "data" / "user_simulation",
            manifest_root="data/user_simulation",
        )

        # 5. Decompose actions from the stable task/environment contract. User
        # simulation is a downstream policy stress test and must not redefine
        # business capabilities or tools.
        business_model = {
            "entities": entity_plan["entities"],
            "tables": [
                {
                    "table_name": table["table_name"],
                    "description": table["description"],
                    "columns": table["columns"],
                    "primary_key": table.get("primary_key", []),
                    "foreign_keys": table.get("foreign_keys", []),
                    "indexes": table.get("indexes", []),
                    "constraints": table.get("constraints", []),
                }
                for table in table_definitions
            ],
        }
        # Freeze machine-checkable business goals before actions, tools,
        # rewards and acceptance are proposed by independent model calls.
        semantic_goal = None
        if environment_plan["mode"] == "stateful":
            from .tasks.task_spec import TaskSpecError, validate_goal_contract
            goal_error = None
            for attempt in range(self.retries):
                candidate = self._call("business_goal_contract", "把题面明确要求的最终业务状态转换为 row_predicates。每项 table/where 定位目标，values 是期望字段值，count 是同时满足 where 和 values 的记录数；删除用 count=0，新增用目标属性定位。仅使用给定表字段和题面目标；初始数据必须尚未满足完整目标。禁止凭空扩大目标。", {
                    "task_description": description, "tables": data_tables,
                    "validation_error": goal_error,
                    "output": {"row_predicates": [{"table": "string", "where": {}, "values": {}, "count": 1}]},
                })
                try:
                    validate_goal_contract(candidate, data_tables)
                    semantic_goal = {"row_predicates": candidate["row_predicates"], "requires_state_change": True}
                    break
                except TaskSpecError as exc:
                    goal_error = str(exc)
            if semantic_goal is None:
                raise PipelineGenerationError(f"business goal contract invalid: {goal_error}")
        action_payload = {
            "goal_fields": [{"table": item["table"], "fields": sorted(set(item["where"]) | set(item["values"]))}
                            for item in (semantic_goal or {}).get("row_predicates", [])],
            "task_description": description, "business_model": business_model,
            "environment_plan": environment_plan,
            "route_plan": description["route_plan"],
            "required_grounding_keywords": self._task_relevant_keywords(
                description, keywords
            ),
            "output": {"actions": [{"name": "string", "description": "string", "atomicity_rationale": "string", "inputs": [{"name": "string", "description": "string"}], "outputs": [{"name": "string", "description": "string"}], "preconditions": ["string"], "effects": ["string"]}]},
        }
        action_list: list[dict[str, Any]] | None = None
        action_error: PipelineGenerationError | None = None
        for action_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if action_error is not None:
                repair_hint = (
                    f"\n上一版动作未通过校验：{action_error}。必须重新输出完整 actions，"
                    "删除与任务实体、输入和目标无关的动作，并补齐遗漏的目标步骤。"
                )
            actions = self._call(
                "agent_actions" if action_attempt == 1 else "agent_actions.repair",
                "根据任务描述、环境计划、route_plan 和业务模型反推 Agent 必须做的原子动作。route_plan.environment_operations 中每个 action_name 必须作为同名动作逐一出现，不得删除、合并或改名；其 dependencies 表示真实的输入输出依赖。不得把用户模拟阶段可能出现的临时扩展请求提升为正式动作。environment_plan.mode=stateless 时禁止生成任何读取或修改数据库、持久化状态或调用外部系统的动作。凡是仅根据用户输入和通用语言能力进行识别、比较、提取、格式化和最终表达的步骤都属于 Agent 自身推理或回答。每个输出动作必须不可再拆，并提供 atomicity_rationale、inputs、outputs、preconditions、effects。动作中的每个业务对象、数值来源和操作都必须能在 task_description 或 business_model 中找到依据，严禁复用其他任务的实体。" + repair_hint,
                action_payload,
            )
            try:
                candidate_actions = self._list(actions.get("actions"), "agent_actions")
                candidate_actions = self._normalize_action_numeric_examples(
                    candidate_actions,
                    task_description=description,
                    grounding_context={
                        "constraints": [
                            constraint
                            for table in business_model.get("tables", [])
                            if isinstance(table, dict)
                            for constraint in table.get("constraints", [])
                        ]
                    },
                )
                self._validate_actions(candidate_actions)
                self._validate_action_grounding(
                    candidate_actions,
                    task_description=description,
                    keywords=keywords,
                )
                self._validate_route_action_coverage(
                    candidate_actions, route_plan=description["route_plan"]
                )
                if environment_plan["mode"] == "external_capability" and not any(
                    any(marker in f"{item.get('name', '')} {item.get('description', '')}".lower()
                        for marker in ("查询", "检索", "搜索", "获取", "lookup", "search", "retrieve"))
                    for item in candidate_actions if isinstance(item, dict)
                ):
                    raise PipelineGenerationError(
                        "external-capability task requires an explicit lookup action"
                    )
                action_audit = self._call(
                    "agent_actions.audit",
                    "独立审查动作是否与任务严格对齐。aligned 表示没有引入任务中不存在的实体、数据集或目标；complete 表示动作足以完成任务目标；atomic 表示动作边界清晰。issues 必须具体指出漂移或遗漏。",
                    {
                        "task_description": description,
                        "business_model": business_model,
                        "actions": candidate_actions,
                        "output": {
                            "aligned": True, "complete": True,
                            "atomic": True, "issues": [],
                        },
                    },
                )
                self._validate_action_alignment_audit(action_audit)
                action_list = candidate_actions
                action_error = None
                break
            except PipelineGenerationError as exc:
                action_error = exc
                action_payload["previous_invalid_actions"] = (
                    candidate_actions if "candidate_actions" in locals() else []
                )
                action_payload["validation_error"] = str(exc)
                logger.warning(
                    "agent action validation failed: attempt=%d/%d error=%s",
                    action_attempt, self.retries, exc,
                )
        if action_list is None:
            logger.warning(
                "agent action proposals remained invalid; using deterministic baseline: %s",
                action_error,
            )
            action_list = self._deterministic_actions(
                task_description=description,
                keywords=keywords,
                environment_mode=environment_plan["mode"],
                route_plan=description["route_plan"],
            )
            self._validate_actions(action_list)
            self._validate_action_grounding(
                action_list, task_description=description, keywords=keywords
            )
        action_list = self._ensure_data_access_action(
            action_list,
            environment_mode=environment_plan["mode"],
            table_names=[str(table.get("table_name")) for table in table_definitions],
        )

        # Classify semantic actions before tool generation.  Environment
        # operations may become tools; reasoning and response composition
        # remain the policy's responsibility and must not be outsourced.
        capability_payload = {
            "training_category": training_category,
            "task_description": description,
            "business_model": business_model,
            "environment_plan": environment_plan,
            "agent_actions": action_list,
            "output": {"capabilities": [{
                "action_name": "string",
                "kind": "environment_operation|agent_reasoning|agent_response",
                "requires_tool": True,
                "dependencies": [],
                "reason": "string",
            }]},
        }
        capability_plan: list[Any] | None = None
        capability_error: PipelineGenerationError | None = None
        for capability_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if capability_error is not None:
                repair_hint = (
                    f"\n上一版能力规划未通过校验：{capability_error}。"
                    "必须逐一覆盖所有输入动作且不得增加动作；kind 与 requires_tool 必须一致。"
                )
            capability_result = self._call(
                "capability_plan" if capability_attempt == 1 else "capability_plan.repair",
                "将每个原子 Agent 动作分类为 environment_operation、agent_reasoning 或 agent_response。只有必须读取或修改沙箱私有业务状态、调用外部系统或使用确定性专用能力的动作才是 environment_operation 且 requires_tool=true。business_model 只包含表和字段定义，不包含 Agent 可见的数据行；environment_plan 为 reference_data 或 stateful 时，读取表中具体记录、属性值、关系或隐藏事实必须分类为 environment_operation，不能伪装成从业务模型推理。比较、分析、筛选、选择、解释、总结和生成最终自然语言回答通常属于 agent_reasoning 或 agent_response，requires_tool=false。必须逐一覆盖输入动作，action_name 必须原样引用，不得新增、删除或改名。" + repair_hint,
                capability_payload,
            )
            try:
                candidate_capabilities = self._list(capability_result.get("capabilities"), "capabilities")
                if source_data is not None:
                    # The source-backed route already fixes the atomic action
                    # set. A wording change must not create a third business
                    # capability for local reasoning such as finding a minimum.
                    known_names = {item["name"] for item in action_list}
                    seen_names: set[str] = set()
                    scoped_capabilities = []
                    for item in candidate_capabilities:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("action_name")
                        if not isinstance(name, str) or name not in known_names or name in seen_names:
                            continue
                        seen_names.add(name)
                        scoped_capabilities.append(item)
                    candidate_capabilities = scoped_capabilities
                candidate_capabilities = self._normalize_data_access_capabilities(
                    candidate_capabilities,
                    actions=action_list,
                    environment_mode=environment_plan["mode"],
                )
                candidate_capabilities = self._normalize_capability_dependencies(
                    candidate_capabilities,
                    actions=action_list,
                )
                candidate_capabilities = self._apply_route_capability_contract(
                    candidate_capabilities,
                    route_plan=description["route_plan"],
                )
                self._validate_capability_plan(
                    candidate_capabilities,
                    action_list,
                    environment_mode=environment_plan["mode"],
                    has_business_data=bool(business_model["tables"]),
                    training_category=training_category,
                )
                capability_plan = candidate_capabilities
                break
            except PipelineGenerationError as exc:
                capability_error = exc
                capability_payload["previous_invalid_capabilities"] = (
                    candidate_capabilities
                    if "candidate_capabilities" in locals() else []
                )
                capability_payload["validation_error"] = str(exc)
                logger.warning("capability plan validation failed: attempt=%d/%d error=%s", capability_attempt, self.retries, exc)
        if capability_plan is None:
            raise capability_error or PipelineGenerationError("capability plan generation failed")
        if environment_plan["mode"] == "stateless":
            scoped_names = {
                item["action_name"] for item in capability_plan if not item["requires_tool"]
            }
            action_list = [item for item in action_list if item["name"] in scoped_names]
            capability_plan = [
                item for item in capability_plan if item["action_name"] in scoped_names
            ]
        tool_action_names = {
            item["action_name"] for item in capability_plan if item["requires_tool"]
        }
        tool_action_list = [item for item in action_list if item["name"] in tool_action_names]
        tool_actions = {"actions": tool_action_list}

        # 6. Generate strict OpenAI Function Tools from the atomic actions.
        noise_count = 0 if self.noise_tool_max == 0 else random_source.randint(min(2, self.noise_tool_max), self.noise_tool_max)
        noise_categories = (["related_irrelevant", "unrelated"] + [
            random_source.choice(("unrelated", "related_irrelevant")) for _ in range(max(0, noise_count - 2))
        ])[:noise_count]
        noise_tool_example = [
            {
                "type": "function",
                "function": {
                    "name": f"semantic_function_name_{index + 1}",
                    "description": "噪声工具示例描述",
                    "parameters": {
                        "type": "object",
                        "description": "噪声工具参数对象",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            for index in range(noise_count)
        ]
        noise_metadata_example = [
            {
                "name": f"semantic_function_name_{index + 1}",
                "category": category,
                "rationale": "说明该工具为什么属于指定噪声类别",
            }
            for index, category in enumerate(noise_categories)
        ]
        tool_prompt = TOOL_DEFINITION_PROMPT
        if environment_plan["mode"] == "stateful":
            tool_prompt += " 当前任务是 stateful：payload.goal_contract 的每个 row_predicate 必须有一个明确的创建、更新或删除业务工具；该工具 parameters 必须覆盖对应 where/values 的每个字段，优先使用目标表字段同名参数，也可使用可唯一映射的 entity_field 或 new_field 命名，以便平台编译 selector 和变更映射。payload.business_model 给出实际表字段，不得编造字段。"
        tool_payload = {"task_description": description,
                        "goal_contract": semantic_goal,
                        "business_model": business_model,
                        "capability_plan": capability_plan,
                        "agent_actions": tool_actions,
                        "training_category": training_category,
                        "noise_spec": {"count": noise_count, "categories": noise_categories,
                                       "category_definitions": {
                                           "unrelated": "与当前任务完全无关",
                                           "related_irrelevant": "与当前任务相关但与任务目标完成无关",
                                       }},
                        "output": {"tools": [], "noise_tools": noise_tool_example,
                                   "noise_tool_metadata": noise_metadata_example,
                                   "tool_bindings": [{"tool_name": "string", "action_name": "string"}]}}
        tool_list: list[dict[str, Any]] | None = None
        tool_bindings: list[dict[str, Any]] = []
        noise_tool_metadata: list[dict[str, Any]] = []
        tool_error: PipelineGenerationError | None = None
        for tool_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if tool_error is not None:
                repair_hint = (
                    f"\n上一版工具定义未通过校验，必须修复以下错误后重新输出完整结果：{tool_error}。"
                    "必须同时返回 tools、noise_tools、noise_tool_metadata 和 tool_bindings。"
                    "noise_tool_metadata 必须是与 noise_tools 数量相同的数组；每项必须包含对应工具 name、"
                    "与 noise_spec.categories 同位置一致的 category，以及非空 rationale。"
                )
            tools = self._call(
                "openai_tools" if tool_attempt == 1 else "openai_tools.repair",
                tool_prompt + repair_hint,
                tool_payload,
            )
            try:
                candidate_tools = self._list(tools.get("tools"), "tools", minimum=0)
                self._fill_tool_schema_descriptions(candidate_tools)
                self._validate_tools(candidate_tools)
                if any(item["function"]["name"] == "ask_user" for item in candidate_tools):
                    raise PipelineGenerationError("ask_user 不再是 LLM Tool，用户交互必须使用 UserSimulator 接口")
                candidate_noise_tools = self._list(tools.get("noise_tools"), "noise_tools", minimum=noise_count)
                if len(candidate_noise_tools) != noise_count:
                    raise PipelineGenerationError(
                        f"noise_tools must contain exactly {noise_count} item(s)"
                    )
                self._fill_tool_schema_descriptions(candidate_noise_tools)
                self._validate_tools(candidate_noise_tools)
                self._validate_noise_tool_safety(candidate_noise_tools)
                if any(item["function"]["name"] == "ask_user" for item in candidate_noise_tools):
                    raise PipelineGenerationError("noise_tools 不能定义 ask_user")
                primary_names = {item["function"]["name"] for item in candidate_tools}
                noise_names = {item["function"]["name"] for item in candidate_noise_tools}
                if primary_names & noise_names:
                    raise PipelineGenerationError("noise tool names must not duplicate task tool names")
                candidate_noise_metadata = tools.get("noise_tool_metadata", [])
                if not isinstance(candidate_noise_metadata, list) or len(candidate_noise_metadata) != noise_count:
                    raise PipelineGenerationError("noise_tool_metadata must match noise_tools")
                for index, metadata in enumerate(candidate_noise_metadata):
                    if not isinstance(metadata, dict) or metadata.get("name") not in noise_names:
                        raise PipelineGenerationError("noise metadata references an unknown tool")
                    if metadata.get("category") != noise_categories[index]:
                        raise PipelineGenerationError("noise metadata category does not match noise_spec")
                    if not isinstance(metadata.get("rationale"), str) or not metadata["rationale"].strip():
                        raise PipelineGenerationError("noise metadata requires rationale")
                if candidate_noise_tools:
                    noise_audit = self._call(
                        "noise_tool_audit",
                        "逐一审查候选噪声工具是否可能实质帮助完成任务。只要工具能提供原因分析证据、关键事实、比较依据、验证手段、排查路径所需资料或最终答案内容，就不是安全噪声，is_safe_noise=false。主题相关但确实不能推进任何任务目标时才可为 true。必须逐一覆盖输入工具并原样返回 name。",
                        {
                            "task_description": description,
                            "agent_actions": action_list,
                            "candidate_noise_tools": candidate_noise_tools,
                            "candidate_metadata": candidate_noise_metadata,
                            "output": {"decisions": [{
                                "name": "string", "is_safe_noise": True, "reason": "string"
                            }]},
                        },
                    )
                    unsafe_noise_names = self._validate_noise_tool_audit(
                        noise_audit.get("decisions"), noise_names=noise_names
                    )
                    if unsafe_noise_names:
                        raise PipelineGenerationError(
                            f"noise tools materially help the task; regenerate distractors: {sorted(unsafe_noise_names)}"
                        )
                candidate_bindings = tools.get("tool_bindings", [])
                if not isinstance(candidate_bindings, list):
                    raise PipelineGenerationError("tool_bindings must be a list")
                normalized_bindings = []
                for binding in candidate_bindings:
                    if not isinstance(binding, dict):
                        raise PipelineGenerationError("each tool binding must be an object")
                    tool_name = binding.get("tool_name")
                    # Accept artifacts produced by the former contract, but
                    # persist only the single canonical action_name field.
                    action_name = binding.get("action_name") or binding.get("trainer_action")
                    if not all(isinstance(value, str) and value.strip() for value in (tool_name, action_name)):
                        raise PipelineGenerationError("each tool binding needs tool_name and action_name")
                    if tool_name not in primary_names:
                        # Providers sometimes bind a generated noise tool even
                        # though noise tools must never map to business actions.
                        # Dropping that binding is lossless; coverage of every
                        # real business tool is checked below.
                        if tool_name in noise_names or tool_name in {
                            item.get("name") for item in candidate_noise_metadata
                            if isinstance(item, dict)
                        }:
                            logger.info(
                                "discarding non-business tool binding: tool=%s action=%s",
                                tool_name, action_name,
                            )
                            continue
                        raise PipelineGenerationError("tool binding references an unknown tool")
                    if action_name not in tool_action_names:
                        raise PipelineGenerationError("tool binding references an action that is not tool-eligible")
                    normalized_bindings.append({"tool_name": tool_name, "action_name": action_name})
                deduplicated_bindings: dict[str, dict[str, str]] = {}
                for binding in normalized_bindings:
                    deduplicated_bindings.setdefault(binding["tool_name"], binding)
                if len(deduplicated_bindings) != len(normalized_bindings):
                    logger.warning(
                        "deduplicated repeated business tool bindings: removed=%d",
                        len(normalized_bindings) - len(deduplicated_bindings),
                    )
                normalized_bindings = list(deduplicated_bindings.values())
                if {item["tool_name"] for item in normalized_bindings} != primary_names:
                    raise PipelineGenerationError("every business tool must have exactly one tool binding")
                business_count = len(normalized_bindings)
                if training_category == "direct_response" and business_count:
                    raise PipelineGenerationError("direct_response must not define business tools")
                if training_category == "simple_agentic" and business_count != 1:
                    raise PipelineGenerationError("simple_agentic requires exactly one business tool")
                if training_category == "multi_step_agentic" and business_count < 2:
                    raise PipelineGenerationError("multi_step_agentic requires at least two business tools")
                if environment_plan["mode"] == "stateful":
                    self._validate_stateful_tool_surface(
                        candidate_tools, semantic_goal=semantic_goal
                    )
                if environment_plan["mode"] == "external_capability" and not primary_names:
                    raise PipelineGenerationError(
                        "external-capability task requires at least one business tool"
                    )
                if (
                    environment_plan["mode"] == "reference_data"
                    and business_model["tables"]
                    and not candidate_tools
                ):
                    data_action = next(
                        (
                            item.get("action_name") for item in capability_plan
                            if isinstance(item, dict)
                            and item.get("kind") == "environment_operation"
                            and item.get("requires_tool") is True
                        ),
                        None,
                    )
                    if not isinstance(data_action, str) or not data_action.strip():
                        raise PipelineGenerationError(
                            "data-backed task has no tool-eligible data access action"
                        )
                    fallback_name = "read_task_reference_data"
                    if fallback_name in noise_names:
                        fallback_name = "read_business_reference_data"
                    fallback_business_tool = {
                        "type": "function",
                        "function": {
                            "name": fallback_name,
                            "description": "读取完成当前任务所需的沙箱业务参考数据。",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "用户目标或需要读取的数据范围。",
                                    }
                                },
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        },
                    }
                    self._validate_tools([fallback_business_tool])
                    candidate_tools = [fallback_business_tool]
                    normalized_bindings = [{
                        "tool_name": fallback_name,
                        "action_name": data_action,
                    }]
                    primary_names = {fallback_name}
                    logger.warning(
                        "injected deterministic data access tool: action=%s tool=%s",
                        data_action, fallback_name,
                    )
                tool_list = candidate_tools + candidate_noise_tools
                tool_bindings = normalized_bindings
                noise_tool_metadata = candidate_noise_metadata
                break
            except PipelineGenerationError as exc:
                tool_error = exc
                tool_payload["previous_proposal"] = tools
                logger.warning("openai tool schema validation failed: attempt=%d/%d error=%s", tool_attempt, self.retries, exc)
        if tool_list is None:
            raise tool_error or PipelineGenerationError("openai tool generation failed")
        # Noise is implemented over independent read-only fixtures. Its public
        # result never contains the private noise label or usefulness verdict.
        for metadata in noise_tool_metadata:

            noise_function = next(tool["function"] for tool in tool_list if tool["function"]["name"] == metadata["name"])
            fixture_error = None
            for attempt in range(self.retries):
                fixture = self._call("noise_tool_fixture", "为此只读工具生成独立的小型合成数据集，至少两行。所有参数必须通过 parameter_columns 映射到记录字段，使用精确匹配过滤。数据必须符合工具描述，不包含噪声标签、任务答案或评价信息。仅支持此查询语义；若工具语义不能用精确过滤表达则返回 supported=false。", {
                    "function": noise_function, "validation_error": fixture_error,
                    "output": {"supported": True, "records": [], "parameter_columns": {}},
                })
                rows, mapping = fixture.get("records"), fixture.get("parameter_columns")
                properties = noise_function.get("parameters", {}).get("properties", {})
                if (fixture.get("supported") is True and isinstance(rows, list) and len(rows) >= 2
                    and all(isinstance(row, dict) for row in rows) and isinstance(mapping, dict)
                    and set(mapping) == set(properties)
                    and all(isinstance(column, str) and all(column in row for row in rows) for column in mapping.values())):
                    metadata.update(records=rows, parameter_columns=mapping)
                    break
                fixture_error = "fixture must support exact filtering, contain two rows and map every parameter to a present column"
            else:
                rows, mapping = self._build_noise_fixture(noise_function)
                metadata.update(records=rows, parameter_columns=mapping)
                logger.warning(
                    "noise fixture proposals remained invalid; using schema-derived fixture: tool=%s error=%s",
                    metadata["name"], fixture_error,
                )
        tools_manifest = self._materialize_tools(tool_list, materialized_dir)

        implementation_result = self._call(
            "tool_implementation_specs",
            "为可以直接由数据表操作实现的业务工具生成声明式实现。operation 支持 select、aggregate_count、insert、update、delete；复杂计算、跨表业务决策、文档生成和噪声工具不要生成 spec。filters.argument 以及 selector/values/changes 的键必须来自工具 parameters，对应值必须是目标表字段。operator 只能是 eq、in、contains、gte、lte。若公开参数是关联实体的可读字段而目标列是外键，filter.resolve 必须声明关联表 table、匹配字段 match_column 和写入外键比较的 value_column。若公开返回字段名与表字段名不同，projection_aliases 使用公开字段名到表字段名的映射。只返回 specs 数组。",
            {
                "business_model": business_model,
                "tools": candidate_tools,
                "output": {"specs": [{
                    "tool_name": "string", "operation": "select", "table": "string",
                    "filters": [{"argument": "string", "column": "string", "operator": "eq", "resolve": {"table": "string", "match_column": "string", "value_column": "string"}}],
                    "projection": ["string"], "projection_aliases": {"public_field": "table_column"},
                    "order_by": ["string"], "result_field": "records",
                    "selector": {"tool_argument": "table_column"},
                    "values": {"tool_argument": "table_column"},
                    "changes": {"tool_argument": "table_column"},
                }]},
            },
        )
        tool_implementations = []
        proposed_specs = implementation_result.get("specs", [])
        for proposed in proposed_specs if isinstance(proposed_specs, list) else []:
            for attempt in range(self.retries):
                try:
                    self._validate_tool_implementations(
                        [*tool_implementations, proposed], tools=candidate_tools, tables=business_model["tables"]
                    )
                    tool_implementations.append(proposed)
                    break
                except PipelineGenerationError as exc:
                    if attempt + 1 == self.retries:
                        logger.warning("single declarative implementation needs custom handler: %s", exc)
                        break
                    repaired = self._call("tool_implementation_specs.repair", "仅修复给定的声明式工具实现，保持工具业务语义。能够表达时返回 supported=true 和 spec 对象；不能表达时返回 supported=false 和空 spec 对象。不要修改其他工具。", {
                        "spec": proposed, "validation_error": str(exc), "business_model": business_model,
                        "tools": candidate_tools, "output": {"supported": True, "spec": {}},
                    })
                    if repaired.get("supported") is False:
                        proposed = None
                        break
                    proposed = repaired.get("spec")

        if environment_plan["mode"] == "stateful":
            tool_implementations = self._complete_stateful_tool_implementations(
                implementations=tool_implementations,
                tools=candidate_tools,
                tables=business_model["tables"],
                semantic_goal=semantic_goal,
            )
            try:
                self._validate_goal_tool_coverage(
                    semantic_goal=semantic_goal,
                    tool_implementations=tool_implementations,
                )
            except PipelineGenerationError as initial_coverage_error:
                coverage_error: PipelineGenerationError = initial_coverage_error
                for coverage_attempt in range(1, self.retries + 1):
                    repaired = self._call(
                        "tool_implementation_specs.coverage_repair",
                        "重新生成完整的声明式业务工具实现，修复 goal_contract 覆盖错误。"
                        "每个 stateful row_predicate 所在表必须由一个 insert、update 或 delete 工具实现；"
                        "update 的 selector 必须定位 where 字段，changes 必须覆盖 values 字段。"
                        "参数键必须来自对应工具 parameters，映射值必须是实际表字段。"
                        "不得为噪声工具生成实现；无法满足时返回空 specs，不得编造表或字段。",
                        {
                            "validation_error": str(coverage_error),
                            "semantic_goal": semantic_goal,
                            "business_model": business_model,
                            "tools": candidate_tools,
                            "previous_specs": tool_implementations,
                            "output": {"specs": [{
                                "tool_name": "string", "operation": "update", "table": "string",
                                "selector": {"tool_argument": "table_column"},
                                "changes": {"tool_argument": "table_column"},
                                "values": {"tool_argument": "table_column"},
                                "filters": [], "projection": [], "order_by": [],
                                "result_field": "records",
                            }]},
                        },
                    )
                    candidate_specs: list[dict[str, Any]] = []
                    for spec in repaired.get("specs", []) if isinstance(repaired.get("specs"), list) else []:
                        try:
                            self._validate_tool_implementations(
                                [*candidate_specs, spec], tools=candidate_tools,
                                tables=business_model["tables"],
                            )
                        except PipelineGenerationError:
                            continue
                        candidate_specs.append(spec)
                    candidate_specs = self._complete_stateful_tool_implementations(
                        implementations=candidate_specs,
                        tools=candidate_tools,
                        tables=business_model["tables"],
                        semantic_goal=semantic_goal,
                    )
                    try:
                        self._validate_goal_tool_coverage(
                            semantic_goal=semantic_goal,
                            tool_implementations=candidate_specs,
                        )
                        tool_implementations = candidate_specs
                        coverage_error = None
                        break
                    except PipelineGenerationError as exc:
                        coverage_error = exc
                        logger.warning(
                            "stateful goal/tool coverage repair failed: attempt=%d/%d error=%s",
                            coverage_attempt, self.retries, exc,
                        )
                if coverage_error is not None:
                    raise coverage_error

        tool_implementations = self._complete_filter_resolvers(
            implementations=tool_implementations,
            tables=business_model["tables"],
        )
        self._validate_tool_implementations(
            tool_implementations,
            tools=candidate_tools,
            tables=business_model["tables"],
        )
        tool_implementations = self._complete_dependency_projections(
            implementations=tool_implementations,
            tools=candidate_tools,
            tables=business_model["tables"],
        )
        self._validate_tool_implementations(
            tool_implementations,
            tools=candidate_tools,
            tables=business_model["tables"],
        )
        projection_gaps = self._tool_chain_projection_gaps(
            route_plan=description["route_plan"],
            tool_bindings=tool_bindings,
            tools=candidate_tools,
            implementations=tool_implementations,
        )
        if projection_gaps:
            original_specs = tool_implementations
            original_operations = {
                spec["tool_name"]: spec["operation"] for spec in original_specs
            }
            repair_error = "dependent tool outputs do not cover downstream required fields"
            for repair_attempt in range(1, self.retries + 1):
                try:
                    repaired = self._call(
                        "tool_implementation_specs.dependency_repair",
                        "修复完整的声明式工具实现，使 route_plan 中有依赖的两个业务工具可以通过公开返回字段连接。"
                        "projection_gaps 列出下游数组项必需但上游未返回的字段。只能使用真实表和列；"
                        "可修改查询表、过滤映射、projection 和 projection_aliases，但必须保持每个已有工具的"
                        "tool_name、operation 和业务语义，不得通过删除工具或编造字段消除错误。"
                        "如果现有工具契约无法表达兼容依赖，返回 supported=false。返回完整 specs。",
                        {
                            "route_plan": description["route_plan"],
                            "tool_bindings": tool_bindings,
                            "business_model": business_model,
                            "tools": candidate_tools,
                            "previous_specs": original_specs,
                            "projection_gaps": projection_gaps,
                            "validation_error": repair_error,
                            "output": {"supported": True, "specs": original_specs},
                        },
                    )
                except PipelineGenerationError as exc:
                    logger.warning("optional tool dependency repair failed: %s", exc)
                    break
                if repaired.get("supported") is False:
                    break
                proposed_specs = repaired.get("specs")
                if (not isinstance(proposed_specs, list)
                    or len(proposed_specs) != len(original_specs)
                    or any(not isinstance(spec, dict) for spec in proposed_specs)
                    or {
                    spec.get("tool_name"): spec.get("operation")
                    for spec in proposed_specs
                } != original_operations):
                    repair_error = "repair changed the set or operation of existing tools"
                    continue
                try:
                    proposed_specs = self._complete_filter_resolvers(
                        implementations=proposed_specs, tables=business_model["tables"],
                    )
                    proposed_specs = self._complete_dependency_projections(
                        implementations=proposed_specs, tools=candidate_tools,
                        tables=business_model["tables"],
                    )
                    self._validate_tool_implementations(
                        proposed_specs, tools=candidate_tools,
                        tables=business_model["tables"],
                    )
                    if environment_plan["mode"] == "stateful":
                        self._validate_goal_tool_coverage(
                            semantic_goal=semantic_goal,
                            tool_implementations=proposed_specs,
                        )
                except PipelineGenerationError as exc:
                    repair_error = str(exc)
                    continue
                remaining = self._tool_chain_projection_gaps(
                    route_plan=description["route_plan"],
                    tool_bindings=tool_bindings,
                    tools=candidate_tools,
                    implementations=proposed_specs,
                )
                if not remaining:
                    tool_implementations = proposed_specs
                    logger.info(
                        "repaired declarative tool dependency projections: attempt=%d gaps=%d",
                        repair_attempt, len(projection_gaps),
                    )
                    break
                repair_error = f"remaining projection gaps: {remaining}"
            else:
                logger.warning("tool dependency projection repair unresolved: %s", repair_error)
        semantic_implementations: list[dict[str, Any]] = []
        tools_by_name = {
            item.get("function", {}).get("name"): item for item in candidate_tools
            if isinstance(item, dict)
        }
        for implementation in tool_implementations:
            tool_name = implementation.get("tool_name") if isinstance(implementation, dict) else None
            tool = tools_by_name.get(tool_name)
            if not isinstance(tool, dict):
                continue
            try:
                self._validate_business_tool_semantics(
                    tools=[tool], implementations=[implementation],
                )
            except PipelineGenerationError as exc:
                logger.warning(
                    "declarative implementation cannot preserve business semantics; "
                    "requiring custom handler: tool=%s error=%s",
                    tool_name, exc,
                )
                continue
            semantic_implementations.append(implementation)
        tool_implementations = semantic_implementations
        if environment_plan["mode"] == "stateful":
            self._validate_goal_tool_coverage(
                semantic_goal=semantic_goal,
                tool_implementations=tool_implementations,
            )

        # 7a. Identify the small set of goal-critical steps before creating
        # process metrics. Ordinary tools are not process-reward candidates.
        if not tool_action_list:
            key_steps: list[Any] = []
            logger.info("reward key-step generation skipped: no tool-eligible business actions")
        else:
            key_step_payload = {
                "task_description": description, "environment": environment,
                "agent_actions": tool_actions, "tools": candidate_tools,
                "output": {"key_steps": [{"step_id": "step-1", "action_name": "string", "rationale": "string", "required_for_goal": True, "dependencies": []}]},
            }
            key_steps = []
            key_step_error: PipelineGenerationError | None = None
            for key_step_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if key_step_error is not None:
                    repair_hint = (
                        f"\n上一版关键步骤未通过校验：{key_step_error}。"
                        "action_name 只能原样引用 payload.agent_actions.actions 中的动作，不得引用噪声工具或新增动作。"
                    )
                key_steps_result = self._call(
                    "reward_key_steps" if key_step_attempt == 1 else "reward_key_steps.repair",
                    "先从任务目标和可工具化环境动作中梳理真正必需的关键步骤。只保留显著推进目标、改变关键业务数据或形成必要中间结果的动作；读取背景信息、噪声工具、可选探索和重复调用通常不是关键步骤。只输出关键步骤定义，不生成 metric 或 reward。" + repair_hint,
                    key_step_payload,
                )
                try:
                    candidate_key_steps = self._list(
                        key_steps_result.get("key_steps"), "key_steps", minimum=0
                    )
                    self._validate_key_steps(candidate_key_steps, tool_action_list)
                    key_steps = candidate_key_steps
                    break
                except PipelineGenerationError as exc:
                    key_step_error = exc
                    logger.warning(
                        "reward key-step validation failed: attempt=%d/%d error=%s",
                        key_step_attempt, self.retries, exc,
                    )
            else:
                raise key_step_error or PipelineGenerationError("reward key-step generation failed")

        # 7b. Design executable observations and rewards from key steps.
        reward_prompt = REWARD_DESIGN_PROMPT
        reward_prompt += " payload.key_steps 是上一阶段确认的关键步骤。process 指标只能评价这些 key_steps 中的 action_name；不得为非关键读取、噪声工具、可选探索或每个工具机械创建过程奖励。先使用 key_steps 判断是否确实需要过程奖励，再生成最少且必要的 process metrics。"
        reward_prompt += " metric 的 rubric、criteria、condition 和 assertion 中不得新增任务描述未提出的数量、字数、比例、时间或最低条目数；只能检验任务中已有的明确约束。"
        reward_prompt += " payload.task_description 只包含 Agent 在运行时可见的公开请求。结果指标不得引用生成阶段隐藏的 goal、context、expected_result 或 requirements，也不得把公开请求未要求的文件格式、Markdown 语法、表格样式、应用场景或措辞设为成功条件。"
        if source_data is not None and source_data.get("verified_reward_facts"):
            reward_prompt += (
                " 对于本次数据集试验，task_description.verified_reward_facts 是从源数据确定性计算的答案键，"
                "可用于核对公开请求明确要求的每一项结果；不得将答案键作为 Agent 的公开输入，"
                "也不得额外增加用户未要求的成功条件。"
            )
        reward_prompt += " 优先使用可执行的 runtime_rule 结果指标，并将多个关键动作合并为最少必要的过程指标。"
        if environment_plan["mode"] == "stateless":
            reward_prompt += " 当前任务是 stateless：不得生成 process 指标，不得以 business_data 或数据库变化作为成功条件。结果指标必须根据运行时 conversation 与 final_agent_response 评价任务完成度、忠实性和格式；噪声工具调用只能作为轨迹惩罚。"
        reward_environment = {
            "data_manifest": data_manifest,
            "tables": business_model["tables"] if "business_model" in locals() else [
                {
                    "table_name": table["table_name"], "description": table.get("description", ""),
                    "columns": table.get("columns", []), "primary_key": table.get("primary_key", []),
                    "foreign_keys": table.get("foreign_keys", []), "constraints": table.get("constraints", []),
                }
                for table in table_definitions
            ],
        }
        public_reward_task = {
            "task": public_input.get("initial_user_message", description.get("task", "")),
            "public_input": public_input,
            "training_category": training_category,
        }
        if source_data is not None and source_data.get("verified_reward_facts"):
            public_reward_task["verified_reward_facts"] = source_data["verified_reward_facts"]
        reward_payload = {"task_description": public_reward_task, "environment": reward_environment,
                          "goal_contract": semantic_goal,
                          "environment_plan": environment_plan,
                          "agent_actions": {"actions": action_list},
                          "tools": tool_list, "key_steps": key_steps, "output": {"observation_schema": {}, "metrics": [{
                              "id": "process_key_tool_action", "category": "process", "type": "hybrid", "scope": "step",
                              "target_action": "key_tool_action",
                              "evaluation_inputs": ["recent_conversation", "public_observation", "tool_call", "tool_arguments"],
                              "criteria": ["当前上下文是否需要执行该关键动作"],
                              "condition": "llm_expected_tool_call_exact_match",
                              "evaluator": {"kind": "hybrid_tool_call", "source": "external_llm", "comparison": "exact_tool_name_and_canonical_arguments", "score_mapping": {"match": 1, "mismatch": 0}},
                              "score_range": [0, 1],
                              "weight": 0.2, "rubric": "过程动作选择正确"}],
                              "reward_formula": {"type": "separate_sign_weighted_sum", "formula": "R = clip(sum(w_i * score_i for category in {process,outcome}) + sum(w_j * score_j for category == penalty), -1, 1)", "positive_weight_sum": 1, "negative_weight_sum": 1, "score_range": [-1, 1]},
                              }}
        rewards: dict[str, Any] | None = None
        reward_error: PipelineGenerationError | None = None
        for reward_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if reward_error is not None:
                repair_hint = (
                    f"\n上一版 observation/reward 设计未通过校验，必须修复以下错误后重新输出完整结果：{reward_error}。"
                    "不要回显输入 payload，也不要返回 task_description、environment、agent_actions、tools 或 output 字段。"
                    "顶层只能返回 observation_schema、metrics、reward_formula；"
                    "metric.type 只能是精确字符串 rule-based、model-based 或 hybrid；"
                    "evaluator.kind 为 business_state_rule、document_rule 或 trajectory_rule 时 type=rule-based；"
                    "evaluator.kind 为 external_llm_judge 时 type=model-based；"
                    "evaluator.kind 为 hybrid_tool_call 或 hybrid_outcome 时 type=hybrid；"
                    "metrics 必须是非空数组，scope 只能是 step、state、terminal、trajectory，"
                    "category 只能是 process、outcome、penalty；process 指标只能对应 payload.key_steps 中的 action_name，且 type 必须为 hybrid；"
                    "每个关键 process 指标必须包含非空 target_action、evaluation_inputs、criteria，且 condition 必须严格为 llm_expected_tool_call_exact_match；"
                    "每个 metric 必须包含 evaluator 对象、kind、source 和 score_mapping；process evaluator 必须是 hybrid_tool_call/external_llm/exact_tool_name_and_canonical_arguments；"
                    "rule-based 结果或惩罚 evaluator 必须声明 business_state_rule、document_rule 或 trajectory_rule 及可执行 assertion；model-based 必须使用 external_llm_judge；"
                    "每个 rule-based 或 hybrid 指标必须补充非空 condition；每个 model-based 或 hybrid 指标必须补充非空 evaluation_inputs 和 criteria 数组，"
                    "不能返回 criteria 缺失、空数组或 null。"
                )
            candidate_rewards = self._call(
                "observations_rewards" if reward_attempt == 1 else "observations_rewards.repair",
                reward_prompt + repair_hint,
                reward_payload,
            )
            try:
                candidate_metrics = self._list(candidate_rewards.get("metrics"), "metrics")
                candidate_metrics = self._normalize_metric_types(candidate_metrics)
                candidate_metrics = self._normalize_metric_evaluators(candidate_metrics)
                candidate_metrics = self._normalize_metrics_for_environment(
                    candidate_metrics, environment_mode=environment_plan["mode"]
                )
                if source_data is not None and source_data.get("verified_reward_criterion"):
                    criterion = str(source_data["verified_reward_criterion"])
                    for metric in candidate_metrics:
                        evaluator = metric.get("evaluator") if isinstance(metric, dict) else None
                        if (isinstance(metric, dict) and metric.get("category") == "outcome"
                                and isinstance(evaluator, dict)
                                and evaluator.get("kind") in {"external_llm_judge", "hybrid_outcome"}):
                            metric["rubric"] = criterion
                            metric["criteria"] = [criterion]
                            if evaluator.get("kind") == "external_llm_judge":
                                evaluator["criteria"] = [criterion]
                candidate_metrics = self._drop_ungrounded_numeric_metrics(
                    candidate_metrics, task_description=public_reward_task
                )
                if noise_tool_metadata:
                    self._ensure_deterministic_noise_penalty(
                        candidate_metrics,
                        noise_names=[str(item["name"]) for item in noise_tool_metadata],
                    )
                candidate_metrics = self._normalize_metric_weights(candidate_metrics)
                candidate_rewards["metrics"] = candidate_metrics
                self._validate_metrics(candidate_metrics, key_steps=key_steps)
                self._validate_metrics_for_environment(
                    candidate_metrics, environment_mode=environment_plan["mode"]
                )
                self._validate_metric_constraint_grounding(
                    candidate_metrics, task_description=public_reward_task
                )
                # Build the formula from the validated metric weights instead
                # of trusting an independently generated expression. Positive
                # process+outcome weights sum to 1; penalty weights sum to 1.
                # Therefore the two signed contributions are already bounded
                # by [0, 1] and [-1, 0], and their sum is bounded by [-1, 1].
                candidate_rewards["reward_formula"] = self._canonical_reward_formula(candidate_metrics)
                self._validate_reward_formula(candidate_rewards["reward_formula"], candidate_metrics)
                if not isinstance(candidate_rewards.get("observation_schema", {}), dict):
                    raise PipelineGenerationError("observation_schema must be an object")
                rewards = candidate_rewards
                break
            except PipelineGenerationError as exc:
                reward_error = exc
                logger.warning("observation/reward validation failed: attempt=%d/%d error=%s", reward_attempt, self.retries, exc)
        if rewards is None:
            logger.warning(
                "observation/reward proposals remained invalid; using deterministic baseline: %s",
                reward_error,
            )
            baseline_metrics = [{
                "id": "outcome_task_completion",
                "category": "outcome",
                "type": "model-based",
                "scope": "terminal",
                "evaluation_inputs": ["conversation", "final_agent_response"],
                "criteria": [
                    "最终回答是否完成任务目标并遵守任务描述中的格式和约束。",
                    "最终回答是否仅使用对话、工具结果和环境中可验证的信息。",
                ],
                "evaluator": {
                    "kind": "external_llm_judge", "source": "external_llm",
                    "score_mapping": {"full": 1, "partial": 0.5, "missing": 0},
                },
                "score_range": [0, 1], "weight": 1,
                "rubric": "完整、可靠地完成用户请求。",
            }]
            if noise_tool_metadata:
                self._ensure_deterministic_noise_penalty(
                    baseline_metrics,
                    noise_names=[str(item["name"]) for item in noise_tool_metadata],
                )
            rewards = {
                "observation_schema": {},
                "metrics": baseline_metrics,
                "reward_formula": self._canonical_reward_formula(baseline_metrics),
            }
            self._validate_metrics(baseline_metrics, key_steps=key_steps)
        if noise_tool_metadata:
            # Already normalized before validation; keep the postcondition
            # explicit for future call-site changes.
            self._validate_metrics(rewards["metrics"], key_steps=key_steps)

        # final_agent_response is stored by the shared runtime as a raw string.  A
        # document_rule such as $.tea_varieties therefore invents structure
        # that does not exist at runtime.  Preserve the semantic criterion by
        # evaluating it with the external judge instead of compiling a bogus
        # JSON path.
        self._promote_unstructured_document_rules(rewards["metrics"])
        self._promote_mixed_response_business_rules(rewards["metrics"])
        rewards["reward_formula"] = self._canonical_reward_formula(rewards["metrics"])

        metric_impl_result = self._call(
            "metric_implementation_specs",
            "为能够从运行时结构化上下文确定性判断的 rule-based 指标生成声明式实现。source 只能是 business_state、trajectory、final_agent_response、observation；path 使用简单 $.field.nested 路径；operator 只能是 eq、ne、gte、lte、contains、exists、count_gte、count_eq。expected 必须是可序列化常量。score_mapping 使用 pass/fail，并且分数必须位于对应 metric.score_range。model-based、hybrid 或无法可靠结构化判断的指标不要生成 spec。只返回 specs 数组。",
            {
                "metrics": rewards["metrics"], "business_model": business_model,
                "output": {"specs": [{
                    "metric_id": "string", "source": "business_state", "path": "$.field",
                    "operator": "eq", "expected": True,
                    "score_mapping": {"pass": 1, "fail": 0},
                }]},
            },
        )
        metric_implementations = metric_impl_result.get("specs", [])
        if not isinstance(metric_implementations, list):
            metric_implementations = []
        rule_metric_ids = {
            item.get("id") for item in rewards["metrics"]
            if isinstance(item, dict) and item.get("type") == "rule-based"
        }
        metric_implementations = [
            item for item in metric_implementations
            if isinstance(item, dict) and item.get("metric_id") in rule_metric_ids
        ]
        if noise_tool_metadata:
            metric_implementations = [
                item for item in metric_implementations
                if isinstance(item, dict) and item.get("metric_id") != "penalty_noise_tool_usage"
            ]
            metric_implementations.append({
                "metric_id": "penalty_noise_tool_usage",
                "source": "trajectory",
                "path": "$.events",
                "operator": "none_tool_calls",
                "expected": [str(item["name"]) for item in noise_tool_metadata],
                "score_mapping": {"pass": 0, "fail": -1},
            })
        try:
            self._validate_metric_implementations(metric_implementations, rewards["metrics"])
            if environment_plan["mode"] == "stateless" and any(
                isinstance(item, dict) and item.get("source") == "business_state"
                for item in metric_implementations
            ):
                raise PipelineGenerationError("stateless metric implementations cannot use business_state")
        except PipelineGenerationError as exc:
            logger.warning("metric implementation specs incomplete; promoting uncovered rules: %s", exc)
            metric_implementations = []
            # The noise predicate is EnvFactory-owned and remains executable
            # even when an unrelated LLM-generated metric spec is malformed.
            if noise_tool_metadata:
                metric_implementations.append({
                    "metric_id": "penalty_noise_tool_usage",
                    "source": "trajectory",
                    "path": "$.events",
                    "operator": "none_tool_calls",
                    "expected": [str(item["name"]) for item in noise_tool_metadata],
                    "score_mapping": {"pass": 0, "fail": -1},
                })
        self._promote_unimplemented_rule_metrics(
            rewards["metrics"], metric_implementations,
            environment_mode=environment_plan["mode"],
        )
        self._prune_literal_payload_process_metrics(
            rewards["metrics"], metric_implementations,
        )
        self._ensure_terminal_outcome_metrics(
            rewards["metrics"], metric_implementations,
            environment_mode=environment_plan["mode"],
        )
        if source_data is not None and source_data.get("verified_reward_criterion"):
            criterion = str(source_data["verified_reward_criterion"])
            for metric in rewards["metrics"]:
                if not isinstance(metric, dict) or metric.get("category") != "outcome":
                    continue
                evaluator = metric.get("evaluator")
                if not isinstance(evaluator, dict) or evaluator.get("kind") not in {
                    "external_llm_judge", "hybrid_outcome",
                }:
                    continue
                metric["rubric"] = criterion
                metric["criteria"] = [criterion]
                if evaluator.get("kind") == "external_llm_judge":
                    evaluator["criteria"] = [criterion]
        self._validate_metrics(rewards["metrics"], key_steps=key_steps)
        self._validate_metrics_for_environment(
            rewards["metrics"], environment_mode=environment_plan["mode"]
        )
        rewards["reward_formula"] = self._canonical_reward_formula(rewards["metrics"])
        rewards["observation_schema"] = self._canonical_observation_schema(
            rewards.get("observation_schema")
        )

        acceptance_seed = self._build_acceptance_contract(
            task_description=description,
            data_manifest=data_manifest,
            data_tables=data_tables,
            actions=action_list,
            tools=tool_list,
            key_steps=key_steps,
            metrics=rewards["metrics"],
            reward_formula=rewards["reward_formula"],
            tool_implementations=tool_implementations,
        )
        acceptance_payload = {
            "goal_contract": semantic_goal,
            "task_description": description,
            "data_manifest": data_manifest,
            "business_tables": data_tables,
            "actions": action_list,
            "tools": tool_list,
            "key_steps": key_steps,
            "metrics": rewards["metrics"],
            "reward_formula": rewards["reward_formula"],
            "baseline_contract": acceptance_seed,
            "output": {"acceptance_contract": acceptance_seed},
        }
        acceptance_contract: dict[str, Any] | None = None
        acceptance_error: PipelineGenerationError | None = None
        for acceptance_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if acceptance_error is not None:
                repair_hint = (
                    f"\n上一版验收契约未通过校验，必须修复以下错误并返回完整 acceptance_contract：{acceptance_error}。"
                    "每个 scenarios 项必须是 object，并且同时包含非空 scenario_id、非空 kind、steps 数组；"
                    "goal_critical 场景还必须引用已有 action_name。每个 tool_cases 项必须包含 case_id、已有 tool_name、"
                    "expected object；每个 reward_cases 项必须包含 case_id、已有 metric_id 和数值 expected_score。"
                    "不要返回缺字段、空字符串或 null。"
                )
            candidate = self._call(
                "acceptance_contract" if acceptance_attempt == 1 else "acceptance_contract.repair",
                "根据任务、完整业务数据、原子动作、工具和奖励指标生成 EnvFactory 独立业务验收契约。该契约供外层黑盒验收使用，不由 Code Agent 修改。必须覆盖真实业务成功路径、失败路径、前置条件、业务数据前后变化、隐藏真值不泄露、工具不返回 reward/observation、reset/episode 隔离、replay、幂等、反事实奖励和 mutation testing。不要只生成 HTTP smoke；每个关键步骤要给出可执行的场景和断言。不要生成新的业务语义，所有表、字段、工具、动作和指标必须来自输入。只输出 acceptance_contract 对象。" + repair_hint,
                acceptance_payload,
            )
            candidate_contract = candidate.get("acceptance_contract", candidate)
            candidate_contract = self._merge_acceptance_contract(candidate_contract, acceptance_seed)
            try:
                self._validate_acceptance_contract(candidate_contract, actions=action_list, tools=tool_list, metrics=rewards["metrics"])
                acceptance_contract = candidate_contract
                break
            except PipelineGenerationError as exc:
                acceptance_error = exc
                logger.warning("acceptance contract validation failed: attempt=%d/%d error=%s", acceptance_attempt, self.retries, exc)
        if acceptance_contract is None:
            # The baseline is generated deterministically from already
            # validated task artifacts and is itself an EnvFactory-owned
            # contract.  It is safer to preserve a valid acceptance contract
            # than to discard the whole task because an LLM omitted a
            # structural field after all repair attempts.
            self._validate_acceptance_contract(
                acceptance_seed, actions=action_list, tools=tool_list, metrics=rewards["metrics"]
            )
            logger.warning(
                "acceptance contract LLM output remained invalid after %d attempts; using validated baseline",
                self.retries,
            )
            acceptance_contract = acceptance_seed

        executable_payload = {
            "goal_contract": semantic_goal,
            "capability_dependencies": key_steps,
                "task_description": description,
                "training_category": training_category,
                "business_tables": data_tables,
                "tools": tool_list,
                "tool_implementations": tool_implementations,
                "noise_tools": noise_tool_metadata,
                "key_steps": key_steps,
                "metrics": rewards["metrics"],
                "output": {"scenarios": [{
                    "scenario_id": "goal_success",
                    "kind": "goal_success",
                    "steps": [
                        {"step_id": "lookup", "operation": "tool_call", "tool_name": "declared_tool", "arguments": {}, "capture": {"record_id": "$.records[0].id"}, "expected_status": 200},
                        {"step_id": "update", "operation": "tool_call", "tool_name": "declared_tool", "arguments": {"id": {"$ref": "record_id"}}, "expected_status": 200},
                    ],
                    "assertions": [{"source": "step:update", "path": "$.status", "operator": "exists", "expected": True}],
                }]},
        }
        business_scenarios: list[Any] | None = None
        executable_error: PipelineGenerationError | None = None
        success_fixture = self._select_success_response_fixture(task_description=description)
        fixture_conversation = [{
            "role": "user",
            "content": str(public_input.get("initial_user_message", "")).strip(),
        }]
        try:
            fixture_result = self._call(
                "acceptance_success_fixture",
                "根据任务描述、用户消息、公开环境数据和结果指标生成一份真正完成任务的最终回答，用作成功验收 fixture。只有任务描述、用户消息和 grounding_environment 才是事实来源；不得沿用模拟 Agent 先前产生的事实。必须忠实保留用户提供的实体、数值、规则和格式范围；不得新增未提供的价格、成分、属性、排行、安全阈值、操作时长或其他业务事实。信息不足时应明确缺口并请求澄清，不能伪造完整答案。回答必须直接解决具体任务，不能只复述输出格式、提纲或评分标准。只返回 content 字段。",
                {
                    "task_description": description,
                    "representative_conversation": fixture_conversation,
                    "grounding_environment": environment,
                    "metrics": [
                        {key: metric.get(key) for key in ("id", "rubric", "criteria")}
                        for metric in rewards["metrics"] if metric.get("category") == "outcome"
                    ],
                    "output": {"content": "完整成功回答"},
                },
            )
            fixture_result = self._unwrap_structured_output(
                fixture_result, expected_fields={"content"}
            )
            candidate_fixture = fixture_result.get("content")
            if isinstance(candidate_fixture, str) and len(candidate_fixture.strip()) >= 100:
                success_fixture = candidate_fixture.strip()
        except PipelineGenerationError as exc:
            logger.warning("success response fixture generation failed; using dialogue-derived fallback: %s", exc)
        fixture_audit_error: PipelineGenerationError | None = None
        for fixture_attempt in range(1, self.retries + 1):
            success_fixture = self._normalize_success_fixture_length(
                success_fixture, task_description=description
            )
            fixture_audit = self._call(
                "acceptance_success_fixture_audit",
                "独立审查成功回答 fixture。latest_instructions_satisfied 表示回答遵循用户消息中最后出现的格式、措辞、增删、纠正和撤销要求；grounded 表示每项价格、属性、适用性、排行、安全阈值、操作时长等事实均可在任务、用户消息或 grounding_environment 中找到，不能把模型常识当作验收真值；goal_completed 表示回答完成当前有证据支持的目标，确实缺少必要输入时准确说明缺口并请求补充也算完成。issues 必须具体指出遗漏或无来源断言。",
                {
                    "task_description": description,
                    "representative_conversation": fixture_conversation,
                    "grounding_environment": environment,
                    "candidate_response": success_fixture,
                    "output": {
                        "latest_instructions_satisfied": True,
                        "grounded": True,
                        "goal_completed": True,
                        "issues": [],
                    },
                },
            )
            fixture_audit = self._unwrap_structured_output(
                fixture_audit,
                expected_fields={
                    "latest_instructions_satisfied", "grounded",
                    "goal_completed", "issues",
                },
            )
            try:
                self._validate_success_fixture_audit(fixture_audit)
                self._validate_success_fixture_constraints(
                    success_fixture, task_description=description
                )
                fixture_audit_error = None
                break
            except PipelineGenerationError as exc:
                fixture_audit_error = exc
                logger.warning(
                    "success fixture audit failed: attempt=%d/%d error=%s",
                    fixture_attempt, self.retries, exc,
                )
                if fixture_attempt >= self.retries:
                    break
                try:
                    repaired_fixture = self._call(
                        "acceptance_success_fixture.repair",
                        "根据审查问题重写完整最终回答。必须执行代表性对话中最后有效的用户指令，后指令覆盖前指令；不得只解释规则，不得保留已被撤销的要求，不得编造事实。只返回 content。",
                        {
                            "task_description": description,
                            "representative_conversation": fixture_conversation,
                            "previous_response": success_fixture,
                            "issues": list(fixture_audit.get("issues", [])) + [str(exc)],
                            "output": {"content": "修复后的完整成功回答"},
                        },
                    )
                    repaired_fixture = self._unwrap_structured_output(
                        repaired_fixture, expected_fields={"content"}
                    )
                except (PipelineGenerationError, ValueError) as repair_exc:
                    logger.warning(
                        "success fixture repair call failed: attempt=%d/%d error=%s",
                        fixture_attempt, self.retries, repair_exc,
                    )
                    continue
                repaired_content = repaired_fixture.get("content")
                if not isinstance(repaired_content, str) or len(repaired_content.strip()) < 20:
                    fixture_audit_error = PipelineGenerationError(
                        "repaired success fixture is empty or too short"
                    )
                    logger.warning(
                        "success fixture repair remained invalid: attempt=%d/%d error=%s",
                        fixture_attempt, self.retries, fixture_audit_error,
                    )
                    continue
                success_fixture = repaired_content.strip()
        if fixture_audit_error is not None:
            latest_user_message = next(
                (
                    turn.get("content") for turn in reversed(fixture_conversation)
                    if isinstance(turn, dict) and turn.get("role") == "user"
                    and isinstance(turn.get("content"), str)
                ),
                "",
            )
            try:
                final_fallback = self._call(
                    "acceptance_success_fixture.final_fallback",
                    "独立生成最终成功回答。latest_user_message 是最高优先级指令；必须直接执行其中已经做出的选择、修改和格式要求，不得再次询问确认，不得列出被拒绝的候选方案。只返回 content。",
                    {
                        "task_description": description,
                        "representative_conversation": fixture_conversation,
                        "latest_user_message": latest_user_message,
                        "grounding_environment": environment,
                        "output": {"content": "严格执行最后用户指令的完整回答"},
                    },
                )
                final_fallback = self._unwrap_structured_output(
                    final_fallback, expected_fields={"content"}
                )
                fallback_content = final_fallback.get("content")
                if not isinstance(fallback_content, str) or len(fallback_content.strip()) < 20:
                    raise PipelineGenerationError("final success fixture fallback is empty or too short")
                fallback_content = self._normalize_success_fixture_length(
                    fallback_content.strip(), task_description=description
                )
                fallback_audit = self._call(
                    "acceptance_success_fixture.final_fallback_audit",
                    "严格审查候选回答是否执行最后用户指令、基于已提供事实并完成目标。只返回指定布尔字段和 issues。",
                    {
                        "task_description": description,
                        "representative_conversation": fixture_conversation,
                        "candidate_response": fallback_content,
                        "grounding_environment": environment,
                        "output": {
                            "latest_instructions_satisfied": True,
                            "grounded": True,
                            "goal_completed": True,
                            "issues": [],
                        },
                    },
                )
                self._validate_success_fixture_audit(fallback_audit)
                self._validate_success_fixture_constraints(
                    fallback_content, task_description=description
                )
                success_fixture = fallback_content
                fixture_audit_error = None
                logger.warning("recovered success fixture with independent final fallback")
            except (PipelineGenerationError, ValueError) as fallback_exc:
                logger.warning("final success fixture fallback failed: %s", fallback_exc)
        if fixture_audit_error is not None:
            raise fixture_audit_error
        executable_baseline = self._build_business_scenario_baseline(
            task_description=description,
            tools=tool_list,
            noise_tools=noise_tool_metadata,
            data_tables=data_tables,
            tool_implementations=tool_implementations,
            semantic_goal=semantic_goal,
            training_category=training_category,
            success_content=success_fixture,
        )
        from .tasks.task_spec import TaskSpecError, compile_task_spec

        def validate_compiled_chain(scenarios: list[dict[str, Any]]) -> None:
            # Use the exact IR compiler consumed by the sandbox builder.  A
            # projection-level audit cannot see capture paths or strict input
            # object schemas after the success trace has been assembled.
            try:
                preview_spec = compile_task_spec(
                    task_description=description,
                    training_category=training_category,
                    environment_plan=environment_plan,
                    data_manifest=data_manifest,
                    data_tables=data_tables,
                    tools=tool_list,
                    noise_tools=noise_tool_metadata,
                    tool_bindings=tool_bindings,
                    tool_implementations=tool_implementations,
                    actions=action_list,
                    key_steps=key_steps,
                    metrics=rewards["metrics"],
                    executable_scenarios=scenarios,
                    semantic_goal=semantic_goal,
                    capability_plan=capability_plan,
                )
            except TaskSpecError as exc:
                raise PipelineGenerationError(f"task_spec compilation failed: {exc}") from exc
            self._validate_compiled_tool_chain(preview_spec, scenarios)

        executable_payload["output"] = {"scenarios": [{"kind": "goal_success", "steps": [
            {"operation": "tool_call", "tool_name": "string", "arguments": {}, "capture": {}}
        ]}]}
        for executable_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if executable_error is not None:
                repair_hint = (
                    f"\n上一版轨迹未通过校验：{executable_error}。必须返回完整 scenarios，"
                    "只修复 goal_success 的业务调用；不要生成平台控制步骤或断言。"
                )
            executable_result = self._call(
                "acceptance_executable_scenarios" if executable_attempt == 1 else "acceptance_executable_scenarios.repair",
                "只规划一条真实业务成功路径。返回 scenarios=[{kind: goal_success, steps: [...]}]，steps 只能包含 operation=tool_call、tool_name、arguments 和可选 capture。不得生成 reset、agent_response、reward 或 assertions，它们由平台编译。参数来自题面和给定数据，capture 为变量名到工具结果 JSONPath 的映射（例如 $.records[0].id），后续参数使用 {$ref: 变量名}。捕获的输出类型、对象字段和下游参数 schema 必须兼容；必要时捕获单个字段并用多个 $ref 组装下游对象。多步任务必须具有真实的数据依赖，禁止凭空凑调用。必须遵守 goal_contract 和 capability_dependencies。工具、表、字段和数据必须来自输入；不允许代码、SQL、占位值或绕过工具直接修改状态。" + repair_hint,
                executable_payload,
            )
            candidate_scenarios = self._normalize_executable_scenarios(
                executable_result.get("scenarios", []),
                success_content=success_fixture,
            )
            candidate_scenarios = self._repair_business_scenario_arguments(
                candidate_scenarios, baseline=executable_baseline, tools=tool_list
            )
            candidate_scenarios = self._compile_business_scenario_structure(
                candidate_scenarios, executable_baseline
            )
            if not candidate_scenarios:
                candidate_scenarios = executable_baseline
            try:
                self._validate_executable_scenarios(
                    candidate_scenarios, tools=tool_list, noise_tools=noise_tool_metadata,
                    training_category=training_category,
                    tool_implementations=tool_implementations,
                    semantic_goal=semantic_goal,
                )
                validate_compiled_chain(candidate_scenarios)
                business_scenarios = candidate_scenarios
                break
            except PipelineGenerationError as exc:
                executable_error = exc
                executable_payload["previous_scenarios"] = candidate_scenarios
                logger.warning("business executable scenarios invalid: attempt=%d/%d error=%s", executable_attempt, self.retries, exc)
        if business_scenarios is None:
            self._validate_executable_scenarios(
                executable_baseline, tools=tool_list, noise_tools=noise_tool_metadata,
                training_category=training_category,
                tool_implementations=tool_implementations,
                semantic_goal=semantic_goal,
            )
            validate_compiled_chain(executable_baseline)
            logger.warning(
                "business executable scenario proposals remained invalid; using deterministic baseline: %s",
                executable_error,
            )
            business_scenarios = executable_baseline
        acceptance_contract["executable_scenarios"] = [
            *acceptance_contract.get("executable_scenarios", []), *business_scenarios
        ]
        acceptance_contract = self._ground_acceptance_probes_from_success(
            acceptance_contract, business_scenarios
        )
        metric_implementations = self._compile_process_metric_implementations(
            metrics=rewards["metrics"],
            metric_implementations=metric_implementations,
            business_scenarios=business_scenarios,
            tool_bindings=tool_bindings,
        )
        self._normalize_compiled_process_metrics(
            rewards["metrics"], metric_implementations
        )
        self._validate_metric_implementations(
            metric_implementations, rewards["metrics"], require_process=True
        )

        requirements = description.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {"input_modalities": ["text"]}
        requirements = dict(requirements)
        requirements.setdefault("media_truth_mode", "programmatic")
        # The HTTP contract is derived after tools and rewards are validated.
        # Keeping it deterministic prevents a second LLM-generated interface
        # description from drifting away from the executable tool schema.
        requirements["runtime_interface"] = self._build_runtime_interface(
            tool_list, rewards.get("reward_formula", {})
        )
        self._validate_runtime_interface(
            requirements["runtime_interface"], tool_list, rewards.get("reward_formula", {})
        )
        declared_complexity = description.get("complexity")
        if declared_complexity not in {"simple", "standard", "complex"}:
            raise PipelineGenerationError("task_description.complexity must be simple, standard or complex")
        complexity = self._derive_complexity(
            business_tool_count=len(tool_bindings),
            metric_count=len(rewards["metrics"]),
            key_step_count=len(key_steps),
        )
        training_profile = self._derive_training_profile(
            business_tool_count=len(tool_bindings),
            noise_tool_count=len(noise_tool_metadata),
            key_step_count=len(key_steps),
        )
        readiness_warnings = []
        if complexity != declared_complexity:
            readiness_warnings.append(
                f"complexity normalized from {declared_complexity} to {complexity}"
            )
        from .tasks.task_routing import training_contract as build_training_contract
        training_blueprint = build_training_contract(training_category)
        scenario_kinds = {
            item.get("kind") for item in business_scenarios if isinstance(item, dict)
        }
        missing_scenarios = set(training_blueprint["required_scenarios"]) - scenario_kinds
        if missing_scenarios:
            raise PipelineGenerationError(
                "final consistency: training skeleton is missing scenarios: "
                + ", ".join(sorted(missing_scenarios))
            )
        self._validate_training_contract_consistency(
            task_description=description,
            training_contract=training_blueprint,
            keywords=keywords,
            environment_mode=environment_plan["mode"],
            data_tables=data_tables,
            actions=action_list,
            tools=tool_list,
            tool_bindings=tool_bindings,
            metrics=rewards["metrics"],
            success_fixture=success_fixture,
        )
        self._validate_task_readiness(
            environment_mode=environment_plan["mode"],
            has_business_data=bool(data_tables),
            capability_plan=capability_plan,
            tool_bindings=tool_bindings,
            noise_tools=noise_tool_metadata,
            metrics=rewards["metrics"],
            metric_implementations=metric_implementations,
            business_scenarios=business_scenarios,
            task_description=description,
            data_tables=data_tables,
            media_generation=media_generation,
            success_fixture=success_fixture,
            require_noise=self.noise_tool_max > 0,
        )
        try:
            task_spec = compile_task_spec(
                task_description=description,
                training_category=training_category,
                environment_plan=environment_plan,
                data_manifest=data_manifest,
                data_tables=data_tables,
                tools=tool_list,
                noise_tools=noise_tool_metadata,
                tool_bindings=tool_bindings,
                tool_implementations=tool_implementations,
                actions=action_list,
                key_steps=key_steps,
                metrics=rewards["metrics"],
                executable_scenarios=business_scenarios,
                semantic_goal=semantic_goal,
                capability_plan=capability_plan,
            )
        except TaskSpecError as exc:
            raise PipelineGenerationError(f"task_spec compilation failed: {exc}") from exc
        output_schemas = {
            contract["name"]: contract["output_contract"].get("schema")
            for contract in task_spec["tool_contracts"]
        }
        for endpoint in requirements["runtime_interface"]["endpoints"]:
            if endpoint.get("kind") == "llm_tool":
                schema = output_schemas.get(endpoint.get("name"))
                if isinstance(schema, dict):
                    endpoint["response_schema"] = schema
        result = {
            "task": task_desc.strip(), "task_type": task_type, "task_intent": task_intent,
            "training_category": training_category,
            "training_contract": training_blueprint,
            "runtime_capabilities": {"environment_modes": sorted(supported_modes)},
            "task_spec": task_spec,
            "complexity": complexity,
            "requirements": requirements,
            "public_input": public_input,
            "environment_plan": environment_plan,
            "environment": self._environment_records(environment_summary, action_list),
            "media_generation": media_generation,
            "data_manifest": data_manifest,
            "user_simulation_manifest": user_simulation_manifest,
            "tools_manifest": tools_manifest,
            "reward_key_steps": key_steps,
            "capability_plan": capability_plan,
            "actions": action_list, "tools": tool_list,
            "tool_bindings": tool_bindings,
            "tool_implementations": tool_implementations,
            "noise_tools": noise_tool_metadata,
            "observation_schema": rewards.get("observation_schema", {}), "metrics": rewards["metrics"],
            "metric_implementations": metric_implementations,
            "reward_formula": rewards.get("reward_formula", {}),
            "acceptance_contract": acceptance_contract,
            "task_readiness": {
                "ready": True,
                "training_profile": training_profile,
                "errors": [],
                "warnings": readiness_warnings,
            },
            "generation_pipeline": {"version": "1.0", "stages": [
                "task_description", "public_input_contract", "environment_plan", "environment_entities", "environment_table_design",
                "environment_table_data", "environment_data_consistency", "environment_data_document",
                "environment_media_generation", "user_profiles", "user_scripts",
                "agent_actions", "capability_plan", "openai_tools", "tool_implementation_specs", "reward_key_steps",
                "observations_rewards", "metric_implementation_specs", "acceptance_contract",
                "acceptance_executable_scenarios",
            ]},
        }
        reward_issues = reward_contract_issues(result)
        if reward_issues:
            raise PipelineGenerationError(
                "reward contract failed final assembly: "
                + ", ".join(item["code"] for item in reward_issues)
            )
        logger.info(
            "task pipeline completed: task_type=%s complexity=%s actions=%d tools=%d metrics=%d duration_ms=%.1f artifact_dir=%s",
            task_type,
            complexity,
            len(action_list),
            len(tool_list),
            len(rewards["metrics"]),
            (time.perf_counter() - pipeline_started) * 1000,
            materialized_dir,
        )
        return result

    @staticmethod
    def _fixture_singular_name(name: str) -> str:
        if name.endswith("ies"):
            return f"{name[:-3]}y"
        if name.endswith(("shes", "ches", "sses", "xes")):
            return name[:-2]
        return name[:-1] if name.endswith("s") and not name.endswith("ss") else name

    @staticmethod
    def _schema_fixture(
        schema: dict[str, Any], *, field_name: str = "",
        fixture_values: dict[str, list[Any]] | None = None,
        fixture_rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Any:
        """Create a non-empty request fixture, preferring generated business rows."""
        fixture_values = fixture_values or {}
        fixture_rows = fixture_rows or {}
        if "default" in schema:
            return schema["default"]
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = next((item for item in kind if item != "null"), None)
        if kind == "object":
            properties = schema.get("properties", {})
            return {
                name: TaskGenerationPipeline._schema_fixture(
                    value, field_name=name, fixture_values=fixture_values,
                    fixture_rows=fixture_rows,
                )
                for name, value in properties.items()
                if name in schema.get("required", [])
            }
        if kind == "array":
            items = schema.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                properties = items.get("properties", {})
                required = set(items.get("required", []))
                if isinstance(properties, dict) and required:
                    singular = TaskGenerationPipeline._fixture_singular_name(field_name)
                    candidates = [field_name, singular, singular.removesuffix("_data")]
                    matching_tables = [
                        name for name, rows in fixture_rows.items()
                        if rows and required <= rows[0].keys()
                    ]
                    if len(matching_tables) == 1:
                        candidates.extend(matching_tables)
                    for name in dict.fromkeys(candidates):
                        rows = fixture_rows.get(name, [])
                        if rows and all(required <= row.keys() for row in rows):
                            return [
                                {key: row[key] for key in properties if key in row}
                                for row in rows[:3]
                            ]
            values = fixture_values.get(field_name, [])
            singular_name = TaskGenerationPipeline._fixture_singular_name(field_name)
            if not values:
                values = fixture_values.get(singular_name, [])
            if not values and singular_name.startswith("target_"):
                values = fixture_values.get(singular_name.removeprefix("target_"), [])
            if values:
                return values[: min(3, len(values))]
            return [TaskGenerationPipeline._schema_fixture(
                schema.get("items", {}), field_name=singular_name,
                fixture_values=fixture_values,
                fixture_rows=fixture_rows,
            )]
        if kind == "integer" or kind == "number":
            values = fixture_values.get(field_name, [])
            if values and isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                return values[0]
            return 1
        if kind == "boolean":
            return True
        values = fixture_values.get(field_name, [])
        if not values and field_name.startswith("selected_"):
            values = fixture_values.get(field_name.removeprefix("selected_"), [])
        if not values and field_name.startswith("target_"):
            values = fixture_values.get(field_name.removeprefix("target_"), [])
        if values:
            return values[0]
        description = str(schema.get("description", ""))
        example_match = re.search(r"(?:例如|示例|如)\s*[:：]?\s*([^，。；,;]+)", description)
        if example_match:
            return example_match.group(1).strip(" '\"`")
        return f"任务输入中的{field_name or '值'}"

    @staticmethod
    def _business_fixture_values(data_tables: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Index real generated row values for deterministic acceptance fixtures."""
        values: dict[str, list[Any]] = {}
        table_names: list[str] = []
        for table in data_tables:
            table_name = table.get("table_name")
            if isinstance(table_name, str) and table_name:
                table_names.append(table_name)
                values.setdefault(f"{table_name}_table", []).append(table_name)
            for row in table.get("rows", []):
                if not isinstance(row, dict):
                    continue
                for name, value in row.items():
                    if value in (None, "", [], {}):
                        continue
                    bucket = values.setdefault(str(name), [])
                    if value not in bucket:
                        bucket.append(value)
                    if str(name).endswith("_name"):
                        stem = str(name).removesuffix("_name")
                        alias_bucket = values.setdefault(stem, [])
                        if value not in alias_bucket:
                            alias_bucket.append(value)
                    # Tool schemas often use localized business labels while
                    # generated tables use storage-oriented snake_case names
                    # (for example 撮口呼样本计数 vs table_a_撮口呼_count).
                    # Index conservative aliases so executable positive paths
                    # use a real correlated row instead of the numeric fallback 1.
                    alias = re.sub(r"^(?:table|表)[_-]?[a-z0-9]+[_-]", "", str(name), flags=re.I)
                    alias = alias.replace("_count", "样本计数").replace("count", "样本计数")
                    alias = alias.replace("_", "")
                    if alias and alias != str(name):
                        alias_bucket = values.setdefault(alias, [])
                        if value not in alias_bucket:
                            alias_bucket.append(value)
        if table_names:
            values["reference_data"] = table_names
            values["data_source"] = table_names
        return values

    @classmethod
    def _business_fixture_context(
        cls, data_tables: list[dict[str, Any]], task_description: dict[str, Any],
    ) -> tuple[dict[str, list[Any]], dict[str, list[dict[str, Any]]]]:
        """Index generated rows and concrete public inputs for executable calls."""
        values = cls._business_fixture_values(data_tables)
        rows = {
            str(table.get("table_name")): [row for row in table.get("rows", []) if isinstance(row, dict)]
            for table in data_tables if isinstance(table, dict) and table.get("table_name")
        }

        def add(name: str, value: Any) -> None:
            if value in (None, "", [], {}):
                return
            bucket = values.setdefault(name, [])
            if value not in bucket:
                bucket.append(value)

        public_input = task_description.get("public_input", {})
        materials = public_input.get("materials", []) if isinstance(public_input, dict) else []
        material_rows: list[dict[str, Any]] = []
        for material in materials if isinstance(materials, list) else []:
            if not isinstance(material, dict):
                continue
            material_rows.append({
                key: value for key, value in material.items()
                if key in {"name", "mime_type", "content"}
                and value not in (None, "", [], {})
            })
            name = material.get("name")
            content = material.get("content")
            if isinstance(name, str) and name.strip():
                add("material_name", name)
            if isinstance(content, str) and content.strip():
                add("corpus_text", content)
            if material.get("mime_type") != "application/json":
                continue
            try:
                document = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                continue
            collections = (
                document.items() if isinstance(document, dict)
                else [(str(name or "records"), document)] if isinstance(document, list) else []
            )
            for collection_name, collection_rows in collections:
                if not (isinstance(collection_name, str) and isinstance(collection_rows, list)
                        and collection_rows and all(isinstance(row, dict) for row in collection_rows)):
                    continue
                rows.setdefault(collection_name, collection_rows)
                for row in collection_rows:
                    for field, value in row.items():
                        add(str(field), value)
        if material_rows:
            rows.setdefault("materials", material_rows)
        return values, rows

    @classmethod
    def _build_acceptance_contract(
        cls,
        *,
        task_description: dict[str, Any],
        data_manifest: dict[str, Any],
        data_tables: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        key_steps: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        reward_formula: dict[str, Any],
        tool_implementations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_cases: list[dict[str, Any]] = []
        fixture_values, fixture_rows = cls._business_fixture_context(
            data_tables, task_description,
        )
        sales_rows = fixture_rows.get("sales", [])
        if sales_rows and all(
            isinstance(row.get("item"), str)
            and isinstance(row.get("quantity"), (int, float))
            and not isinstance(row.get("quantity"), bool)
            for row in sales_rows
        ):
            totals: dict[str, list[float]] = {}
            for row in sales_rows:
                totals.setdefault(row["item"], []).append(float(row["quantity"]))
            fixture_rows.setdefault("average_daily_sales", [
                {"item": item, "average_daily_sales": sum(values) / len(values)}
                for item, values in totals.items()
            ])
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        for tool in tools:
            function = tool["function"]
            parameters = function["parameters"]
            required = list(parameters.get("required", []))
            arguments_template = cls._schema_fixture(
                parameters, fixture_values=fixture_values, fixture_rows=fixture_rows,
            )
            implementation = implementation_by_tool.get(function["name"], {})
            for mapping_name in ("selector", "changes", "values"):
                mapping = implementation.get(mapping_name)
                if not isinstance(mapping, dict):
                    continue
                for argument_name, column_name in mapping.items():
                    if (isinstance(arguments_template, dict)
                            and isinstance(arguments_template.get(argument_name), str)
                            and arguments_template[argument_name].startswith("任务输入中的")
                            and isinstance(column_name, str) and fixture_values.get(column_name)):
                        arguments_template[argument_name] = fixture_values[column_name][0]
            tool_cases.append({
                "case_id": f"{function['name']}.valid_shape",
                "tool_name": function["name"],
                "kind": "schema_and_business_smoke",
                "arguments_template": arguments_template,
                "expected": {"status_class": [200], "must_not_return": ["observation", "reward"]},
            })
            if required:
                tool_cases.append({
                    "case_id": f"{function['name']}.missing_required",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {}, "expected": {"status_class": [400, 422]},
                })
            if parameters.get("additionalProperties") is False:
                tool_cases.append({
                    "case_id": f"{function['name']}.unexpected_property",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {"__outer_invalid_property__": True}, "expected": {"status_class": [400, 422]},
                })
        table_names = [table.get("table_name") for table in data_tables]
        critical_fields = {
            table.get("table_name"): [column.get("name") for column in table.get("columns", [])]
            for table in data_tables
        }
        reward_cases = []
        for metric in metrics:
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [value for value in mapping.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not values:
                continue
            reward_cases.append({
                "case_id": f"{metric['id']}.pass",
                "metric_id": metric["id"],
                "expected_score": max(values),
                "fixture": "success",
            })
            reward_cases.append({
                "case_id": f"{metric['id']}.fail",
                "metric_id": metric["id"],
                "expected_score": min(values),
                "fixture": "failure",
            })
        argument_probes = [
            {"probe_id": f"{case['case_id']}.argument_sensitivity", "tool_name": case["tool_name"], "arguments": case["arguments_template"]}
            for case in tool_cases
            if case.get("kind") == "schema_and_business_smoke"
            and isinstance(case.get("arguments_template"), dict) and case["arguments_template"]
        ]
        executable_scenarios = [{
            "scenario_id": "runtime_reset_replay",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "acceptance-episode", "seed": 17}, "expected_status": 200},
                {"operation": "observation", "expected_status": 200},
                {"operation": "replay", "expected_status": 200},
            ],
            "assertions": [{"path": "$.episode_id", "operator": "eq", "expected": "acceptance-episode"}],
        }, *[
            {
                "scenario_id": f"invalid_{case['case_id']}",
                "steps": [{"step_id": "invalid_call", "operation": "tool_call", "tool_name": case["tool_name"], "arguments": case.get("arguments", {}), "expected_status": case["expected"]["status_class"][0]}],
                "assertions": [{"source": "step:invalid_call", "path": "$.error", "operator": "exists", "expected": True}],
            }
            for case in tool_cases if case.get("kind") == "invalid_input"
        ]]
        # Match ManifestDataStore exactly: the runtime hashes the loaded table
        # mapping, not the richer generation-time schema/table documents.
        initial_tables = {
            str(table.get("table_name")): table.get("rows", [])
            for table in data_tables if isinstance(table, dict) and table.get("table_name")
        }
        initial_data_hash = hashlib.sha256(
            json.dumps(
                initial_tables, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        return {
            "version": "1.0",
            "authority": "env_factory_outer_workflow",
            "task_goal": task_description.get("goal") or task_description.get("task"),
            "fixtures": {
                "data_manifest": data_manifest,
                "tables": table_names,
                "critical_fields": critical_fields,
                "initial_data_hash": initial_data_hash,
            },
            "scenarios": [
                {"scenario_id": "reset_isolation", "kind": "lifecycle", "steps": ["reset(seed=17)", "write_episode_a", "reset(seed=17)", "assert_episode_a_not_visible"]},
                {"scenario_id": "replay_integrity", "kind": "replay", "steps": ["reset(seed=17)", "execute_tool_cases", "get_replay", "assert_trace_hash"]},
                *[{"scenario_id": f"key_step_{step['step_id']}", "kind": "goal_critical", "action_name": step["action_name"], "required_for_goal": step["required_for_goal"]} for step in key_steps],
            ],
            "tool_cases": tool_cases,
            "argument_probes": argument_probes,
            "executable_scenarios": executable_scenarios,
            "invariants": [
                "tool response never contains observation or reward",
                "failed tool call does not partially mutate business data",
                "hidden truth is absent from public observation",
                "reward is computed only by explicit reward endpoint",
            ],
            "mutations": [
                {"mutation_id": f"mutate_{table}", "table": table, "strategy": "change_one_critical_field", "fields": critical_fields.get(table, [])}
                for table in table_names
            ],
            "reward_cases": reward_cases,
            "reward_formula": reward_formula,
            "mutation_tests": [
                "replace_tool_result_with_constant",
                "skip_business_write",
                "return_constant_reward",
                "ignore_tool_arguments",
                "bypass_trainer_auth",
            ],
            "actions": [{"name": action["name"], "preconditions": action.get("preconditions", []), "effects": action.get("effects", [])} for action in actions],
        }

    @staticmethod
    def _ground_acceptance_probes_from_success(
        contract: dict[str, Any], scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reuse validated success-call values without carrying runtime refs into smoke probes."""
        success_calls: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            if not isinstance(scenario, dict) or scenario.get("kind") != "goal_success":
                continue
            for step in scenario.get("steps", []):
                if (isinstance(step, dict) and step.get("operation") == "tool_call"
                        and isinstance(step.get("tool_name"), str)
                        and isinstance(step.get("arguments"), dict)):
                    success_calls.setdefault(step["tool_name"], step["arguments"])

        def concrete(value: Any, fallback: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                return copy.deepcopy(fallback)
            if isinstance(value, dict) and isinstance(fallback, dict):
                return {
                    key: concrete(value[key], item) if key in value else copy.deepcopy(item)
                    for key, item in fallback.items()
                }
            if isinstance(value, list):
                if not value:
                    return copy.deepcopy(fallback)
                template = fallback if isinstance(fallback, list) else []
                return [
                    concrete(item, template[min(index, len(template) - 1)] if template else None)
                    for index, item in enumerate(value)
                ]
            return copy.deepcopy(value)

        grounded = copy.deepcopy(contract)
        for case in grounded.get("tool_cases", []):
            if (isinstance(case, dict) and case.get("kind") == "schema_and_business_smoke"
                    and case.get("tool_name") in success_calls):
                case["arguments_template"] = concrete(
                    success_calls[case["tool_name"]], case.get("arguments_template", {})
                )
        for probe in grounded.get("argument_probes", []):
            if isinstance(probe, dict) and probe.get("tool_name") in success_calls:
                probe["arguments"] = concrete(
                    success_calls[probe["tool_name"]], probe.get("arguments", {})
                )
        return grounded

    @staticmethod
    def _merge_acceptance_contract(candidate: Any, baseline: dict[str, Any]) -> dict[str, Any]:
        """Merge an LLM proposal onto the deterministic valid contract.

        The baseline supplies structural completeness.  Candidate fields are
        retained, so the LLM may add richer business assertions without being
        allowed to remove required scenario/tool/reward cases accidentally.
        """
        if not isinstance(candidate, dict):
            return dict(baseline)
        merged = dict(baseline)
        for key, value in candidate.items():
            if key not in {"scenarios", "tool_cases", "reward_cases", "argument_probes", "executable_scenarios"}:
                if isinstance(merged.get(key), dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value

        identity = {
            "scenarios": "scenario_id",
            "tool_cases": "case_id",
            "reward_cases": "case_id",
        }

        def safe_overlay(base_item: Any, candidate_item: Any) -> dict[str, Any]:
            if not isinstance(base_item, dict):
                return dict(candidate_item) if isinstance(candidate_item, dict) else {}
            if not isinstance(candidate_item, dict):
                return dict(base_item)
            result = dict(base_item)
            for field, value in candidate_item.items():
                if value is None or value == "":
                    continue
                if field == "steps" and not isinstance(value, list):
                    continue
                if field == "expected" and not isinstance(value, dict):
                    continue
                result[field] = value
            return result

        for key, id_field in identity.items():
            base_items = baseline.get(key, [])
            candidate_items = candidate.get(key, [])
            if not isinstance(base_items, list):
                base_items = []
            if not isinstance(candidate_items, list):
                candidate_items = []
            by_id = {
                item.get(id_field): item
                for item in candidate_items
                if isinstance(item, dict) and isinstance(item.get(id_field), str) and item.get(id_field)
            }
            result: list[Any] = []
            used: set[int] = set()
            for index, base_item in enumerate(base_items):
                overlay = by_id.get(base_item.get(id_field)) if isinstance(base_item, dict) else None
                if overlay is not None:
                    for candidate_index, candidate_item in enumerate(candidate_items):
                        if candidate_item is overlay:
                            used.add(candidate_index)
                            break
                if overlay is None and index < len(candidate_items) and isinstance(candidate_items[index], dict):
                    overlay = candidate_items[index]
                    used.add(index)
                if isinstance(base_item, dict):
                    combined = safe_overlay(base_item, overlay)
                    if key == "tool_cases":
                        # The executable request and status expectation are
                        # derived from validated tools and business fixtures.
                        # Model prose may enrich a case but cannot replace
                        # its authoritative input with a placeholder.
                        for field in ("tool_name", "kind", "arguments_template", "arguments", "expected"):
                            if field in base_item:
                                combined[field] = base_item[field]
                    result.append(combined)
                else:
                    result.append(overlay or base_item)
            for index, item in enumerate(candidate_items):
                if index not in used and isinstance(item, dict) and item not in result:
                    result.append(item)
            merged[key] = result
        return merged

    @staticmethod
    def _validate_acceptance_contract(contract: Any, *, actions: list[dict[str, Any]], tools: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
        if not isinstance(contract, dict) or contract.get("authority") != "env_factory_outer_workflow":
            raise PipelineGenerationError("acceptance_contract must be EnvFactory-owned")
        for key in ("fixtures", "scenarios", "tool_cases", "argument_probes", "executable_scenarios", "invariants", "mutations", "reward_cases", "mutation_tests"):
            if not isinstance(contract.get(key), list | dict):
                raise PipelineGenerationError(f"acceptance_contract.{key} is invalid")
        if not contract["scenarios"] or (tools and not contract["tool_cases"]) or not contract["reward_cases"]:
            raise PipelineGenerationError("acceptance_contract requires scenarios, tool_cases and reward_cases")
        action_names = {action.get("name") for action in actions}
        tool_names = {tool.get("function", {}).get("name") for tool in tools}
        metric_by_id = {metric.get("id"): metric for metric in metrics}
        metric_ids = set(metric_by_id)
        for case in contract["tool_cases"]:
            if not isinstance(case, dict) or not case.get("case_id") or case.get("tool_name") not in tool_names or not isinstance(case.get("expected"), dict):
                raise PipelineGenerationError("acceptance_contract references unknown tool")
        for case in contract["reward_cases"]:
            if (
                not isinstance(case, dict) or not case.get("case_id")
                or case.get("metric_id") not in metric_ids
                or isinstance(case.get("expected_score"), bool)
                or not isinstance(case.get("expected_score"), (int, float))
            ):
                raise PipelineGenerationError("acceptance_contract references unknown metric")
            metric = metric_by_id[case["metric_id"]]
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [
                value for value in mapping.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            expected = max(values) if str(case["case_id"]).endswith(".pass") else min(values)
            if values and case["expected_score"] != expected:
                raise PipelineGenerationError(
                    f"acceptance_contract reward case {case['case_id']} has reversed expected score"
                )
        for scenario in contract["scenarios"]:
            if not isinstance(scenario, dict) or not scenario.get("scenario_id") or not scenario.get("kind") or not isinstance(scenario.get("steps"), list):
                raise PipelineGenerationError("acceptance_contract scenario is incomplete")
            if scenario.get("kind") == "goal_critical" and scenario.get("action_name") not in action_names:
                raise PipelineGenerationError("acceptance_contract references unknown action")
        if not contract.get("invariants") or not contract.get("mutation_tests"):
            raise PipelineGenerationError("acceptance_contract requires invariants and mutation_tests")

    @staticmethod
    def _validate_executable_scenarios(
        scenarios: Any, *, tools: list[dict[str, Any]], noise_tools: list[dict[str, Any]],
        training_category: str | None = None,
        tool_implementations: list[dict[str, Any]] | None = None,
        semantic_goal: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(scenarios, list) or not scenarios:
            raise PipelineGenerationError("executable business scenarios must be a non-empty list")
        tool_schemas = {
            tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
        }
        noise_names = {item.get("name") for item in noise_tools if isinstance(item, dict)}
        allowed_operations = {
            "reset", "tool_call", "agent_response", "observation", "reward", "replay",
            "business_snapshot", "mutate_business_state",
        }
        allowed_operators = {
            "eq", "ne", "gte", "lte", "contains", "exists", "count_gte",
            "count_eq", "changed", "unchanged", "subset",
        }
        kinds: set[str] = set()
        ids: set[str] = set()
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }

        def validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                if not isinstance(value["$ref"], str) or not value["$ref"].strip():
                    raise PipelineGenerationError(f"{path} has an invalid reference")
                return
            kind = schema.get("type")
            if kind == "array":
                if not isinstance(value, list) or not value:
                    raise PipelineGenerationError(f"{path} must be a non-empty array")
                for item_index, item in enumerate(value):
                    validate_value(item, schema.get("items", {}), f"{path}[{item_index}]")
            elif kind == "object":
                if not isinstance(value, dict) or not value:
                    raise PipelineGenerationError(f"{path} must be a non-empty object")
                properties = schema.get("properties", {})
                for required_name in schema.get("required", []):
                    if required_name not in value:
                        raise PipelineGenerationError(f"{path} misses required field {required_name}")
                for child_name, child_value in value.items():
                    if child_name in properties:
                        validate_value(child_value, properties[child_name], f"{path}.{child_name}")
            elif kind == "string":
                if not isinstance(value, str) or not value.strip():
                    raise PipelineGenerationError(f"{path} must be a non-empty string")
                lowered = value.strip().lower()
                if any(marker in lowered for marker in (
                    "fixture-value", "placeholder", "todo", "sample-value", "test-value",
                    "任务输入中", "待提供", "请填写", "示例值",
                )):
                    raise PipelineGenerationError(f"{path} uses a placeholder value")
        def references(value: Any) -> set[str]:
            if isinstance(value, dict):
                if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
                    return {value["$ref"]}
                return set().union(*(references(item) for item in value.values())) if value else set()
            if isinstance(value, list):
                return set().union(*(references(item) for item in value)) if value else set()
            return set()
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or not isinstance(scenario.get("scenario_id"), str):
                raise PipelineGenerationError(f"executable_scenarios[{index}] is invalid")
            if scenario["scenario_id"] in ids or scenario.get("kind") not in {"goal_success", "goal_failure", "noise_selection", "counterfactual"}:
                raise PipelineGenerationError(f"executable_scenarios[{index}] id/kind is invalid")
            ids.add(scenario["scenario_id"])
            kinds.add(scenario["kind"])
            steps = scenario.get("steps")
            if not isinstance(steps, list) or not steps:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires steps")
            captures: set[str] = set()
            capture_shapes: dict[str, set[str]] = {}
            success_business_calls = 0
            has_dependency_edge = False
            for step in steps:
                if not isinstance(step, dict) or step.get("operation") not in allowed_operations:
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid operation")
                if step["operation"] == "tool_call":
                    name, arguments = step.get("tool_name"), step.get("arguments", {})
                    if name not in tool_schemas or not isinstance(arguments, dict):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] references invalid tool")
                    if scenario["kind"] == "goal_success" and name in noise_names:
                        raise PipelineGenerationError("goal_success scenario cannot use noise tools")
                    if scenario["kind"] == "goal_success":
                        success_business_calls += 1
                        step_refs = references(arguments)
                        if step_refs:
                            if not step_refs <= captures:
                                raise PipelineGenerationError(
                                    f"executable_scenarios[{index}] references an uncaptured value"
                                )
                            has_dependency_edge = True
                    schema = tool_schemas[name]
                    if set(arguments) - set(schema.get("properties", {})):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] has unknown tool arguments")
                    if set(schema.get("required", [])) - set(arguments):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] misses required tool arguments")
                    if scenario["kind"] == "goal_success":
                        for argument_name in schema.get("required", []):
                            validate_value(
                                arguments[argument_name],
                                schema.get("properties", {}).get(argument_name, {}),
                                f"executable_scenarios[{index}].{name}.{argument_name}",
                            )
                            argument = arguments[argument_name]
                            argument_schema = schema.get("properties", {}).get(argument_name, {})
                            if isinstance(argument, dict) and set(argument) == {"$ref"}:
                                projected = capture_shapes.get(argument["$ref"])
                                item_schema = argument_schema.get("items", {})
                                required_fields = set(item_schema.get("required", [])) if (
                                    argument_schema.get("type") == "array"
                                    and isinstance(item_schema, dict)
                                ) else set()
                                if projected is not None and not required_fields <= projected:
                                    missing = sorted(required_fields - projected)
                                    raise PipelineGenerationError(
                                        f"executable_scenarios[{index}] dependency into "
                                        f"{name}.{argument_name} misses projected fields: {missing}"
                                    )
                if step["operation"] == "agent_response" and (
                    not isinstance(step.get("content"), str) or not step["content"].strip()
                ):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid agent response")
                capture = step.get("capture") or {}
                if not isinstance(capture, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in capture.items()):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] capture is invalid")
                if step.get("operation") == "tool_call":
                    implementation = implementation_by_tool.get(step.get("tool_name"), {})
                    result_field = implementation.get("result_field")
                    projection = implementation.get("projection")
                    if isinstance(result_field, str) and isinstance(projection, list):
                        aliases = implementation.get("projection_aliases", {})
                        for variable, json_path in capture.items():
                            if json_path == f"$.{result_field}":
                                capture_shapes[variable] = {
                                    field for field in projection if isinstance(field, str)
                                }
                                if isinstance(aliases, dict):
                                    capture_shapes[variable].update(
                                        alias for alias in aliases if isinstance(alias, str)
                                    )
                captures.update(capture)
            if scenario["kind"] == "goal_success":
                if training_category == "direct_response" and success_business_calls:
                    raise PipelineGenerationError("direct_response success must not call business tools")
                if training_category == "simple_agentic" and success_business_calls != 1:
                    raise PipelineGenerationError("simple_agentic success requires exactly one business tool call")
                if training_category == "multi_step_agentic" and (
                    success_business_calls < 2 or not has_dependency_edge
                ):
                    raise PipelineGenerationError(
                        "multi_step_agentic success requires two business calls and a capture/$ref dependency"
                    )
                TaskGenerationPipeline._validate_success_scenario_goal_coverage(
                    scenario,
                    semantic_goal=semantic_goal,
                    tool_implementations=tool_implementations or [],
                )
            assertions = scenario.get("assertions", [])
            if not isinstance(assertions, list) or not assertions:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires assertions")
            if any(not isinstance(item, dict) or item.get("operator") not in allowed_operators for item in assertions):
                raise PipelineGenerationError(f"executable_scenarios[{index}] assertion is invalid")
            reward_assertions = [
                item for item in assertions
                if isinstance(item, dict) and item.get("path") == "$.reward"
            ]
            if scenario["kind"] == "goal_success" and not any(
                item.get("operator") == "gte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) >= 0.5
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_success must assert reward >= 0.5")
            if scenario["kind"] == "goal_failure" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0.2
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_failure must assert reward <= 0.2")
            if scenario["kind"] == "noise_selection" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0
                for item in reward_assertions
            ):
                raise PipelineGenerationError("noise_selection must assert non-positive reward")
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("executable scenarios require goal_success and goal_failure")
        if noise_names and "noise_selection" not in kinds:
            raise PipelineGenerationError("tasks with noise tools require a noise_selection scenario")

    @staticmethod
    def _validate_success_scenario_goal_coverage(
        scenario: dict[str, Any], *, semantic_goal: dict[str, Any] | None,
        tool_implementations: list[dict[str, Any]],
    ) -> None:
        """Prove that the success trace can establish every declared state goal."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        if not predicates:
            return
        implementations = {
            item.get("tool_name"): item for item in tool_implementations
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        calls = [
            step for step in scenario.get("steps", [])
            if isinstance(step, dict) and step.get("operation") == "tool_call"
        ]
        used: set[int] = set()

        def matches(call: dict[str, Any], predicate: dict[str, Any]) -> bool:
            implementation = implementations.get(call.get("tool_name"), {})
            operation = implementation.get("operation")
            expected_operation = "delete" if predicate.get("count") == 0 else None
            if implementation.get("table") != predicate.get("table"):
                return False
            if expected_operation and operation != expected_operation:
                return False
            if not expected_operation and operation not in {"update", "insert"}:
                return False
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                return False
            selector = implementation.get("selector", {})
            changes = implementation.get("changes", {})
            values = implementation.get("values", {})

            def establishes(expected: dict[str, Any], mappings: list[Any], *, allow_ref: bool) -> bool:
                for column, value in expected.items():
                    candidates = [
                        argument for mapping in mappings if isinstance(mapping, dict)
                        for argument, mapped_column in mapping.items()
                        if mapped_column == column
                    ]
                    if not candidates:
                        return False
                    actual = arguments.get(candidates[0])
                    if actual == value:
                        continue
                    if allow_ref and isinstance(actual, dict) and set(actual) == {"$ref"}:
                        continue
                    return False
                return True

            return (
                establishes(predicate.get("where", {}), [selector, values], allow_ref=True)
                and establishes(predicate.get("values", {}), [changes, values], allow_ref=False)
            )

        for index, predicate in enumerate(predicates):
            if not isinstance(predicate, dict):
                continue
            match = next((
                call_index for call_index, call in enumerate(calls)
                if call_index not in used and matches(call, predicate)
            ), None)
            if match is None:
                raise PipelineGenerationError(
                    f"goal_success does not establish row_predicates[{index}]"
                )
            used.add(match)

    @classmethod
    def _build_business_scenario_baseline(
        cls,
        *,
        task_description: dict[str, Any],
        tools: list[dict[str, Any]],
        noise_tools: list[dict[str, Any]],
        data_tables: list[dict[str, Any]] | None = None,
        tool_implementations: list[dict[str, Any]] | None = None,
        semantic_goal: dict[str, Any] | None = None,
        training_category: str | None = None,
        success_content: str | None = None,
    ) -> list[dict[str, Any]]:
        noise_names = {
            item.get("name") for item in noise_tools if isinstance(item, dict)
        }
        business_tools = [
            tool for tool in tools if tool.get("function", {}).get("name") not in noise_names
        ]
        fixture_values, fixture_rows = cls._business_fixture_context(
            data_tables or [], task_description,
        )
        success_steps: list[dict[str, Any]] = [
            {"operation": "reset", "body": {"episode_id": "goal-success", "seed": 17}, "expected_status": 200}
        ]
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        prior_outputs: list[tuple[str, list[str]]] = []
        dependency_added = False
        tool_step_index = 0
        goal_predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        for tool in business_tools:
            function = tool["function"]
            fixture_arguments = cls._schema_fixture(
                function["parameters"], fixture_values=fixture_values,
                fixture_rows=fixture_rows,
            )
            implementation = implementation_by_tool.get(function["name"], {})
            # Business tool argument names often differ from source table columns.
            # Bind schema placeholders through the declared selector before asking
            # the model to invent a value for an executable acceptance trace.
            selector_mapping = implementation.get("selector", {})
            for argument_name, column_name in (selector_mapping.items()
                                               if isinstance(selector_mapping, dict) else []):
                if (argument_name in fixture_arguments and isinstance(column_name, str)
                        and isinstance(fixture_arguments[argument_name], str)
                        and fixture_arguments[argument_name].startswith("任务输入中的")
                        and fixture_values.get(column_name)):
                    fixture_arguments[argument_name] = fixture_values[column_name][0]
            predicates = [
                item for item in goal_predicates
                if isinstance(item, dict) and item.get("table") == implementation.get("table")
            ]
            argument_variants = [fixture_arguments]
            if implementation.get("operation") in {"insert", "update", "delete"} and predicates:
                argument_variants = []
                for predicate in predicates:
                    arguments = copy.deepcopy(fixture_arguments)
                    for mapping_name, source in (
                        ("selector", predicate.get("where", {})),
                        ("changes", predicate.get("values", {})),
                        ("values", {**predicate.get("where", {}), **predicate.get("values", {})}),
                    ):
                        mapping = implementation.get(mapping_name, {})
                        if isinstance(mapping, dict) and isinstance(source, dict):
                            for argument, column in mapping.items():
                                if column in source:
                                    arguments[argument] = source[column]
                    argument_variants.append(arguments)
            for arguments in argument_variants:
                if training_category == "multi_step_agentic" and prior_outputs and not dependency_added:
                    required = function["parameters"].get("required", [])
                    properties = function["parameters"].get("properties", {})
                    dependency = cls._match_baseline_dependency(
                        required=required, properties=properties, prior_outputs=prior_outputs
                    )
                    if dependency is not None:
                        argument_name, variable_name, json_path = dependency
                        arguments[argument_name] = {"$ref": variable_name}
                        prior_step = next(
                            step for step in reversed(success_steps)
                            if step.get("operation") == "tool_call"
                        )
                        prior_step.setdefault("capture", {})[variable_name] = json_path
                        dependency_added = True
                tool_step_index += 1
                success_steps.append({
                    "step_id": f"business_tool_{tool_step_index}",
                    "operation": "tool_call",
                    "tool_name": function["name"],
                    "arguments": arguments,
                    "expected_status": 200,
                })
            result_field = implementation.get("result_field")
            projection = implementation.get("projection", [])
            if isinstance(result_field, str) and result_field.strip():
                fields = [item for item in projection if isinstance(item, str)]
                aliases = implementation.get("projection_aliases", {})
                if isinstance(aliases, dict):
                    fields.extend(alias for alias in aliases if isinstance(alias, str))
                prior_outputs.append((result_field, fields))
        success_steps.extend([
            {
                "step_id": "final_answer",
                "operation": "agent_response",
                "content": str(success_content or task_description.get("expected_result") or task_description.get("goal") or "已完成任务并给出依据。"),
                "expected_status": 200,
            },
            {"step_id": "success_reward", "operation": "reward", "expected_status": 200},
        ])
        scenarios: list[dict[str, Any]] = [{
            "scenario_id": "deterministic_goal_success",
            "kind": "goal_success",
            "steps": success_steps,
            "assertions": [
                {"source": "step:success_reward", "path": "$.reward", "operator": "gte", "expected": 0.6}
            ],
        }, {
            "scenario_id": "deterministic_goal_failure",
            "kind": "goal_failure",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "goal-failure", "seed": 17}, "expected_status": 200},
                {"step_id": "failure_reward", "operation": "reward", "expected_status": 200},
            ],
            "assertions": [
                {"source": "step:failure_reward", "path": "$.reward", "operator": "lte", "expected": 0.2}
            ],
        }]
        if noise_tools:
            noise_name = str(noise_tools[0]["name"])
            noise_tool = next(tool for tool in tools if tool.get("function", {}).get("name") == noise_name)
            scenarios.append({
                "scenario_id": "deterministic_noise_selection",
                "kind": "noise_selection",
                "steps": [
                    {"operation": "reset", "body": {"episode_id": "noise-selection", "seed": 17}, "expected_status": 200},
                    {
                        "step_id": "noise_call", "operation": "tool_call", "tool_name": noise_name,
                        "arguments": cls._schema_fixture(noise_tool["function"]["parameters"]), "expected_status": 200,
                    },
                    {"step_id": "noise_reward", "operation": "reward", "expected_status": 200},
                ],
                "assertions": [
                    {"source": "step:noise_reward", "path": "$.reward", "operator": "lte", "expected": 0}
                ],
            })
        return scenarios

    @staticmethod
    def _match_baseline_dependency(
        *, required: list[Any], properties: dict[str, Any],
        prior_outputs: list[tuple[str, list[str]]],
    ) -> tuple[str, str, str] | None:
        """Find a schema-compatible, fixture-backed edge for a multi-tool trace."""
        for result_field, projection in reversed(prior_outputs):
            for argument_name in required:
                if not isinstance(argument_name, str):
                    continue
                if argument_name in projection:
                    variable = f"upstream_{argument_name}"
                    return argument_name, variable, f"$.{result_field}[0].{argument_name}"
            for argument_name in required:
                schema = properties.get(argument_name, {})
                if not isinstance(argument_name, str) or schema.get("type") != "array":
                    continue
                item_schema = schema.get("items", {})
                if not isinstance(item_schema, dict) or item_schema.get("type") != "object":
                    continue
                item_required = item_schema.get("required", [])
                if (not isinstance(item_required, list)
                        or any(not isinstance(field, str) for field in item_required)
                        or not set(item_required) <= set(projection)):
                    continue
                variable = f"upstream_{result_field}"
                return argument_name, variable, f"$.{result_field}"
        return None

    @staticmethod
    def _select_success_response_fixture(
        *, task_description: dict[str, Any]
    ) -> str:
        return str(
            task_description.get("expected_result")
            or task_description.get("goal")
            or "已完成任务并给出依据。"
        )

    @staticmethod
    def _compile_business_scenario_structure(
        proposals: list[dict[str, Any]], baseline: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Models fill business calls; platform owns lifecycle and assertions."""
        import copy
        result = copy.deepcopy(baseline)
        success = next((item for item in proposals if item.get("kind") == "goal_success"), None)
        if success is None:
            return result
        calls = [copy.deepcopy(step) for step in success.get("steps", [])
                 if isinstance(step, dict) and step.get("operation") == "tool_call"]
        compiled = next(item for item in result if item.get("kind") == "goal_success")
        for index, step in enumerate(calls):
            step["step_id"] = f"business_tool_{index + 1}"
            step["expected_status"] = 200
        compiled["steps"] = [compiled["steps"][0], *calls, *compiled["steps"][-2:]]
        return result

    @staticmethod
    def _repair_business_scenario_arguments(
        proposals: list[dict[str, Any]], *, baseline: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Absorb malformed argument serialization without changing a valid plan.

        Tool schemas and fixture-backed baseline calls are already validated by
        the platform.  We use them only to fill missing, empty, or wrong-typed
        values.  Valid model values and capture references remain authoritative.
        """
        import copy

        schemas = {
            item.get("function", {}).get("name"): item.get("function", {}).get("parameters", {})
            for item in tools if isinstance(item, dict)
        }
        baseline_success = next(
            (item for item in baseline if isinstance(item, dict) and item.get("kind") == "goal_success"),
            {},
        )
        fallback_arguments = {
            step.get("tool_name"): step.get("arguments", {})
            for step in baseline_success.get("steps", []) if isinstance(step, dict)
            and step.get("operation") == "tool_call"
        }

        def repair(value: Any, schema: dict[str, Any], fallback: Any, field_name: str = "") -> Any:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.strip():
                    return value
            kind = schema.get("type")
            if isinstance(kind, list):
                kind = next((item for item in kind if item != "null"), None)
            if kind == "object":
                properties = schema.get("properties", {})
                current = value if isinstance(value, dict) else {}
                base = fallback if isinstance(fallback, dict) else {}
                result = {
                    name: repair(child, properties.get(name, {}), base.get(name), name)
                    for name, child in current.items() if name in properties
                }
                for name in schema.get("required", []):
                    if name not in result:
                        result[name] = repair(None, properties.get(name, {}), base.get(name), name)
                return result
            if kind == "array":
                if not isinstance(value, list) or not value:
                    if not isinstance(fallback, list) or not fallback:
                        fallback = TaskGenerationPipeline._schema_fixture(
                            schema, field_name=field_name
                        )
                    return copy.deepcopy(fallback)
                item_schema = schema.get("items", {})
                fallback_items = fallback if isinstance(fallback, list) else []
                return [
                    repair(item, item_schema, fallback_items[min(index, len(fallback_items) - 1)] if fallback_items else None, field_name)
                    for index, item in enumerate(value)
                ]
            if kind == "string":
                if isinstance(value, str) and value.strip() and not any(
                    marker in value.strip().lower() for marker in (
                        "fixture-value", "placeholder", "todo", "sample-value", "test-value",
                        "任务输入中", "待提供", "请填写", "示例值",
                    )
                ):
                    return value
                if not isinstance(fallback, str) or not fallback.strip():
                    fallback = TaskGenerationPipeline._schema_fixture(
                        schema, field_name=field_name
                    )
                return copy.deepcopy(fallback)
            if kind in {"integer", "number"}:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                return value if valid else copy.deepcopy(fallback)
            if kind == "boolean":
                return value if isinstance(value, bool) else copy.deepcopy(fallback)
            return value if value is not None else copy.deepcopy(fallback)

        repaired = copy.deepcopy(proposals)
        for scenario in repaired:
            if not isinstance(scenario, dict) or scenario.get("kind") != "goal_success":
                continue
            for step in scenario.get("steps", []):
                if not isinstance(step, dict) or step.get("operation") != "tool_call":
                    continue
                name = step.get("tool_name")
                schema = schemas.get(name)
                if not isinstance(schema, dict):
                    continue
                step["arguments"] = repair(
                    step.get("arguments"), schema, fallback_arguments.get(name, {})
                )
        return repaired

    @staticmethod
    def _normalize_executable_scenarios(
        scenarios: Any, *, success_content: str = ""
    ) -> list[dict[str, Any]]:
        """Canonicalize business scenario identity without changing its semantics."""
        if not isinstance(scenarios, list):
            return []
        aliases = {
            "success": "goal_success",
            "successful": "goal_success",
            "goal-success": "goal_success",
            "failure": "goal_failure",
            "failed": "goal_failure",
            "goal-failure": "goal_failure",
            "noise": "noise_selection",
            "noise-selection": "noise_selection",
            "counter-factual": "counterfactual",
        }
        allowed = {"goal_success", "goal_failure", "noise_selection", "counterfactual"}
        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        counts: dict[str, int] = {}
        for raw in scenarios:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            raw_kind = item.get("kind")
            kind = aliases.get(str(raw_kind).strip().lower(), raw_kind)
            scenario_id = item.get("scenario_id")
            identity = str(scenario_id or "").lower()
            if kind not in allowed:
                if "success" in identity:
                    kind = "goal_success"
                elif "fail" in identity:
                    kind = "goal_failure"
                elif "noise" in identity:
                    kind = "noise_selection"
                elif "counter" in identity or "mutation" in identity:
                    kind = "counterfactual"
                else:
                    # Models sometimes echo lifecycle/replay entries from the
                    # baseline. They are already retained by the outer
                    # contract and are not business success/failure scenarios.
                    continue
            item["kind"] = kind
            counts[kind] = counts.get(kind, 0) + 1
            base_id = str(scenario_id).strip() if isinstance(scenario_id, str) else ""
            if not base_id:
                base_id = f"{kind}_{counts[kind]}"
            unique_id = base_id
            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            item["scenario_id"] = unique_id
            if kind == "goal_success" and success_content.strip():
                steps = item.get("steps")
                if isinstance(steps, list):
                    for step in steps:
                        if (
                            isinstance(step, dict)
                            and step.get("operation") == "agent_response"
                            and (
                                not isinstance(step.get("content"), str)
                                or not step["content"].strip()
                            )
                        ):
                            step["content"] = success_content.strip()
            used_ids.add(unique_id)
            normalized.append(item)
        return normalized

    @staticmethod
    def _fill_tool_schema_descriptions(tools: list[Any]) -> None:
        """Fill cosmetic JSON Schema descriptions without another LLM call."""
        def fill_schema(schema: Any, label: str) -> None:
            if not isinstance(schema, dict):
                return
            if not isinstance(schema.get("description"), str) or not schema["description"].strip():
                schema["description"] = f"{label} 的输入值。"
            properties = schema.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    fill_schema(child, str(name))
            items = schema.get("items")
            if isinstance(items, dict):
                fill_schema(items, f"{label} 数组元素")

        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name") if isinstance(function.get("name"), str) else "工具"
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                function["description"] = f"执行 {name} 操作。"
            fill_schema(function.get("parameters"), f"{name} 参数")

    @staticmethod
    def _validate_noise_tool_audit(decisions: Any, *, noise_names: set[str]) -> set[str]:
        if not isinstance(decisions, list):
            raise PipelineGenerationError("noise tool audit must return decisions")
        seen: set[str] = set()
        unsafe: set[str] = set()
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                raise PipelineGenerationError(f"noise audit decisions[{index}] must be an object")
            name = decision.get("name")
            if name not in noise_names or name in seen:
                raise PipelineGenerationError("noise audit references an unknown or duplicate tool")
            if not isinstance(decision.get("is_safe_noise"), bool):
                raise PipelineGenerationError("noise audit requires boolean is_safe_noise")
            if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise PipelineGenerationError("noise audit requires reason")
            if not decision["is_safe_noise"]:
                unsafe.add(str(name))
            seen.add(str(name))
        if seen != noise_names:
            raise PipelineGenerationError(
                f"noise audit must cover every noise tool; missing={sorted(noise_names - seen)}"
            )
        return unsafe

    @staticmethod
    def _build_noise_fixture(function: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Build a minimal exact-match fixture from a validated noise schema."""
        properties = function.get("parameters", {}).get("properties", {})
        mapping = {name: name for name in properties}
        rows: list[dict[str, Any]] = []
        for index in range(2):
            row: dict[str, Any] = {}
            for name, schema in properties.items():
                value = TaskGenerationPipeline._schema_fixture(schema, field_name=name)
                if isinstance(value, str) and not schema.get("enum"):
                    value = f"{value}-{index + 1}"
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    value += index
                row[name] = value
            row["result"] = f"{function.get('name', 'noise_lookup')} 的合成参考结果 {index + 1}"
            rows.append(row)
        return rows, mapping

    @staticmethod
    def _validate_noise_tool_safety(tools: list[Any]) -> None:
        """Noise tools must not claim real-world writes the shared runtime cannot perform."""
        mutation_name_prefixes = (
            "book_", "reserve_", "create_", "update_", "delete_", "submit_", "send_",
            "purchase_", "order_",
        )
        read_only_name_prefixes = (
            "search_", "lookup_", "get_", "list_", "find_", "estimate_",
            "calculate_", "check_", "convert_", "preview_", "inspect_",
        )
        embedded_mutation_name = re.compile(
            r"(?:^|_)(?:book|reserve|create|update|delete|submit|send|purchase|order)(?:_|$)"
        )
        # Descriptions often mention read-only fields such as "booking status" or
        # "预订信息".  Match an explicit operation claim instead of rejecting any
        # occurrence of a mutation-related noun.
        mutation_description_patterns = (
            r"\b(?:book|reserve|create|update|delete|submit|send|purchase|order)s?\b"
            r"\s+(?:a|an|the|new|user|customer|external|real-world)?\s*"
            r"(?:booking|reservation|record|request|message|email|order|purchase|resource)",
            r"(?:执行|进行|发起|完成)(?:实际|外部|真实)?"
            r"(?:预订|预约|创建|修改|更新|删除|提交|发送|购买|下单)",
            r"(?:预订|预约|创建|修改|更新|删除|提交|发送|购买|下单)"
            r"(?:场地|座位|记录|请求|消息|邮件|订单|商品|资源)",
        )
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = str(function.get("name", "")).lower()
            description = str(function.get("description", "")).lower()
            if name.startswith(read_only_name_prefixes) and not embedded_mutation_name.search(name):
                claims_mutation = False
            else:
                claims_mutation = name.startswith(mutation_name_prefixes) or any(
                    re.search(pattern, description, re.I)
                    for pattern in mutation_description_patterns
                )
            if claims_mutation:
                raise PipelineGenerationError(
                    f"noise tool {function.get('name')} claims a side effect; use a read-only distractor"
                )



    @staticmethod
    def _validate_tools(tools: list[Any]) -> None:
        names: set[str] = set()
        for index, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
                raise PipelineGenerationError(f"tools[{index}] is not an OpenAI function tool")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise PipelineGenerationError(f"tools[{index}].function.name is invalid or duplicated")
            if re.fullmatch(
                r"(?:noise_)?tool_?\d*|function_?\d*|semantic_function_name_?\d*",
                name,
                flags=re.IGNORECASE,
            ):
                raise PipelineGenerationError(
                    f"tool {name} uses a placeholder name; use a concrete semantic function name"
                )
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                raise PipelineGenerationError(f"tool {name} has no description")
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise PipelineGenerationError(f"tool {name} parameters must be an object schema")
            required = parameters.get("required", [])
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict) or not isinstance(required, list) or any(item not in properties for item in required):
                raise PipelineGenerationError(f"tool {name} has invalid properties/required")
            forbidden = {"acceptance_evidence_id", "acceptance_message_id", "ground_truth", "user_accepted"}
            if forbidden & set(properties):
                raise PipelineGenerationError(f"tool {name} exposes forbidden internal input")
            def check_properties(items: dict[str, Any], prefix: str) -> None:
                for property_name, schema in items.items():
                    if not isinstance(schema, dict) or not isinstance(schema.get("description"), str) or not schema["description"].strip():
                        raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} requires a description")
                    if schema.get("type") == "object":
                        nested = schema.get("properties", {})
                        if not isinstance(nested, dict):
                            raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} has invalid nested properties")
                        check_properties(nested, prefix + property_name + ".")
                    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
                        check_properties({"[]": schema["items"]}, prefix + property_name)
            check_properties(properties, "")
            names.add(name)

    @staticmethod
    def _validate_tool_implementations(
        specs: list[Any], *, tools: list[dict[str, Any]], tables: list[dict[str, Any]]
    ) -> None:
        tool_parameters = {
            tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
            for tool in tools
        }
        table_columns = {
            table["table_name"]: {column["name"] for column in table.get("columns", [])}
            for table in tables
        }
        seen: set[str] = set()
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"tool_implementations[{index}] must be an object")
            name, table = spec.get("tool_name"), spec.get("table")
            if not isinstance(name, str) or name not in tool_parameters or name in seen:
                raise PipelineGenerationError(f"tool_implementations[{index}] references an invalid tool")
            operation = spec.get("operation")
            if (
                operation not in {"select", "aggregate_count", "insert", "update", "delete"}
                or not isinstance(table, str)
                or table not in table_columns
            ):
                raise PipelineGenerationError(f"tool_implementations[{index}] operation/table is invalid")
            if not isinstance(spec.get("result_field"), str) or not spec["result_field"]:
                raise PipelineGenerationError(f"tool_implementations[{index}] requires result_field")
            filters = spec.get("filters", [])
            if not isinstance(filters, list):
                raise PipelineGenerationError(f"tool_implementations[{index}].filters must be a list")
            for rule in filters:
                if (
                    not isinstance(rule, dict)
                    or not isinstance(rule.get("argument"), str)
                    or not isinstance(rule.get("column"), str)
                    or rule.get("argument") not in tool_parameters[name]
                    or rule.get("column") not in table_columns[table]
                    or rule.get("operator") not in {"eq", "in", "contains", "gte", "lte"}
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}] has an invalid filter")
                resolver = rule.get("resolve")
                if resolver is not None:
                    if not isinstance(resolver, dict):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] has an invalid filter resolver"
                        )
                    resolver_table = resolver.get("table")
                    match_column = resolver.get("match_column")
                    value_column = resolver.get("value_column")
                    if (
                        resolver_table not in table_columns
                        or match_column not in table_columns[resolver_table]
                        or value_column not in table_columns[resolver_table]
                    ):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] resolver references an invalid table column"
                        )
                    target_table = next(
                        item for item in tables if item.get("table_name") == table
                    )
                    foreign_keys = target_table.get("foreign_keys", [])
                    if not any(
                        isinstance(foreign, dict)
                        and foreign.get("column") == rule.get("column")
                        and (foreign.get("ref_table") or foreign.get("references_table"))
                            == resolver_table
                        and (foreign.get("ref_column") or foreign.get("references_column"))
                            == value_column
                        for foreign in foreign_keys
                    ):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] resolver is not backed by a foreign key"
                        )
            for field in ("projection", "order_by"):
                values = spec.get(field, [])
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or value not in table_columns[table]
                    for value in values
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            projection_aliases = spec.get("projection_aliases", {})
            if not isinstance(projection_aliases, dict) or any(
                not isinstance(alias, str) or not alias
                or not isinstance(column, str) or column not in table_columns[table]
                or alias in spec.get("projection", [])
                for alias, column in projection_aliases.items()
            ):
                raise PipelineGenerationError(
                    f"tool_implementations[{index}].projection_aliases is invalid"
                )
            for field in ("selector", "values", "changes"):
                mapping = spec.get(field, {})
                if not isinstance(mapping, dict) or any(
                    not isinstance(argument, str)
                    or not isinstance(column, str)
                    or argument not in tool_parameters[name]
                    or column not in table_columns[table]
                    for argument, column in mapping.items()
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            if operation in {"update", "delete"} and not spec.get("selector"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires selector")
            if operation == "insert" and not spec.get("values"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires values")
            if operation == "update" and not spec.get("changes"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires changes")
            seen.add(name)

    @staticmethod
    def _complete_filter_resolvers(
        *, implementations: list[dict[str, Any]], tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve human-readable related-entity arguments through declared FKs."""
        table_map = {
            table.get("table_name"): table for table in tables
            if isinstance(table, dict) and isinstance(table.get("table_name"), str)
        }
        completed = copy.deepcopy(implementations)
        for spec in completed:
            target = table_map.get(spec.get("table"), {})
            foreign_keys = target.get("foreign_keys", []) if isinstance(target, dict) else []
            for rule in spec.get("filters", []):
                if not isinstance(rule, dict) or rule.get("resolve") is not None:
                    continue
                foreign = next((
                    item for item in foreign_keys
                    if isinstance(item, dict) and item.get("column") == rule.get("column")
                ), None)
                if not foreign:
                    continue
                related_name = foreign.get("ref_table") or foreign.get("references_table")
                value_column = foreign.get("ref_column") or foreign.get("references_column")
                related = table_map.get(related_name)
                argument = rule.get("argument")
                if not isinstance(related, dict) or not isinstance(argument, str):
                    continue
                related_columns = {
                    column.get("name") for column in related.get("columns", [])
                    if isinstance(column, dict)
                }
                candidates = [argument]
                prefix = f"{related_name}_"
                if isinstance(related_name, str) and argument.startswith(prefix):
                    candidates.append(argument[len(prefix):])
                match_column = next(
                    (candidate for candidate in candidates if candidate in related_columns), None
                )
                if match_column and value_column in related_columns:
                    rule["resolve"] = {
                        "table": related_name,
                        "match_column": match_column,
                        "value_column": value_column,
                    }
        return completed

    @staticmethod
    def _validate_business_tool_semantics(
        *, tools: list[dict[str, Any]], implementations: list[dict[str, Any]],
    ) -> None:
        """Reject declarative tools whose public API overclaims their behavior."""
        by_name = {
            item.get("tool_name"): item for item in implementations
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        read_tokens = {"get", "query", "search", "lookup", "list", "find", "fetch", "inspect"}
        mutation_tokens = {
            "create", "insert", "add", "update", "modify", "correct", "set", "save", "mark",
            "delete", "remove",
        }
        semantic_tokens = {
            "calculate", "validate", "verify", "compare", "recommend", "classify", "summarize",
            "estimate", "score", "evaluate",
        }
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = function.get("name")
            spec = by_name.get(name)
            # No declarative spec means the sandbox must supply a real custom
            # handler; the independent semantic review owns that path.
            if not isinstance(name, str) or not isinstance(spec, dict):
                continue
            parameters = function.get("parameters", {}).get("properties", {})
            parameter_names = set(parameters) if isinstance(parameters, dict) else set()
            consumed = {
                rule.get("argument") for rule in spec.get("filters", [])
                if isinstance(rule, dict) and isinstance(rule.get("argument"), str)
            }
            for field in ("selector", "values", "changes"):
                mapping = spec.get(field, {})
                if isinstance(mapping, dict):
                    consumed.update(mapping)
            unused = sorted(parameter_names - consumed)
            if unused:
                raise PipelineGenerationError(
                    f"business tool {name} exposes parameters ignored by its implementation: {unused}"
                )
            operation = spec.get("operation")
            lowered = name.lower()
            name_tokens = set(lowered.split("_"))
            if operation == "select" and (
                name_tokens & semantic_tokens or not name_tokens & read_tokens
            ):
                raise PipelineGenerationError(
                    f"business tool {name} overclaims a plain select implementation"
                )
            if operation == "aggregate_count" and "count" not in name_tokens:
                raise PipelineGenerationError(
                    f"business tool {name} must expose count semantics"
                )
            if operation in {"insert", "update", "delete"} and not name_tokens & mutation_tokens:
                raise PipelineGenerationError(
                    f"business tool {name} does not name its mutation semantics"
                )

    @staticmethod
    def _validate_goal_tool_coverage(
        *, semantic_goal: dict[str, Any] | None,
        tool_implementations: list[Any],
    ) -> None:
        """Require every compiled mutation to implement an asserted goal delta."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        by_table: dict[str, list[dict[str, Any]]] = {}
        for predicate in predicates:
            if isinstance(predicate, dict) and isinstance(predicate.get("table"), str):
                by_table.setdefault(predicate["table"], []).append(predicate)

        mutations = [
            item for item in tool_implementations
            if isinstance(item, dict)
            and item.get("operation") in {"insert", "update", "delete"}
        ]
        mutation_tables = {
            item.get("table") for item in mutations if isinstance(item.get("table"), str)
        }
        uncovered_goal_tables = sorted(set(by_table) - mutation_tables)
        if uncovered_goal_tables:
            raise PipelineGenerationError(
                "stateful goal_contract has no declarative mutation for tables: "
                f"{uncovered_goal_tables}"
            )
        for implementation in mutations:
            table = str(implementation.get("table", ""))
            tool_name = implementation.get("tool_name")
            table_predicates = by_table.get(table, [])
            if not table_predicates:
                raise PipelineGenerationError(
                    f"stateful tool {tool_name} mutates table {table} but goal_contract "
                    "does not assert its final state"
                )
            if implementation.get("operation") != "update":
                continue
            changed_columns = {
                column for column in implementation.get("changes", {}).values()
                if isinstance(column, str)
            }
            asserted_values = {
                field for predicate in table_predicates
                for field in predicate.get("values", {})
            }
            missing = asserted_values - changed_columns
            if missing:
                raise PipelineGenerationError(
                    f"stateful tool {tool_name} cannot establish goal fields: {sorted(missing)}"
                )

    @staticmethod
    def _validate_stateful_tool_surface(
        tools: list[Any], *, semantic_goal: dict[str, Any] | None
    ) -> None:
        """Require a compilable mutation surface before accepting tool schemas."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        mutation_markers = (
            "create", "insert", "add", "update", "modify", "correct", "delete", "remove",
            "创建", "新增", "添加", "更新", "修改", "修正", "删除", "移除",
        )
        surfaces: list[set[str]] = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            label = f"{function.get('name', '')} {function.get('description', '')}".lower()
            if not any(marker in label for marker in mutation_markers):
                continue
            properties = function.get("parameters", {}).get("properties", {})
            if isinstance(properties, dict):
                surfaces.append(set(properties))
        for predicate in predicates if isinstance(predicates, list) else []:
            if not isinstance(predicate, dict):
                continue
            required = set(predicate.get("where", {})) | set(predicate.get("values", {}))
            def covers(surface: set[str]) -> bool:
                return all(
                    len([
                        argument for argument in surface
                        if argument == field
                        or argument.endswith(f"_{field}")
                        or argument.removeprefix("new_") == field
                        or argument.removeprefix("target_") == field
                    ]) == 1
                    for field in required
                )
            if not required or not any(covers(surface) for surface in surfaces):
                raise PipelineGenerationError(
                    "stateful tool schema has no compilable mutation parameters for "
                    f"table {predicate.get('table')}: {sorted(required)}"
                )

    @staticmethod
    def _validate_compiled_tool_chain(
        task_spec: dict[str, Any], scenarios: list[dict[str, Any]],
    ) -> None:
        from .contracts.tool_chain_contract import tool_chain_issues

        issues = tool_chain_issues({
            "task_spec": task_spec,
            "acceptance_contract": {"executable_scenarios": scenarios},
        })
        if issues:
            details = [{"code": issue["code"], **issue["evidence"]} for issue in issues]
            raise PipelineGenerationError(
                f"compiled tool-chain schema conflicts: {details}"
            )

    @staticmethod
    def _tool_chain_projection_gaps(
        *, route_plan: dict[str, Any], tool_bindings: list[dict[str, Any]],
        tools: list[dict[str, Any]], implementations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find unambiguous route edges whose required record shape is unavailable."""
        bound: dict[str, list[str]] = {}
        for binding in tool_bindings:
            if not isinstance(binding, dict):
                continue
            action, name = binding.get("action_name"), binding.get("tool_name")
            if isinstance(action, str) and isinstance(name, str):
                bound.setdefault(action, []).append(name)
        schemas = {
            tool.get("function", {}).get("name"): tool.get("function", {}).get("parameters", {})
            for tool in tools if isinstance(tool, dict)
        }
        specs = {
            spec.get("tool_name"): spec for spec in implementations
            if isinstance(spec, dict)
        }
        gaps: list[dict[str, Any]] = []
        operations = route_plan.get("environment_operations", [])
        for operation in operations if isinstance(operations, list) else []:
            if not isinstance(operation, dict):
                continue
            dependencies = operation.get("dependencies", [])
            downstream_names = bound.get(operation.get("action_name"), [])
            if not isinstance(dependencies, list) or len(dependencies) != 1 or len(downstream_names) != 1:
                continue
            upstream_names = bound.get(dependencies[0], [])
            if len(upstream_names) != 1:
                continue
            upstream_name, downstream_name = upstream_names[0], downstream_names[0]
            upstream = specs.get(upstream_name, {})
            if upstream.get("operation") != "select" or not isinstance(upstream.get("result_field"), str):
                continue
            projected = {
                field for field in upstream.get("projection", []) if isinstance(field, str)
            }
            aliases = upstream.get("projection_aliases", {})
            if isinstance(aliases, dict):
                projected.update(field for field in aliases if isinstance(field, str))
            parameters = schemas.get(downstream_name, {})
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                continue
            # A matching scalar can carry the dependency even when a separate
            # record array is supplied from public input.
            if any(
                name in projected
                and isinstance(properties.get(name), dict)
                and properties[name].get("type") != "array"
                for name in required if isinstance(name, str)
            ):
                continue
            arrays: list[tuple[str, set[str]]] = []
            for name in required:
                schema = properties.get(name, {})
                if not isinstance(name, str) or not isinstance(schema, dict) or schema.get("type") != "array":
                    continue
                item = schema.get("items", {})
                fields = item.get("required", []) if isinstance(item, dict) and item.get("type") == "object" else []
                if isinstance(fields, list) and fields and all(isinstance(field, str) for field in fields):
                    arrays.append((name, set(fields)))
            if len(arrays) != 1:
                continue
            argument, fields = arrays[0]
            missing = sorted(fields - projected)
            if missing:
                gaps.append({
                    "upstream_tool": upstream_name,
                    "upstream_table": upstream.get("table"),
                    "downstream_tool": downstream_name,
                    "downstream_argument": argument,
                    "missing_fields": missing,
                    "projected_fields": sorted(projected),
                })
        return gaps

    @staticmethod
    def _complete_dependency_projections(
        *, implementations: list[dict[str, Any]], tools: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose downstream fields and labels for selected business entities."""
        import copy

        schema_fields: set[str] = set()
        def collect(schema: Any) -> None:
            if not isinstance(schema, dict):
                return
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                schema_fields.update(str(name) for name in properties)
                for child in properties.values():
                    collect(child)
            collect(schema.get("items"))
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            collect(function.get("parameters"))

        table_columns = {
            table.get("table_name"): [
                str(column.get("name")) for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            ]
            for table in tables if isinstance(table, dict)
        }
        table_primary_keys = {
            table.get("table_name"): {
                key for key in table.get("primary_key", []) if isinstance(key, str)
            }
            for table in tables if isinstance(table, dict)
            and isinstance(table.get("primary_key", []), list)
        }
        completed = copy.deepcopy(implementations)
        for spec in completed:
            if not isinstance(spec, dict) or spec.get("operation") != "select":
                continue
            available = table_columns.get(spec.get("table"), [])
            required = [name for name in available if name in schema_fields]
            projection = spec.get("projection", [])
            if not isinstance(projection, list):
                projection = []
            table_name = str(spec.get("table", ""))
            # A record identified only by its id is hard to use in a named
            # report. Include a same-entity display label when the select
            # already exposes that entity's primary key. This does not add
            # unrelated table columns or infer hidden business truth.
            label_names = {
                "name", "title", "label", f"{table_name}_name",
                f"{table_name.removesuffix('s')}_name",
            }
            primary_keys = table_primary_keys.get(table_name, set())
            projected_names = {name for name in projection if isinstance(name, str)}
            labels = [
                name for name in available if name in label_names
                and primary_keys and primary_keys <= projected_names
            ]
            projection = list(dict.fromkeys([*projection, *required, *labels]))
            if projection:
                spec["projection"] = projection
            aliases = spec.get("projection_aliases", {})
            if not isinstance(aliases, dict):
                aliases = {}
            for public_name in sorted(schema_fields - set(available)):
                candidates = [
                    column for column in available
                    if public_name == f"{table_name}_{column}"
                ]
                if len(candidates) == 1 and public_name not in projection:
                    aliases.setdefault(public_name, candidates[0])
            if aliases:
                spec["projection_aliases"] = aliases
        return completed

    @staticmethod
    def _complete_stateful_tool_implementations(
        *, implementations: list[dict[str, Any]], tools: list[dict[str, Any]],
        tables: list[dict[str, Any]], semantic_goal: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Compile obvious one-table updates when a model omits a valid spec.

        This is intentionally conservative: argument and column names must be
        identical, selector and changed fields must both be present, and their
        primitive types must agree. More complex semantics remain custom-handler
        work and therefore fail the stateful buildability gate.
        """
        completed = [dict(item) for item in implementations]
        raw_predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        predicates = [
            item for item in raw_predicates if isinstance(item, dict)
        ] if isinstance(raw_predicates, list) else []
        goals_by_table: dict[str, list[dict[str, Any]]] = {}
        for predicate in predicates:
            table = predicate.get("table")
            if isinstance(table, str):
                goals_by_table.setdefault(table, []).append(predicate)

        table_by_name = {
            table.get("table_name"): table for table in tables
            if isinstance(table, dict) and isinstance(table.get("table_name"), str)
        }
        numeric_types = ("INT", "DECIMAL", "NUMERIC", "REAL", "FLOAT", "DOUBLE")

        def compatible(schema: dict[str, Any], sql_type: str) -> bool:
            json_type = schema.get("type")
            is_numeric = any(token in sql_type.upper() for token in numeric_types)
            return json_type in ({"number", "integer"} if is_numeric else {"string"})

        def argument_for(field: str, properties: dict[str, Any], columns: dict[str, Any]) -> str | None:
            candidates = [
                argument for argument, schema in properties.items()
                if isinstance(argument, str)
                and (
                    argument == field
                    or argument.endswith(f"_{field}")
                    or argument.removeprefix("new_") == field
                    or argument.removeprefix("target_") == field
                )
                and field in columns
                and isinstance(schema, dict)
                and compatible(schema, str(columns[field].get("type", "")))
            ]
            return candidates[0] if len(candidates) == 1 else None

        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = function.get("name")
            if not isinstance(name, str):
                continue
            label = f"{name} {function.get('description', '')}".lower()
            if not any(marker in label for marker in ("update", "modify", "更新", "修改", "修正")):
                continue
            existing = next(
                (item for item in completed if item.get("tool_name") == name), None
            )
            if isinstance(existing, dict) and existing.get("operation") in {
                "insert", "update", "delete",
            }:
                continue
            properties = function.get("parameters", {}).get("properties", {})
            if not isinstance(properties, dict):
                continue
            candidates: list[dict[str, Any]] = []
            for table_name, table_predicates in goals_by_table.items():
                table = table_by_name.get(table_name, {})
                columns = {
                    column.get("name"): column
                    for column in table.get("columns", [])
                    if isinstance(column, dict)
                    if isinstance(column.get("name"), str)
                }
                where_fields = {
                    field for predicate in table_predicates
                    for field in predicate.get("where", {})
                }
                value_fields = {
                    field for predicate in table_predicates
                    for field in predicate.get("values", {})
                }
                selector = {
                    argument: field for field in where_fields
                    if (argument := argument_for(field, properties, columns)) is not None
                }
                changes = {
                    argument: field for field in value_fields
                    if (argument := argument_for(field, properties, columns)) is not None
                }
                if (
                    selector and changes
                    and where_fields <= set(selector.values())
                    and value_fields <= set(changes.values())
                ):
                    candidates.append({
                        "tool_name": name, "operation": "update", "table": table_name,
                        "selector": selector, "changes": changes, "result_field": "records",
                    })
            if len(candidates) == 1:
                self_spec = candidates[0]
                completed = [
                    item for item in completed if item.get("tool_name") != name
                ]
                TaskGenerationPipeline._validate_tool_implementations(
                    [*completed, self_spec], tools=tools, tables=tables
                )
                completed.append(self_spec)
                logger.warning(
                    "compiled deterministic stateful implementation: tool=%s table=%s",
                    name, self_spec["table"],
                )
        return completed

    @staticmethod
    def _build_runtime_interface(
        tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the HTTP contract consumed by sandbox builders and trainers.

        This is deliberately derived from validated artifacts rather than
        requested from the LLM.  Every generated LLM tool is listed with its
        exact request schema, and the single runtime reward function has an
        explicit endpoint.
        """
        tool_endpoints = []
        for tool in tools:
            function = tool["function"]
            tool_endpoints.append({
                "name": function["name"],
                "kind": "llm_tool",
                "method": "POST",
                "path": "/v1/tools/{tool_name}",
                "request_schema": function["parameters"],
                "response_schema": {
                    "type": "object",
                    "description": "本次工具调用的业务结果；不包含 observation 或 reward。",
                },
            })
        return {
            "protocol": "http",
            "base_path": "/v1",
            "version": "1.0",
            "shared_runtime": {
                "module": "sandbox_runtime.py",
                "version": "1.2",
                "required_components": [
                    "EpisodeStore",
                    "ManifestDataStore",
                    "DeclarativeToolCompiler",
                    "SandboxApplication",
                    "ContractToolRegistry",
                    "ContractRewardAggregator",
                    "ContractRewardGate",
                    "DeclarativeMetricEvaluator",
                    "AcceptanceScenarioRunner",
                    "ContractUserSimulator",
                    "validate_json_schema",
                ],
            },
            "errors": {
                "content_type": "application/json",
                "schema": {
                    "type": "object",
                    "required": ["error"],
                    "error_required": ["code", "message", "request_id"],
                },
                "status_codes": {"invalid_argument": 400, "unauthorized": 401, "forbidden": 403, "not_found": 404, "conflict": 409, "server_error": 500},
            },
            "observability": {
                "request_id_header": "X-Request-ID",
                "required_fields": ["request_id", "episode_id", "tool_call_id", "timestamp", "duration_ms"],
                "credential_redaction": True,
                "structured_format": "json",
            },
            "security": {
                "trainer": {
                    "scheme": "bearer",
                    "environment_variable": "SANDBOX_TRAINER_API_KEY",
                    "required_for": ["reset", "observation", "state", "user_simulator", "agent_response", "reward", "replay"],
                },
                "agent": {
                    "scheme": "none",
                    "allowed_kinds": ["llm_tool"],
                    "cannot_access": ["state", "user_simulator", "agent_response", "reward", "replay"],
                },
            },
            "episode": {
                "isolation": "per_episode",
                "reset_accepts_seed": True,
                "deterministic_replay": True,
                "idempotency_header": "Idempotency-Key",
                "persist_trace": True,
            },
            "llm_runtime": {
                "provider": "openai_compatible_chat_completions",
                "api_key_environment_variable": "SANDBOX_LLM_API_KEY",
                "base_url_environment_variable": "SANDBOX_LLM_BASE_URL",
                "model_environment_variable": "SANDBOX_LLM_MODEL",
                "timeout_environment_variable": "SANDBOX_LLM_TIMEOUT_SECONDS",
                "max_retries_environment_variable": "SANDBOX_LLM_MAX_RETRIES",
                "never_persist_credentials": True,
            },
            "evaluator_runtime": {
                "external_llm": "SANDBOX_LLM_API_KEY",
                "mock_mode_environment_variable": "SANDBOX_EVALUATOR_MOCK",
                "record_call_trace": True,
                "cache_by_context_hash": True,
            },
            "mutation_testing": {
                "environment_variable": "SANDBOX_MUTATION_MODE",
                "modes": ["constant_tool_result", "skip_business_write", "constant_reward", "ignore_tool_arguments", "bypass_trainer_auth"],
                "production_default": "disabled",
                "must_be_observable_in_test_only": True,
            },
            "launcher": {
                "command": ["python", "app.py", "--port", "{port}"],
                "health_path": "/health",
            },
            "endpoints": [
                {"name": "health", "kind": "system", "method": "GET", "path": "/health"},
                {
                    "name": "reset", "kind": "system", "method": "POST", "path": "/v1/reset",
                    "access": "rl_trainer_only",
                    "request_schema": {
                        "type": "object", "description": "创建或重置独立 episode。",
                        "properties": {
                            "episode_id": {"type": "string", "description": "可选的 episode 标识。"},
                            "seed": {"type": "integer", "description": "可选的确定性随机种子。"},
                        }, "required": [], "additionalProperties": False,
                    },
                },
                {"name": "observation", "kind": "system", "access": "rl_trainer_only", "method": "GET", "path": "/v1/observation"},
                {
                    "name": "state", "kind": "trainer_evidence",
                    "access": "rl_trainer_only", "method": "GET", "path": "/v1/state",
                    "response_schema": {
                        "type": "object",
                        "description": "仅供 Trainer 记录初末业务状态；不得进入 policy observation。",
                    },
                },
                {"name": "tools", "kind": "system", "method": "GET", "path": "/v1/tools"},
                {
                    "name": "user_simulator",
                    "kind": "user_simulator",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/user_simulator",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                                "items": {
                                    "type": "object",
                                    "description": "一条对话消息。",
                                    "properties": {
                                        "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                                        "content": {"type": "string", "description": "消息文本内容。"},
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["messages"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "User Simulator 生成的用户查询和终止标志。",
                        "properties": {
                            "user_query": {"type": "string", "description": "模拟用户生成的自然语言查询。"},
                            "should_end": {"type": "boolean", "description": "用户是否停止继续提问。"},
                            "match_status": {"type": "string", "description": "实时对话为 matched、unmatched 或 ambiguous。"},
                            "outcome_category": {"type": "string", "description": "固定的八类对话结果之一。"},
                            "reason_code": {"type": "string", "description": "分支匹配或恢复原因。"},
                            "fsm_script_id": {"type": "string", "description": "Trainer-only：本轮使用的 FSM 脚本身份。"},
                            "fsm_transition_id": {"type": ["string", "null"], "description": "Trainer-only：matched 时实际应用的迁移；恢复结果为空。"},
                            "fsm_state_before": {"type": "string", "description": "Trainer-only：迁移前状态。"},
                            "fsm_state_after": {"type": "string", "description": "Trainer-only：迁移后状态。"},
                            "fsm_transition_applied": {"type": "boolean", "description": "Trainer-only：是否应用了声明式迁移。"},
                            "fsm_recovery_count": {"type": "integer", "minimum": 0, "description": "Trainer-only：连续恢复次数。"},
                            "attachments": {"type": "array", "items": {"type": "object", "description": "附件对象（当前不支持）"}, "maxItems": 0,
                                            "description": "当前文本/结构化环境不接受模型生成附件，固定为空。"},
                            "termination_reason": {"type": "string", "description": "结束时为 completed 或 unresolved_dialogue。"},
                        },
                        "required": [
                            "user_query", "should_end", "match_status",
                            "outcome_category", "reason_code", "attachments",
                            "fsm_script_id", "fsm_transition_id",
                            "fsm_state_before", "fsm_state_after",
                            "fsm_transition_applied", "fsm_recovery_count",
                        ],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "agent_response",
                    "kind": "agent_response",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/agent_response",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 提交待训练 Agent 的最终自然语言回答。",
                        "properties": {
                            "content": {"type": "string", "description": "Agent 的非空最终回答。"},
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "确认最终回答已写入当前 episode。",
                    },
                },
                *tool_endpoints,
                {
                    "name": "reward",
                    "kind": "reward_function",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/reward",
                    "formula": reward_formula,
                    "response_schema": {
                        "type": "object",
                        "description": "调用该接口时统一计算并返回当前会话奖励和各指标得分。",
                    },
                },
                {
                    "name": "replay",
                    "kind": "replay",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/replay",
                    "response_schema": {
                        "type": "object",
                        "description": "返回当前 episode 的可回放轨迹、seed、版本和数据哈希。",
                    },
                },
            ],
            "llm_tools": [endpoint["name"] for endpoint in tool_endpoints],
            "reward_functions": [{"name": "reward", "endpoint": "/v1/reward"}],
        }

    @staticmethod
    def _validate_runtime_interface(
        interface: Any, tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> None:
        if not isinstance(interface, dict) or interface.get("protocol") != "http":
            raise PipelineGenerationError("requirements.runtime_interface must declare http")
        shared_runtime = interface.get("shared_runtime")
        required_shared = {
            "EpisodeStore", "ManifestDataStore", "DeclarativeToolCompiler", "SandboxApplication", "ContractToolRegistry",
            "ContractRewardAggregator", "ContractRewardGate", "DeclarativeMetricEvaluator", "AcceptanceScenarioRunner", "ContractUserSimulator",
            "validate_json_schema"
        }
        if (
            not isinstance(shared_runtime, dict)
            or shared_runtime.get("module") != "sandbox_runtime.py"
            or not required_shared <= set(shared_runtime.get("required_components", []))
        ):
            raise PipelineGenerationError("runtime_interface shared runtime contract is incomplete")
        endpoints = interface.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            raise PipelineGenerationError("runtime_interface.endpoints must be non-empty")
        endpoint_keys = {(item.get("name"), item.get("method"), item.get("path"))
                         for item in endpoints if isinstance(item, dict)}
        for name, method, path in (
            ("health", "GET", "/health"),
            ("reset", "POST", "/v1/reset"),
            ("observation", "GET", "/v1/observation"),
            ("state", "GET", "/v1/state"),
            ("tools", "GET", "/v1/tools"),
            ("user_simulator", "POST", "/v1/user_simulator"),
            ("agent_response", "POST", "/v1/agent_response"),
            ("reward", "GET", "/v1/reward"),
            ("replay", "GET", "/v1/replay"),
        ):
            if (name, method, path) not in endpoint_keys:
                raise PipelineGenerationError(f"runtime_interface missing {method} {path}")
        expected_tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        declared_tools = [
            item for item in endpoints
            if isinstance(item, dict) and item.get("kind") == "llm_tool"
        ]
        if len(declared_tools) != len(expected_tools):
            raise PipelineGenerationError("runtime_interface must list every LLM tool exactly once")
        for item in declared_tools:
            name = item.get("name")
            if name not in expected_tools or item.get("method") != "POST" or item.get("path") != "/v1/tools/{tool_name}":
                raise PipelineGenerationError(f"invalid runtime endpoint for tool {name!r}")
            if item.get("request_schema") != expected_tools[name]:
                raise PipelineGenerationError(f"runtime schema does not match tool {name}")
        if interface.get("llm_tools") != list(expected_tools):
            raise PipelineGenerationError("runtime_interface.llm_tools is inconsistent with tools")
        user_simulator_endpoint = next(
            item for item in endpoints if item.get("name") == "user_simulator"
        )
        if user_simulator_endpoint.get("kind") != "user_simulator" or user_simulator_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime_interface user_simulator endpoint has invalid kind")
        expected_user_request = {
            "type": "object",
            "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                    "items": {
                        "type": "object",
                        "description": "一条对话消息。",
                        "properties": {
                            "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                            "content": {"type": "string", "description": "消息文本内容。"},
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["messages"],
            "additionalProperties": False,
        }
        if user_simulator_endpoint.get("request_schema") != expected_user_request:
            raise PipelineGenerationError("user_simulator request schema must use complete messages")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("reward endpoint must be Trainer-only")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("formula") != reward_formula:
            raise PipelineGenerationError("runtime reward endpoint formula is inconsistent")
        replay_endpoint = next(item for item in endpoints if item.get("name") == "replay")
        if replay_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime replay endpoint must be Trainer-only")
        security = interface.get("security")
        if not isinstance(security, dict):
            raise PipelineGenerationError("runtime_interface.security is required")
        trainer_security = security.get("trainer")
        if not isinstance(trainer_security, dict) or trainer_security.get("scheme") != "bearer" \
                or trainer_security.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
            raise PipelineGenerationError("runtime_interface trainer authentication is invalid")
        episode = interface.get("episode")
        if not isinstance(episode, dict) or episode.get("isolation") != "per_episode" \
                or episode.get("reset_accepts_seed") is not True \
                or episode.get("deterministic_replay") is not True:
            raise PipelineGenerationError("runtime_interface episode isolation/replay contract is invalid")
        llm_runtime = interface.get("llm_runtime")
        if not isinstance(llm_runtime, dict) or llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
            raise PipelineGenerationError("runtime_interface LLM credential boundary is invalid")
        evaluator_runtime = interface.get("evaluator_runtime")
        if not isinstance(evaluator_runtime, dict) or evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
            raise PipelineGenerationError("runtime_interface evaluator mock contract is invalid")
        mutation_testing = interface.get("mutation_testing")
        if not isinstance(mutation_testing, dict) or mutation_testing.get("environment_variable") != "SANDBOX_MUTATION_MODE" \
                or not isinstance(mutation_testing.get("modes"), list) or not mutation_testing["modes"]:
            raise PipelineGenerationError("runtime_interface mutation testing contract is invalid")
        launcher = interface.get("launcher")
        if not isinstance(launcher, dict) or launcher.get("command") != ["python", "app.py", "--port", "{port}"]:
            raise PipelineGenerationError("runtime_interface launcher contract is invalid")
        errors = interface.get("errors")
        if not isinstance(errors, dict) or errors.get("content_type") != "application/json" \
                or not isinstance(errors.get("schema"), dict):
            raise PipelineGenerationError("runtime_interface error protocol is invalid")
        observability = interface.get("observability")
        if not isinstance(observability, dict) or observability.get("request_id_header") != "X-Request-ID" \
                or observability.get("credential_redaction") is not True:
            raise PipelineGenerationError("runtime_interface observability contract is invalid")

    @staticmethod
    def _validate_actions(actions: list[Any]) -> None:
        """Validate the action decomposition before tools are generated."""
        names: set[str] = set()
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise PipelineGenerationError(f"agent_actions[{index}] must be an object")
            name = action.get("name") or action.get("action")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"agent_actions[{index}] has invalid or duplicate name")
            if not isinstance(action.get("atomicity_rationale"), str) or not action["atomicity_rationale"].strip():
                raise PipelineGenerationError(f"agent action {name} requires atomicity_rationale")
            for field in ("inputs", "outputs"):
                values = action.get(field)
                if not isinstance(values, list):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list")
                for item_index, item in enumerate(values):
                    if not isinstance(item, dict):
                        raise PipelineGenerationError(f"agent action {name}.{field}[{item_index}] must be an object")
                    for item_field in ("name", "description"):
                        if not isinstance(item.get(item_field), str) or not item[item_field].strip():
                            raise PipelineGenerationError(
                                f"agent action {name}.{field}[{item_index}] requires {item_field}"
                            )
            for field in ("preconditions", "effects"):
                values = action.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list of non-empty strings")
            names.add(name)

    @staticmethod
    def _deterministic_actions(
        *, task_description: dict[str, Any], keywords: list[str], environment_mode: str,
        route_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a minimal grounded plan after repeated action-model drift."""
        relevant = TaskGenerationPipeline._task_relevant_keywords(
            task_description, keywords
        )
        subject = relevant[0] if relevant else str(task_description.get("task", "任务"))[:16]
        actions: list[dict[str, Any]] = []

        def action(name: str, description: str, output: str) -> dict[str, Any]:
            return {
                "name": name,
                "description": description,
                "atomicity_rationale": "该动作只完成一个明确阶段，不合并后续判断或表达。",
                "inputs": [{"name": "task_input", "description": "用户提供的任务输入与约束。"}],
                "outputs": [{"name": output, "description": f"完成{name}后得到的可验证结果。"}],
                "preconditions": ["已获得当前阶段所需输入。"],
                "effects": [f"产生{output}，不引入任务外事实。"],
            }

        route_operations = (
            route_plan.get("environment_operations", [])
            if isinstance(route_plan, dict) else []
        )
        if route_operations:
            for index, operation in enumerate(route_operations, start=1):
                actions.append(action(
                    str(operation["action_name"]),
                    str(operation.get("purpose", "执行任务所需的环境操作。")),
                    f"environment_result_{index}",
                ))
        elif environment_mode in {"reference_data", "stateful"}:
            actions.append(action(
                f"读取{subject}任务数据", f"读取沙箱中与{subject}任务直接相关的记录。", "task_records"
            ))
        elif environment_mode == "external_capability":
            actions.append(action(
                f"查询{subject}外部信息", f"通过声明的外部能力获取与{subject}直接相关的信息。", "external_facts"
            ))
        actions.extend([
            action(f"分析{subject}任务输入", f"依据任务约束分析与{subject}相关的输入。", "analysis_result"),
            action(f"生成{subject}任务结果", f"根据已验证信息完成{subject}任务并遵守输出格式。", "final_result"),
        ])
        return actions

    @staticmethod
    def _ensure_data_access_action(
        actions: list[Any], *, environment_mode: str, table_names: list[str]
    ) -> list[Any]:
        """Add a stable read boundary when a data-backed task omitted one."""
        if environment_mode not in {"reference_data", "stateful"} or not table_names:
            return actions
        chinese_markers = ("读取", "查询", "检索", "查找", "获取")
        english_pattern = re.compile(r"\b(?:read|query|search|lookup|get)\b", re.I)
        if any(
            isinstance(action, dict)
            and (
                any(marker in f"{action.get('name', '')} {action.get('description', '')}" for marker in chinese_markers)
                or english_pattern.search(f"{action.get('name', '')} {action.get('description', '')}")
            )
            for action in actions
        ):
            return actions
        name = "读取任务参考数据"
        used = {str(action.get("name")) for action in actions if isinstance(action, dict)}
        suffix = 2
        while name in used:
            name = f"读取任务参考数据{suffix}"
            suffix += 1
        return [
            {
                "name": name,
                "description": f"从沙箱只读数据表中获取完成任务所需的记录：{', '.join(table_names)}。",
                "atomicity_rationale": "该动作仅负责读取环境事实，不进行比较、决策或最终回答。",
                "inputs": [{"name": "query", "description": "用户目标和筛选条件"}],
                "outputs": [{"name": "records", "description": "与任务相关的只读业务记录"}],
                "preconditions": ["沙箱参考数据已初始化"],
                "effects": ["向 Agent 提供可验证的环境事实，不修改业务状态"],
            },
            *actions,
        ]

    @staticmethod
    def _normalize_data_access_capabilities(
        capabilities: list[Any], *, actions: list[Any], environment_mode: str
    ) -> list[Any]:
        if environment_mode not in {"reference_data", "stateful"}:
            return capabilities
        action_text = {
            str(action.get("name")): f"{action.get('name', '')} {action.get('description', '')}"
            for action in actions if isinstance(action, dict)
        }
        chinese_markers = ("读取", "查询", "检索", "查找", "获取")
        english_pattern = re.compile(r"\b(?:read|query|search|lookup|get)\b", re.I)
        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            text = action_text.get(str(item.get("action_name")), "")
            if any(marker in text for marker in chinese_markers) or english_pattern.search(text):
                item["kind"] = "environment_operation"
                item["requires_tool"] = True
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    item["reason"] = "该动作必须读取沙箱中的参考或业务数据。"
            normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_capability_dependencies(
        capabilities: list[Any], *, actions: list[Any]
    ) -> list[Any]:
        """Translate common ordinal dependency aliases to canonical action names.

        Dependency topology belongs to the action/capability contract, while
        weaker models often serialize an intended edge as ``step-1`` or
        ``action_1``.  Canonicalizing those identifiers preserves the edge;
        unknown values remain untouched so semantic validation still rejects
        invented dependencies.
        """
        action_names = [
            str(action["name"]) for action in actions
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        ]

        def key(value: str) -> str:
            return re.sub(r"[\s_-]+", "-", value.strip().lower())

        aliases: dict[str, str] = {}
        for index, action_name in enumerate(action_names, start=1):
            aliases[key(action_name)] = action_name
            for alias in (str(index), f"step-{index}", f"action-{index}"):
                aliases[key(alias)] = action_name
        for action in actions:
            if not isinstance(action, dict) or action.get("name") not in action_names:
                continue
            for field in ("id", "action_id", "step_id"):
                value = action.get(field)
                if isinstance(value, str) and value.strip():
                    aliases[key(value)] = str(action["name"])

        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            dependencies = item.get("dependencies", [])
            if isinstance(dependencies, list):
                canonical: list[Any] = []
                for dependency in dependencies:
                    value = aliases.get(key(dependency), dependency) if isinstance(dependency, str) else dependency
                    if value not in canonical:
                        canonical.append(value)
                item["dependencies"] = canonical
            normalized.append(item)
        return normalized

    @staticmethod
    def _apply_route_capability_contract(
        capabilities: list[Any], *, route_plan: dict[str, Any]
    ) -> list[Any]:
        """Project the validated route skeleton onto capability classification."""
        operations = {
            item["action_name"]: item
            for item in route_plan.get("environment_operations", [])
            if isinstance(item, dict) and isinstance(item.get("action_name"), str)
        }
        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            route_operation = operations.get(item.get("action_name"))
            if route_operation is not None:
                item["kind"] = "environment_operation"
                item["requires_tool"] = True
                item["dependencies"] = list(route_operation.get("dependencies", []))
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    item["reason"] = str(route_operation.get("purpose", "环境操作"))
            normalized.append(item)
        return normalized

    @staticmethod
    def _validate_capability_plan(
        capabilities: list[Any],
        actions: list[Any],
        *,
        environment_mode: str | None = None,
        has_business_data: bool = False,
        training_category: str | None = None,
    ) -> None:
        action_names = {
            action.get("name") for action in actions
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        }
        seen: set[str] = set()
        response_markers = (
            "生成markdown", "生成表格", "格式化", "输出最终", "生成最终",
            "生成推荐", "生成说明", "撰写", "总结", "generate markdown",
            "format output", "final response", "write summary",
        )
        for index, capability in enumerate(capabilities):
            if not isinstance(capability, dict):
                raise PipelineGenerationError(f"capabilities[{index}] must be an object")
            action_name = capability.get("action_name")
            kind = capability.get("kind")
            requires_tool = capability.get("requires_tool")
            if action_name not in action_names or action_name in seen:
                raise PipelineGenerationError(f"capabilities[{index}] has unknown or duplicate action_name")
            if kind not in {"environment_operation", "agent_reasoning", "agent_response"}:
                raise PipelineGenerationError(f"capabilities[{index}] has invalid kind")
            if not isinstance(requires_tool, bool):
                raise PipelineGenerationError(f"capabilities[{index}] requires boolean requires_tool")
            if requires_tool != (kind == "environment_operation"):
                raise PipelineGenerationError(f"capabilities[{index}] kind/requires_tool conflict")
            if requires_tool and any(marker in str(action_name).lower() for marker in response_markers):
                raise PipelineGenerationError(
                    f"capabilities[{index}] delegates response composition to a tool"
                )
            if not isinstance(capability.get("reason"), str) or not capability["reason"].strip():
                raise PipelineGenerationError(f"capabilities[{index}] requires reason")
            seen.add(action_name)
        if seen != action_names:
            missing = sorted(action_names - seen)
            raise PipelineGenerationError(f"capability plan must cover every action; missing={missing}")
        environment_actions = {item["action_name"] for item in capabilities if item["requires_tool"]}
        dependency_graph = {}
        for item in capabilities:
            dependencies = item.get("dependencies", [])
            if not isinstance(dependencies, list) or any(not isinstance(name, str) or name not in action_names for name in dependencies):
                raise PipelineGenerationError("capability dependencies must reference known action names")
            dependency_graph[item["action_name"]] = set(dependencies)
        remaining = set(dependency_graph)
        while remaining:
            roots = {name for name in remaining if not dependency_graph[name] & remaining}
            if not roots:
                raise PipelineGenerationError("capability dependency graph contains a cycle")
            remaining -= roots
        if training_category == "multi_step_agentic" and not any(
            dependency_graph[name] & environment_actions for name in environment_actions
        ):
            raise PipelineGenerationError("multi_step_agentic capability plan requires a real environment-action dependency; declare dependencies using action names, not reward step IDs")
        if environment_mode in {"reference_data", "stateful"} and has_business_data and not any(
            isinstance(item, dict) and item.get("requires_tool") is True
            for item in capabilities
        ):
            raise PipelineGenerationError(
                "capability plan for a data-backed environment must expose at least one data access operation"
            )

    @staticmethod
    def _validate_data_grounding_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("environment data grounding audit must be an object")
        fields = ("task_supported", "decision_determinate", "facts_consistent")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError(
                "environment data grounding audit requires boolean verdicts"
            )
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError(
                "environment data grounding audit requires a list of issue strings"
            )
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"environment data does not ground the task: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_task_grounding_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("task grounding audit must be an object")
        fields = (
            "self_contained", "no_unprovided_facts",
            "expected_result_derivable", "internally_consistent",
        )
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("task grounding audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("task grounding audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"task description is not grounded: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_task_description_consistency(description: Any) -> None:
        """Check structure only; task-specific semantics belong to grounding audits."""
        if not isinstance(description, dict):
            raise PipelineGenerationError("task description must be an object")
        if not isinstance(description.get("task"), str) or not description["task"].strip():
            raise PipelineGenerationError("task description requires non-empty task")
        if not isinstance(description.get("requirements", {}), dict):
            raise PipelineGenerationError("task requirements must be an object")

    @staticmethod
    def _validate_route_plan(route_plan: Any, training_category: str) -> None:
        if not isinstance(route_plan, dict):
            raise PipelineGenerationError("task route_plan must be an object")
        operations = route_plan.get("environment_operations")
        if not isinstance(operations, list):
            raise PipelineGenerationError(
                "task route_plan.environment_operations must be a list"
            )
        names: list[str] = []
        dependencies: dict[str, list[str]] = {}
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] must be an object"
                )
            name = operation.get("action_name")
            purpose = operation.get("purpose")
            declared = operation.get("dependencies")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] has invalid action_name"
                )
            if not isinstance(purpose, str) or not purpose.strip():
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] requires purpose"
                )
            if not isinstance(declared, list) or any(
                not isinstance(item, str) or item not in names for item in declared
            ):
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] dependencies must reference earlier actions"
                )
            names.append(name)
            dependencies[name] = list(dict.fromkeys(declared))
        if training_category == "direct_response" and operations:
            raise PipelineGenerationError(
                "direct_response route_plan must not contain environment operations"
            )
        if training_category == "simple_agentic" and len(operations) != 1:
            raise PipelineGenerationError(
                "simple_agentic route_plan requires exactly one environment operation"
            )
        if training_category == "multi_step_agentic" and (
            len(operations) < 2 or not any(dependencies.values())
        ):
            raise PipelineGenerationError(
                "multi_step_agentic route_plan requires dependent environment operations"
            )

    @staticmethod
    def _validate_route_input_boundary(
        description: dict[str, Any], training_category: str
    ) -> None:
        """Reject tool routes whose complete truth is already in public input."""
        if training_category == "direct_response":
            return
        public_input = description.get("public_input", {})
        corpus = " ".join(
            str(description.get(key, ""))
            for key in ("task", "goal", "context", "expected_result", "requirements")
        )
        if isinstance(public_input, dict):
            corpus += " " + str(public_input.get("initial_user_message", ""))
        complete_input_supplied = any(
            re.search(pattern, corpus) is not None
            for pattern in (
                r"(?:基于|仅依赖|仅使用|只依赖|只使用)用户提供的.{0,12}(?:规格|数据|资料|文本|列表|清单)",
                r"基于以下提供的.{0,8}(?:规格|数据|资料|文本)",
                r"从用户提供的.{0,8}(?:列表|清单|文本|数据)中",
            )
        )
        operations = description.get("route_plan", {}).get("environment_operations", [])
        route_text = json.dumps(operations, ensure_ascii=False).lower()
        private_markers = (
            "内部", "私有", "沙箱", "数据库", "业务数据", "库存", "目录", "历史记录",
            "系统记录", "internal", "private", "sandbox", "database", "inventory",
            "catalog", "system record",
        )
        has_private_route = any(marker in route_text for marker in private_markers)
        if complete_input_supplied and not has_private_route:
            raise PipelineGenerationError(
                f"{training_category} task is fully solvable from public user input; "
                "make at least one route operation depend on sandbox-private business data"
            )

    @staticmethod
    def _validate_no_deferred_business_truth(
        description: dict[str, Any], task_intent: str
    ) -> None:
        """Do not make a future simulator invent authoritative mutation inputs.

        Clarification dialogue is still useful for preferences and presentation
        details.  It cannot supply a value that determines the sandbox's target
        business state unless that value (or a deterministic derivation rule) is
        already part of the immutable task contract.
        """
        if task_intent not in {"modify", "execute", "schedule"}:
            return
        corpus = json.dumps({
            key: description.get(key)
            for key in ("task", "goal", "expected_result")
        } | {
            "initial_user_message": (
                description.get("public_input", {}).get("initial_user_message")
                if isinstance(description.get("public_input"), dict) else None
            )
        }, ensure_ascii=False).lower()
        unresolved_markers = (
            "暂时没想好", "先问我", "稍后提供", "之后提供", "待我提供",
            "需要向用户询问", "询问正确", "ask me", "provide later",
            "to be provided",
        )
        if any(marker in corpus for marker in unresolved_markers):
            raise PipelineGenerationError(
                "stateful task defers required business truth to an unspecified "
                "future user reply"
            )

    @staticmethod
    def _validate_route_action_coverage(
        actions: list[Any], *, route_plan: dict[str, Any]
    ) -> None:
        action_names = {
            action.get("name") for action in actions if isinstance(action, dict)
        }
        required = {
            operation.get("action_name")
            for operation in route_plan.get("environment_operations", [])
            if isinstance(operation, dict)
        }
        missing = sorted(required - action_names)
        if missing:
            raise PipelineGenerationError(
                f"agent actions do not implement route_plan operations: {missing}"
            )

    @staticmethod
    def _validate_action_alignment_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("agent action audit must be an object")
        fields = ("aligned", "complete", "atomic")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("agent action audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("agent action audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"agent actions are not task-aligned: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_action_grounding(
        actions: list[Any], *, task_description: dict[str, Any], keywords: list[str]
    ) -> None:
        """Reject action plans that import entities or numbers from another task."""
        task_text = json.dumps(task_description, ensure_ascii=False).lower()
        action_core = []
        for item in actions:
            if not isinstance(item, dict):
                continue
            action_core.append({
                "name": item.get("name"),
                "description": item.get("description"),
                "outputs": item.get("outputs", []),
                "effects": item.get("effects", []),
            })
        action_text = json.dumps(action_core, ensure_ascii=False).lower()
        relevant_keywords = [
            str(keyword).strip().lower() for keyword in keywords
            if len(str(keyword).strip()) >= 2 and str(keyword).strip().lower() in task_text
        ]
        if relevant_keywords and not any(keyword in action_text for keyword in relevant_keywords):
            raise PipelineGenerationError(
                "agent actions omit every task-specific keyword and may belong to another task"
            )

        action_names = [
            str(item.get("name", "")).strip() for item in actions if isinstance(item, dict)
        ]
        generic_count = sum(
            name in GENERIC_ACTION_NAMES
            or any(marker.lower() in name.lower() for marker in GENERIC_ACTION_MARKERS)
            for name in action_names
        )
        if action_names and generic_count * 2 >= len(action_names):
            raise PipelineGenerationError(
                "agent actions are predominantly generic placeholders instead of task-specific steps"
            )
        allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", task_text))
        # Numeric suffixes in stable action identifiers (for example
        # lookup_stage_1) are graph identity, not user-facing constraints.
        numeric_semantics = json.dumps([
            {
                "description": item.get("description"),
                "inputs": item.get("inputs", []),
                "outputs": item.get("outputs", []),
                "preconditions": item.get("preconditions", []),
                "effects": item.get("effects", []),
            }
            for item in actions if isinstance(item, dict)
        ], ensure_ascii=False).lower()
        action_numbers = set(re.findall(r"\d+(?:\.\d+)?", numeric_semantics))
        invented = sorted(action_numbers - allowed_numbers)
        if invented:
            raise PipelineGenerationError(
                f"agent actions invent numeric constraints absent from the task: {invented}"
            )

    @staticmethod
    def _normalize_action_numeric_examples(
        actions: list[Any], *, task_description: dict[str, Any],
        grounding_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Remove model-invented example numbers without changing action structure."""
        allowed = set(re.findall(
            r"\d+(?:\.\d+)?",
            json.dumps(
                {"task": task_description, "grounding": grounding_context or {}},
                ensure_ascii=False,
            ),
        ))

        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return re.sub(
                    r"\d+(?:\.\d+)?",
                    lambda match: match.group(0) if match.group(0) in allowed else "用户提供值",
                    value,
                )
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: (item if key == "name" else clean(item)) for key, item in value.items()}
            return value

        return [clean(action) for action in actions]

    @staticmethod
    def _task_relevant_keywords(
        task_description: dict[str, Any], keywords: list[str]
    ) -> list[str]:
        task_text = json.dumps(task_description, ensure_ascii=False).lower()
        return [
            str(keyword).strip().lower() for keyword in keywords
            if len(str(keyword).strip()) >= 2 and str(keyword).strip().lower() in task_text
        ]

    @staticmethod
    def _validate_data_keyword_alignment(
        data_tables: list[Any], *, task_description: dict[str, Any], keywords: list[str]
    ) -> None:
        """Keep generated business fixtures in the task's semantic domain."""
        relevant = TaskGenerationPipeline._task_relevant_keywords(
            task_description, keywords
        )
        if not relevant or not data_tables:
            return
        data_text = json.dumps(data_tables, ensure_ascii=False).lower()
        if not any(keyword in data_text for keyword in relevant):
            raise PipelineGenerationError(
                "environment data omits every task-specific keyword and may belong to another task"
            )

    @staticmethod
    def _validate_stateful_preconditions(
        data_tables: list[Any], *, task_description: dict[str, Any], environment_mode: str
    ) -> None:
        """Reject baselines that already contain an explicitly requested end state."""
        if environment_mode != "stateful":
            return
        task_text = json.dumps(task_description, ensure_ascii=False)
        changes = re.findall(
            r"从\s*[“\"]([^”\"]+)[”\"]\s*(?:更正|修改|更新|调整|改|变更)?\s*为\s*[“\"]([^”\"]+)[”\"]",
            task_text,
        )
        if not changes:
            return
        rows_text = json.dumps(data_tables, ensure_ascii=False)
        missing_old = [old for old, _new in changes if old not in rows_text]
        if missing_old:
            raise PipelineGenerationError(
                "stateful baseline omits explicit precondition value(s): "
                + ", ".join(sorted(set(missing_old)))
            )

    @staticmethod
    def _validate_training_contract_consistency(
        *,
        task_description: dict[str, Any],
        training_contract: dict[str, Any],
        keywords: list[str],
        environment_mode: str,
        data_tables: list[Any],
        actions: list[Any],
        tools: list[Any],
        tool_bindings: list[Any],
        metrics: list[Any],
        success_fixture: str,
    ) -> None:
        """Final cross-stage gate before a task is accepted for sandbox construction."""
        category = training_contract.get("category")
        allowed_modes = training_contract.get("allowed_environment_modes", [])
        if environment_mode not in allowed_modes:
            raise PipelineGenerationError(
                f"final consistency: {category} does not allow environment mode {environment_mode}"
            )
        TaskGenerationPipeline._validate_task_description_consistency(task_description)
        task_text = json.dumps(task_description, ensure_ascii=False).lower()
        action_text = json.dumps(actions, ensure_ascii=False).lower()
        if environment_mode == "stateless" and any(
            marker in task_text for marker in ("用户提供", "给定的", "provided")
        ) and any(
            marker in action_text for marker in ("生成候选列表", "创建候选列表", "invent candidates")
        ):
            raise PipelineGenerationError(
                "final consistency: stateless actions invent candidates instead of using user input"
            )
        TaskGenerationPipeline._validate_action_grounding(
            actions, task_description=task_description, keywords=keywords
        )
        if environment_mode in {"reference_data", "stateful"}:
            TaskGenerationPipeline._validate_data_keyword_alignment(
                data_tables, task_description=task_description, keywords=keywords
            )
        tool_names = {
            item.get("function", {}).get("name") for item in tools
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        action_names = {
            item.get("name") for item in actions if isinstance(item, dict)
        }
        for binding in tool_bindings:
            if not isinstance(binding, dict):
                raise PipelineGenerationError("final consistency: invalid tool binding")
            if binding.get("tool_name") not in tool_names or binding.get("action_name") not in action_names:
                raise PipelineGenerationError(
                    "final consistency: tool binding references an unknown tool or action"
                )
        bounds = training_contract.get("business_tools", {})
        count = len(tool_bindings)
        minimum, maximum = bounds.get("min", 0), bounds.get("max")
        if count < minimum or (maximum is not None and count > maximum):
            raise PipelineGenerationError(
                f"final consistency: {category} business tool count {count} violates [{minimum}, {maximum}]"
            )
        if not isinstance(metrics, list) or not metrics:
            raise PipelineGenerationError("final consistency: task has no reward metrics")
        TaskGenerationPipeline._validate_success_fixture_constraints(
            success_fixture, task_description=task_description
        )

    @staticmethod
    def _validate_task_generation_scope(
        task_description: dict[str, Any], *, graph_context: dict[str, Any]
    ) -> None:
        text = json.dumps(task_description, ensure_ascii=False).lower()
        chain_of_thought_markers = (
            "思考过程", "推理过程", "内心推理", "逐步思考",
            "chain of thought", "hidden reasoning", "show your reasoning",
        )
        if any(marker in text for marker in chain_of_thought_markers):
            raise PipelineGenerationError(
                "task must request concise rationale, not hidden chain-of-thought"
            )
        if TaskGenerationPipeline._is_high_stakes_task(task_description):
            evidence = json.dumps(graph_context, ensure_ascii=False)
            if not re.search(r"https?://[^\s\"']+", evidence, re.I):
                raise PipelineGenerationError(
                    "high-stakes task requires authoritative source URLs in graph_context"
                )

    @staticmethod
    def _is_high_stakes_task(task_description: dict[str, Any]) -> bool:
        """Classify the user-facing decision, not incidental generated metadata."""
        text = json.dumps({
            key: task_description.get(key)
            for key in ("task", "goal", "context", "requirements")
        }, ensure_ascii=False).lower()
        if any(marker in text for marker in HIGH_STAKES_MARKERS):
            return True
        return (
            any(marker in text for marker in HIGH_STAKES_CHEMICAL_MARKERS)
            and any(marker in text for marker in HIGH_STAKES_CHEMICAL_RISK_MARKERS)
        )

    @staticmethod
    def _normalize_reasoning_request(value: Any) -> Any:
        """Rewrite requests for hidden reasoning into an observable short rationale."""
        replacements = (
            ("逐步思考", "给出简要依据"), ("思考过程", "简要依据"),
            ("推理过程", "简要依据"), ("内心推理", "简要依据"),
            ("chain of thought", "concise rationale"),
            ("hidden reasoning", "concise rationale"),
            ("show your reasoning", "give a concise rationale"),
        )
        if isinstance(value, str):
            result = value
            for old, new in replacements:
                result = re.sub(re.escape(old), new, result, flags=re.I)
            return result
        if isinstance(value, list):
            return [TaskGenerationPipeline._normalize_reasoning_request(item) for item in value]
        if isinstance(value, dict):
            return {
                key: TaskGenerationPipeline._normalize_reasoning_request(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _normalize_unsourced_governance_task(value: Any) -> Any:
        """Turn unsourced governance wording into fixture-owned rule checks.

        Medical, financial, tax and hazardous-chemical decisions are not
        rewritten here; without authoritative sources they remain rejected.
        """
        replacements = (
            ("法律法规", "内部业务规则"), ("法律规定", "内部业务规则"),
            ("法规要求", "内部规则要求"), ("政策文件", "内部规则文档"),
            ("监管要求", "内部审核要求"), ("合规性", "内部规则一致性"),
            ("合规", "符合内部规则"),
            ("legal compliance", "internal rule consistency"),
            ("regulatory compliance", "internal rule consistency"),
        )
        if isinstance(value, str):
            result = value
            for old, new in replacements:
                result = re.sub(re.escape(old), new, result, flags=re.I)
            return result
        if isinstance(value, list):
            return [
                TaskGenerationPipeline._normalize_unsourced_governance_task(item)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: TaskGenerationPipeline._normalize_unsourced_governance_task(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _unwrap_task_description(value: dict[str, Any]) -> dict[str, Any]:
        """Accept common repair envelopes while keeping one canonical shape."""
        if isinstance(value.get("output"), dict) and isinstance(value["output"].get("task"), str):
            return value["output"]
        if isinstance(value.get("task_description"), dict) and isinstance(
            value["task_description"].get("task"), str
        ):
            return value["task_description"]
        return value



    @staticmethod
    def _unwrap_structured_output(
        value: Any, *, expected_fields: set[str]
    ) -> Any:
        """Unwrap common model response envelopes without weakening validation."""
        current = value
        for _ in range(3):
            if not isinstance(current, dict):
                break
            if expected_fields.issubset(current):
                return current
            nested = current.get("output")
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except (TypeError, ValueError, json.JSONDecodeError):
                    break
            if not isinstance(nested, dict):
                break
            current = nested
        return current

    @staticmethod
    def _resolve_environment_plan(
        candidate: Any, *, task_description: dict[str, Any], task_intent: str
    ) -> dict[str, Any]:
        modes = {"stateless", "reference_data", "stateful", "external_capability"}
        value = dict(candidate) if isinstance(candidate, dict) else {}
        mode = value.get("mode")
        fallback_reason: str | None = None
        aliases = {
            "none": "stateless", "no_data": "stateless",
            "no_business_data": "stateless", "text_only": "stateless",
            "pure_text": "stateless", "无状态": "stateless", "纯文本": "stateless",
            "reference": "reference_data", "read_only": "reference_data",
            "readonly": "reference_data", "read_only_data": "reference_data",
            "只读数据": "reference_data", "参考数据": "reference_data",
            "persistent": "stateful", "persistence": "stateful",
            "business_state": "stateful", "有状态": "stateful", "持久化": "stateful",
            "external": "external_capability", "external_tool": "external_capability",
            "external_api": "external_capability", "外部能力": "external_capability",
        }
        if isinstance(mode, str):
            normalized_mode = re.sub(r"[\s-]+", "_", mode.strip().lower())
            mode = aliases.get(normalized_mode, normalized_mode)
        if mode not in modes:
            mode = TaskGenerationPipeline._infer_environment_mode(
                task_description=task_description, task_intent=task_intent
            )
            fallback_reason = (
                "模型返回了未知环境模式，已根据任务中明确的外部能力、持久化和只读资料需求"
                "确定性选择最小必要环境。"
            )
            logger.warning(
                "environment_plan.mode is invalid; using deterministic fallback: raw=%r resolved=%s",
                value.get("mode"), mode,
            )
        inferred_mode = TaskGenerationPipeline._infer_environment_mode(
            task_description=task_description, task_intent=task_intent
        )
        if mode == "stateless" and inferred_mode in {"external_capability", "reference_data"}:
            mode = inferred_mode
            fallback_reason = (
                "任务需要可验证的外部或参考事实，已确定性升级为相应的数据环境。"
            )
            logger.warning(
                "environment_plan understated external dependency; upgraded to external_capability"
            )
        task_text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result", "context", "requirements")
        )
        persistence_markers = (
            "写入", "保存", "持久化", "更新", "修改", "删除", "创建记录", "提交订单",
            "数据库", "审批", "排程", "预订", "write", "save", "persist", "update",
            "delete", "create record", "database", "approve", "schedule", "book",
        )
        external_markers = (
            "天气", "实时", "当前市场", "目前市场", "最新", "联网", "搜索", "汇率",
            "预估总价", "估算价格", "查询价格",
            "weather", "current market", "latest", "search", "exchange rate",
        )
        explicitly_persistent = any(marker in task_text.lower() for marker in persistence_markers)
        explicitly_external = any(marker in task_text.lower() for marker in external_markers)
        runtime_candidates_supplied = any(
            marker in task_text.lower()
            for marker in (
                "提供的规格", "提供的候选", "给定的候选", "给定选项", "两个方案",
                "两款", "候选清单", "provided specifications", "given candidates",
                "provided options", "two options",
            )
        ) or re.search(
            r"(?:用户)?(?:明确)?提供的.{0,8}(?:候选|选项|列表|清单|规格|属性)",
            task_text,
        ) is not None
        stateless_intents = {"extract", "summarize", "classify", "transform", "explain"}
        common_knowledge_explanation = (
            task_intent == "explain"
            and any(
                marker in task_text.lower()
                for marker in (
                    "常见用途", "基本用途", "日常用途", "基本构造", "基本结构",
                    "一般原理", "基础原理", "常见特点", "common uses",
                    "basic structure", "basic principle",
                )
            )
            and not any(
                marker in task_text.lower()
                for marker in (
                    "历史", "来源", "出处", "资料", "记录", "数据", "文献",
                    "权威", "给定", "提供的", "history", "source", "reference",
                    "record", "data", "provided",
                )
            )
        )
        if (
            task_intent in stateless_intents
            and (inferred_mode == "stateless" or common_knowledge_explanation)
            and not explicitly_persistent
            and not explicitly_external
        ):
            mode = "stateless"
        if (
            task_intent in {"compare", "recommend", "decide"}
            and runtime_candidates_supplied
            and not explicitly_persistent
            and not explicitly_external
        ):
            mode = "stateless"
        requires_business_data = mode in {"reference_data", "stateful"}
        requires_persistence = mode == "stateful"
        reason = fallback_reason or value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "由任务明确目标和预期结果推导运行时环境模式。"
        return {
            "mode": mode,
            "requires_business_data": requires_business_data,
            "requires_persistence": requires_persistence,
            "reason": reason.strip(),
        }

    @staticmethod
    def _align_environment_plan_with_route(
        environment_plan: dict[str, Any], *, route_plan: dict[str, Any],
        task_intent: str, supported_modes: set[str],
    ) -> dict[str, Any]:
        """Make the validated tool route authoritative over a noisy mode label.

        A non-empty route plan represents operations that must cross the sandbox
        boundary.  Treating such a task as stateless contradicts that contract.
        The correction is deterministic and chooses the smallest environment
        capable of implementing the route; it does not invent an unavailable
        external service.
        """
        operations = route_plan.get("environment_operations", [])
        if not isinstance(operations, list) or not operations:
            raise PipelineGenerationError(
                "agentic route requires at least one environment operation"
            )
        if environment_plan.get("mode") != "stateless":
            return environment_plan

        mutating_intents = {"modify", "execute", "schedule"}
        required_mode = "stateful" if task_intent in mutating_intents else "reference_data"
        if required_mode not in supported_modes:
            raise PipelineGenerationError(
                "task buildability: route requires environment mode "
                f"{required_mode!r}, but supported={sorted(supported_modes)}"
            )
        logger.warning(
            "environment_plan conflicts with non-empty route; aligned mode from "
            "stateless to %s for intent=%s",
            required_mode, task_intent,
        )
        return {
            "mode": required_mode,
            "requires_business_data": True,
            "requires_persistence": required_mode == "stateful",
            "reason": (
                "已验证的训练路由包含必须跨越沙箱边界的业务操作；"
                "按任务意图确定性选择最小可实现环境。"
            ),
        }

    @staticmethod
    def _infer_environment_mode(
        *, task_description: dict[str, Any], task_intent: str
    ) -> str:
        """Conservatively infer a mode when the model emits an unknown label."""
        text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result", "context", "requirements")
        ).lower()
        external_markers = (
            "实时天气", "实时汇率", "实时价格", "联网搜索", "网络搜索", "外部 api",
            "当前市场", "目前市场", "最新", "最新列表", "最新名单", "公开查询", "公开可查", "官方网站", "官网查询",
            "所属省级行政区", "行政区归属", "地名归属",
            "预估总价", "估算价格", "查询价格",
            "现行法律", "当前开放", "开放时间", "公共交通", "权威来源",
            "weather forecast", "exchange rate", "web search", "external api",
            "latest list", "current list", "official website", "publicly available",
        )
        persistence_markers = (
            "写入数据库", "保存到", "持久化", "更新记录", "修改记录", "删除记录",
            "创建记录", "提交订单", "审批申请", "预订", "写入日历",
            "write to", "save to", "persist", "update record", "delete record",
            "create record", "submit order", "book ",
        )
        reference_markers = (
            "资料库", "知识库", "档案库", "商品目录", "内部文档", "业务记录中",
            "训练计划", "锻炼计划", "运动计划", "康复计划", "疼痛", "伤病",
            "reference data", "knowledge base", "catalog", "internal document",
            "training plan", "exercise plan", "rehabilitation plan", "pain", "injury",
        )
        if any(marker in text for marker in external_markers):
            return "external_capability"
        if any(marker in text for marker in persistence_markers):
            return "stateful"
        if any(marker in text for marker in reference_markers):
            return "reference_data"
        if task_intent in {"recommend", "compare", "explain", "troubleshoot", "decide"}:
            return "reference_data"
        # A schedule/create/plan intent describes what the Agent produces, not
        # whether the result must be persisted. Without an explicit state
        # boundary, the minimal environment is stateless.
        return "stateless"

    @staticmethod
    def _validate_success_fixture_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("success fixture audit must be an object")
        fields = ("latest_instructions_satisfied", "grounded", "goal_completed")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("success fixture audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("success fixture audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"success fixture is inconsistent: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_success_fixture_constraints(
        content: Any, *, task_description: dict[str, Any]
    ) -> None:
        """Enforce simple objective constraints without trusting an LLM judge."""
        if not isinstance(content, str) or not content.strip():
            raise PipelineGenerationError("success fixture content must be non-empty")
        source = json.dumps(task_description, ensure_ascii=False)
        compact = re.sub(r"\s+", "", content)
        ranges = re.findall(r"(\d+)\s*[-–—~～至到]\s*(\d+)\s*(?:个)?字", source)
        for lower_text, upper_text in ranges:
            lower, upper = int(lower_text), int(upper_text)
            if lower <= upper and not lower <= len(compact) <= upper:
                raise PipelineGenerationError(
                    f"success fixture length {len(compact)} violates required {lower}-{upper} characters"
                )

    @staticmethod
    def _normalize_success_fixture_length(
        content: str, *, task_description: dict[str, Any]
    ) -> str:
        """Deterministically trim an overlong fixture to an explicit char limit."""
        if not isinstance(content, str):
            return content
        source = json.dumps(task_description, ensure_ascii=False)
        ranges = re.findall(r"(\d+)\s*[-–—~～至到]\s*(\d+)\s*(?:个)?字", source)
        upper_bounds = [int(upper) for lower, upper in ranges if int(lower) <= int(upper)]
        if not upper_bounds:
            return content.strip()
        upper = min(upper_bounds)
        if len(re.sub(r"\s+", "", content)) <= upper:
            return content.strip()
        kept: list[str] = []
        count = 0
        for char in content.strip():
            if not char.isspace():
                if count >= upper:
                    break
                count += 1
            kept.append(char)
        normalized = "".join(kept).rstrip()
        logger.info(
            "trimmed success fixture to explicit character limit: upper=%d", upper
        )
        return normalized

    @staticmethod
    def _validate_metric_constraint_grounding(
        metrics: list[Any], *, task_description: dict[str, Any]
    ) -> None:
        """Reject thresholds invented by reward generation.

        We inspect only human-facing metric semantics, deliberately excluding
        weights, score ranges and score mappings which contain evaluator
        mechanics rather than task requirements.
        """
        task_text = json.dumps(task_description, ensure_ascii=False)
        allowed = {
            Decimal(number)
            for number in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", task_text)
        }
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            evaluator = metric.get("evaluator", {})
            semantic_parts: list[str] = []
            for key in ("rubric", "condition"):
                if isinstance(metric.get(key), str):
                    semantic_parts.append(metric[key])
            criteria = metric.get("criteria")
            if isinstance(criteria, list):
                semantic_parts.extend(item for item in criteria if isinstance(item, str))
            if isinstance(evaluator, dict):
                for key in ("assertion", "criteria"):
                    value = evaluator.get(key)
                    if isinstance(value, str):
                        semantic_parts.append(value)
                    elif isinstance(value, list):
                        semantic_parts.extend(item for item in value if isinstance(item, str))
            mentioned = set(re.findall(
                r"(?<![\w.])\d+(?:\.\d+)?", " ".join(semantic_parts)
            ))
            # Zero and one frequently express boolean/control-state assertions
            # (for example count == 0 or completed == 1), not user-facing
            # quantitative requirements. Treat them as evaluator mechanics.
            invented = sorted(
                number for number in mentioned
                if Decimal(number) not in allowed and Decimal(number) not in {0, 1}
            )
            if invented:
                raise PipelineGenerationError(
                    f"metrics[{index}] invents numeric constraints absent from the task: {invented}"
                )

    @classmethod
    def _drop_ungrounded_numeric_metrics(
        cls, metrics: list[Any], *, task_description: dict[str, Any]
    ) -> list[Any]:
        """Discard isolated reward thresholds invented by the model."""
        retained: list[Any] = []
        for metric in metrics:
            try:
                cls._validate_metric_constraint_grounding(
                    [metric], task_description=task_description
                )
            except PipelineGenerationError as exc:
                logger.warning(
                    "discarding metric with ungrounded numeric constraint: metric=%s error=%s",
                    metric.get("id") if isinstance(metric, dict) else None, exc,
                )
                continue
            retained.append(metric)
        return retained

    @staticmethod
    def _derive_complexity(
        *, business_tool_count: int, metric_count: int, key_step_count: int
    ) -> str:
        if business_tool_count <= 1 and metric_count <= 4 and key_step_count <= 1:
            return "simple"
        if business_tool_count <= 4 and metric_count <= 8 and key_step_count <= 4:
            return "standard"
        return "complex"

    @staticmethod
    def _derive_training_profile(
        *, business_tool_count: int, noise_tool_count: int, key_step_count: int
    ) -> str:
        if business_tool_count == 0:
            return "tool_abstention" if noise_tool_count else "language_only"
        if key_step_count > 1 or business_tool_count > 1:
            return "multi_step_tool_use"
        return "single_step_tool_use"

    @staticmethod
    def _requires_file_deliverable(task_description: dict[str, Any]) -> bool:
        """Detect tasks whose goal is an actual file, not merely textual content."""
        requirements = task_description.get("requirements")
        requirements = requirements if isinstance(requirements, dict) else {}
        output_format = str(requirements.get("output_format", "")).lower()
        text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result")
        ).lower()
        text = f"{text} {output_format}"
        file_markers = (
            "pdf", "png", "jpg", "jpeg", "svg", "docx", "xlsx", "pptx",
            "音频文件", "视频文件", "图片文件", "可打印文件", "生成文件",
        )
        return any(marker in text for marker in file_markers)

    @staticmethod
    def _validate_media_generation(
        media_generation: dict[str, Any],
        *,
        input_media_required: bool,
        deliverable_required: bool,
        task_description: dict[str, Any],
    ) -> None:
        code = str(media_generation.get("code", ""))
        lower = code.lower()
        if deliverable_required and not media_generation.get("output_dir"):
            raise PipelineGenerationError(
                "file deliverable requires a non-empty media_generation.output_dir"
            )
        if deliverable_required:
            requirements = task_description.get("requirements")
            requirements = requirements if isinstance(requirements, dict) else {}
            expected = str(requirements.get("output_format", "")).lower()
            extensions = {
                "pdf": ".pdf", "png": ".png", "jpg": ".jpg", "jpeg": ".jpeg",
                "svg": ".svg", "docx": ".docx", "xlsx": ".xlsx", "pptx": ".pptx",
            }
            required_extensions = {
                extension for label, extension in extensions.items() if label in expected
            }
            if required_extensions and not any(extension in lower for extension in required_extensions):
                raise PipelineGenerationError(
                    "media_generation code does not create the declared file format"
                )
        if input_media_required:
            fabricated_markers = (
                "模拟数据", "生成一个模拟", "mock data", "dummy data",
                "placeholder data", "synthetic user", "假设最大", "示例数据：",
            )
            if any(marker in lower for marker in fabricated_markers):
                raise PipelineGenerationError(
                    "media_generation fabricates runtime user input; read inputs from arguments instead"
                )
            input_markers = (
                "argparse", "sys.argv", "input_path", "input_file", "source_path",
                "image_path", "file_path", "json.load", "csv.", "read_csv",
                "image.open", "soundfile.read", "wave.open",
            )
            if not any(marker in lower for marker in input_markers):
                raise PipelineGenerationError(
                    "media_generation for user-provided media must read an explicit runtime input"
                )

    @staticmethod
    def _validate_authoritative_source_schema(
        *, task_description: dict[str, Any], table_definitions: list[Any]
    ) -> None:
        """Require traceable provenance for generated high-stakes reference facts."""
        if not TaskGenerationPipeline._is_high_stakes_task(task_description):
            return
        column_names = {
            str(column.get("name", "")).lower()
            for table in table_definitions if isinstance(table, dict)
            for column in table.get("columns", []) if isinstance(column, dict)
        }
        aliases = {
            "source_url": {"source_url", "official_url", "url"},
            "retrieved_at": {"retrieved_at", "accessed_at", "fetched_at"},
            "content_hash": {"content_hash", "source_hash", "sha256"},
        }
        missing = [
            canonical for canonical, names in aliases.items()
            if not column_names.intersection(names)
        ]
        if missing:
            raise PipelineGenerationError(
                "authoritative reference data requires provenance columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _normalize_public_input(task_description: dict[str, Any]) -> dict[str, Any]:
        """Compile the public episode payload without copying hidden truth."""
        task = str(task_description.get("task") or "").strip()
        candidate = task_description.get("public_input")
        candidate = candidate if isinstance(candidate, dict) else {}
        message = candidate.get("initial_user_message")
        if not isinstance(message, str) or not message.strip():
            message = task
        materials: list[dict[str, str]] = []
        raw_materials = candidate.get("materials")
        if isinstance(raw_materials, list):
            for index, item in enumerate(raw_materials, start=1):
                if isinstance(item, str):
                    item = {"name": f"material-{index}", "mime_type": "text/plain", "content": item}
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                name = item.get("name")
                mime_type = item.get("mime_type")
                materials.append({
                    "name": name.strip() if isinstance(name, str) and name.strip() else f"material-{index}",
                    "mime_type": mime_type.strip() if isinstance(mime_type, str) and mime_type.strip() else "text/plain",
                    "content": content.strip(),
                })
        return {"initial_user_message": message.strip(), "materials": materials}

    @staticmethod
    def _validate_public_input(
        *, task_description: dict[str, Any], public_input: dict[str, Any], environment_mode: str,
    ) -> None:
        message = public_input.get("initial_user_message")
        if not isinstance(message, str) or not message.strip():
            raise PipelineGenerationError("public_input.initial_user_message must be non-empty")
        materials = public_input.get("materials")
        if not isinstance(materials, list):
            raise PipelineGenerationError("public_input.materials must be a list")
        text = json.dumps({
            "task": task_description.get("task"),
            "context": task_description.get("context"),
            "requirements": task_description.get("requirements"),
        }, ensure_ascii=False).lower()
        references_runtime_material = re.search(
            r"(?:以下|下列|上述|这段|这些|给定|提供|附上|附件).{0,12}"
            r"(?:文本|资料|数据|列表|清单|内容|说明|笔记|记录|规格|描述|选项)",
            text,
        ) is not None
        concrete = [
            item for item in materials
            if isinstance(item, dict)
            and isinstance(item.get("content"), str)
            and len(item["content"].strip()) >= 8
            and not re.fullmatch(r"(?:用户)?(?:已)?提供(?:的)?(?:文本|资料|数据|内容|说明|笔记|记录|规格|描述)", item["content"].strip())
        ]
        if references_runtime_material and not concrete:
            raise PipelineGenerationError(
                "public_input is missing the concrete material referenced by the task"
            )
        if environment_mode == "stateless" and not message.strip():
            raise PipelineGenerationError("stateless task requires a complete public user message")

    @staticmethod
    def _validate_authoritative_task_input(
        *,
        task_description: dict[str, Any],
        environment_mode: str,
        graph_context: dict[str, Any],
    ) -> None:
        """Do not let an LLM invent the source corpus for high-stakes lookup tasks."""
        if environment_mode not in {"reference_data", "stateful"}:
            return
        if not TaskGenerationPipeline._is_high_stakes_task(task_description):
            return
        evidence = json.dumps(graph_context, ensure_ascii=False)
        if not re.search(r"https?://[^\s\"']+", evidence, re.I):
            raise PipelineGenerationError(
                "authoritative reference task requires source URLs in graph_context; "
                "LLM-generated authority data is not allowed"
            )

    @staticmethod
    def _validate_task_readiness(
        *,
        capability_plan: list[Any],
        tool_bindings: list[Any],
        noise_tools: list[Any],
        metrics: list[Any],
        metric_implementations: list[Any],
        business_scenarios: list[Any],
        environment_mode: str = "stateless",
        has_business_data: bool = False,
        task_description: dict[str, Any] | None = None,
        data_tables: list[Any] | None = None,
        media_generation: dict[str, Any] | None = None,
        success_fixture: str = "",
        require_noise: bool = True,
    ) -> None:
        eligible = {
            item.get("action_name") for item in capability_plan
            if isinstance(item, dict) and item.get("requires_tool") is True
        }
        bound = {
            item.get("action_name") for item in tool_bindings if isinstance(item, dict)
        }
        if not bound <= eligible:
            raise PipelineGenerationError("task readiness: a tool delegates agent reasoning or response composition")
        if environment_mode == "external_capability" and not bound:
            raise PipelineGenerationError(
                "task readiness: external-capability task exposes no business tool"
            )
        if environment_mode in {"reference_data", "stateful"} and has_business_data and not bound:
            raise PipelineGenerationError(
                "task readiness: data-backed task exposes no business data access tool"
            )
        if environment_mode == "reference_data":
            leaked = [
                metric.get("id") for metric in metrics
                if isinstance(metric, dict)
                and metric.get("category") == "outcome"
                and isinstance(metric.get("evaluator"), dict)
                and metric["evaluator"].get("kind") == "business_state_rule"
            ]
            if leaked:
                raise PipelineGenerationError(
                    "task readiness: read-only reference data cannot by itself prove agent outcome; "
                    f"metrics={leaked}"
                )
        if require_noise and not noise_tools:
            raise PipelineGenerationError("task readiness: tool-selection tasks require at least one noise tool")
        rule_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("type") == "rule-based"
        }
        implemented = {
            item.get("metric_id") for item in metric_implementations if isinstance(item, dict)
        }
        if not rule_ids <= implemented:
            raise PipelineGenerationError("task readiness: rule-based metrics lack executable implementations")
        kinds = {
            item.get("kind") for item in business_scenarios if isinstance(item, dict)
        }
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("task readiness: success and failure trajectories are required")
        if noise_tools and "noise_selection" not in kinds:
            raise PipelineGenerationError("task readiness: a noise-selection trajectory is required")
        TaskGenerationPipeline._validate_metric_runtime_semantics(
            metrics=metrics,
            metric_implementations=metric_implementations,
            noise_tools=noise_tools,
        )
        description = task_description or {}
        if TaskGenerationPipeline._requires_file_deliverable(description):
            media = media_generation or {}
            if media.get("required") is not True or not str(media.get("code", "")).strip():
                raise PipelineGenerationError(
                    "task readiness: file deliverable has no executable artifact generation"
                )
            file_metric_text = json.dumps(metrics, ensure_ascii=False).lower()
            if any(marker in file_metric_text for marker in ("file_generation", "pdf_file", "生成文件", "pdf文件")):
                model_only = [
                    metric.get("id") for metric in metrics
                    if isinstance(metric, dict)
                    and any(marker in json.dumps(metric, ensure_ascii=False).lower()
                            for marker in ("file_generation", "pdf_file", "生成文件", "pdf文件"))
                    and isinstance(metric.get("evaluator"), dict)
                    and metric["evaluator"].get("kind") == "external_llm_judge"
                ]
                if model_only:
                    raise PipelineGenerationError(
                        "task readiness: file existence cannot be evaluated only by an LLM judge; "
                        f"metrics={model_only}"
                    )
            if re.search(r"(?:已|已经).{0,12}(?:生成|创建|提供).{0,30}\.(?:pdf|png|jpg|docx|xlsx|pptx)", success_fixture, re.I):
                # A claimed artifact is acceptable only because executable media
                # generation was required and validated above. Keep this branch
                # explicit so future artifact-manifest support can tighten it.
                pass

    @staticmethod
    def _normalize_metric_weights(metrics: list[Any]) -> list[Any]:
        """Normalize positive and negative metric weights independently.

        LLMs often emit proportional weights such as 0.4/0.5/1.0 instead of
        already-normalized weights. Since the ratio is the meaningful part,
        normalize each sign group before strict contract validation.
        """
        positive = 0.0
        negative = 0.0
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("weight"), (int, float)):
                continue
            category = metric.get("category")
            if category in {"process", "outcome"} and metric["weight"] >= 0:
                positive += float(metric["weight"])
            elif category == "penalty" and metric["weight"] >= 0:
                negative += float(metric["weight"])
        if positive > 0 and not math.isclose(positive, 1.0, abs_tol=1e-6):
            logger.info("normalized positive metric weights: original_sum=%s", positive)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") in {"process", "outcome"} and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / positive
        if negative > 0 and not math.isclose(negative, 1.0, abs_tol=1e-6):
            logger.info("normalized penalty metric weights: original_sum=%s", negative)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") == "penalty" and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / negative
        return metrics

    @staticmethod
    def _normalize_metric_types(metrics: list[Any]) -> list[Any]:
        """Derive the canonical metric type from an unambiguous evaluator."""
        evaluator_to_type = {
            "business_state_rule": "rule-based",
            "document_rule": "rule-based",
            "trajectory_rule": "rule-based",
            "external_llm_judge": "model-based",
            "hybrid_tool_call": "hybrid",
            "hybrid_outcome": "hybrid",
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") in {"rule-based", "model-based", "hybrid"}:
                continue
            evaluator = metric.get("evaluator")
            kind = evaluator.get("kind") if isinstance(evaluator, dict) else None
            canonical = evaluator_to_type.get(kind)
            if canonical:
                logger.info(
                    "normalized metric type from evaluator: metric=%s original=%r normalized=%s",
                    metric.get("id"), metric.get("type"), canonical,
                )
                metric["type"] = canonical
        return metrics

    @staticmethod
    def _normalize_metric_evaluators(metrics: list[Any]) -> list[Any]:
        """Canonicalize duplicated rule text emitted at either contract level."""
        rule_kinds = {"business_state_rule", "document_rule", "trajectory_rule"}
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                continue
            condition = metric.get("condition")
            assertion = evaluator.get("assertion")
            if evaluator.get("kind") in rule_kinds:
                if not isinstance(assertion, str) or not assertion.strip():
                    if isinstance(condition, str) and condition.strip():
                        evaluator["assertion"] = condition.strip()
                        logger.info(
                            "normalized rule assertion from condition: metric=%s",
                            metric.get("id"),
                        )
                elif not isinstance(condition, str) or not condition.strip():
                    metric["condition"] = assertion.strip()
            elif evaluator.get("kind") == "hybrid_outcome":
                rule = evaluator.get("rule")
                if isinstance(rule, dict):
                    rule_assertion = rule.get("assertion")
                    if (not isinstance(rule_assertion, str) or not rule_assertion.strip()) and isinstance(condition, str) and condition.strip():
                        rule["assertion"] = condition.strip()
                # A prose assertion is not a deterministic program. Avoid a
                # hybrid declaration that the runtime cannot honestly execute
                # as two independent branches; retain all criteria in one
                # explicit semantic judge instead.
                criteria: list[str] = []
                for value in metric.get("criteria", []):
                    if isinstance(value, str) and value.strip() and value.strip() not in criteria:
                        criteria.append(value.strip())
                if isinstance(rule, dict) and isinstance(rule.get("assertion"), str):
                    value = rule["assertion"].strip()
                    if value and value not in criteria:
                        criteria.append(value)
                semantic = evaluator.get("external_llm")
                if isinstance(semantic, dict):
                    values = semantic.get("criteria", [])
                    if isinstance(values, str):
                        values = [values]
                    judge = semantic.get("judge_criteria")
                    if isinstance(judge, str):
                        values = [*values, judge]
                    for value in values:
                        if isinstance(value, str) and value.strip() and value.strip() not in criteria:
                            criteria.append(value.strip())
                metric["type"] = "model-based"
                metric.pop("condition", None)
                metric["criteria"] = criteria or [str(metric.get("rubric") or "任务结果满足目标")]
                metric["evaluator"] = {
                    "kind": "external_llm_judge",
                    "source": "external_llm",
                    "criteria": list(metric["criteria"]),
                    "score_mapping": copy.deepcopy(evaluator.get("score_mapping", {"pass": 1, "fail": 0})),
                }
                logger.info(
                    "normalized non-executable hybrid outcome to semantic judge: metric=%s",
                    metric.get("id"),
                )
        return metrics

    @staticmethod
    def _normalize_metrics_for_environment(
        metrics: list[Any], *, environment_mode: str
    ) -> list[Any]:
        """Prevent read-only source data from being treated as an Agent outcome."""
        if environment_mode != "reference_data":
            return metrics
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "outcome":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "business_state_rule":
                continue
            criterion = (
                metric.get("rubric") or evaluator.get("assertion")
                or "最终回答是否基于参考数据完成用户目标"
            )
            metric["type"] = "model-based"
            metric["scope"] = "terminal"
            metric.pop("condition", None)
            metric["evaluation_inputs"] = ["final_agent_response", "business_data"]
            metric["criteria"] = [str(criterion)]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "criteria": [str(criterion)],
                "score_mapping": {"pass": 1, "fail": 0},
            }
            logger.info(
                "promoted reference-data outcome to response judge: metric=%s",
                metric.get("id"),
            )
        return metrics

    @staticmethod
    def _ensure_deterministic_noise_penalty(
        metrics: list[Any], *, noise_names: list[str]
    ) -> None:
        retained: list[Any] = []
        for metric in metrics:
            if not isinstance(metric, dict):
                retained.append(metric)
                continue
            text = json.dumps(metric, ensure_ascii=False, sort_keys=True).lower()
            if metric.get("category") == "penalty" and (
                "noise" in text
                or "噪声" in text
                or "tool_call" in text
                or "tool use" in text
                or "工具调用" in text
                or "工具使用" in text
                or any(name.lower() in text for name in noise_names)
            ):
                continue
            retained.append(metric)
        retained.append({
            "id": "penalty_noise_tool_usage",
            "category": "penalty",
            "type": "rule-based",
            "scope": "trajectory",
            "condition": "trajectory contains no calls to declared noise tools",
            "evaluator": {
                "kind": "trajectory_rule",
                "source": "runtime_rule",
                "assertion": "no declared noise tool appears in trajectory events",
                "score_mapping": {"pass": 0, "fail": -1},
            },
            "score_range": [-1, 0],
            "weight": 1,
            "rubric": "调用任一声明的噪声工具时施加确定性惩罚。",
        })
        metrics[:] = retained

    @staticmethod
    def _promote_unstructured_document_rules(metrics: list[Any]) -> None:
        """Use a semantic judge for facts that only exist inside document text."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "document_rule":
                continue
            criterion = evaluator.get("assertion") or metric.get("rubric") or metric.get("condition")
            mapping = evaluator.get("score_mapping", {"pass": 1, "fail": 0})
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = ["conversation", "final_agent_response"]
            metric["criteria"] = [str(criterion or "判断最终文档是否满足该指标。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(mapping) if isinstance(mapping, dict) else {"pass": 1, "fail": 0},
            }
            metric.pop("condition", None)
            logger.info(
                "promoted document rule over raw final_agent_response to external judge: metric=%s",
                metric.get("id"),
            )

    @staticmethod
    def _promote_mixed_response_business_rules(metrics: list[Any]) -> None:
        """A textual answer cannot be joined to business tables by the path DSL."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "business_state_rule":
                continue
            source_fields = evaluator.get("source_fields", [])
            assertion = str(evaluator.get("assertion", ""))
            mixes_response = (
                isinstance(source_fields, list) and "final_agent_response" in source_fields
            ) or "final_agent_response" in assertion
            if not mixes_response:
                continue
            criterion = assertion or metric.get("rubric") or metric.get("condition")
            mapping = evaluator.get("score_mapping", {"pass": 1, "fail": 0})
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = ["business_data", "final_agent_response"]
            metric["criteria"] = [str(criterion or "判断最终回答是否与业务数据一致。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(mapping) if isinstance(mapping, dict) else {"pass": 1, "fail": 0},
            }
            metric.pop("condition", None)
            logger.info(
                "promoted mixed response/business rule to external judge: metric=%s",
                metric.get("id"),
            )

    @staticmethod
    def _canonical_observation_schema(candidate: Any) -> dict[str, Any]:
        """Return one stable observation shape for every generated task."""
        descriptions = {
            "episode_id": "当前 episode 标识。",
            "conversation": "当前 episode 的完整对话消息。",
            "public_observation": "沙箱允许 Agent 或 Trainer 观察的公开环境信息。",
            "available_tools": "当前提供给 Agent 的 OpenAI Function Tool 定义。",
            "tool_call": "最近一次工具调用；未调用时为空。",
            "tool_results": "当前 episode 的工具调用结果列表。",
            "final_agent_response": "Agent 提交的最终自然语言回答。",
        }
        properties: dict[str, Any] = {}
        if isinstance(candidate, dict) and isinstance(candidate.get("properties"), dict):
            properties.update(candidate["properties"])
        field_types = {
            "episode_id": "string", "conversation": "array", "public_observation": "object",
            "available_tools": "array", "tool_call": "object",
            "tool_results": "array", "final_agent_response": "string",
        }
        for name, description in descriptions.items():
            # These fields are emitted by the shared runtime protocol.  Model
            # proposals may add task-specific observation fields, but must not
            # redefine platform-owned types (for example tool_results as an
            # object when the runtime always emits an array).
            properties[name] = {
                "type": field_types[name],
                "description": description,
            }
        return {
            "type": "object",
            "properties": properties,
            "required": [
                "episode_id", "conversation", "available_tools", "tool_results",
                "public_observation", "final_agent_response",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _validate_metric_runtime_semantics(
        *, metrics: list[Any], metric_implementations: list[Any], noise_tools: list[Any]
    ) -> None:
        """Reject contracts whose valid-looking DSL would produce reversed rewards."""
        metric_by_id = {
            item.get("id"): item for item in metrics
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        noise_names = {
            item.get("name") for item in noise_tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        noise_penalties = 0
        for index, spec in enumerate(metric_implementations):
            if not isinstance(spec, dict):
                continue
            metric = metric_by_id.get(spec.get("metric_id"), {})
            mapping = spec.get("score_mapping", {})
            if metric.get("category") == "penalty":
                if mapping.get("pass") != 0 or not isinstance(mapping.get("fail"), (int, float)) or mapping.get("fail") >= 0:
                    raise PipelineGenerationError(
                        f"task readiness: penalty metric {spec.get('metric_id')} has reversed score mapping"
                    )
            elif isinstance(mapping, dict) and mapping.get("pass", 0) < mapping.get("fail", 0):
                raise PipelineGenerationError(
                    f"task readiness: positive metric {spec.get('metric_id')} has reversed score mapping"
                )
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"task readiness: metric_implementations[{index}] addresses fields on raw final_agent_response"
                )
            if spec.get("source") == "business_state" and "final_agent_response" in str(spec.get("path")):
                raise PipelineGenerationError(
                    f"task readiness: metric_implementations[{index}] mixes response data into business_state"
                )
            expected = spec.get("expected")
            targets_noise = (
                spec.get("operator") == "none_tool_calls"
                or (isinstance(expected, str) and expected in noise_names)
                or (isinstance(expected, list) and bool(set(expected) & noise_names))
            )
            if metric.get("category") == "penalty" and targets_noise:
                noise_penalties += 1
        if noise_names and noise_penalties != 1:
            raise PipelineGenerationError(
                f"task readiness: expected exactly one executable noise penalty, found {noise_penalties}"
            )

    @staticmethod
    def _validate_key_steps(key_steps: list[Any], actions: list[Any]) -> None:
        action_names = {
            str(action.get("name") or action.get("action"))
            for action in actions
            if isinstance(action, dict)
        }
        step_ids: set[str] = set()
        declared_dependencies: dict[str, list[str]] = {}
        for index, step in enumerate(key_steps):
            if not isinstance(step, dict):
                raise PipelineGenerationError(f"key_steps[{index}] must be an object")
            step_id = step.get("step_id")
            action_name = step.get("action_name")
            if not isinstance(step_id, str) or not step_id.strip() or step_id in step_ids:
                raise PipelineGenerationError(f"key_steps[{index}] has invalid or duplicate step_id")
            if not isinstance(action_name, str) or action_name not in action_names:
                raise PipelineGenerationError(f"key_steps[{index}] references an unknown action")
            if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
                raise PipelineGenerationError(f"key_steps[{index}] requires rationale")
            if not isinstance(step.get("required_for_goal"), bool):
                raise PipelineGenerationError(f"key_steps[{index}] requires boolean required_for_goal")
            dependencies = step.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) or not item.strip() for item in dependencies
            ):
                raise PipelineGenerationError(f"key_steps[{index}].dependencies must be a list")
            unknown = [item for item in dependencies if item not in step_ids]
            if unknown:
                raise PipelineGenerationError(
                    f"key_steps[{index}] dependencies must reference earlier steps: {unknown}"
                )
            declared_dependencies[step_id] = list(dependencies)
            step_ids.add(step_id)

    @staticmethod
    def _validate_metrics(metrics: list[Any], *, key_steps: list[Any] | None = None) -> None:
        ids: set[str] = set()
        category_weights = {"process": 0.0, "outcome": 0.0, "penalty": 0.0}
        key_action_names = {
            str(step.get("action_name"))
            for step in (key_steps or [])
            if isinstance(step, dict) and isinstance(step.get("action_name"), str)
        }
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise PipelineGenerationError(f"metrics[{index}] must be an object")
            metric_id = metric.get("id")
            if not isinstance(metric_id, str) or not metric_id.strip() or metric_id in ids:
                raise PipelineGenerationError(f"metrics[{index}] has invalid or duplicate id")
            category = metric.get("category")
            if category not in category_weights:
                raise PipelineGenerationError(
                    f"metrics[{index}] has invalid category: {category!r}; "
                    "expected one of 'process', 'outcome', 'penalty'"
                )
            metric_type = metric.get("type")
            if metric_type not in {"rule-based", "model-based", "hybrid"}:
                raise PipelineGenerationError(f"metrics[{index}] has invalid type")
            if metric.get("scope") not in {"step", "state", "terminal", "trajectory"}:
                raise PipelineGenerationError(f"metrics[{index}] has invalid scope")
            if not isinstance(metric.get("rubric"), str) or not metric["rubric"].strip():
                raise PipelineGenerationError(f"metrics[{index}] requires rubric")
            weight = metric.get("weight")
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise PipelineGenerationError(f"metrics[{index}] weight must be a finite non-negative number")
            score_range = metric.get("score_range")
            expected_range = [-1, 0] if category == "penalty" else [0, 1]
            if score_range != expected_range:
                raise PipelineGenerationError(f"metrics[{index}] score_range must be {expected_range}")
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                raise PipelineGenerationError(f"metrics[{index}] requires executable evaluator")
            if not isinstance(evaluator.get("kind"), str) or not evaluator["kind"].strip():
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires kind")
            if evaluator.get("source") not in {"runtime_rule", "external_llm"}:
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.source must be runtime_rule or external_llm"
                )
            score_mapping = evaluator.get("score_mapping")
            if not isinstance(score_mapping, dict) or not score_mapping:
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires score_mapping")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < expected_range[0]
                or float(value) > expected_range[1]
                for value in score_mapping.values()
            ):
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.score_mapping values must be within {expected_range}"
                )
            if category == "process":
                if (
                    evaluator.get("kind") != "hybrid_tool_call"
                    or evaluator.get("source") != "external_llm"
                    or evaluator.get("comparison") != "exact_tool_name_and_canonical_arguments"
                ):
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator must use external exact tool-call comparison"
                    )
                if set(evaluator.get("score_mapping", {})) != {"match", "mismatch"}:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator score_mapping must contain match and mismatch"
                    )
            elif metric_type == "rule-based":
                if evaluator.get("kind") not in {
                    "business_state_rule", "document_rule", "trajectory_rule"
                } or evaluator.get("source") != "runtime_rule":
                    raise PipelineGenerationError(
                        f"metrics[{index}] rule-based evaluator must be an executable runtime rule"
                    )
                if not isinstance(evaluator.get("assertion"), str) or not evaluator["assertion"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] rule-based evaluator requires assertion")
            elif metric_type == "model-based":
                if evaluator.get("kind") != "external_llm_judge" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] model-based evaluator must use external_llm_judge"
                    )
            elif metric_type == "hybrid":
                if evaluator.get("kind") != "hybrid_outcome" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid non-process evaluator must use hybrid_outcome"
                    )
                if not isinstance(evaluator.get("rule"), dict) or not isinstance(evaluator.get("external_llm"), dict):
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid evaluator requires rule and external_llm sections"
                    )
            evaluation_inputs = metric.get("evaluation_inputs")
            if isinstance(evaluation_inputs, list):
                simulated_inputs = {
                    "user_profiles",
                    "user_scripts",
                    "simulated_dialogue",
                    "user_simulation",
                }
                forbidden = sorted(
                    str(item) for item in evaluation_inputs if str(item) in simulated_inputs
                )
                if forbidden:
                    raise PipelineGenerationError(
                        f"metrics[{index}] cannot depend on generated simulation inputs: {forbidden}"
                    )
            if category == "process":
                if metric_type != "hybrid":
                    raise PipelineGenerationError("process metrics must be hybrid")
                if not isinstance(metric.get("target_action"), str) or not metric["target_action"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] process metric requires target_action")
                if key_steps is not None and metric["target_action"] not in key_action_names:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process target_action must reference a key step"
                    )
                if metric.get("condition") != "llm_expected_tool_call_exact_match":
                    raise PipelineGenerationError(
                        f"metrics[{index}] process condition must be llm_expected_tool_call_exact_match"
                    )
            if metric_type in {"model-based", "hybrid"}:
                if not isinstance(metric.get("evaluation_inputs"), list) or not metric["evaluation_inputs"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires evaluation_inputs")
                if not isinstance(metric.get("criteria"), list) or not metric["criteria"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires criteria")
            if metric_type in {"rule-based", "hybrid"}:
                if not isinstance(metric.get("condition"), str) or not metric["condition"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] requires condition")
            category_weights[category] += float(weight)
            ids.add(metric_id)
        if not category_weights["outcome"]:
            raise PipelineGenerationError("metrics must include an outcome metric")
        if not category_weights["penalty"]:
            raise PipelineGenerationError("metrics must include a penalty metric")
        positive_weight_sum = category_weights["process"] + category_weights["outcome"]
        if not math.isclose(positive_weight_sum, 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"positive metric weights must sum to 1, got {positive_weight_sum}")
        if not math.isclose(category_weights["penalty"], 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"penalty metric weights must sum to 1, got {category_weights['penalty']}")
        if category_weights["outcome"] <= category_weights["process"]:
            raise PipelineGenerationError("outcome metric weight must exceed process metric weight")

    @staticmethod
    def _validate_metrics_for_environment(
        metrics: list[Any], *, environment_mode: str
    ) -> None:
        if environment_mode != "stateless":
            return
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            if metric.get("category") == "process":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks must not reward tool processes"
                )
            evaluator = metric.get("evaluator", {})
            if isinstance(evaluator, dict) and evaluator.get("kind") == "business_state_rule":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot depend on business state"
                )
            inputs = metric.get("evaluation_inputs", [])
            if isinstance(inputs, list) and "business_data" in inputs:
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot evaluate business_data"
                )

    @staticmethod
    def _validate_metric_implementations(
        specs: list[Any], metrics: list[Any], *, require_process: bool = False
    ) -> None:
        from .sandbox_runtime import DeclarativeMetricEvaluator, SandboxError

        by_id = {
            metric.get("id"): metric for metric in metrics
            if isinstance(metric, dict) and isinstance(metric.get("id"), str)
        }
        seen: set[str] = set()
        allowed_sources = {"business_state", "trajectory", "final_agent_response", "observation"}
        allowed_operators = {
            "eq", "ne", "gte", "lte", "contains", "exists", "count_gte",
            "count_eq", "none_tool_calls", "contains_tool_call",
        }
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"metric_implementations[{index}] must be an object")
            metric_id = spec.get("metric_id")
            metric = by_id.get(metric_id)
            is_process_call = (
                isinstance(metric, dict)
                and metric.get("category") == "process"
                and spec.get("operator") == "contains_tool_call"
            )
            if metric is None or metric_id in seen or (
                metric.get("type") != "rule-based" and not is_process_call
            ):
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] references an unsupported metric"
                )
            if spec.get("source") not in allowed_sources or spec.get("operator") not in allowed_operators:
                raise PipelineGenerationError(f"metric_implementations[{index}] source/operator is invalid")
            if not isinstance(spec.get("path"), str) or not spec["path"]:
                raise PipelineGenerationError(f"metric_implementations[{index}] path is invalid")
            try:
                DeclarativeMetricEvaluator.path_tokens(spec["path"])
            except (SandboxError, ValueError, SyntaxError) as exc:
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] unsupported path {spec['path']!r}; "
                    "use fields, integer indices or literal equality filters"
                ) from exc
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] cannot address fields on raw final_agent_response"
                )
            if spec.get("source") == "business_state" and "final_agent_response" in str(spec.get("path")):
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] cannot address final_agent_response through business_state"
                )
            if is_process_call:
                expected = spec.get("expected")
                if (
                    spec.get("source") != "trajectory"
                    or spec.get("path") != "$.events"
                    or not isinstance(expected, dict)
                    or not isinstance(expected.get("tool_name"), str)
                    or not isinstance(expected.get("arguments"), dict)
                    or not isinstance(expected.get("captures", []), list)
                ):
                    raise PipelineGenerationError(
                        f"metric_implementations[{index}] process call contract is invalid"
                    )
            mapping = spec.get("score_mapping")
            low, high = metric.get("score_range", [0, 1])
            if (
                not isinstance(mapping, dict) or set(mapping) != {"pass", "fail"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high for value in mapping.values())
            ):
                raise PipelineGenerationError(f"metric_implementations[{index}] score_mapping is invalid")
            seen.add(metric_id)
        required = {
            metric_id for metric_id, metric in by_id.items()
            if metric.get("type") == "rule-based"
            or (require_process and metric.get("category") == "process")
        }
        if seen != required:
            raise PipelineGenerationError(
                "rule-based and process metrics require executable implementations; "
                f"missing={sorted(required - seen)}"
            )

    @staticmethod
    def _compile_process_metric_implementations(
        *,
        metrics: list[Any],
        metric_implementations: list[Any],
        business_scenarios: list[Any],
        tool_bindings: list[Any],
    ) -> list[Any]:
        """Compile process rewards from the accepted success trajectory.

        The success scenario is already the executable source of truth for tool
        names and canonical arguments. Reusing it removes a redundant runtime
        LLM judgment while retaining support for arguments captured from an
        earlier tool result.
        """
        result = [dict(item) for item in metric_implementations if isinstance(item, dict)]
        result = [
            item for item in result
            if not any(
                isinstance(metric, dict)
                and metric.get("id") == item.get("metric_id")
                and metric.get("category") == "process"
                for metric in metrics
            )
        ]
        action_to_tool = {
            item.get("action_name"): item.get("tool_name")
            for item in tool_bindings
            if isinstance(item, dict)
            and isinstance(item.get("action_name"), str)
            and isinstance(item.get("tool_name"), str)
        }
        success = next((
            scenario for scenario in business_scenarios
            if isinstance(scenario, dict) and scenario.get("kind") == "goal_success"
        ), None)
        steps = success.get("steps", []) if isinstance(success, dict) else []
        if not isinstance(steps, list):
            steps = []
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "process":
                continue
            target = metric.get("target_action")
            tool_name = action_to_tool.get(target, target)
            target_index = next((
                index for index, step in enumerate(steps)
                if isinstance(step, dict)
                and step.get("operation") == "tool_call"
                and step.get("tool_name") == tool_name
            ), None)
            if target_index is None:
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} has no matching success tool call"
                )
            target_step = steps[target_index]
            arguments = target_step.get("arguments", {})
            if not isinstance(arguments, dict):
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} success arguments are invalid"
                )
            referenced = {
                value.get("$ref")
                for value in TaskGenerationPipeline._walk_values(arguments)
                if isinstance(value, dict) and set(value) == {"$ref"}
                and isinstance(value.get("$ref"), str)
            }
            captures: list[dict[str, str]] = []
            for step in steps[:target_index]:
                if not isinstance(step, dict) or step.get("operation") != "tool_call":
                    continue
                declared = step.get("capture", {})
                if not isinstance(declared, dict):
                    continue
                for name, path in declared.items():
                    if name in referenced and isinstance(path, str):
                        captures.append({
                            "name": name,
                            "tool_name": str(step.get("tool_name")),
                            "path": path,
                        })
            if referenced != {item["name"] for item in captures}:
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} has unresolved success references"
                )
            result.append({
                "metric_id": metric["id"],
                "source": "trajectory",
                "path": "$.events",
                "operator": "contains_tool_call",
                "expected": {
                    "tool_name": tool_name,
                    "arguments": copy.deepcopy(arguments),
                    "captures": captures,
                },
                "score_mapping": {"pass": 1, "fail": 0},
            })
        return result

    @staticmethod
    def _normalize_compiled_process_metrics(
        metrics: list[Any], metric_implementations: list[Any]
    ) -> None:
        """Make the metric declaration agree with its compiled runtime rule."""
        compiled = {
            item.get("metric_id"): item
            for item in metric_implementations
            if isinstance(item, dict) and item.get("operator") == "contains_tool_call"
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "process":
                continue
            spec = compiled.get(metric.get("id"))
            if not isinstance(spec, dict):
                continue
            expected = spec.get("expected", {})
            metric["type"] = "rule-based"
            metric["condition"] = "trajectory_contains_exact_compiled_tool_call"
            metric["evaluator"] = {
                "kind": "trajectory_rule",
                "source": "runtime_rule",
                "assertion": (
                    "trajectory contains the accepted success-path call to "
                    f"{expected.get('tool_name')} with exact canonical arguments"
                ),
                "score_mapping": {"pass": 1, "fail": 0},
            }
            metric.pop("evaluation_inputs", None)
            metric.pop("criteria", None)

    @staticmethod
    def _walk_values(value: Any) -> list[Any]:
        values = [value]
        if isinstance(value, dict):
            for item in value.values():
                values.extend(TaskGenerationPipeline._walk_values(item))
        elif isinstance(value, list):
            for item in value:
                values.extend(TaskGenerationPipeline._walk_values(item))
        return values

    @staticmethod
    def _promote_unimplemented_rule_metrics(
        metrics: list[Any], specs: list[Any], *, environment_mode: str | None = None
    ) -> None:
        """Never leave a declared rule metric without executable semantics.

        A rule that could not be compiled to the constrained DSL becomes an
        external judge.  This is slower, but preserves evaluability and is
        safer than asking the sandbox builder to interpret free-form text.
        """
        implemented = {
            spec.get("metric_id") for spec in specs if isinstance(spec, dict)
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            metric_id = metric.get("id")
            if metric_id in implemented:
                continue
            evaluator = metric.get("evaluator")
            assertion = evaluator.get("assertion") if isinstance(evaluator, dict) else None
            criterion = assertion if isinstance(assertion, str) and assertion.strip() else metric.get("rubric")
            scope = metric.get("scope")
            if scope == "trajectory":
                inputs = ["conversation", "tool_call", "tool_results", "final_agent_response"]
            elif scope == "terminal":
                inputs = ["final_agent_response", "terminal_observation"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            else:
                inputs = ["public_observation", "tool_results"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = inputs
            metric["criteria"] = [str(criterion or "判断该指标是否满足。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(evaluator.get("score_mapping", {}))
                if isinstance(evaluator, dict) else {"pass": 1, "fail": 0},
            }
            logger.info("promoted uncompiled rule metric to external judge: metric=%s", metric_id)

    @staticmethod
    def _ensure_terminal_outcome_metrics(
        metrics: list[Any], specs: list[Any], *, environment_mode: str,
    ) -> None:
        """Make user-facing answers part of the declared reward contract."""
        if environment_mode == "stateful":
            return
        outcome_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("category") == "outcome"
        }
        if declared_terminal_outcome_ids(metrics, specs):
            return
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "outcome":
                continue
            rubric = str(metric.get("rubric") or "完成用户目标")
            metric["type"] = "model-based"
            metric["scope"] = "terminal"
            metric["evaluation_inputs"] = [
                "conversation", "tool_results", "final_agent_response",
            ]
            metric["criteria"] = [
                "最终回答必须完成用户当前目标；仅调用工具或检索到记录不能代替回答。",
                "回答中的结论必须由公开输入或工具结果支持，不得编造事实。",
                f"业务目标：{rubric}",
            ]
            metric["evaluator"] = {
                "kind": "external_llm_judge", "source": "external_llm",
                "score_mapping": {"full": 1, "partial": 0.5, "missing": 0},
            }
            metric["rubric"] = f"最终回答完成目标并有证据支持：{rubric}"
            metric.pop("condition", None)
        specs[:] = [
            spec for spec in specs
            if not isinstance(spec, dict) or spec.get("metric_id") not in outcome_ids
        ]
        logger.warning("promoted outcome reward to declared terminal evaluator: metrics=%s", sorted(outcome_ids))

    @staticmethod
    def _prune_literal_payload_process_metrics(
        metrics: list[Any], specs: list[Any],
    ) -> None:
        """Avoid process credit tied to one serialized free-form payload.

        A later process call that captures the parser's result already proves
        the parsing step occurred and supplies the causal dependency.  Only in
        that case can the brittle exact-input metric be removed safely.
        """
        brittle: set[str] = set()
        for spec in specs:
            if not has_literal_payload_argument(spec):
                continue
            tool_name = spec["expected"].get("tool_name")
            downstream = any(
                other is not spec and isinstance(other, dict)
                and isinstance(other.get("expected"), dict)
                and any(
                    isinstance(capture, dict)
                    and capture.get("tool_name") == tool_name
                    for capture in other["expected"].get("captures", [])
                )
                for other in specs
            )
            if not downstream:
                raise PipelineGenerationError(
                    f"process metric {spec.get('metric_id')} matches a literal payload "
                    "without downstream capture evidence"
                )
            brittle.add(str(spec.get("metric_id")))
        if not brittle:
            return
        metrics[:] = [
            metric for metric in metrics
            if not isinstance(metric, dict) or metric.get("id") not in brittle
        ]
        specs[:] = [
            spec for spec in specs
            if not isinstance(spec, dict) or spec.get("metric_id") not in brittle
        ]
        TaskGenerationPipeline._normalize_metric_weights(metrics)
        logger.warning("removed literal-payload process metrics with downstream capture proof: %s", sorted(brittle))

    @staticmethod
    def _validate_reward_formula(formula: Any, metrics: list[Any]) -> None:
        if not isinstance(formula, dict):
            raise PipelineGenerationError("reward_formula must be an object")
        if formula.get("type") != "separate_sign_weighted_sum":
            raise PipelineGenerationError("reward_formula.type must be separate_sign_weighted_sum")
        if not isinstance(formula.get("formula"), str) or not formula["formula"].strip():
            raise PipelineGenerationError("reward_formula.formula must be non-empty")
        if formula.get("score_range") != [-1, 1]:
            raise PipelineGenerationError("reward_formula.score_range must be [-1, 1]")
        if formula.get("positive_weight_sum") != 1 or formula.get("negative_weight_sum") != 1:
            raise PipelineGenerationError("reward_formula positive and negative weights must each sum to 1")

    @staticmethod
    def _canonical_reward_formula(metrics: list[Any]) -> dict[str, Any]:
        """Return a bounded formula derived from the validated metric weights.

        Process and outcome are one positive feedback pool. Penalty is the
        independent negative feedback pool. Dividing process and outcome into
        separate averages would make the positive contribution reach 2, so
        the canonical formula uses the already-normalized metric weights once.
        """
        positive_terms: list[str] = []
        negative_terms: list[str] = []
        for metric in metrics:
            metric_id = str(metric["id"])
            term = f"{float(metric['weight']):.12g}*score({metric_id})"
            if metric["category"] in {"process", "outcome"}:
                positive_terms.append(term)
            else:
                negative_terms.append(term)
        positive_expression = " + ".join(positive_terms) or "0"
        negative_expression = " + ".join(negative_terms) or "0"
        return {
            "type": "separate_sign_weighted_sum",
            "formula": f"R = clip(({positive_expression}) + ({negative_expression}), -1, 1)",
            "positive_weight_sum": 1,
            "negative_weight_sum": 1,
            "score_range": [-1, 1],
            "positive_categories": ["process", "outcome"],
            "negative_categories": ["penalty"],
            "normalization": "metric weights are normalized once across all positive categories and independently across penalty metrics",
        }

    @staticmethod
    def _materialize_tools(tools: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
        """Persist only standard OpenAI tool definitions.

        Bindings, Trainer actions, and implementation details deliberately stay
        outside this file. Code Agent consumes this schema and implements the
        corresponding sandbox behavior separately.
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "tools.json"
        path.write_text(
            json.dumps(tools, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "version": "1.0",
            "file": str(path.relative_to(artifact_dir)),
            "format": "openai_function_tools",
            "tool_count": len(tools),
        }

    def _generate_table_rows(
        self, table: dict[str, Any], *, task_description: dict[str, Any],
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Repair one table's rows without regenerating unrelated tables."""
        previous_rows: Any = None
        validation_error: str | None = None
        for attempt in range(1, self.retries + 1):
            result = self._call(
                f"environment_table_data.{table['table_name']}"
                + (".repair" if attempt > 1 else ""),
                "根据已确认的单张表结构生成完整但紧凑、可直接插入数据库的初始业务数据。"
                "只返回顶层 {\"rows\": [...]}；rows 必须非空，每行覆盖全部字段，主键唯一，"
                "每个值必须满足该表的 CHECK 约束。创建、修改、审批、排程等 stateful 任务的 rows "
                "必须是 Agent 执行之前的基线，不能预先写入目标终态。不要返回 schema 或解释。"
                "如果收到 validation_error，针对错误修复 previous_rows，保持其他合法事实。",
                {
                    "task_description": task_description, "entities": entities,
                    "table": table, "previous_rows": previous_rows,
                    "validation_error": validation_error,
                    "output": {"rows": []},
                },
            )
            previous_rows = result.get("rows")
            try:
                self._validate_generated_table_rows(table, previous_rows)
                return table | {"rows": previous_rows}
            except PipelineGenerationError as exc:
                validation_error = str(exc)
                logger.warning(
                    "environment table rows invalid: table=%s attempt=%d/%d error=%s",
                    table["table_name"], attempt, self.retries, exc,
                )
        raise PipelineGenerationError(
            f"environment_table_data.{table['table_name']} exhausted "
            f"{self.retries} row candidates: {validation_error}"
        )

    @staticmethod
    def _validate_generated_table_rows(
        table: dict[str, Any], rows: Any, *, check_values: bool = True,
    ) -> None:
        from .sandbox_runtime import ManifestDataStore, SandboxError

        name = str(table["table_name"])
        if not isinstance(rows, list) or not rows:
            raise PipelineGenerationError(f"data table {name} requires complete non-empty rows")
        columns = {str(column["name"]): column for column in table["columns"]}
        primary = table["primary_key"]
        keys: set[tuple[Any, ...]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != set(columns):
                raise PipelineGenerationError(f"data table {name}.rows[{index}] is not a complete row")
            if not check_values:
                continue
            key = tuple(row[field] for field in primary)
            try:
                duplicate = key in keys
            except TypeError as exc:
                raise PipelineGenerationError(f"data table {name} has a non-scalar primary key") from exc
            if any(value is None for value in key) or duplicate:
                raise PipelineGenerationError(f"data table {name} has null or duplicate primary keys")
            keys.add(key)
            for column_name, column in columns.items():
                if row[column_name] is None and column.get("nullable") is False:
                    raise PipelineGenerationError(
                        f"data table {name}.rows[{index}].{column_name} violates NOT NULL"
                    )
            for constraint in table.get("constraints", []):
                expression = constraint if isinstance(constraint, str) else constraint.get("expression")
                try:
                    valid = ManifestDataStore._check_constraint(row, expression)
                except (SandboxError, TypeError, ValueError) as exc:
                    raise PipelineGenerationError(
                        f"data table {name} has unsupported CHECK constraint: {expression}"
                    ) from exc
                if not valid:
                    raise PipelineGenerationError(
                        f"data table {name}.rows[{index}] violates CHECK constraint: {expression}"
                    )

    @staticmethod
    def _validate_data_tables(tables: Any) -> None:
        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("business data tables must be a non-empty list")
        TaskGenerationPipeline._validate_table_definitions(tables)
        for table in tables:
            TaskGenerationPipeline._validate_generated_table_rows(
                table, table.get("rows"), check_values=False,
            )

    @staticmethod
    def _validate_relational_data(tables: list[dict[str, Any]]) -> None:
        """Validate generated rows with the same relational invariants as runtime."""
        TaskGenerationPipeline._validate_data_tables(tables)
        by_name = {str(table["table_name"]): table for table in tables}
        for table in tables:
            name = str(table["table_name"])
            rows = table["rows"]
            TaskGenerationPipeline._validate_generated_table_rows(table, rows)
            for foreign in table.get("foreign_keys", []):
                if not isinstance(foreign, dict):
                    raise PipelineGenerationError(f"data table {name} has an invalid foreign key")
                field = foreign.get("column")
                parent_name = foreign.get("ref_table") or foreign.get("references_table")
                target = foreign.get("ref_column") or foreign.get("references_column")
                parent = by_name.get(str(parent_name))
                parent_columns = {
                    item.get("name") for item in parent.get("columns", [])
                } if isinstance(parent, dict) else set()
                own_columns = {item.get("name") for item in table.get("columns", [])}
                if field not in own_columns or target not in parent_columns:
                    raise PipelineGenerationError(
                        f"data table {name} foreign key declaration is invalid"
                    )
                available = {row.get(target) for row in parent["rows"]}
                missing = sorted({row.get(field) for row in rows if row.get(field) is not None} - available, key=str)
                if missing:
                    raise PipelineGenerationError(
                        f"data table {name}.{field} has missing foreign values: {missing[:5]}"
                    )

    @staticmethod
    def _validate_entities(entities: Any) -> None:
        if not isinstance(entities, list) or not entities:
            raise PipelineGenerationError("environment_entities.entities must be a non-empty list")
        ids: set[str] = set()
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise PipelineGenerationError(f"entities[{index}] must be an object")
            entity_id = entity.get("entity_id") or entity.get("name")
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in ids:
                raise PipelineGenerationError(f"entities[{index}] has invalid or duplicate entity_id/name")
            if not isinstance(entity.get("description"), str) or not entity["description"].strip():
                raise PipelineGenerationError(f"entities[{index}] requires a description")
            if not isinstance(entity.get("required_facts", []), list):
                raise PipelineGenerationError(f"entities[{index}].required_facts must be a list")
            ids.add(entity_id)

    @staticmethod
    def _validate_table_definitions(tables: Any) -> None:
        from .sandbox_runtime import ManifestDataStore, SandboxError

        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("environment_table_design.tables must be a non-empty list")
        declared_names = [
            table.get("table_name") for table in tables if isinstance(table, dict)
        ]
        if any(not isinstance(name, str) or not name.strip() for name in declared_names):
            raise PipelineGenerationError("table definitions contain an invalid table_name")
        if len(declared_names) != len(set(declared_names)):
            raise PipelineGenerationError("table definitions contain duplicate table_name values")
        known_tables = set(declared_names)
        names: set[str] = set()
        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                raise PipelineGenerationError(f"table definitions[{index}] must be an object")
            name = table.get("table_name")
            columns = table.get("columns")
            primary_key = table.get("primary_key")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"table definitions[{index}] has invalid or duplicate table_name")
            if not isinstance(columns, list) or not columns:
                raise PipelineGenerationError(f"table definitions[{index}] requires columns")
            if not isinstance(primary_key, list) or not primary_key:
                raise PipelineGenerationError(f"table definitions[{index}] requires primary_key")
            column_names: set[str] = set()
            for column in columns:
                if not isinstance(column, dict) or not isinstance(column.get("name"), str) or not column["name"].strip():
                    raise PipelineGenerationError(f"table definitions[{index}] contains an invalid column")
                if column["name"] in column_names:
                    raise PipelineGenerationError(f"table definitions[{index}] contains duplicate columns")
                column_names.add(column["name"])
            if any(key not in column_names for key in primary_key):
                raise PipelineGenerationError(f"table definitions[{index}] primary_key references an unknown column")
            for field in ("foreign_keys", "indexes", "constraints"):
                if field in table and not isinstance(table[field], list):
                    raise PipelineGenerationError(f"table definitions[{index}].{field} must be a list")
            for foreign in table.get("foreign_keys", []):
                ref_table = foreign.get("ref_table") or foreign.get("references_table") if isinstance(foreign, dict) else None
                ref_column = foreign.get("ref_column") or foreign.get("references_column") if isinstance(foreign, dict) else None
                if (
                    not isinstance(foreign, dict)
                    or not isinstance(foreign.get("column"), str)
                    or foreign.get("column") not in column_names
                    or ref_table not in known_tables
                    or not isinstance(ref_column, str)
                    or not ref_column.strip()
                ):
                    raise PipelineGenerationError(
                        f"table definitions[{index}] contains an invalid foreign key"
                    )
            for constraint in table.get("constraints", []):
                if isinstance(constraint, str):
                    expression = constraint.strip()
                elif isinstance(constraint, dict) and str(constraint.get("type", "CHECK")).upper() == "CHECK":
                    expression = constraint.get("expression")
                else:
                    expression = None
                if not isinstance(expression, str) or not expression.strip():
                    raise PipelineGenerationError(
                        f"table definitions[{index}] contains an invalid CHECK constraint"
                    )
                for clause in re.split(r"\s+AND\s+", expression.strip(), flags=re.I):
                    nullable_in_match = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*?)\)\s+OR\s+\1\s+IS\s+NULL\s*",
                        clause, flags=re.I,
                    )
                    in_match = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*)\)\s*",
                        clause, flags=re.I,
                    )
                    comparison = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|<>|!=|=|>|<)\s*(.*?)\s*",
                        clause,
                    )
                    match = nullable_in_match or in_match or comparison
                    if not match or match.group(1) not in column_names:
                        raise PipelineGenerationError(
                            f"table definitions[{index}] has unsupported CHECK constraint: {expression}"
                        )
                    try:
                        if nullable_in_match or in_match:
                            import ast
                            ast.literal_eval(f"({match.group(2)},)")
                        else:
                            ManifestDataStore._constraint_literal(comparison.group(3))
                    except (ValueError, SyntaxError, SandboxError) as exc:
                        raise PipelineGenerationError(
                            f"table definitions[{index}] has unsupported CHECK constraint: {expression}"
                        ) from exc
            names.add(name)

    @staticmethod
    def _normalize_structural_constraints(tables: Any) -> Any:
        """Compile common descriptive UNIQUE/NOT NULL text into schema fields.

        Mid-tier models occasionally put structural constraints such as
        ``name 唯一且非空`` in CHECK constraints.  Those are not executable
        CHECK expressions, but their meaning is unambiguous and already has a
        canonical representation in the schema.
        """
        if not isinstance(tables, list):
            return tables
        normalized = copy.deepcopy(tables)
        for table in normalized:
            if not isinstance(table, dict):
                continue
            columns = {
                column.get("name"): column for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            }
            indexes = table.setdefault("indexes", [])
            kept = []
            for constraint in table.get("constraints", []):
                if not isinstance(constraint, str):
                    kept.append(constraint)
                    continue
                lowered = constraint.casefold()
                unique = "唯一" in constraint or bool(re.search(r"\bunique\b", lowered))
                non_null = (
                    "非空" in constraint
                    or bool(re.search(r"\bnot[ _-]?null\b", lowered))
                    or "不能为空" in constraint
                )
                mentioned = [
                    name for name in columns
                    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", constraint)
                ]
                if (unique or non_null) and len(mentioned) == 1:
                    column_name = mentioned[0]
                    if unique and not any(
                        isinstance(index, dict)
                        and index.get("unique") is True
                        and index.get("columns") == [column_name]
                        for index in indexes
                    ):
                        indexes.append({
                            "name": f"uq_{table.get('table_name', 'table')}_{column_name}",
                            "columns": [column_name], "unique": True,
                        })
                    if non_null:
                        columns[column_name]["nullable"] = False
                    continue
                kept.append(constraint)
            table["constraints"] = kept
        return normalized

    @staticmethod
    def _materialize_business_data(
        tables: list[dict[str, Any]],
        data_document: str,
        artifact_dir: Path,
        *,
        environment_mode: str = "business_data",
        manifest_root: str | None = None,
        data_governance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write table schemas/rows to files and return the compact task manifest."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_dir = artifact_dir / "schemas"
        rows_dir = artifact_dir / "rows"
        schema_dir.mkdir(exist_ok=True)
        rows_dir.mkdir(exist_ok=True)
        used_names: set[str] = set()
        manifest_tables: list[dict[str, Any]] = []

        for index, table in enumerate(tables):
            raw_name = str(table["table_name"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._") or f"table_{index + 1}"
            if safe_name in used_names:
                safe_name = f"{safe_name}_{index + 1}"
            used_names.add(safe_name)
            schema = {key: value for key, value in table.items() if key != "rows"}
            schema_path = schema_dir / f"{safe_name}.json"
            rows_path = rows_dir / f"{safe_name}.jsonl"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with rows_path.open("w", encoding="utf-8") as handle:
                for row in table["rows"]:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            manifest_tables.append({
                "table_name": raw_name,
                "schema_file": str(schema_path.relative_to(artifact_dir)),
                "rows_file": str(rows_path.relative_to(artifact_dir)),
                "row_count": len(table["rows"]),
                "primary_key": table["primary_key"],
            })

        document_path = artifact_dir / "data_document.md"
        document_path.write_text(data_document.rstrip() + "\n", encoding="utf-8")
        return {
            "version": "1.0",
            "environment_mode": environment_mode,
            "data_governance": data_governance or {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            },
            "root": manifest_root or str(artifact_dir),
            "document_file": str(document_path.relative_to(artifact_dir)),
            "tables": manifest_tables,
        }

    @staticmethod
    def _environment_records(environment: dict[str, Any], actions: list[Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in environment.get("records", []):
            if isinstance(item, dict) and {"type", "field", "description", "value"} <= set(item):
                records.append({k: item[k] for k in ("type", "field", "description", "value")})
        for table in environment.get("data_tables", []):
            if isinstance(table, dict) and isinstance(table.get("table_name"), str):
                records.append({
                    "type": "business_table",
                    "field": table["table_name"],
                    "description": str(table.get("description") or "任务业务数据表"),
                    "value": table,
                })
        for action in actions:
            if isinstance(action, dict):
                name = action.get("name") or action.get("action")
                if isinstance(name, str) and name.strip():
                    records.append({"type": "action", "name": name, "description": action.get("description", name)})
        return records
