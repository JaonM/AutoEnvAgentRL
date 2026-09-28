"""Source-backed task generation from a local table or Kaggle catalog entry."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from env_factory.generation.task_generator import TaskGenerationError, TaskGenerator
from env_factory.generation.dataset_formats import SUPPORTED_SOURCE_EXTENSIONS, source_extension
from env_factory.generation.dataset_source_registry import (
    eligible_hk_ids, verified_hk_source,
)
from env_factory.generation.dataset_table_reader import DEFAULT_MAX_SOURCE_BYTES, sample_table, selected_member
from env_factory.graph.knowledge_graph import TaskType
from env_factory.task_pipeline import TaskGenerationPipeline
from env_factory.tasks.task import Task


PROJECT = Path(__file__).resolve().parents[3]
KAGGLE_ROOT = PROJECT / "data" / "sources" / "kaggle"
DEFAULT_ALLOWLIST = PROJECT / "config" / "dataset_generation_allowlist.json"
logger = logging.getLogger(__name__)
PRIVATE_COLUMNS = re.compile(r"name|email|phone|address|passport|ssn|birth|contact|comment|review|姓名|邮箱|电话|手机|住址|地址|身份证|护照|出生|联系人|评论", re.I)
KEY_COLUMNS = re.compile(r"(^id$|[_ -]id$|order|booking|invoice|transaction|record|ticket|订单号|交易号|预订号|票号|单号|流水号|编号)", re.I)
GROUP_COLUMNS = re.compile(r"category|type|region|city|status|product|department|store|channel|类别|分类|地区|城市|状态|商品|部门|店铺|渠道|目的地", re.I)
VALUE_COLUMNS = re.compile(r"price|amount|cost|sales|quantity|revenue|fare|total|stock|units|金额|价格|费用|销售额|收入|数量|总价|票价|成本|库存", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_name(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not result or result[0].isdigit() or any(ord(char) > 127 for char in value):
        result = ("field_" + result + "_" if result else "field_") + hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:10]
    return result


def _number(value: str) -> float | None:
    try:
        number = float(value.replace(",", "").replace("$", "")
                       .replace("¥", "").replace("￥", "").strip())
    except (ValueError, AttributeError):
        return None
    return number if math.isfinite(number) else None


def _sample_source(path: Path, *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES) -> tuple[list[str], list[dict[str, str]]]:
    """Load a bounded table sample and check normalized column identities."""
    headers, rows = sample_table(path, max_source_bytes=max_source_bytes)
    if any(not _column_name(header) for header in headers):
        raise TaskGenerationError("dataset table has invalid headers")
    if len({_column_name(header) for header in headers}) != len(headers):
        raise TaskGenerationError("dataset table has colliding normalized headers")
    return headers, rows


def _columns(headers: list[str], rows: list[dict[str, str]]) -> tuple[str, str, str]:
    def rank(header: str, patterns: tuple[str, ...]) -> int:
        normalized = _column_name(header)
        return next((index for index, pattern in enumerate(patterns)
                     if re.search(pattern, normalized)), len(patterns))

    keys = [h for h in headers if KEY_COLUMNS.search(h) and not PRIVATE_COLUMNS.search(h)
            and all(str(row.get(h) or "").strip() for row in rows)
            and len({str(row[h]).strip() for row in rows}) >= max(2, len(rows) // 4)]
    key = min(keys, key=lambda h: rank(h, (r"booking_id", r"order_id", r"transaction_id",
                                           r"invoice", r"record_id", r"_id$")), default=None)
    groups = [h for h in headers if h != key and GROUP_COLUMNS.search(h)
              and not PRIVATE_COLUMNS.search(h)
              and 2 <= len({str(row.get(h) or "").strip() for row in rows}) <= 40
              and all(str(row.get(h) or "").strip() for row in rows)]
    group = min(groups, key=lambda h: (rank(h, (r"destination_city", r"category", r"status",
                                              r"region", r"product", r"department", r"store",
                                              r"channel", r"city", r"type")),
                                       abs(len({str(row[h]).strip() for row in rows}) - 8)), default=None)
    numeric_fields = [h for h in headers if h not in (key, group) and VALUE_COLUMNS.search(h)
                      and not PRIVATE_COLUMNS.search(h)
                      and all(_number(str(row.get(h) or "")) is not None for row in rows)
                      and len({_number(str(row[h])) for row in rows}) > 2]
    numeric = min(numeric_fields, key=lambda h: rank(h, (r"total.*(cost|price|amount)", r"total.*sales",
                                                       r"revenue", r"price", r"fare", r"cost",
                                                       r"sales", r"amount", r"quantity", r"stock",
                                                       r"discount")), default=None)
    if not all((key, group, numeric)):
        raise TaskGenerationError("dataset lacks a unique ID, repeated category and numeric business value")
    return key, group, numeric


def _choose_rows(rows: list[dict[str, str]], key: str, group: str, numeric: str,
                 category: str, rng: random.Random, *, extreme: str = "minimum") -> list[dict[str, str]]:
    if category == "direct_response":
        pairs = [(a, b) for a in rows[:80] for b in rows[:80]
                 if a is not b and _number(a[numeric]) != _number(b[numeric])]
        if not pairs:
            raise TaskGenerationError("dataset has no comparable distinct values")
        return list(rng.choice(pairs))
    if category == "simple_agentic":
        starter = rng.choice(rows)
        return [starter] + [row for row in rows if row is not starter][:19]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[group], []).append(row)
    eligible = [members for members in groups.values() if len(members) >= 3
                and len({_number(row[numeric]) for row in members}) >= 2]
    if not eligible:
        raise TaskGenerationError("dataset has no group with a unique minimum and usable dependency")
    members = rng.choice(eligible)
    if extreme in {"minimum", "maximum"}:
        selected_extreme = (min if extreme == "minimum" else max)(
            members, key=lambda row: _number(row[numeric]))
        other = [row for row in members if _number(row[numeric]) != _number(selected_extreme[numeric])]
        starter = rng.choice(other)
        members = [starter, selected_extreme] + [row for row in other if row is not starter][:18]
    else:
        starter = rng.choice(members)
        members = [starter] + [row for row in members if row is not starter][:19]
    others = [row for row in rows if row[group] != members[0][group]][:10]
    return members + others


def _project(rows: list[dict[str, str]], key: str, group: str, numeric: str) -> list[dict[str, Any]]:
    projected = [{_column_name(key): str(row[key]).strip(),
                  _column_name(group): str(row[group]).strip(),
                  _column_name(numeric): _number(row[numeric])} for row in rows]
    for row in projected:
        for field in (_column_name(key), _column_name(group)):
            value = row[field]
            if (len(value) > 80 or re.search(r"[\x00-\x1f\x7f]|https?://|@|\b\d{10,}\b", value, re.I)):
                raise TaskGenerationError("projected source value may contain private or unsafe text")
    return projected


def _business_terms(key: str, group: str, numeric: str) -> tuple[str, str, str, str]:
    if "booking" in key or "ticket" in key:
        entity, identifier = "行程", "预订号"
    elif "transaction" in key or "order" in key or "invoice" in key:
        entity, identifier = "消费", "交易号"
    else:
        entity, identifier = "业务", "凭证号"
    if "destination" in group and "city" in group:
        group_label = "目的地"
    elif "category" in group or "product" in group:
        group_label = "商品类别"
    elif "city" in group or "region" in group:
        group_label = "地区"
    else:
        group_label = "类别"
    if "cost" in numeric or "fare" in numeric:
        value_label = "总费用"
    elif "quantity" in numeric or "units" in numeric or "stock" in numeric:
        value_label = "数量"
    else:
        value_label = "金额"
    return entity, identifier, group_label, value_label


def _description(category: str, rows: list[dict[str, Any]], key: str, group: str,
                 numeric: str, title: str, scene: dict[str, str] | None = None,
                 *, extreme: str = "minimum", comparison: str = "lower") -> tuple[dict, dict]:
    del title  # Source identity belongs in provenance and the private business model.
    first = rows[0]
    key_value, group_value, number_value = first[key], first[group], first[numeric]
    entity, identifier, group_label, value_label = _business_terms(key, group, numeric)
    if scene:
        entity = scene["entity_name"]
        identifier = scene["identifier_label"]
        group_label = scene["group_label"]
        value_label = scene["value_label"]
    common = {"context": [],
              "requirements": {"input_modalities": ["text"], "output_format": "text"}}
    if category == "direct_response":
        second = rows[1]
        selected = ((first if number_value < second[numeric] else second)
                    if comparison == "lower" else
                    (first if number_value > second[numeric] else second))
        difference = abs(number_value - second[numeric])
        direction = "较低" if comparison == "lower" else "较高"
        request = f"比较两笔{entity}的{value_label}，说出{direction}的一笔和差额。"
        public = [{"标签": label, value_label: row[numeric]}
                  for label, row in zip(("A", "B"), rows)]
        selected_label = "A" if selected is first else "B"
        facts = {f"{comparison}_label": selected_label, "difference": difference}
        return ({**common, "task": request, "task_intent": "compare", "goal": request,
                 "public_input": {"initial_user_message": request, "materials": [{
                     "name": f"两笔{entity}.json", "mime_type": "application/json",
                     "content": json.dumps(public, ensure_ascii=False),
                 }]}, "route_plan": {"environment_operations": []},
                 "expected_result": f"{selected_label} 的{value_label}{direction}，相差 {difference:g}。",
                 "complexity": "simple"}, facts)
    voucher = {"name": f"{entity}信息.json", "mime_type": "application/json",
               "content": json.dumps({identifier: key_value}, ensure_ascii=False)}
    if category == "simple_agentic":
        request = f"查询这笔{entity}的{group_label}和{value_label}。"
        facts = {"record_id": key_value, group: group_value, numeric: number_value}
        return ({**common, "task": request, "task_intent": "query", "goal": request,
                 "public_input": {"initial_user_message": request, "materials": [voucher]},
                 "route_plan": {"environment_operations": [{
                     "action_name": "lookup_record", "purpose": "Read one private business record by voucher ID", "dependencies": [],
                 }]}, "expected_result": f"{group_label}为 {group_value}，{value_label}为 {number_value:g}。",
                 "complexity": "simple"}, facts)
    same = [row for row in rows if row[group] == group_value]
    if extreme in {"average", "count"}:
        if extreme == "average":
            statistic = sum(row[numeric] for row in same) / len(same)
            request = f"这笔{entity}属于哪个{group_label}？同类{entity}的平均{value_label}是多少？"
            expected = f"{group_value}的平均{value_label}为 {statistic:g}。"
            facts = {"starting_record_id": key_value, "group_value": group_value,
                     "average_value": statistic}
        else:
            request = f"这笔{entity}属于哪个{group_label}？同类{entity}共有多少笔？"
            expected = f"{group_value}共有 {len(same)} 笔{entity}。"
            facts = {"starting_record_id": key_value, "group_value": group_value,
                     "group_count": len(same)}
        return ({**common, "task": request, "task_intent": "calculate", "goal": request,
                 "public_input": {"initial_user_message": request, "materials": [voucher]},
                 "route_plan": {"environment_operations": [
                     {"action_name": "lookup_record", "purpose": "Read the group field of a private business record", "dependencies": []},
                     {"action_name": "list_group_records", "purpose": "Read private records in the returned group to calculate the requested statistic",
                      "dependencies": ["lookup_record"]},
                 ]}, "expected_result": expected, "complexity": "standard"}, facts)
    lowest = (min if extreme == "minimum" else max)(same, key=lambda row: row[numeric])
    adjective = "最低" if extreme == "minimum" else "最高"
    fact_prefix = "lowest" if extreme == "minimum" else "highest"
    request = (f"查询这笔{entity}的{group_label}，再找同类{entity}中"
               f"{value_label}{adjective}的那笔及其数值。")
    facts = {"starting_record_id": key_value, "group_value": group_value,
             f"{fact_prefix}_record_id": lowest[key], f"{fact_prefix}_value": lowest[numeric]}
    return ({**common, "task": request, "task_intent": "query", "goal": request,
             "public_input": {"initial_user_message": request, "materials": [voucher]},
             "route_plan": {"environment_operations": [
                 {"action_name": "lookup_record", "purpose": "Read the group field of a private business record", "dependencies": []},
                 {"action_name": "list_group_records", "purpose": "Search private business records using the group returned by lookup_record",
                  "dependencies": ["lookup_record"]},
             ]}, "expected_result": f"{group_value}中{value_label}{adjective}的是 {lowest[key]}，数值为 {lowest[numeric]:g}。",
             "complexity": "standard"}, facts)


def _validate_scene(scene: dict[str, Any]) -> dict[str, str]:
    fields = ("entity_name", "identifier_label", "group_label", "value_label",
              "user_role", "business_situation")
    result: dict[str, str] = {}
    for field in fields:
        value = scene.get(field)
        if not isinstance(value, str) or not 2 <= len(value.strip()) <= 60:
            raise TaskGenerationError(f"dataset business scenario has invalid {field}")
        result[field] = value.strip()
    if any(re.search(r"数据集|CSV|表字段|沙箱|评测|训练|记录", value, re.I)
           for value in result.values()):
        raise TaskGenerationError("dataset business scenario uses implementation language")
    return result


def _supported_business_need(category: str, scene: dict[str, str], *, extreme: str = "minimum",
                             comparison: str = "lower") -> str:
    entity, group, value = (scene["entity_name"], scene["group_label"], scene["value_label"])
    if category == "direct_response":
        direction = "较低" if comparison == "lower" else "较高"
        return f"核对两笔{entity}的{value}差异，找出{direction}的一笔"
    if category == "simple_agentic":
        return f"核对这笔{entity}的{group}归属与{value}"
    if extreme == "average":
        return f"了解这笔{entity}所属的{group}，并计算同类{entity}的平均{value}"
    if extreme == "count":
        return f"了解这笔{entity}所属的{group}，并统计同类{entity}的数量"
    adjective = "最低" if extreme == "minimum" else "最高"
    return f"了解这笔{entity}所属的{group}，并比较同类{entity}的{adjective}{value}"


def _validate_voice(message: str, original: str, category: str, rows: list[dict[str, Any]],
                    key: str, facts: dict[str, Any], title: str,
                    scene: dict[str, str] | None = None, *, extreme: str = "minimum",
                    comparison: str = "lower") -> None:
    if not 12 <= len(message) <= 260 or message == original:
        raise TaskGenerationError("dataset task voice is invalid or unchanged")
    if re.search(r"数据集|\b(?:record|dataset)\b|记录|编号|附上|附件|凭证", message, re.I) or title in message:
        raise TaskGenerationError("dataset task voice exposes source-oriented wording")
    if category != "direct_response" and re.search(
        r"(?:这|那|附上|提供|手头|有|的).{0,5}(?:两|二|2)(?:笔|张|份|个)"
        r"|(?:两|二|2)(?:笔|张|份|个).{0,8}(?:凭证|小票|订单|消费|行程)|分别", message
    ):
        raise TaskGenerationError("dataset task voice invents a second starting voucher")
    if any(re.search(rf"(?<![\w]){re.escape(str(row[key]))}(?![\w])", message)
           for row in rows):
        raise TaskGenerationError("dataset task voice repeats a source record ID")
    if re.search(r"帮我(?:查一下|看看|比较一下)|再(?:查|看看)|先.{0,20}(?:再|然后)|然后", message):
        raise TaskGenerationError("dataset task voice describes a lookup procedure")
    if re.search(r"营销|活动|退款|退货|折扣|利润|预算|贡献|表现|汇总|上周|昨天|去年", message):
        raise TaskGenerationError("dataset task voice adds unsupported business context")
    if category == "direct_response" and scene and scene["group_label"] in message:
        raise TaskGenerationError("direct dataset task mentions a group absent from public input")
    if not re.search(r"我在|我想|我需要|我要|为了|准备|核对|对账|想弄清|想知道|需要确认", message):
        raise TaskGenerationError("dataset task voice lacks a business motivation")
    if category == "direct_response" and not re.search(r"两笔|两单|两次|两个|这两|A.{0,8}B", message):
        raise TaskGenerationError("dataset task voice loses the two-item comparison")
    if category == "direct_response":
        desired = r"低|少|便宜" if comparison == "lower" else r"高|多|贵"
        opposite = r"高|最多|最贵" if comparison == "lower" else r"低|最少|最便宜"
        if not re.search(desired, message) or re.search(opposite, message):
            raise TaskGenerationError("dataset task voice changes the comparison direction")
    extreme_words = {
        "minimum": r"最低|最少|最便宜", "maximum": r"最高|最多|最贵",
        "average": r"平均|均值", "count": r"多少笔|几笔|数量|总共有多少|一共有多少",
    }[extreme]
    if category == "multi_step_agentic" and not re.search(extreme_words, message):
        raise TaskGenerationError(f"dataset task voice loses the group {extreme} goal")
    if category == "multi_step_agentic" and extreme in {"minimum", "maximum"} and not re.search(r"哪(?:一)?(?:笔|单|条|个|次)", message):
        raise TaskGenerationError("dataset task voice omits the minimum item's identity")
    hidden = ([str(facts["difference"])] if category == "direct_response"
              else [str(value) for name, value in facts.items()
                    if name not in ("record_id", "starting_record_id") and value is not None])
    if any(value and re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", message) for value in hidden):
        raise TaskGenerationError("dataset task voice reveals a hidden answer")
    if re.search(r"lookup_record|list_group_records|\$ref|capture", message, re.I):
        raise TaskGenerationError("dataset task voice includes implementation details")


class DatasetTaskGenerator:
    """Use source rows as the business truth for the existing downstream pipeline."""

    def __init__(self, llm: Any, *, dataset_ref: str | None = None,
                 dataset_file: Path | None = None, source_url: str | None = None,
                 dataset_id: str | None = None,
                 max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                 user_script_count: int = 3, noise_tool_max: int = 3) -> None:
        self.dataset_ref = dataset_ref
        self.dataset_id = dataset_id
        self.dataset_file = dataset_file
        self.source_url = source_url
        self.max_source_bytes = max_source_bytes
        self.pipeline = TaskGenerationPipeline(llm, script_count=user_script_count,
                                               noise_tool_max=noise_tool_max)

    def _source(self, rng: random.Random, platform: str = "kaggle") -> tuple[Path, str, str, str | None]:
        if self.dataset_file:
            if not self.dataset_file.is_file():
                raise TaskGenerationError(f"dataset file not found: {self.dataset_file}")
            return (self.dataset_file, self.dataset_file.stem,
                    self.source_url or self.dataset_file.resolve().as_uri(), None)
        if platform == "data_gov_hk":
            candidates = eligible_hk_ids()
            if not candidates:
                raise TaskGenerationError("DATA.GOV.HK catalog has no approved datasets")
            dataset_id = self.dataset_id or rng.choice(candidates)
            if dataset_id not in candidates:
                raise TaskGenerationError(f"DATA.GOV.HK dataset is not approved: {dataset_id}")
            source = verified_hk_source(dataset_id)
            if source is None:
                result = subprocess.run([
                    sys.executable, str(PROJECT / "scripts" / "diagnostics" / "download_data_gov_hk_dataset.py"),
                    dataset_id, "--max-bytes", str(self.max_source_bytes),
                ], capture_output=True, text=True, check=False)
                if result.returncode:
                    raise TaskGenerationError(
                        f"DATA.GOV.HK download failed for {dataset_id}: {result.stderr[-500:]}"
                    )
                source = verified_hk_source(dataset_id)
            if source is None:
                raise TaskGenerationError(f"DATA.GOV.HK source verification failed: {dataset_id}")
            return source
        if platform != "kaggle":
            raise TaskGenerationError(f"unsupported dataset platform: {platform}")
        ref = self.dataset_ref
        approved = None
        if ref is None:
            if not DEFAULT_ALLOWLIST.is_file():
                raise TaskGenerationError(f"approved dataset list not found: {DEFAULT_ALLOWLIST}")
            approved_sources = json.loads(DEFAULT_ALLOWLIST.read_text(encoding="utf-8"))["datasets"]
            if not approved_sources:
                raise TaskGenerationError("approved dataset list is empty")
            approved = rng.choice(approved_sources)
            ref = approved["ref"]
        if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ref):
            raise TaskGenerationError("dataset ref must be Kaggle owner/slug")
        source_dir = KAGGLE_ROOT / ref
        manifests = list(source_dir.glob("v*/source_manifest.json"))
        if not manifests:
            result = subprocess.run([
                sys.executable, str(PROJECT / "scripts" / "diagnostics" / "download_kaggle_dataset.py"),
                ref, "--max-bytes", str(self.max_source_bytes),
            ], capture_output=True, text=True, check=False)
            if result.returncode:
                raise TaskGenerationError(f"Kaggle download failed for {ref}: {result.stderr[-500:]}")
            manifests = list(source_dir.glob("v*/source_manifest.json"))
        if not manifests:
            raise TaskGenerationError(f"Kaggle download has no manifest: {ref}")
        if approved is not None:
            manifests = [path for path in manifests
                         if path.parent.name == f"v{approved['version']}"]
            if not manifests:
                raise TaskGenerationError(f"approved Kaggle version is unavailable: {ref}")
        manifest_path = max(manifests, key=lambda path: int(path.parent.name[1:]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_files = [manifest_path.parent / "raw" / entry["path"]
                     for entry in manifest.get("files", [])
                     if isinstance(entry.get("path"), str)
                     and source_extension(entry["path"]) in SUPPORTED_SOURCE_EXTENSIONS]
        if not source_files:
            raise TaskGenerationError(f"Kaggle dataset contains no supported table file: {ref}")
        if approved is not None:
            if manifest.get("license") != approved["license"]:
                raise TaskGenerationError(f"approved dataset license changed: {ref}")
            approved_hash = approved.get("source_sha256") or approved.get("csv_sha256")
            source_files = [path for path in source_files if _sha256(path) == approved_hash]
            if not source_files:
                raise TaskGenerationError(f"approved dataset source hash changed: {ref}")
        preference = {suffix: index for index, suffix in enumerate((
            ".csv", ".tsv", ".jsonl", ".ndjson", ".json", ".xlsx",
            ".parquet", ".sqlite", ".sqlite3", ".db",
            ".zip", ".gz", ".tar", ".tar.gz", ".tgz",
        ))}
        source_files.sort(key=lambda path: (preference[source_extension(path.name)], str(path)))
        return source_files[0], str(manifest.get("title") or ref), str(manifest["source"]), manifest.get("license")

    def generate(self, hops: int = 0, task_type: str | None = None,
                 task_style: str | None = None, artifact_dir: str | Path | None = None,
                 task_intent: str | None = None, training_category: str = "multi_step_agentic",
                 seed: int | None = None, dataset_platform: str = "kaggle",
                 graph_link: Any | None = None) -> Task:
        del hops
        if task_type and task_type != TaskType.QA.value:
            raise TaskGenerationError("dataset generation currently supports task type QA")
        rng = random.Random(seed)
        if graph_link and training_category == "multi_step_agentic":
            operations = (("average", "count") if task_intent == "calculate" else
                          ("minimum", "maximum") if task_intent == "query" else
                          ("minimum", "maximum", "average", "count"))
            extreme = rng.choice(operations)
        else:
            extreme = "minimum"
        comparison = rng.choice(("lower", "higher")) if graph_link and training_category == "direct_response" else "lower"
        expected_intent = ("compare" if training_category == "direct_response" else
                           "calculate" if extreme in {"average", "count"} else "query")
        if task_intent and task_intent != expected_intent:
            raise TaskGenerationError(f"{training_category} requires task intent {expected_intent}")
        source_error: TaskGenerationError | None = None
        for _ in range(1 if self.dataset_ref or self.dataset_id or self.dataset_file else 6):
            try:
                source, title, source_url, license_name = self._source(rng, dataset_platform)
                if graph_link and _sha256(source) != graph_link.source_sha256:
                    raise TaskGenerationError("graph link source hash differs from reviewed source")
                headers, original_rows = _sample_source(
                    source, max_source_bytes=getattr(self, "max_source_bytes", DEFAULT_MAX_SOURCE_BYTES)
                )
                key_col, group_col, numeric_col = _columns(headers, original_rows)
                if graph_link and graph_link.group_field:
                    if group_col != graph_link.group_field:
                        raise TaskGenerationError("graph link group field is no longer selected")
                    original_rows = [row for row in original_rows
                                     if row[group_col] == graph_link.group_value]
                    if len(original_rows) < 3:
                        raise TaskGenerationError("graph link group value is absent from source")
                distinct_rows = list({str(row[key_col]).strip(): row for row in reversed(original_rows)}.values())
                distinct_rows.reverse()
                selected = _choose_rows(distinct_rows, key_col, group_col, numeric_col,
                                        training_category, rng, extreme=extreme)
                break
            except TaskGenerationError as exc:
                source_error = exc
                if self.dataset_ref or self.dataset_id or self.dataset_file:
                    raise
                logger.warning("dataset candidate rejected before LLM: %s", exc)
        else:
            raise TaskGenerationError(f"no usable dataset after six candidates: {source_error}")
        rows = _project(selected, key_col, group_col, numeric_col)
        key, group, numeric = map(_column_name, (key_col, group_col, numeric_col))
        scene = _validate_scene(self.pipeline._call(
            "dataset_business_scenario",
            "你是业务分析员。根据数据集标题、可用字段与少量已筛选的样本值，判断这些行在真实业务中代表什么。"
            "只推断字段能支持的场景，不能臆造日期、客户、商品明细、退款、折扣或其他未提供事实。"
            "输出简短中文：entity_name 为一条业务数据的自然称呼，identifier_label 为标识的业务名称，"
            "group_label 和 value_label 为业务字段的自然称呼，user_role 和 business_situation 描述谁会在什么场景提出查询。"
            "entity_name 也不要包含‘记录’。不要在输出中使用数据集、CSV、表字段、沙箱或训练术语。",
            {"dataset_title": title,
             "reviewed_graph_scene": graph_link.scene_name if graph_link else None,
             "graph_relation_evidence": graph_link.evidence if graph_link else None,
             "available_columns": [key_col, group_col, numeric_col],
             "selected_fields": {"identifier": key_col, "group": group_col, "value": numeric_col},
             "safe_examples": _project(original_rows[:8], key_col, group_col, numeric_col),
             "output": {"entity_name": "string", "identifier_label": "string",
                        "group_label": "string", "value_label": "string",
                        "user_role": "string", "business_situation": "string"}},
        ))
        description, facts = _description(training_category, rows, key, group, numeric, title,
                                          scene, extreme=extreme, comparison=comparison)
        style = task_style or rng.choice(TaskGenerator.STYLES)
        if style not in TaskGenerator.STYLES:
            raise TaskGenerationError(f"unsupported task style: {style}")
        route_goal = ({"compare_two_visible_values": True, "report_lower_item": True,
                       "report_exact_difference": True} if training_category == "direct_response" else
                      {"find_current_item_group": True, "report_current_item_value": True}
                      if training_category == "simple_agentic" else
                      {"find_current_item_group": True, "find_same_group_minimum": True,
                       "report_minimum_item": True, "report_minimum_value": True})
        if graph_link and training_category == "multi_step_agentic" and extreme == "maximum":
            route_goal = {"find_current_item_group": True, "find_same_group_maximum": True,
                          "report_maximum_item": True, "report_maximum_value": True}
        if graph_link and training_category == "multi_step_agentic" and extreme == "average":
            route_goal = {"find_current_item_group": True, "calculate_same_group_average": True,
                          "report_average_value": True}
        if graph_link and training_category == "multi_step_agentic" and extreme == "count":
            route_goal = {"find_current_item_group": True, "count_same_group_items": True,
                          "report_group_count": True}
        if graph_link and training_category == "direct_response" and comparison == "higher":
            route_goal = {"compare_two_visible_values": True, "report_higher_item": True,
                          "report_exact_difference": True}
        design_system = (
            "你是该业务场景中的真实用户。根据 supported_business_need 和可支持的目标，"
            "用第一人称写两句自然中文：先说你正在做的业务核对或比较，再提出想知道的结果。"
            "运行界面已向助手展示 public_materials 中的业务信息，用户可以说‘这单’‘这笔’‘这两笔’，"
            "无需解释附件、凭证、数据集或编号。业务动机仅限 supported_business_need 的含义，"
            "不能引入营销、活动、利润、销售贡献等材料中没有的业务背景。"
            "不要写‘帮我查一下’‘帮我看看’‘再看看’等执行步骤，也不得写‘附上的凭证’‘附件’"
            "‘记录编号’、原始字段名、工具名、答案或评测术语。"
            "直接回答只比较两笔可见值，并遵循 supported_goal 中的比较方向；单工具只问当前一笔的分组和数值；多步需要当前分组、"
            "多步任务需围绕当前分组完成 supported_goal 指定的同组极值、平均值或数量目标；"
            "只有极值目标才询问对应的那一笔。不要用‘先…然后…’写成操作流程。"
            "只返回 JSON 对象 user_message。"
        )
        design_payload = {
            "style": style, "category": training_category,
            "reviewed_graph_scene": graph_link.scene_name if graph_link else None,
            "business_scenario": {name: scene[name] for name in
                                  ("entity_name", "identifier_label", "group_label", "value_label", "user_role")},
            "supported_business_need": _supported_business_need(training_category, scene,
                                                                  extreme=extreme, comparison=comparison),
            "supported_goal": description["task"], "required_outputs": route_goal,
            "public_materials": description["public_input"]["materials"],
            "output": {"user_message": "string"},
        }
        design_error: TaskGenerationError | None = None
        for attempt in range(3):
            voice = self.pipeline._call("dataset_task_design", design_system,
                                        {**design_payload, "revision_feedback": str(design_error or "")})
            message = str(voice.get("user_message") or "").strip()
            try:
                _validate_voice(message, description["task"], training_category,
                                rows, key, facts, title, scene, extreme=extreme,
                                comparison=comparison)
                break
            except TaskGenerationError as exc:
                design_error = exc
                logger.warning("dataset task design rejected: attempt=%d/3 reason=%s", attempt + 1, exc)
        else:
            raise design_error or TaskGenerationError("dataset task design failed")
        description["task"] = message
        description["public_input"]["initial_user_message"] = message
        source_record = {"source_url": source_url, "source_file": str(source),
                         "source_sha256": _sha256(source), "source_format": source_extension(source.name),
                         "source_member": selected_member(
                             source, max_source_bytes=getattr(self, "max_source_bytes", DEFAULT_MAX_SOURCE_BYTES)
                         ),
                         "license": license_name,
                         "business_scenario": scene,
                         "original_columns": [key_col, group_col, numeric_col],
                         "selected_ids": [row[key] for row in rows],
                         "verified_reward_facts": facts}
        if graph_link:
            source_record["graph_plan"] = {
                "scene": graph_link.scene_name, "dataset_key": graph_link.dataset_key,
                "evidence": graph_link.evidence, "group_field": graph_link.group_field,
                "group_value": graph_link.group_value, "extreme": extreme,
                "comparison": comparison,
            }
        hostname = (urlparse(source_url).hostname or "").lower()
        provider = ("kaggle" if hostname in {"kaggle.com", "www.kaggle.com"}
                    else "data_gov_hk" if hostname == "data.gov.hk"
                    else None)
        governance = {
            "origin": "public_dataset" if provider else "local_dataset_unverified",
            "provider": provider,
            "source_url": source_url,
            "source_sha256": source_record["source_sha256"],
            "source_format": source_record["source_format"],
            "source_member": source_record["source_member"],
            "license": license_name,
            "contains_real_user_data": "undetermined",
            "intended_use": "agentic_rl_training_material",
        }
        if artifact_dir:
            (Path(artifact_dir) / "source_selection.json").write_text(
                json.dumps(source_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        answer_facts = {name: value for name, value in facts.items()
                        if name not in ("record_id", "starting_record_id")}
        source_data: dict[str, Any] = {
            "data_governance": governance,
            "verified_reward_facts": facts,
            "verified_reward_criterion": "Correctly report the requested source-backed facts: "
                                         + json.dumps(answer_facts, ensure_ascii=False),
        }
        if training_category != "direct_response":
            source_data.update({
                "entities": [{"entity_id": "source_record", "name": title[:60],
                              "description": "One projected original source record",
                              "required_facts": [key, group, numeric], "relationships": []}],
                "data_tables": [{"table_name": "source_records", "description": f"Projected source rows from {title}",
                                 "columns": [{"name": key, "type": "TEXT", "nullable": False,
                                              "description": f"Original {key_col}"},
                                             {"name": group, "type": "TEXT", "nullable": False,
                                              "description": f"Original {group_col}"},
                                             {"name": numeric, "type": "REAL", "nullable": False,
                                              "description": f"Original {numeric_col}"}],
                                 "primary_key": [key], "foreign_keys": [], "indexes": [], "constraints": [],
                                 "rows": rows}],
                "data_document": (f"# {title}\n\nBusiness entity: {scene['entity_name']}; "
                                  f"group: {scene['group_label']}; value: {scene['value_label']}. "
                                  f"Original rows projected to {key_col}, {group_col}, {numeric_col}. "
                                  f"No business values are synthesized. Source: {source_url}. "
                                  f"License: {license_name or 'unverified'}. Reset each episode."),
            })
        artifacts = self.pipeline.generate(
            keywords=[title[:60], group_col, numeric_col], task_type=TaskType.QA.value,
            style=style, task_intent=expected_intent,
            graph_context={"dataset": title, "source_url": source_url,
                           "reviewed_scene": graph_link.scene_name if graph_link else None,
                           "dataset_key": graph_link.dataset_key if graph_link else None,
                           "business_scenario": {name: scene[name] for name in
                                                 ("entity_name", "identifier_label",
                                                  "group_label", "value_label")}},
            artifact_dir=artifact_dir, training_category=training_category, rng=rng,
            available_environment_modes=("stateless", "reference_data"),
            description_override=description, source_data=source_data,
        )
        final_message = artifacts.get("public_input", {}).get("initial_user_message", "")
        if final_message != message or artifacts.get("task") != message:
            raise TaskGenerationError("downstream pipeline changed the approved public business request")
        artifacts["dataset_source"] = {
            "source_url": source_url, "source_sha256": source_record["source_sha256"],
            "source_format": source_record["source_format"],
            "source_member": source_record["source_member"],
            "license": license_name, "provider": provider,
        }
        if graph_link:
            artifacts["graph_plan"] = source_record["graph_plan"]
        return Task(artifacts["task"], artifacts["environment"], artifacts["metrics"],
                    task_type=TaskType.QA, task_intent=artifacts["task_intent"],
                    complexity=artifacts["complexity"], artifacts=artifacts)
