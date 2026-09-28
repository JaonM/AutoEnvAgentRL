"""Generate source-backed tasks from datasets, with an optional legacy graph route."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
import logging
import os
import random
import re
import shutil
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from env_factory import (
    LLMClient,
    Neo4jGraphStore,
    PipelineGenerationError,
    TaskGenerationError,
    TaskGenerator,
    TaskType,
)
from env_factory.tasks.task_routing import (
    TRAINING_CATEGORIES,
    allocate_training_routes,
    compatible_training_categories,
    parse_training_mix,
    select_training_intent,
)
from env_factory.evidence.data_governance import provider_identity
from env_factory.llm import capture_llm_trace, summarize_llm_trace
from env_factory.generation.dataset_task_generator import DatasetTaskGenerator


_TASK_DIR_PATTERN = re.compile(r"task-(\d+)")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_task_artifact(task_dir: Path, task: object, training_category: str) -> Path:
    """Materialize the candidate before its buildability gate runs."""
    pipeline_artifacts = task.artifacts or {}
    artifact_manifest = {
        key: pipeline_artifacts[key]
        for key in (
            "data_manifest", "user_simulation_manifest", "tools_manifest",
            "media_generation", "generation_pipeline", "dataset_source",
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


def _reset_reserved_directory(path: Path) -> None:
    """Remove partial artifacts while preserving the experiment sample identity."""
    for child in path.iterdir():
        if child.name == "sample_manifest.json":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _validate_generated_candidate(task_dir: Path) -> None:
    """Reject a generated candidate before it consumes a production sample slot."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from scripts.sandbox.assess_task_buildability import assess

    result = assess(task_dir)
    if result.get("buildable"):
        return
    codes = sorted({
        str(issue.get("code", "TASK_BUILDABILITY"))
        for issue in result.get("issues", []) if isinstance(issue, dict)
    })
    raise TaskGenerationError(
        "task buildability gate failed: " + ", ".join(codes or ["UNKNOWN"])
    )


def _generation_failure_class(exc: BaseException) -> str:
    """Observability taxonomy only; it never changes generation behavior."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name in {"ServiceUnavailable", "SessionExpired", "ConnectionError", "TimeoutError"}:
        return "INFRA"
    if any(marker in text for marker in ("schema", "must be", "requires", "invalid", "non-empty list")):
        return "GEN_SCHEMA"
    if any(marker in text for marker in ("external capability", "buildability", "unsupported", "missing input")):
        return "TASK_BUILDABILITY"
    if isinstance(exc, TaskGenerationError) and ("path" in text or "keyword" in text):
        return "INPUT_SAMPLING"
    return "GEN_SEMANTIC"


def _existing_task_numbers(artifact_root: Path) -> list[int]:
    if not artifact_root.is_dir():
        return []
    numbers: list[int] = []
    for path in artifact_root.iterdir():
        match = _TASK_DIR_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _reserve_task_directories(artifact_root: Path, count: int) -> list[tuple[int, Path]]:
    """Atomically reserve monotonically increasing task directories.

    ``mkdir(exist_ok=False)`` also prevents two concurrently running CLI
    processes from selecting the same task number.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidate = max(_existing_task_numbers(artifact_root), default=0) + 1
    reserved: list[tuple[int, Path]] = []
    while len(reserved) < count:
        task_dir = artifact_root / f"task-{candidate}"
        try:
            task_dir.mkdir()
        except FileExistsError:
            candidate += 1
            continue
        reserved.append((candidate, task_dir))
        candidate += 1
    return reserved


def main() -> int:
    parser = argparse.ArgumentParser(description="从数据集生成 Agentic RL 任务")
    parser.add_argument("--generation-source", choices=("dataset", "graph"), default="dataset",
                        help="默认从数据集生成；graph 为旧知识图谱路径")
    parser.add_argument("--dataset-ref", help="指定 Kaggle owner/slug；缺省时从已核验清单选取")
    parser.add_argument("--dataset-file", type=Path, help="指定本地表格或 ZIP/GZIP/TAR 压缩文件")
    parser.add_argument("--dataset-url", help="本地数据文件的原始数据集来源链接")
    parser.add_argument("--dataset-platform", choices=("kaggle", "data_gov_hk", "balanced"),
                        default="kaggle", help="数据集来源；balanced 要求偶数任务并按 1:1 分配")
    parser.add_argument("--dataset-max-gb", type=float, default=5.0,
                        help="数据源压缩文件、单个表格及解压内容的大小上限，默认 5 GB")
    parser.add_argument("--hops", type=int, default=3, help="随机路径最大跳数，实际范围为 0 到该值，默认 3")
    parser.add_argument("--count", type=int, default=1, help="生成任务数量，默认 1")
    parser.add_argument("--max-workers", type=int, default=4, help="任务生成并发数，默认 4")
    parser.add_argument(
        "--path-query-timeout",
        type=float,
        default=10.0,
        help="Neo4j 随机路径查询超时时间（秒），默认 10.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="任务输出根路径；每次运行在 task_artifacts 下追加新的 task-N，默认 output",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("output/task_generation.log"),
        help="任务生成日志文件，默认 output/task_generation.log",
    )
    parser.add_argument(
        "--task-type",
        help="任务类型；支持逗号分隔多选，例如 QA,Event；默认从全部类型随机选择",
    )
    parser.add_argument(
        "--task-style",
        choices=TaskGenerator.STYLES,
        help="任务表达风格；默认随机选择",
    )
    parser.add_argument(
        "--task-intent",
        choices=TaskGenerator.INTENTS,
        help="任务意图；默认随机选择。可选：query、explain、compare、recommend、diagnose、modify、execute、plan、summarize、create、extract、classify、validate、audit、calculate、estimate、schedule、monitor、troubleshoot、transform、decide、simulate",
    )
    parser.add_argument(
        "--user-script-count",
        type=int,
        default=3,
        help="每个任务生成的用户 FSM 数量，默认 3",
    )
    parser.add_argument(
        "--noise-tool-max",
        type=int,
        default=3,
        help="每个任务最多生成的噪声工具数量，实际数量随机为 0..N；噪声工具由共享运行时安全执行，默认 3",
    )
    parser.add_argument("--training-category", choices=TRAINING_CATEGORIES, help="固定全部任务的训练路由类别")
    parser.add_argument(
        "--training-mix",
        default="direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50",
        help="批次训练类别比例，默认 20/30/50",
    )
    parser.add_argument("--route-attempts", type=int, default=3, help="每个训练路由候选最多重采样次数，默认 3")
    parser.add_argument("--seed", type=int, help="实验随机种子；省略时生成并记录一个随机种子")
    parser.add_argument("--stage-cache-dir", type=Path, help="可选的阶段检查点目录；相同模型、代码、提示和输入复用结果，并重新执行语义校验")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.max_workers <= 0:
        parser.error("--max-workers 必须大于 0")
    if args.path_query_timeout <= 0:
        parser.error("--path-query-timeout 必须大于 0")
    if args.noise_tool_max < 0:
        parser.error("--noise-tool-max 不能小于 0")
    if args.route_attempts <= 0:
        parser.error("--route-attempts 必须大于 0")
    if not 0 < args.dataset_max_gb <= 100:
        parser.error("--dataset-max-gb 必须大于 0 且不超过 100")
    if args.generation_source == "dataset" and args.dataset_ref and args.dataset_file:
        parser.error("--dataset-ref 与 --dataset-file 不能同时使用")
    if args.generation_source == "dataset" and args.dataset_platform == "balanced":
        if args.count % 2 or args.dataset_ref or args.dataset_file:
            parser.error("balanced 需要偶数任务且不能指定单个数据集")
    if args.generation_source == "dataset" and args.dataset_platform == "data_gov_hk" and args.dataset_ref:
        parser.error("DATA.GOV.HK 来源不能使用 Kaggle --dataset-ref")
    if args.generation_source == "dataset" and args.task_type not in (None, "QA"):
        parser.error("数据集生成目前只支持 --task-type QA")
    if args.generation_source == "dataset" and args.task_intent not in (None, "query", "compare"):
        parser.error("数据集生成目前只支持 query 或 compare 意图")
    if args.generation_source == "graph" and (args.dataset_ref or args.dataset_file or args.dataset_url):
        parser.error("graph 路径不接受数据集参数")
    try:
        training_mix = parse_training_mix(args.training_mix)
    except ValueError as exc:
        parser.error(str(exc))
    if args.training_category and args.task_intent:
        try:
            select_training_intent(args.training_category, args.task_intent)
        except ValueError as exc:
            parser.error(str(exc))
    load_dotenv()
    if args.stage_cache_dir:
        os.environ["ENVFACTORY_STAGE_CACHE_DIR"] = str(args.stage_cache_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    external_fixture = os.getenv("SANDBOX_EXTERNAL_FIXTURES", "").strip()
    external_available = bool(os.getenv("SANDBOX_EXTERNAL_CAPABILITY_URL", "").strip()) or bool(
        external_fixture and Path(external_fixture).is_file()
    )
    available_environment_modes = (("stateless", "reference_data") if args.generation_source == "dataset"
                                   else ("stateless", "reference_data", "stateful") + (
                                       ("external_capability",) if external_available else ()))
    logging.getLogger(__name__).info(
        "task generation run started: count=%d max_workers=%d output=%s log_file=%s",
        args.count, args.max_workers, args.output, args.log_file,
    )
    task_root = args.output if args.output.suffix == "" else args.output.parent
    task_root.mkdir(parents=True, exist_ok=True)
    artifact_root = task_root / "task_artifacts"
    store_context = (Neo4jGraphStore(
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        path_query_timeout=args.path_query_timeout,
    ) if args.generation_source == "graph" else nullcontext(None))
    with store_context as store:
        generator = (TaskGenerator(
            store, llm, user_script_count=args.user_script_count,
            noise_tool_max=args.noise_tool_max,
            available_environment_modes=available_environment_modes,
        ) if args.generation_source == "graph" else DatasetTaskGenerator(
            llm, dataset_ref=args.dataset_ref, dataset_file=args.dataset_file,
            source_url=args.dataset_url, max_source_bytes=int(args.dataset_max_gb * 1_000_000_000),
            user_script_count=args.user_script_count,
            noise_tool_max=args.noise_tool_max,
        ))
        run_seed = args.seed if args.seed is not None else secrets.randbits(63)
        run_rng = random.Random(run_seed)
        reserved_tasks = _reserve_task_directories(artifact_root, args.count)
        routes = (
            [args.training_category] * args.count
            if args.training_category else allocate_training_routes(
                args.count,
                training_mix,
                allowed_categories=(
                    compatible_training_categories(args.task_intent)
                    if args.task_intent else None
                ),
                rng=run_rng,
            )
        )
        sample_seeds = [run_rng.randrange(0, 2**63) for _ in reserved_tasks]
        if args.generation_source == "dataset":
            from env_factory.generation.dataset_source_registry import balanced_platforms
            platforms = (balanced_platforms(args.count, rng=run_rng)
                         if args.dataset_platform == "balanced"
                         else [args.dataset_platform] * args.count)
        else:
            platforms = [None] * args.count
        generation_provider = provider_identity(llm.base_url, llm.model)
        for batch_index, ((task_number, task_dir), training_category, sample_seed, platform) in enumerate(
            zip(reserved_tasks, routes, sample_seeds, platforms), start=1
        ):
            _write_json(task_dir / "sample_manifest.json", {
                "version": "2.0",
                "task_id": f"task-{task_number}",
                "batch_index": batch_index,
                "run_seed": run_seed,
                "sample_seed": sample_seed,
                "training_category": training_category,
                "requested_task_intent": args.task_intent,
                "requested_task_style": args.task_style,
                "requested_task_type": args.task_type,
                "generation_source": args.generation_source,
                "dataset_platform": platform,
                "dataset_ref": args.dataset_ref,
                "dataset_file": str(args.dataset_file) if args.dataset_file else None,
                "hops": args.hops if args.generation_source == "graph" else None,
                "available_environment_modes": list(available_environment_modes),
                "generator_provider": generation_provider,
                "generation_settings": {
                    "route_attempt_limit": args.route_attempts,
                    "timeout_seconds": llm.timeout,
                    "network_retries": llm.network_retries,
                },
                "status": "reserved",
                "attempts": [],
            })
        logging.getLogger(__name__).info(
            "reserved incremental task ids: %s",
            [task_number for task_number, _ in reserved_tasks],
        )
        with ThreadPoolExecutor(max_workers=min(args.max_workers, args.count)) as executor:
            def generate_one(
                batch_index: int, task_number: int, task_dir: Path,
                training_category: str, sample_seed: int, platform: str | None,
            ):
                logging.getLogger(__name__).info(
                    "task generation started: batch=%d/%d task_id=task-%d",
                    batch_index, args.count, task_number,
                )
                last_error = None
                for route_attempt in range(1, args.route_attempts + 1):
                    manifest_path = task_dir / "sample_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    attempt_seed = sample_seed + route_attempt - 1
                    llm_trace = []
                    try:
                        with capture_llm_trace() as llm_trace:
                            task = generator.generate(
                                args.hops,
                                args.task_type,
                                args.task_style,
                                artifact_dir=task_dir,
                                task_intent=args.task_intent,
                                training_category=training_category,
                                seed=attempt_seed,
                                **({"dataset_platform": platform} if platform else {}),
                            )
                        _write_task_artifact(task_dir, task, training_category)
                        _validate_generated_candidate(task_dir)
                        manifest["attempts"].append({
                            "attempt": route_attempt,
                            "seed": attempt_seed,
                            "status": "completed",
                            "llm_trace": summarize_llm_trace(llm_trace),
                        })
                        manifest["successful_attempt"] = route_attempt
                        manifest["status"] = "generated"
                        _write_json(manifest_path, manifest)
                        return task
                    except (TaskGenerationError, PipelineGenerationError) as exc:
                        last_error = exc
                        manifest["attempts"].append({
                            "attempt": route_attempt,
                            "seed": attempt_seed,
                            "status": "rejected",
                            "failure_class": _generation_failure_class(exc),
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:2000],
                            "llm_trace": summarize_llm_trace(llm_trace),
                        })
                        manifest["status"] = "retrying" if route_attempt < args.route_attempts else "failed"
                        _write_json(manifest_path, manifest)
                        logging.getLogger(__name__).warning(
                            "training route candidate rejected: task_id=task-%d category=%s attempt=%d/%d reason=%s",
                            task_number, training_category, route_attempt, args.route_attempts, exc,
                        )
                        _reset_reserved_directory(task_dir)
                raise TaskGenerationError(
                    f"{training_category} route exhausted {args.route_attempts} candidate(s): {last_error}"
                )

            futures = {
                executor.submit(
                    generate_one, batch_index, task_number, task_dir,
                    routes[batch_index - 1], sample_seeds[batch_index - 1], platforms[batch_index - 1]
                ): (
                    batch_index, task_number, task_dir, routes[batch_index - 1]
                )
                for batch_index, (task_number, task_dir) in enumerate(reserved_tasks, start=1)
            }
            failed = 0
            for completed, future in enumerate(as_completed(futures), start=1):
                batch_index, task_number, task_dir, training_category = futures[future]
                try:
                    task = future.result()
                except TaskGenerationError as exc:
                    failed += 1
                    failure = {
                        "failure_class": _generation_failure_class(exc),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:4000],
                        "training_category": training_category,
                        "batch_index": batch_index,
                    }
                    _write_json(task_dir / "failure.json", failure)
                    logging.getLogger(__name__).warning(
                        "task generation skipped: batch=%d/%d task_id=task-%d reason=%s",
                        batch_index, args.count, task_number, exc,
                    )
                    continue
                except Exception as exc:
                    failed += 1
                    manifest_path = task_dir / "sample_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["status"] = "failed"
                    manifest["attempts"].append({
                        "attempt": len(manifest.get("attempts", [])) + 1,
                        "status": "failed",
                        "failure_class": _generation_failure_class(exc),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:2000],
                    })
                    _write_json(manifest_path, manifest)
                    _write_json(task_dir / "failure.json", {
                        "failure_class": _generation_failure_class(exc),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:4000],
                        "training_category": training_category,
                        "batch_index": batch_index,
                    })
                    logging.getLogger(__name__).exception(
                        "task generation failed: batch=%d/%d task_id=task-%d",
                        batch_index, args.count, task_number,
                    )
                    continue
                task_path = task_dir / "task.json"
                manifest_path = task_dir / "sample_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update({
                    "status": "completed",
                    "resolved_task_intent": task.task_intent,
                    "complexity": task.complexity,
                    "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                _write_json(manifest_path, manifest)
                logging.getLogger(__name__).info(
                    "task artifact written: batch=%d/%d task_id=task-%d complexity=%s task_file=%s",
                    batch_index, args.count, task_number, task.complexity, task_path,
                )
                logging.getLogger(__name__).info(
                    "task generation progress: completed=%d/%d success=%d failed=%d",
                    completed, args.count, completed - failed, failed,
                )
    logging.getLogger(__name__).info(
        "task generation run completed: success=%d failed=%d output=%s",
        args.count - failed, failed, task_root,
    )
    print(f"任务生成结束：成功={args.count - failed}，失败={failed}，目录={task_root / 'task_artifacts'}，日志={args.log_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
