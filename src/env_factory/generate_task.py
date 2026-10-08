"""Generate tasks through Scene paths with randomly sampled hop counts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import random
import re
import shutil
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from env_factory import (
    Neo4jGraphStore,
    PipelineGenerationError,
    TaskGenerationError,
    TaskGenerator,
)
from env_factory.tasks.task_routing import (
    TRAINING_CATEGORIES,
    allocate_training_routes,
    compatible_training_categories,
    parse_training_mix,
    select_training_intent,
    training_contract,
)
from env_factory.evidence.data_governance import provider_identity
from env_factory.llm import capture_llm_trace, summarize_llm_trace
from env_factory.generation.pipeline_errors import is_transient_llm_error


_TASK_DIR_PATTERN = re.compile(r"artifact-(\d+)")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


from env_factory.generation.artifacts import write_task_artifact as _write_task_artifact


def _archive_rejected_candidate(task_dir: Path, attempt: int) -> Path:
    """Keep complete rejected evidence outside the next candidate's delivery."""
    archive = task_dir.parent / 'rejected_generations' / f'attempt-{attempt}'
    archive.mkdir(parents=True, exist_ok=False)
    for child in task_dir.iterdir():
        if child.name == 'sample_manifest.json':
            shutil.copy2(child, archive / child.name)
        else:
            child.rename(archive / child.name)
    return archive


def _retry_seed_blocks(rng: random.Random, count: int, attempts: int) -> list[int]:
    """Reserve disjoint seed ranges for every candidate and its retries."""
    if attempts <= 0 or count < 0:
        raise ValueError("count and route attempts must be valid")
    slots = (2**63 - 1) // attempts
    if count > slots:
        raise ValueError("too many task candidates for disjoint retry seed ranges")
    return [slot * attempts for slot in rng.sample(range(slots), count)]


def _validate_generated_candidate(task_dir: Path) -> None:
    """Reject a generated candidate before it consumes a production sample slot."""
    project_root = str(Path(__file__).resolve().parents[2])
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
        + "; issues=" + json.dumps(result.get('issues', []), ensure_ascii=False)
    )


def _generation_failure_history(task_dir: Path) -> list[dict]:
    """Retain validation causes when a later timeout becomes the terminal error."""
    try:
        evidence = json.loads((task_dir / "code_agent_generation.json").read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(evidence, dict):
        return []
    return [{key: attempt[key] for key in
             ("index", "status", "error", "issue_ids", "repair_stop_reason") if key in attempt}
            for attempt in evidence.get("attempts", [])
            if isinstance(attempt, dict) and attempt.get("error")]


def _generation_failure_class(exc: BaseException) -> str:
    """Observability taxonomy only; it never changes generation behavior."""
    name = type(exc).__name__
    text = str(exc).lower()
    if 'code_agent_capacity' in text:
        return 'INFRA'
    if 'code_agent_startup_stalled' in text:
        return 'GEN_STARTUP_STALL'  # No candidate evidence; provider cause is unknown.
    if "code_agent_timeout" in text:
        return "GEN_BUDGET"
    if "source_semantic_review_failed" in text or "open_explanation_exact_match" in text:
        return "GEN_SEMANTIC"
    # These prefixes describe a missing/unusable independent judgment, not a
    # defect established in the candidate. Check before generic "invalid" or
    # "schema" matching: reviewer output can itself violate its report schema.
    if "source_review_" in text or "semantic_calibration_unavailable" in text:
        return "INFRA"
    if any(is_transient_llm_error(error) for error in (
        exc, exc.__cause__, exc.__context__,
    ) if isinstance(error, Exception)):
        return "INFRA"
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
        artifact_dir = artifact_root / f"artifact-{candidate}"
        task_dir = artifact_dir / "task"
        try:
            artifact_dir.mkdir()
        except FileExistsError:
            candidate += 1
            continue
        task_dir.mkdir()
        reserved.append((candidate, task_dir))
        candidate += 1
    return reserved


def main() -> int:
    parser = argparse.ArgumentParser(description="从多跳 Scene 图谱生成 Agentic RL 任务")
    parser.add_argument("--hops", type=int, default=3, help="Scene 随机跳数上限 N，从 0..N 均匀抽取；N 范围 0..20，默认 3")
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
        help="任务输出根路径；每次运行在 artifacts/artifact-N/task 下追加新任务，默认 output",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("output/task_generation.log"),
        help="任务生成日志文件，默认 output/task_generation.log",
    )
    parser.add_argument(
        "--task-type",
        help="任务类型；支持逗号分隔多选，例如 QA,Event；默认随机选择",
    )
    parser.add_argument(
        "--task-style",
        choices=TaskGenerator.STYLES,
        help="任务表达风格；默认随机选择",
    )
    parser.add_argument(
        "--task-intent",
        choices=TaskGenerator.INTENTS,
        help="任务意图；默认按训练类别选择",
    )
    parser.add_argument(
        "--user-script-count",
        type=int,
        default=3,
        help="每个任务生成的用户 FSM 数量，默认 3",
    )
    parser.add_argument("--training-category", choices=TRAINING_CATEGORIES, help="固定全部任务的训练路由类别")
    parser.add_argument('--difficulty-mix', default='standard=1',
                        help='难度分配比例，如 standard=4,long_horizon_planning=2,large_scale_retrieval=2,complex_write_transactions=2')
    parser.add_argument("--environment-mode", choices=("stateless", "reference_data", "stateful", "external_capability"),
                        help="要求实际环境模式；验证写入任务时使用 stateful，不能仅依赖 modify 意图标签")
    parser.add_argument("--code-agent", choices=("codex", "claude", "opencode"), default="codex")
    parser.add_argument("--code-agent-model", default=None)
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--code-agent-timeout", type=float, default=600,
                        help="单样本 Code Agent 防卡死超时（秒），默认 600；5 分钟不是质量淘汰线")
    parser.add_argument(
        "--training-mix",
        default="direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50",
        help="批次训练类别比例，默认 20/30/50",
    )
    parser.add_argument("--route-attempts", type=int, default=3, help="每个训练路由候选最多重采样次数，默认 3")
    parser.add_argument("--seed", type=int, help="随机种子；省略时生成并记录一个随机种子")
    args = parser.parse_args()
    if not args.code_agent_model:
        if args.code_agent != "codex":
            parser.error("--code-agent-model is required for claude/opencode")
        args.code_agent_model = "gpt-6.1-sol"
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.max_workers <= 0:
        parser.error("--max-workers 必须大于 0")
    if args.path_query_timeout <= 0:
        parser.error("--path-query-timeout 必须大于 0")
    if args.route_attempts <= 0:
        parser.error("--route-attempts 必须大于 0")
    if not 0 <= args.hops <= 20:
        parser.error("--hops 上限必须在 0 到 20 之间")
    if args.task_type:
        try:
            TaskGenerator._select_task_type(args.task_type)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        training_mix = parse_training_mix(args.training_mix)
        from env_factory.generation.difficulty_profiles import parse_mix as parse_difficulty_mix
        difficulty_mix = parse_difficulty_mix(args.difficulty_mix)
    except ValueError as exc:
        parser.error(str(exc))
    if args.environment_mode:
        routes = [args.training_category] if args.training_category else [
            category for category, weight in training_mix.items() if weight > 0]
        if any(args.environment_mode not in training_contract(category)["allowed_environment_modes"] for category in routes):
            parser.error("requested environment mode is incompatible with the training categories")
    if args.training_category and args.task_intent:
        try:
            select_training_intent(args.training_category, args.task_intent)
        except ValueError as exc:
            parser.error(str(exc))
    load_dotenv()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    external_fixture = os.getenv("SANDBOX_EXTERNAL_FIXTURES", "").strip()
    external_available = bool(os.getenv("SANDBOX_EXTERNAL_CAPABILITY_URL", "").strip()) or bool(
        external_fixture and Path(external_fixture).is_file()
    )
    available_environment_modes = ("stateless", "reference_data", "stateful") + (
        ("external_capability",) if external_available else ()
    )
    if args.environment_mode:
        if args.environment_mode not in available_environment_modes:
            parser.error("requested environment mode is unavailable")
        available_environment_modes = (args.environment_mode,)
    logging.getLogger(__name__).info(
        "task generation run started: count=%d max_workers=%d output=%s log_file=%s",
        args.count, args.max_workers, args.output, args.log_file,
    )
    task_root = args.output if args.output.suffix == "" else args.output.parent
    task_root.mkdir(parents=True, exist_ok=True)
    artifact_root = task_root / "artifacts"
    store_context = Neo4jGraphStore(
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        path_query_timeout=args.path_query_timeout,
    )
    with store_context as store:
        generator = TaskGenerator(
            store, user_script_count=args.user_script_count,
            available_environment_modes=available_environment_modes,
            code_agent_timeout=args.code_agent_timeout,
            code_agent_model=args.code_agent_model, language=args.language, code_agent=args.code_agent,
        )
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
        from env_factory.generation.realism_requirements import allocate_realism_requirements
        realism = allocate_realism_requirements(routes, available_modes=available_environment_modes, rng=run_rng)
        from env_factory.generation.difficulty_profiles import allocate as allocate_difficulty, augment_realism, requirements as difficulty_requirements
        profiles = allocate_difficulty(routes, difficulty_mix, rng=run_rng,
            available_modes=available_environment_modes, task_intent=args.task_intent)
        realism = [augment_realism({'difficulty_profile': profile, 'training_category': category,
            'available_environment_modes': available_environment_modes, 'task_intent': args.task_intent,
            'realism_requirements': assigned}) for profile, category, assigned in zip(profiles, routes, realism)]
        sample_seeds = _retry_seed_blocks(
            run_rng, len(reserved_tasks), args.route_attempts,
        )
        generation_provider = provider_identity(f"{args.code_agent}://cli", args.code_agent_model)
        for batch_index, ((task_number, task_dir), training_category, sample_seed) in enumerate(
            zip(reserved_tasks, routes, sample_seeds), start=1
        ):
            _write_json(task_dir / "sample_manifest.json", {
                "version": "2.0",
                "task_id": f"task-{task_number}",
                "batch_index": batch_index,
                "run_seed": run_seed,
                "sample_seed": sample_seed,
                "training_category": training_category,
                "realism_requirements": realism[batch_index - 1],
                "difficulty_profile": profiles[batch_index - 1],
                "difficulty_requirements": difficulty_requirements(profiles[batch_index - 1]),
                "requested_task_intent": args.task_intent,
                "requested_task_style": args.task_style,
                "requested_task_type": args.task_type,
                "generation_source": "graph_scene_path",
                "generation_backend": "code_agent",
                "max_hops": args.hops,
                "available_environment_modes": list(available_environment_modes),
                "generator_provider": generation_provider,
                "generation_settings": {
                    "route_attempt_limit": args.route_attempts,
                    "timeout_seconds": args.code_agent_timeout,
                    "network_retries": 0,
                },
                "status": "reserved",
                "reserved_at": datetime.now(timezone.utc).isoformat(),
                "attempts": [],
            })
        logging.getLogger(__name__).info(
            "reserved incremental task ids: %s",
            [task_number for task_number, _ in reserved_tasks],
        )
        with ThreadPoolExecutor(max_workers=min(args.max_workers, args.count)) as executor:
            def generate_one(
                batch_index: int, task_number: int, task_dir: Path,
                training_category: str, sample_seed: int,
            ):
                logging.getLogger(__name__).info(
                    "task generation started: batch=%d/%d task_id=task-%d",
                    batch_index, args.count, task_number,
                )
                sample_started = time.monotonic()
                last_error = None
                resume_checkpoint = None
                for route_attempt in range(1, args.route_attempts + 1):
                    attempt_started = time.monotonic()
                    manifest_path = task_dir / "sample_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    attempt_seed = sample_seed + route_attempt - 1
                    if resume_checkpoint:
                        from env_factory.generation.authoring_checkpoint import load_checkpoint
                        attempt_seed = load_checkpoint(resume_checkpoint)['request'].get('seed', sample_seed)
                    llm_trace = []
                    author_root = task_dir if resume_checkpoint is None else task_dir / 'generation_retries' / f'attempt-{route_attempt}'
                    try:
                        with capture_llm_trace() as llm_trace:
                            task = generator.resume(resume_checkpoint, artifact_dir=author_root) if resume_checkpoint else generator.generate(
                                args.hops,
                                args.task_type,
                                args.task_style,
                                artifact_dir=author_root,
                                task_intent=args.task_intent,
                                training_category=training_category,
                                seed=attempt_seed,
                                realism_requirements=realism[batch_index - 1],
                                difficulty_profile=profiles[batch_index - 1],
                            )
                        if resume_checkpoint:
                            # Manifest paths stay relative to task.json. Keep the
                            # old data as evidence before publishing resumed data.
                            import shutil
                            previous = author_root / 'previous_delivery'
                            for name in ('data', 'tools.json'):
                                incoming, target = author_root / name, task_dir / name
                                if incoming.exists():
                                    if target.exists():
                                        previous.mkdir(exist_ok=True)
                                        target.rename(previous / name)
                                    if incoming.is_dir():
                                        shutil.copytree(incoming, target)
                                    else:
                                        shutil.copy2(incoming, target)
                        _write_task_artifact(task_dir, task, training_category)
                        _validate_generated_candidate(task_dir)
                        manifest["attempts"].append({
                            "attempt": route_attempt,
                            "seed": attempt_seed,
                            "status": "completed",
                            "continued_fixed_design": resume_checkpoint is not None,
                            "llm_trace": summarize_llm_trace(llm_trace),
                            "duration_seconds": round(time.monotonic() - attempt_started, 3),
                        })
                        manifest["successful_attempt"] = route_attempt
                        manifest["status"] = "generated"
                        manifest["generation_seconds"] = round(time.monotonic() - sample_started, 3)
                        _write_json(manifest_path, manifest)
                        return task
                    except (TaskGenerationError, PipelineGenerationError) as exc:
                        last_error = exc
                        manifest["attempts"].append({
                            "attempt": route_attempt,
                            "seed": attempt_seed,
                            "status": "rejected",
                            "continued_fixed_design": resume_checkpoint is not None,
                            "failure_class": _generation_failure_class(exc),
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:2000],
                            "llm_trace": summarize_llm_trace(llm_trace),
                            "duration_seconds": round(time.monotonic() - attempt_started, 3),
                        })
                        structural_failure = str(exc).startswith(("SPEC_", "CODE_AGENT_"))
                        checkpoint_path = author_root / 'authoring_checkpoint' / 'manifest.json'
                        resumable = (checkpoint_path.is_file() and any(code in str(exc) for code in (
                            'CODE_AGENT_TIMEOUT:', 'CODE_AGENT_EXEC_FAILED:', 'CODE_AGENT_NO_COMPLETION:')))
                        if resumable:
                            from env_factory.generation.authoring_checkpoint import load_checkpoint
                            saved = load_checkpoint(checkpoint_path)
                            resumable = saved['manifest']['source_attempts'] < 3
                        manifest['attempts'][-1]['resume_checkpoint'] = str(checkpoint_path) if resumable else None
                        manifest["status"] = "retrying" if route_attempt < args.route_attempts and (resumable or not structural_failure) else "failed"
                        if manifest['status'] == 'failed':
                            manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
                        manifest["generation_seconds"] = round(time.monotonic() - sample_started, 3)
                        _write_json(manifest_path, manifest)
                        logging.getLogger(__name__).warning(
                            "training route candidate rejected: task_id=task-%d category=%s attempt=%d/%d reason=%s",
                            task_number, training_category, route_attempt, args.route_attempts, exc,
                        )
                        if resumable and route_attempt < args.route_attempts:
                            resume_checkpoint = checkpoint_path
                            continue
                        if structural_failure:
                            raise TaskGenerationError(str(exc)) from exc
                        if route_attempt < args.route_attempts:
                            _archive_rejected_candidate(task_dir, route_attempt)
                            resume_checkpoint = None
                raise TaskGenerationError(
                    f"{training_category} route exhausted {args.route_attempts} candidate(s): {last_error}"
                )

            # Preserve reserved IDs and seeds while giving the scarce model
            # workers to the Agentic curriculum first. The main pipeline
            # consumes completed samples as soon as they materialize.
            generation_order = sorted(
                enumerate(reserved_tasks, start=1),
                key=lambda item: ({"multi_step_agentic": 0, "simple_agentic": 1,
                                   "direct_response": 2}.get(routes[item[0] - 1], 3), item[0]),
            )
            futures = {
                executor.submit(
                    generate_one, batch_index, task_number, task_dir,
                    routes[batch_index - 1], sample_seeds[batch_index - 1]
                ): (
                    batch_index, task_number, task_dir, routes[batch_index - 1]
                )
                for batch_index, (task_number, task_dir) in generation_order
            }
            failed = 0
            for completed, future in enumerate(as_completed(futures), start=1):
                batch_index, task_number, task_dir, training_category = futures[future]
                try:
                    task = future.result()
                except TaskGenerationError as exc:
                    failed += 1
                    manifest_path = task_dir / 'sample_manifest.json'
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    manifest['status'] = 'failed'
                    manifest.setdefault('completed_at', datetime.now(timezone.utc).isoformat())
                    _write_json(manifest_path, manifest)
                    failure = {
                        "failure_class": _generation_failure_class(exc),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:4000],
                        "failure_history": _generation_failure_history(task_dir),
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
                    manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
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
                        "failure_history": _generation_failure_history(task_dir),
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
                pipeline = (task.artifacts or {}).get("generation_pipeline", {})
                if pipeline.get("backend") == "code_agent":
                    manifest["code_agent_provenance"] = {key: pipeline[key] for key in (
                        "model", "agent_invocations", "completed_turns", "events_sha256",
                        "source_sha256", "request_sha256", "compiler_sha256",
                    )}
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
    print(f"任务生成结束：成功={args.count - failed}，失败={failed}，目录={artifact_root}，日志={args.log_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
