"""Training configuration and unchanged command-line defaults."""

import math
from dataclasses import dataclass


@dataclass
class Config:
    sandbox: str = ""
    output: str = ""
    tasks: str = ""
    eval_episodes: int = 3
    eval_interval_steps: int = 100
    metrics_export_interval: int = 0
    group_artifacts_keep: int = 256
    zero_variance_policy: str = "retry_skip"
    # GRPO advantage scale: "group" (std of the group) or "task" (std of the recent rewards of the same
    # task across groups, AutoForge ERPO: less noise with small groups; falls back to the group early on)
    advantage_std: str = "task"
    # GRPO loss aggregation: "sequence" (original GRPO: each episode's tokens share the episode's weight) or "token"
    # (DAPO: every token of a group weighs the same, so long failing loops are penalised token by token).
    loss_aggregation: str = "sequence"
    # SPO's probability mask: the advantage applies only to tokens the behaviour policy sampled with probability
    # below this (0 = off; 0.9 in SPO), renormalised over those tokens; near-certain tokens (JSON, set phrases) carry
    # no credit.
    low_prob_mask: float = 0.0
    # rollout_mode "tree": rollout_group trunks, then tree_forks fork points with tree_branches continuations each
    # (rollout/scheduler.py); tree_advantage "segment" (value differences between fork points) or "tree_grpo".
    rollout_mode: str = "chain"
    tree_forks: int = 4
    tree_branches: int = 2
    tree_advantage: str = "segment"
    # Reverse curriculum (rollout/curriculum.py): a start library (experiments/v3/start_states.py); its tasks start
    # late in recorded successes and move earlier once groups succeed at curriculum_threshold.
    reverse_curriculum: str = ""
    curriculum_threshold: float = 0.6
    context_limit_policy: str = "finish"
    numerical_check_mode: str = "periodic"
    numerical_check_interval: int = 20
    reference_cache_tokens: int = 1000000
    heartbeat_interval: float = 1.0
    model_load_timeout: float = 600.0
    group_timeout: float = 3600.0
    train_progress_timeout: float = 3600.0
    checkpoint_keep: int = 3
    replay_min_ess: float = 0.01
    replay_max_log_ratio: float = 60.0
    model: str = "models/Qwen3-4B-Instruct-2507-4bit"
    algorithm: str = "grpo"
    replay_source: str = ""
    replay_capacity: int = 32
    replay_max_uses: int = 4
    replay_max_age: int = 32
    replay_batches_per_batch: int = 0
    tuning: str = "full"
    thinking_mode: str = "no-thinking"
    qat_scope: str = "projections"
    lora_targets: str = "self_attn.q_proj,self_attn.v_proj"
    lora_scale: float = 16.0
    lora_dropout: float = 0.0
    lora_merge_export: bool = False
    lora_requantize_export: bool = False
    gradient_checkpointing: bool = False
    logits_chunk_size: int = 128
    prefill_chunk_size: int = 512
    prefix_cache_tokens: int = 0
    packed_inference: bool = False
    profile_memory: bool = False
    checkpoint_interval_steps: int = 1
    policy_publish_interval_steps: int = 1
    updates: int = 0
    rollout_workers: int = 1
    rollout_concurrency: int = 8
    watchdog_interval: float = 1.0
    micro_batch_size: int = 4
    max_tokens_per_micro_batch: int = 32768
    batch_logp_tolerance: float = 0.01
    sandbox_backend: str = "local"
    sandbox_images: str = ""
    sandbox_max_active: int = 4
    sandbox_startup_workers: int = 2
    sandbox_idle_timeout: int = 300
    sandbox_cpus: str = "1"
    sandbox_memory: str = "1g"
    sandbox_services: str = ""
    environment_factory: str = "rl.environment:SandboxEpisode"
    rollout_group: int = 8
    over_sampling_batch_size: int = 0
    rollout_max_attempts: int = 3
    batch_size: int = 8
    mini_batch_size: int = 8
    queue_size: int = 2
    max_policy_lag: int = 1
    max_groups: int = 0
    max_steps: int = 30
    max_tokens: int = 1024
    max_context: int = 32768
    layers: int = 1
    rank: int = 8
    bits: int = 4
    temperature: float = 1.0
    learning_rate: float = 1e-6
    clip: float = 0.2
    beta: float = 0.01
    gamma: float = 1.0
    discount_unit: str = "action"
    target_kl: float = 0.05
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    epochs: int = 2
    optimization_passes: int = 1
    seed: int = 42
    rollout_timeout: int = 1200
    max_rollout_restarts: int = 2
    tensorboard: bool = True
    log_flush_seconds: int = 5
    rollout_trace_samples: int = 1
    rollout_trace_max_chars: int = 16000
    resume: bool = False

    @property
    def sampling_pool_size(self):
        return self.over_sampling_batch_size or 2 * self.batch_size

    @property
    def training_batch_size(self):
        return self.batch_size

    @property
    def training_mini_batch_size(self):
        return self.mini_batch_size

    def validate(self):
        if (
            min(
                self.logits_chunk_size,
                self.prefill_chunk_size,
                self.checkpoint_interval_steps,
                self.policy_publish_interval_steps,
            )
            < 1
            or self.prefix_cache_tokens < 0
        ):
            raise ValueError("invalid memory/scheduling budgets")
        if (
            not math.isfinite(self.lora_scale)
            or self.lora_scale <= 0
            or not self.lora_targets.strip()
        ):
            raise ValueError("invalid LoRA scale or targets")
        if self.lora_requantize_export and not self.lora_merge_export:
            raise ValueError("LoRA requantization requires merged export")
        if self.lora_dropout != 0:
            raise ValueError(
                "RL requires lora_dropout=0 to preserve behavior/target probability consistency"
            )
        for name in (
            "heartbeat_interval",
            "model_load_timeout",
            "group_timeout",
            "train_progress_timeout",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.heartbeat_interval >= self.rollout_timeout:
            raise ValueError("heartbeat_interval must be below rollout_timeout")
        if (
            min(self.eval_interval_steps, self.metrics_export_interval, self.reference_cache_tokens)
            < 0
        ):
            raise ValueError(
                "evaluation/export intervals and reference cache budget must be nonnegative"
            )
        if self.group_artifacts_keep < 1 or self.numerical_check_interval < 1:
            raise ValueError("artifact retention and numerical check interval must be positive")
        if self.advantage_std not in {"group", "task"}:
            raise ValueError("advantage_std must be group or task")
        if self.zero_variance_policy not in {"retry_skip", "retry_fail", "skip", "fail"}:
            raise ValueError("invalid zero_variance_policy")
        if self.context_limit_policy not in {"finish", "fail"}:
            raise ValueError("invalid context_limit_policy")
        if self.numerical_check_mode not in {"strict", "periodic"}:
            raise ValueError("invalid numerical_check_mode")
        if (
            self.log_flush_seconds < 1
            or self.rollout_trace_max_chars < 256
            or self.rollout_trace_samples < 0
        ):
            raise ValueError(
                "require log_flush_seconds >= 1, rollout_trace_max_chars >= 256, rollout_trace_samples >= 0"
            )
        if bool(self.sandbox) == bool(self.tasks) or not self.output:
            raise ValueError("provide output and exactly one of sandbox or tasks")
        for name in (
            "rollout_workers",
            "rollout_group",
            "queue_size",
            "batch_size",
            "mini_batch_size",
            "optimization_passes",
            "max_steps",
            "max_tokens",
            "max_context",
            "layers",
            "rank",
            "micro_batch_size",
            "max_tokens_per_micro_batch",
            "rollout_concurrency",
            "epochs",
            "rollout_timeout",
            "eval_episodes",
            "replay_capacity",
            "replay_max_uses",
            "replay_max_age",
            "checkpoint_keep",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sandbox_backend not in ("local", "docker"):
            raise ValueError("sandbox_backend must be local or docker")
        if (
            min(self.sandbox_max_active, self.sandbox_startup_workers, self.sandbox_idle_timeout)
            <= 0
        ):
            raise ValueError("sandbox manager limits must be positive")
        if ":" not in self.environment_factory:
            raise ValueError("environment_factory must be module:callable")
        if not math.isfinite(self.batch_logp_tolerance) or self.batch_logp_tolerance <= 0:
            raise ValueError("batch_logp_tolerance must be positive and finite")
        if self.rollout_max_attempts < 1 or self.over_sampling_batch_size < 0:
            raise ValueError(
                "rollout_max_attempts must be positive and over_sampling_batch_size nonnegative"
            )
        if self.sampling_pool_size < self.batch_size:
            raise ValueError("over_sampling_batch_size must be at least batch_size")
        if self.mini_batch_size > self.batch_size:
            raise ValueError("mini_batch_size cannot exceed batch_size")
        if min(self.updates, self.max_groups, self.replay_batches_per_batch) < 0:
            raise ValueError("optional budgets and replay batches must be nonnegative")
        if self.replay_source and not self.replay_batches_per_batch:
            raise ValueError("replay_source requires replay_batches_per_batch > 0")
        if not math.isfinite(self.watchdog_interval) or self.watchdog_interval <= 0:
            raise ValueError("watchdog_interval must be positive and finite")
        if self.algorithm not in {"ppo", "grpo"} or self.tuning not in {"qat", "lora", "full"}:
            raise ValueError("unsupported algorithm or tuning mode")
        if self.qat_scope not in {"projections", "full"} or (
            self.qat_scope == "full" and self.tuning != "qat"
        ):
            raise ValueError("qat_scope full requires tuning qat")
        if self.rollout_mode not in {"chain", "tree"} or self.tree_advantage not in {"segment", "tree_grpo"}:
            raise ValueError("rollout_mode must be chain or tree, tree_advantage segment or tree_grpo")
        if self.tree_forks < 0 or self.tree_branches < 1:
            raise ValueError("tree_forks must be nonnegative and tree_branches positive")
        if self.loss_aggregation not in {"sequence", "token"}:
            raise ValueError("loss_aggregation must be sequence or token")
        if not 0.0 <= self.low_prob_mask <= 1.0:
            raise ValueError("low_prob_mask must be in [0, 1]")
        if self.thinking_mode not in {"auto", "thinking", "no-thinking"}:
            raise ValueError("unsupported thinking mode")
        if not 0 <= self.replay_min_ess <= 1 or not 0 < self.replay_max_log_ratio <= 80:
            raise ValueError("replay ESS must be in [0,1] and log ratio bound in (0,80]")
        if self.algorithm == "grpo" and self.rollout_group < 2:
            raise ValueError("GRPO requires rollout_group >= 2")
        if self.max_rollout_restarts < 0:
            raise ValueError("max_rollout_restarts must be nonnegative")
        if self.max_policy_lag < 0:
            raise ValueError("require max_policy_lag >= 0")
        if self.max_context <= self.max_tokens or self.bits not in {4, 8}:
            raise ValueError("require max_context > max_tokens and bits in {4, 8}")
        for name in (
            "learning_rate",
            "clip",
            "beta",
            "gamma",
            "gae_lambda",
            "value_coefficient",
            "max_grad_norm",
            "temperature",
            "target_kl",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            self.temperature == 0
            or self.learning_rate == 0
            or self.max_grad_norm == 0
            or not 0 < self.clip < 1
        ):
            raise ValueError("require positive learning rate/gradient norm and clip in (0, 1)")
        if self.discount_unit not in {"action", "token"}:
            raise ValueError("discount_unit must be action or token")
        if self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError("discounts must be in [0, 1]")
