"""Parse the existing RL CLI without loading a model."""

import argparse
from rl.config import Config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bounded asynchronous PPO/GRPO training on qualified local sandboxes.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sandbox", default="", help="qualified sandbox directory; training_ready must be true"
    )
    parser.add_argument("--output", required=True, help="new training output directory")
    for name, field in Config.__dataclass_fields__.items():
        if name in {"sandbox", "output"}:
            continue
        default = "docker" if name == "sandbox_backend" else field.default
        if name == "tensorboard":
            parser.add_argument(
                "--tensorboard",
                action=argparse.BooleanOptionalAction,
                default=default,
                help="write live TensorBoard scalars and rollout text",
            )
            continue
        if isinstance(default, bool):
            parser.add_argument(
                "--" + name.replace("_", "-"), action="store_true", help=name.replace("_", " ")
            )
            continue
        choices = {
            "sandbox_backend": ["local", "docker"],
            "algorithm": ["ppo", "grpo"],
            "tuning": ["qat", "lora", "full"],
            "thinking_mode": ["auto", "thinking", "no-thinking"],
            "qat_scope": ["projections", "full"],
            "bits": [4, 8],
            "discount_unit": ["action", "token"],
            "zero_variance_policy": ["retry_skip", "retry_fail", "skip", "fail"],
            "advantage_std": ["group", "task"],
            "loss_aggregation": ["sequence", "token"],
            "context_limit_policy": ["finish", "fail"],
            "numerical_check_mode": ["strict", "periodic"],
        }.get(name)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(default),
            default=default,
            choices=choices,
            help={
                "over_sampling_batch_size": "candidate pool size; 0 means twice batch_size",
                "rollout_max_attempts": "maximum sampling attempts per sandbox visit, including initial attempt",
                "batch_size": "completed sandbox groups per batch within an epoch (keeps final partial batch)",
                "rollout_workers": "sampling processes; actor means policy trainer",
                "micro_batch_size": "maximum action segments per physical tensor batch",
                "max_tokens_per_micro_batch": "maximum padded input tokens per physical tensor batch",
                "batch_logp_tolerance": "maximum batch/cached logprob discrepancy; sampled logprobs are never replaced",
                "environment_factory": "AgentRuntime factory as module:callable",
                "sandbox_backend": "docker auto-manages local containers; local uses in-process runtime",
                "sandbox_images": "optional task-ID to Docker image JSON; defaults to qualified image metadata",
                "sandbox_services": "existing Docker/Kubernetes services.json; skips auto-management",
                "mini_batch_size": "sandboxes per optimizer step; all their rollouts stay together",
                "rollout_group": "rollout trajectories sampled per sandbox visit",
                "epochs": "complete passes through the training sandbox dataset",
                "optimization_passes": "optimization passes over a collected batch; separate from dataset epochs",
                "updates": "optional policy update cap; 0 means no extra cap",
                "max_groups": "optional fresh sandbox-visit cap; 0 means no extra cap",
                "replay_batches_per_batch": "extra replay batches after each fresh dataset batch; does not advance epoch",
            }.get(name, name.replace("_", " ")),
        )
    config = Config(**vars(parser.parse_args(argv)))
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    return config
