"""Training coordinator. Optimizer progress, publication and durable commits remain separate."""

from rl.trainer.io import write_json
from rl.trainer.initialization import initialize
from rl.trainer import periodic, optimization, persistence, publication, finalization


class TrainingSession:
    def __init__(self, config, run_id, checkpoint_saver):
        initialize(self, config, run_id, checkpoint_saver)

    def run(self):
        from rl.data.contracts import attach_contract
        from rl.rollout.scheduler import environment_overlap
        from rl.checkpointing.retention import prune_live_artifacts
        from rl.rollout.admission import rejection_action
        from rl.checkpointing.checkpoint import sha256

        try:
            if not self.config.resume:
                self.evaluate_periodically(force=True)
                self.commit(force=True)
            if self.state.dataset_cursor < len(self.schedule.batches):
                self.pool.start_all()
            while self.state.dataset_cursor < len(self.schedule.batches):
                jobs = self.schedule.batches[self.state.dataset_cursor]
                if not self.state.inflight:
                    self.state.inflight = {
                        "records": [],
                        "next_mini": 0,
                        "steps_start": self.state.optimizer_step,
                        "passes_completed": 0,
                        "replay_completed": 0,
                        "stop_fresh": False,
                    }
                remaining = len(jobs) - sum(
                    (record.get("trained", True) for record in self.state.inflight["records"])
                )
                if (
                    self.config.max_groups
                    and self.state.groups
                    + remaining
                    - sum(
                        (
                            not record.get("trained", True)
                            for record in self.state.inflight["records"]
                        )
                    )
                    > self.config.max_groups
                ):
                    raise RuntimeError("fresh group budget cannot cover next dataset batch")
                while remaining:
                    wanted = min(self.config.mini_batch_size, remaining)
                    fresh = self.prepared_records(only_untrained=True)
                    while len(fresh) < wanted:
                        if self.config.max_groups and self.state.groups >= self.config.max_groups:
                            raise RuntimeError(
                                "oversampling exhausted max_groups before filling qualified batch"
                            )
                        with self.times.measure("actor_wait_ready_minibatch"):
                            ready = next(self.loader.ready_count(jobs[0]["dataset_epoch"], 1, 1))
                        group = ready[0]
                        self.state.groups += 1
                        index = group["dataset_index"]
                        path = self.output / f"group-{self.state.groups:04d}.json"
                        attach_contract(group)
                        with self.times.measure("actor_rollout_io"):
                            write_json(path, group)
                        self.times.ingest_rollout(group)
                        self.monitor.group(group, self.state.groups, self.state.updates)
                        self.metrics.append(
                            {
                                "event": "rollout_received",
                                "task_id": group["task_id"],
                                "dataset_index": index,
                                "group": self.state.groups,
                                "started_at": group.get("started_at"),
                                "finished_at": group.get("finished_at"),
                                "environment_overlap": environment_overlap([group]),
                            }
                        )
                        with self.times.measure("actor_prepare_targets"):
                            item = self.prepare(group, path)
                        if "rejected" in item:
                            attempt = group.get("rollout_attempt", 0)
                            disposition = rejection_action(
                                item["rejected"],
                                self.config.zero_variance_policy,
                                attempt,
                                self.config.rollout_max_attempts,
                            )
                            if disposition == "fail":
                                raise RuntimeError(
                                    f"oversampling exhausted {attempt + 1} attempts for sandbox {group['task_id']}: {item['rejected']}"
                                )
                            if disposition == "skip":
                                self.state.consumed_jobs.add(index)
                                self.state.skipped_jobs[str(index)] = item["rejected"]
                                self.state.inflight["records"].append(
                                    {
                                        "dataset_index": index,
                                        "path": str(path),
                                        "sha256": sha256(path),
                                        "eligible": False,
                                        "trained": True,
                                    }
                                )
                                remaining -= 1
                                wanted = min(self.config.mini_batch_size, remaining)
                                self.monitor.runtime.emit(
                                    "task/skipped",
                                    text=item["rejected"],
                                    dataset_index=index,
                                    task_id=group["task_id"],
                                    metrics={
                                        "admission/skipped_jobs": len(self.state.skipped_jobs)
                                    },
                                )
                            else:
                                self.state.rollout_attempts[str(index)] = attempt + 1
                                self.loader.completed.discard(index)
                            self.commit()
                            continue
                        self.state.consumed_jobs.add(index)
                        self.replay.add(path, group, self.state.updates)
                        self.state.inflight["records"].append(
                            {
                                "dataset_index": index,
                                "path": str(path),
                                "sha256": sha256(path),
                                "eligible": True,
                                "drift": item["drift"],
                                "lag": item["lag"],
                                "trained": False,
                            }
                        )
                        fresh.append(item)
                    if not fresh:
                        break
                    if not self.state.inflight["stop_fresh"]:
                        self.state.inflight["stop_fresh"] = self.optimize(
                            fresh,
                            jobs[0]["dataset_epoch"],
                            0,
                            mini_offset=self.state.inflight["next_mini"],
                        )
                    for record in self.state.inflight["records"]:
                        record["trained"] = True
                    self.state.inflight["next_mini"] += 1
                    remaining -= wanted
                    self.commit()
                self.state.inflight["passes_completed"] = max(
                    1, self.state.inflight["passes_completed"]
                )
                while self.state.inflight[
                    "passes_completed"
                ] < self.config.optimization_passes and (not self.state.inflight["stop_fresh"]):
                    number = self.state.inflight["passes_completed"] + 1
                    self.state.inflight["stop_fresh"] = self.optimize(
                        self.prepared_records(), jobs[0]["dataset_epoch"], 0, number
                    )
                    self.state.inflight["passes_completed"] = number
                    self.commit()
                while (
                    self.state.inflight["replay_completed"] < self.config.replay_batches_per_batch
                ):
                    cycle = self.state.inflight["replay_completed"] + 1
                    replay_batch = []
                    for record in self.state.inflight["records"]:
                        if not record["eligible"]:
                            continue
                        job = self.schedule.jobs[record["dataset_index"]]
                        selected = self.replay.sample(
                            self.state.updates, self.tasks, task_id=job["task_id"]
                        )
                        if selected:
                            group, entry = selected
                            item = self.prepare(group, entry["path"], entry)
                            if "rejected" not in item:
                                replay_batch.append(item)
                    for number in range(1, self.config.optimization_passes + 1):
                        if self.optimize(replay_batch, jobs[0]["dataset_epoch"], cycle, number):
                            break
                    self.state.inflight["replay_completed"] = cycle
                    self.commit()
                self.publish_policy()
                self.metrics.append(
                    {
                        "event": "dataset_batch_completed",
                        "epoch": jobs[0]["dataset_epoch"],
                        "dataset_batch": self.state.dataset_cursor,
                        "sandbox_ids": [
                            self.schedule.jobs[record["dataset_index"]]["task_id"]
                            for record in self.state.inflight["records"]
                        ],
                        "fresh_rollouts": len(jobs) * self.config.rollout_group,
                        "group_paths": [
                            record["path"] for record in self.state.inflight["records"]
                        ],
                    }
                )
                self.state.dataset_cursor += 1
                self.state.inflight = {}
                self.evaluate_periodically()
                self.commit()
                self.mx.clear_cache()
            self.evaluate_periodically(force=True)
            self.commit(force=True)
        finally:
            self.pool.close()
            prune_live_artifacts(
                self.output,
                self.task_queue,
                self.state.committed_version,
                keep=self.config.checkpoint_keep,
                group_keep=self.config.group_artifacts_keep,
            )
            self.metrics.export()
            self.optimizer_metrics.export()
        return self.finalize()

    def evaluate_periodically(self, *, force=False):
        return periodic.evaluate_periodically(self, force=force)

    def measure_drift(self, group):
        return optimization.measure_drift(self, group)

    def prepare(self, group, path, replay_entry=None):
        return optimization.prepare(self, group, path, replay_entry)

    def optimize(
        self, batch_groups, dataset_epoch, replay_cycle, optimization_pass=1, mini_offset=0
    ):
        return optimization.optimize(
            self, batch_groups, dataset_epoch, replay_cycle, optimization_pass, mini_offset
        )

    def prepared_records(self, *, only_untrained=False):
        return optimization.prepared_records(self, only_untrained=only_untrained)

    def validate_actor_group(self, group):
        return optimization.validate_actor_group(self, group)

    def commit(self, *, force=False):
        return persistence.commit(self, force=force)

    def publish_policy(self):
        return publication.publish_policy(self)

    def finalize(self):
        return finalization.finalize(self)
