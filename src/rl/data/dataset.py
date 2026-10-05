"""Finite, reproducible sandbox epochs with bounded asynchronous prefetch."""
import queue
import random
import time


class DatasetSchedule:
    def __init__(self, tasks, *, epochs, batch_size, seed):
        train = [task for task in tasks if task.split == 'train']
        if not train or min(epochs, batch_size) < 1:
            raise ValueError('dataset requires training sandboxes and positive epoch/batch sizes')
        self.jobs, self.batches = [], []
        for epoch in range(1, epochs + 1):
            ordered = list(train)
            random.Random(f'{seed}:dataset:{epoch}').shuffle(ordered)
            for start in range(0, len(ordered), batch_size):
                jobs = []
                for task in ordered[start:start + batch_size]:
                    ordinal = len(self.jobs)
                    job = {'dataset_index': ordinal, 'dataset_epoch': epoch,
                           'dataset_batch': len(self.batches), 'task_id': task.id,
                           'seed': seed + ordinal}
                    jobs.append(job)
                    self.jobs.append(job)
                self.batches.append(jobs)

    def prefetch_limit(self, next_batch):
        """Current batch plus at most one future batch; never unbounded reordering."""
        if next_batch >= len(self.batches):
            return len(self.jobs)
        return self.batches[min(next_batch + 1, len(self.batches) - 1)][-1]['dataset_index'] + 1


class DatasetResults:
    def __init__(self, schedule, results, pool, positions, *, rollout_workers, timeout, validate,
                 next_index=0, completed=(), task_queue=None, on_poll=None, progress_timeout=None):
        self.schedule, self.results, self.pool = schedule, results, pool
        self.positions, self.rollout_workers, self.timeout = positions, rollout_workers, timeout
        self.validate = validate
        self.completed = set(range(next_index)) | set(completed)
        self.pending, self.acks = {}, {}
        self.task_queue = task_queue
        self.on_poll = on_poll or (lambda: None)
        self.progress_timeout = timeout if progress_timeout is None else progress_timeout

    def _wait_for_progress(self, allowed, progress):
        now = time.monotonic()
        ready = frozenset(set(self.pending) & allowed)
        if ready != progress[0]:
            progress[:] = [ready, now]
        if now - progress[1] >= self.progress_timeout:
            details = self.pool.diagnostics() if hasattr(self.pool, 'diagnostics') else {}
            raise TimeoutError(f'no rollout result progress for {self.progress_timeout}s; '
                               f'waiting={sorted(allowed - ready)}, pending={sorted(ready)}, workers={details}')
        self._receive()


    def _receive(self):
        self.pool.raise_if_failed()
        self.on_poll()
        if self.task_queue is not None:
            ready = self.task_queue.ready(self.completed | set(self.pending))
            for group in ready:
                self._accept(group)
            if ready:
                return
        try:
            group = self.results.get(timeout=min(.1, self.timeout))
        except queue.Empty:
            return
        if 'rollout_error' in group:
            self.pool.handle_error(group)
            return
        self._accept(group)

    def _accept(self, group):
        self.validate(group)
        index = group.get('dataset_index')
        if not isinstance(index, int) or not 0 <= index < len(self.schedule.jobs):
            raise ValueError('invalid dataset index from rollout worker')
        expected = self.schedule.jobs[index]
        attempt = group.get('rollout_attempt', 0)
        if self.task_queue is not None and attempt != self.task_queue.attempt(index):
            return  # Stale notification from a rejected attempt.
        expected = {**expected, 'seed':expected['seed'] + attempt * 1000003}
        if any(group.get(key) != expected[key] for key in ('task_id', 'dataset_epoch', 'dataset_batch')):
            raise ValueError('rollout group does not match scheduled sandbox')
        if not group.get('shared_task') and (group['rollout_worker'] != index % self.rollout_workers or group['group_index'] != index // self.rollout_workers
                or any(e['seed'] != expected['seed'] for e in group['episodes'])):
            raise ValueError('rollout assignment or seed does not match dataset schedule')
        if any(e['seed'] != expected['seed'] for e in group['episodes']):
            raise ValueError('rollout seed does not match dataset schedule')
        if index not in self.completed:
            self.pending.setdefault(index, group)

    def _consume(self, index):
        group = self.pending.pop(index)
        self.completed.add(index)
        if group.get('shared_task'):
            return group
        worker = group['rollout_worker']
        acknowledged = self.acks.setdefault(worker, {})
        acknowledged[group['group_index']] = group['rollout_rng_after']
        next_group = self.positions.get(str(worker), {}).get('next_group', 0)
        last_rng = None
        # Do not skip an unfinished group on recovery even if later results arrive first.
        while next_group in acknowledged:
            last_rng = acknowledged.pop(next_group)
            next_group += 1
        if last_rng is not None:
            self.pool.record_position(worker, {'next_group': next_group, 'rng': last_rng})
        return group

    def ready_minibatches(self, jobs, size):
        remaining = {job['dataset_index'] for job in jobs} - self.completed
        progress = [frozenset(set(self.pending) & remaining), time.monotonic()]
        while remaining:
            wanted = min(size, len(remaining))
            ready = [index for index in self.pending if index in remaining]
            if len(ready) < wanted:
                self._wait_for_progress(remaining, progress)
                continue
            selected = ready[:wanted]
            remaining.difference_update(selected)
            yield [self._consume(index) for index in selected]

    def ready_count(self, epoch, count, size):
        jobs = [job for job in self.schedule.jobs if job['dataset_epoch'] == epoch]
        remaining = count
        progress = [frozenset(), time.monotonic()]
        while remaining:
            wanted = min(size, remaining)
            allowed = {job['dataset_index'] for job in jobs} - self.completed
            if wanted > len(allowed):
                raise ValueError('requested more results than remaining epoch coverage')
            ready = [index for index in self.pending if index in allowed]
            if len(ready) < wanted:
                self._wait_for_progress(allowed, progress)
                continue
            selected = ready[:wanted]
            remaining -= len(selected)
            yield [self._consume(index) for index in selected]

    def take(self, job):
        # Single-job adapter for callers that explicitly require ordered consumption.
        if job['dataset_index'] in self.completed:
            raise ValueError('dataset job already consumed')
        return next(self.ready_minibatches([job], 1))[0]
