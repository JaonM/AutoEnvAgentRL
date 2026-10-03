"""Per-task evaluation with explicit train/held-out split reporting."""
from dataclasses import asdict
from contextlib import contextmanager
from .rollout_worker import rollout
from .environment import SandboxEpisode
from .contracts import create_runtime


def evaluate(policy, tasks, config):
    config = asdict(config) if not isinstance(config, dict) else config
    episodes = []
    for task in tasks:
        environment = (SandboxEpisode(task.sandbox) if config.get('environment_factory', 'rl.environment:SandboxEpisode') == 'rl.environment:SandboxEpisode'
                       else create_runtime(config['environment_factory'], task.sandbox))
        try:
            for offset in range(config.get('eval_episodes', 1)):
                episode = rollout(policy, environment, config, config['seed'] + 999 + offset, greedy=True)
                episode.update(task_id=task.id, split=task.split)
                episodes.append(episode)
        finally:
            environment.close()
    def aggregate(rows):
        return {'episodes':len(rows), 'mean_reward':sum(e['final_reward'] for e in rows)/len(rows),
                'success_rate':sum(e['terminated'] and e['final_reward'] >= 1.-1e-9 for e in rows)/len(rows)}
    return {'episodes':episodes, 'by_task':{t.id:aggregate([e for e in episodes if e['task_id']==t.id]) for t in tasks},
            'by_split':{split:aggregate([e for e in episodes if e['split']==split])
                        for split in sorted({t.split for t in tasks})},
            'mean_reward':aggregate(episodes)['mean_reward'],
            'independent_eval':any(t.split=='eval' for t in tasks)}


class PeriodicEvaluation:
    """Fixed held-out task/seed evaluation at safe learner boundaries."""
    def __init__(self, output, tasks, config, monitor, state=None):
        from pathlib import Path
        self.output = Path(output)
        self.tasks = [task for task in tasks if task.split == 'eval']
        self.config, self.monitor = config, monitor
        self.state = dict(state or {})
        self.publish_best()

    def publish_best(self):
        from .checkpoint import atomic_json, sha256
        pointer = self.output / 'best_policy.json'
        if self.state.get('best'):
            best = self.state['best']
            if sha256(best['path']) != best['sha256']:
                raise ValueError('best evaluation policy checksum mismatch')
            atomic_json(pointer, best)
        elif pointer.exists():
            pointer.unlink()

    def run(self, policy, step, *, evaluator=evaluate, force=False):
        import math
        from .checkpoint import atomic_json, sha256
        interval = self.config.eval_interval_steps
        if not interval or not self.tasks or step == self.state.get('last_step'):
            return False
        if not force and step - self.state.get('last_step', -interval) < interval:
            return False
        result = evaluator(policy, self.tasks, self.config)
        reward = result['mean_reward']
        if not math.isfinite(reward):
            raise ValueError('non-finite held-out evaluation reward')
        self.monitor.evaluation('periodic', result, step)
        atomic_json(self.output / 'evaluations' / f'step-{step:08d}.json', result)
        self.state['last_step'] = step
        if not self.state.get('best') or reward > self.state['best']['mean_reward']:
            path = policy.save_snapshot(self.output / 'best_policy', step)
            self.state['best'] = {'optimizer_step': step, 'mean_reward': reward,
                                  'path': path, 'sha256': sha256(path),
                                  'scope': 'held-out policy weights; use main checkpoints for optimizer resume'}
        return True


@contextmanager
def preserve_mlx_rng():
    """Evaluation must not change the learner's stochastic continuation, even on error."""
    import mlx.core as mx
    saved = [key.tolist() for key in mx.random.state]
    try:
        yield
    finally:
        for key, value in zip(mx.random.state, saved):
            key[...] = mx.array(value, dtype=mx.uint32)
