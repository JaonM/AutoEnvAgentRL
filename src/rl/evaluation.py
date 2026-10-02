"""Per-task evaluation with explicit train/held-out split reporting."""
from dataclasses import asdict
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
