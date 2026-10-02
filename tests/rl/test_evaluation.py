from rl import evaluation
from rl.tasks import TaskSpec


def test_evaluation_aggregates_tasks_and_closes_environments(monkeypatch):
    closed = []
    class Environment:
        def __init__(self, root):
            self.root = root
        def close(self):
            closed.append(self.root)
    def rollout(policy, environment, config, seed, *, greedy):
        assert greedy
        return {'final_reward': 1. if environment.root == 'train' else .2,
                'terminated': environment.root == 'train', 'seed': seed}
    monkeypatch.setattr(evaluation, 'SandboxEpisode', Environment)
    monkeypatch.setattr(evaluation, 'rollout', rollout)
    result = evaluation.evaluate(None, [TaskSpec('a','train'),TaskSpec('b','eval',split='eval')],
                                 {'seed':42,'eval_episodes':2})
    assert result['independent_eval'] is True
    assert result['by_split']['train']['success_rate'] == 1
    assert result['by_split']['eval']['success_rate'] == 0
    assert result['by_task']['b']['episodes'] == 2
    assert closed == ['train','eval']
