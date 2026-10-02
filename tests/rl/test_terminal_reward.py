import pytest
from rl.rollout_worker import rollout
from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore


@pytest.mark.parametrize('terminal', [True, False])
def test_rollout_scores_only_after_last_action_and_attaches_terminal_reward(terminal):
    class Policy:
        def sample(self, *args, **kwargs):
            return {'text': 'answer', 'tokens': [1], 'old_logp': [0.]}
        def encode(self, messages): return [1]
        def value(self, *args): raise AssertionError('scored finite horizon must not bootstrap')
    class Environment:
        messages = []
        def reset(self, seed): self.steps = 0; self.terminated = False
        def step(self, text):
            self.steps += 1
            self.terminated = terminal and self.steps == 2
            return 0., self.terminated
        def finish(self):
            assert self.steps == 2
            return {'final_reward': .7, 'terminated': self.terminated}
    result = rollout(Policy(), Environment(), {'max_steps': 2, 'max_tokens': 10,
        'max_context': 20, 'algorithm': 'ppo'}, 1)
    assert [a['reward'] for a in result['actions']] == [0., .7]
    assert result['bootstrap'] == 0
    assert result['truncated'] == (not terminal)


def test_protocol_final_submission_does_not_evaluate_reward(tmp_path):
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    store.reset(episode_id='terminal-test', seed=1)
    store.set_state('reward_mode', 'episode_end')
    def forbidden(): raise AssertionError('reward evaluated before episode ended')
    user = ContractUserSimulator.__new__(ContractUserSimulator)
    user.episode_store, user.completion_check = store, forbidden
    script = {'interaction_protocol': {'stages': []}}
    state = {'protocol_index': 0, 'turn_index': 0, 'script_id': 'test', 'state_id': 'review'}
    reply = user._protocol_turn(script, state, [{'role': 'assistant', 'content': 'submitted answer'}])
    assert reply['should_end'] is True
    assert reply['outcome_category'] == 'agent_submitted'
    # Certification retains its existing success-based check outside training.
    store.set_state('reward_mode', 'continuous')
    with pytest.raises(AssertionError, match='before episode ended'):
        user._protocol_turn(script, state, [{'role': 'assistant', 'content': 'answer'}])


def test_rl_calls_real_reward_endpoint_once_after_episode_end(tmp_path, monkeypatch):
    from env_factory.sandbox_runtime import SandboxApplication, ContractToolRegistry
    from rl.environment import SandboxEpisode
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'test-only')
    store = EpisodeStore(tmp_path / 'http.sqlite3')
    evaluations = []
    def score():
        evaluations.append(store.get_state('final_agent_response'))
        return {'reward': .8, 'components': {'answer': .8}}
    app = SandboxApplication(episode_store=store,
        tool_registry=ContractToolRegistry([], {}), observation=lambda: {}, reward=score,
        user_turn=lambda messages: {'user_query': '收到', 'should_end': True})
    episode = SandboxEpisode.__new__(SandboxEpisode)
    episode.app, episode.task = app, {'task': 'test'}
    episode.headers = {'Authorization': 'Bearer test-only'}
    episode.reset(7)
    assert store.get_state('reward_mode') == 'episode_end'
    assert evaluations == []
    assert episode.step('{"kind":"respond","content":"answer"}') == (0., True)
    assert evaluations == []
    assert episode.finish()['final_reward'] == .8
    assert episode.finish()['final_reward'] == .8
    assert evaluations == ['answer']
    with pytest.raises(RuntimeError, match='already scored'):
        episode.step('{"kind":"respond","content":"another"}')
