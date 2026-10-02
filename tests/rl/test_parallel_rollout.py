import os
import time
from pathlib import Path
from rl.parallel_rollout import EnvironmentPool,parallel_rollouts


def barrier_environment(connection,root,parent_pid):
    messages=[]
    while True:
        operation,value=connection.recv()
        started=time.time()
        if operation=='close':break
        if operation=='reset':
            messages=[{'role':'user','content':f'{os.getpid()}:{value}'}]
            reply=None
        elif operation=='step':
            # Requires two distinct environment processes to reach step together.
            (Path(root)/str(os.getpid())).touch()
            deadline=time.monotonic()+5
            while len(list(Path(root).iterdir()))<2:
                if time.monotonic()>deadline:raise TimeoutError('episodes were serialized')
                time.sleep(.01)
            time.sleep(.05)
            reply=(1.,True)
        else:
            reply={'final_reward':1.,'terminated':True,'trace':[],'worker_pid':os.getpid()}
        connection.send({'value':reply,'messages':messages,'terminated':operation!='reset',
                         'started_at':started,'finished_at':time.time()})
    connection.close()


class Policy:
    def iter_sample(self,messages,max_tokens,max_context):
        yield None
        return {'prompt':[1],'tokens':[2],'text':'finish','old_logp':[-.1],'old_values':[0.],
                'context':messages[0]['content']}


def test_rollouts_overlap_with_isolated_workers_and_same_group_seed(tmp_path):
    pool=EnvironmentPool(str(tmp_path),2,target=barrier_environment)
    try:
        episodes=parallel_rollouts(Policy(),pool,{'rollout_group':2,'max_steps':2,'max_tokens':8,
            'max_context':32,'algorithm':'grpo','rollout_timeout':10},42)
    finally:pool.close()
    assert len({e['worker_pid'] for e in episodes})==2
    assert [e['seed'] for e in episodes]==[42,42]
    assert all(e['actions'][0]['context'].endswith(':42') for e in episodes)
    steps=[next(x for x in e['environment_intervals'] if x['operation']=='step') for e in episodes]
    assert max(x['started_at'] for x in steps)<min(x['finished_at'] for x in steps)


class BatchedPolicy:
    def __init__(self):
        self.decoder = None

    def batch_decoder(self):
        self.decoder = ReadyDecoder()
        return self.decoder


class ReadyDecoder:
    def __init__(self):
        self.requests = {}
        self.batch_sizes = []

    def add(self, key, messages, max_tokens, max_context):
        self.requests[key] = [messages, 0]

    def tick(self):
        self.batch_sizes.append(len(self.requests))
        finished = {}
        for key, state in self.requests.items():
            state[1] += 1
            # Allow both reset RPCs to arrive; GPU work would take time too.
            time.sleep(.005)
            if state[1] >= 12:
                finished[key] = {'prompt':[1], 'tokens':[2], 'text':'finish', 'old_logp':[-.1],
                                 'context':state[0][0]['content']}
        for key in finished:
            del self.requests[key]
        return finished


def test_scheduler_uses_one_decode_tick_for_ready_trajectories(tmp_path):
    pool = EnvironmentPool(str(tmp_path), 2, target=barrier_environment)
    policy = BatchedPolicy()
    try:
        episodes = parallel_rollouts(policy, pool, {'rollout_group':2, 'max_steps':2,
            'max_tokens':8, 'max_context':32, 'algorithm':'grpo', 'rollout_timeout':10}, 42)
    finally:
        pool.close()
    assert max(policy.decoder.batch_sizes) == 2
    assert len(episodes) == 2
    assert all(e['actions'][0]['context'].endswith(':42') for e in episodes)
