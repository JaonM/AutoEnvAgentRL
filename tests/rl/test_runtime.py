import multiprocessing as mp
import os
import pytest
from rl.runtime import RolloutPool


def exiting_worker(index,config,version,results,stop):
    os._exit(3)


def test_dead_actor_restarted_with_bounded_budget(tmp_path):
    ctx=mp.get_context('spawn');queue=ctx.Queue(maxsize=2);stop=ctx.Event();version=ctx.Value('i',0)
    pool=RolloutPool(ctx,{'output':str(tmp_path),'rollout_workers':1,'max_rollout_restarts':1,'rollout_timeout':30},
                   version,queue,stop,{},target=exiting_worker)
    try:
        pool.start_all();pool.processes[0].join(timeout=10)
        assert pool.processes[0].exitcode==3
        pool.check(queue_empty=True)
        pool.processes[0].join(timeout=10)
        assert pool.restarts=={'0':1}
        with pytest.raises(RuntimeError,match='exhausted'):
            pool.check(queue_empty=True)
    finally:
        pool.close()


def test_resume_cannot_reset_uncommitted_restart_budget(tmp_path):
    import json
    (tmp_path/'rollout_recovery.json').write_text(json.dumps([
        {'rollout_worker':0,'restart':1,'reason':'failure','at':1.},
        {'rollout_worker':0,'restart':2,'reason':'failure','at':2.}]))
    pool=RolloutPool(None,{'output':str(tmp_path),'resume':True,'max_rollout_restarts':2},
                   None,None,None,{},restarts={'0':1},events=[{'rollout_worker':0,'restart':1,'reason':'failure','at':1.}])
    assert pool.restarts=={'0':2} and len(pool.events)==2
    with pytest.raises(RuntimeError,match='exhausted'):
        pool.restart(0,'retry')


def mixed_workers(index,config,version,results,stop):
    import time
    import queue
    from pathlib import Path
    from rl.checkpoint import atomic_json
    heartbeat=Path(config['output'])/'rollout_workers'/f'worker-{index}.json'
    atomic_json(heartbeat,{'rollout_worker':index,'pid':os.getpid(),'stage':'environment','at':time.time()})
    while not stop.wait(.02):
        if index==1:
            atomic_json(heartbeat,{'rollout_worker':index,'pid':os.getpid(),'stage':'queue','at':time.time()})
            try:results.put({'healthy':True},timeout=.01)
            except queue.Full:pass


def test_watchdog_recovers_stall_while_queue_stays_full(tmp_path):
    import time
    from rl.process_control import StopSignal
    ctx=mp.get_context('spawn');queue=ctx.Queue(maxsize=1);stop=StopSignal(ctx);version=ctx.Value('i',0)
    pool=RolloutPool(ctx,{'output':str(tmp_path),'rollout_workers':2,'max_rollout_restarts':1,'rollout_timeout':1.,
                       'watchdog_interval':.02},version,queue,stop,{},target=mixed_workers)
    try:
        pool.start_all()
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not pool.checkpoint_state()['rollout_restarts'].get('0'):
            time.sleep(.02)
        state=pool.checkpoint_state()
        assert state['rollout_restarts'].get('0')==1
        assert state['rollout_restarts'].get('1',0)==0
        assert queue.get(timeout=1)=={'healthy':True}
    finally:pool.close()
