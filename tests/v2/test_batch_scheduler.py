import sys
import threading
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments'))
from v2_batch_scheduler import run_batch


def test_two_workers_overlap_but_publish_on_controller_and_preserve_all_samples():
    barrier=threading.Barrier(2,timeout=3);controller=threading.get_ident()
    samples=[{'id':i} for i in range(4)];started_ids=[];done=[]
    def started(s):
        assert threading.get_ident()==controller
        started_ids.append(s['id'])
    def execute(s):
        assert threading.get_ident()!=controller
        barrier.wait()
        return {'id':s['id'],'qualified':s['id']%2==0}
    def completed(s,v):
        assert threading.get_ident()==controller
        s.update(v);done.append(s['id'])
    run_batch(samples,concurrency=2,execute=execute,started=started,completed=completed)
    assert started_ids==list(range(4)) and sorted(done)==list(range(4))
    assert sum(s['qualified'] for s in samples)==2


def test_single_worker_preserves_sequential_order():
    events=[]
    run_batch([{'id':i} for i in range(3)],concurrency=1,
        started=lambda s:events.append(('start',s['id'])),execute=lambda s:s,
        completed=lambda s,v:events.append(('done',v['id'])))
    assert events==[(phase,i) for i in range(3) for phase in ['start','done']]


def test_unbounded_concurrency_is_rejected():
    with pytest.raises(ValueError):run_batch([],concurrency=3,execute=None,started=None,completed=None)
