import pytest
mx=pytest.importorskip('mlx.core',exc_type=ImportError)
nn=pytest.importorskip('mlx.nn',exc_type=ImportError)
import mlx.optimizers as optim
from rl.checkpoint import save_checkpoint,restore_checkpoint,read_checkpoint


def test_resume_restores_adam_moments_step_weights_and_rng(tmp_path):
    mx.random.seed(42)
    model=nn.Linear(2,1)
    optimizer=optim.Adam(.01)
    optimizer.init(model.trainable_parameters())
    grad={'weight':mx.ones_like(model.weight),'bias':mx.ones_like(model.bias)}
    optimizer.update(model,grad);mx.eval(model.parameters(),optimizer.state)
    save_checkpoint(tmp_path,model,optimizer,{'optimizer_step':1,'updates':1})
    expected_random=mx.random.uniform(shape=(3,));mx.eval(expected_random)
    optimizer.update(model,grad);mx.eval(model.parameters(),optimizer.state)
    expected=mx.array(model.weight)
    restored_model=nn.Linear(2,1)
    restored_optimizer=optim.Adam(.01)
    restored_optimizer.init(restored_model.trainable_parameters())
    state=restore_checkpoint(tmp_path,restored_model,restored_optimizer)
    assert state['optimizer_step']==1
    assert mx.random.uniform(shape=(3,)).tolist()==expected_random.tolist()
    restored_optimizer.update(restored_model,grad)
    mx.eval(restored_model.parameters(),restored_optimizer.state)
    assert bool(mx.all(restored_model.weight==expected))
    assert int(restored_optimizer.step)==2


def test_corruption_and_uncommitted_checkpoint_rejected_or_ignored(tmp_path):
    model=nn.Linear(2,1);opt=optim.Adam(.01);opt.init(model.trainable_parameters())
    path=save_checkpoint(tmp_path,model,opt,{'optimizer_step':0})
    (tmp_path/'.incomplete').mkdir()
    assert read_checkpoint(tmp_path)[0]==path
    (path/'state.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum mismatch'):
        read_checkpoint(tmp_path)
