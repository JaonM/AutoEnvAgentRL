import subprocess
import sys


def test_removed_mode_is_not_interpreted_as_model_abbreviation():
    result = subprocess.run([sys.executable,'-m','rl.train','--sandbox','unused','--output','unused',
                             '--mode','off-policy'],capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert 'unrecognized arguments: --mode off-policy' in result.stderr


def test_help_lists_only_supported_algorithms():
    result=subprocess.run([sys.executable,'-m','rl.train','--help'],capture_output=True,text=True,timeout=10)
    assert result.returncode == 0
    assert '--algorithm {ppo,grpo}' in result.stdout
    assert '--mode ' not in result.stdout and 'vtrace' not in result.stdout
