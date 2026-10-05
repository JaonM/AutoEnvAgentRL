"""Compatibility CLI: python -m rl.train."""
import sys
from rl.trainer import runner as _runner

if __name__ == "__main__":
    _runner.main()
else:
    sys.modules[__name__] = _runner
