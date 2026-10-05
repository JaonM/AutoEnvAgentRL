"""Compatibility entry point for :mod:`rl.rollout.scheduler`."""
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module('rl.rollout.scheduler')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
