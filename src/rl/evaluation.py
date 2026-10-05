"""Compatibility entry point for :mod:`rl.trainer.evaluation`."""
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module('rl.trainer.evaluation')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
