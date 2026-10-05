"""Compatibility entry point for :mod:`rl.environments.services.sandbox_service`."""
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module('rl.environments.services.sandbox_service')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
