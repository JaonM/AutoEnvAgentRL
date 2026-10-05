"""Compatibility entry point for :mod:`rl.environments.tool_transport`."""
import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module('rl.environments.tool_transport')
if __name__ == "__main__":
    _impl.main()
else:
    _sys.modules[__name__] = _impl
