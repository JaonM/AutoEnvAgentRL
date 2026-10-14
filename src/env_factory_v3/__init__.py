"""Compatibility alias for code written before the rename to `env_factory`.

Sandboxes built earlier contain generated env.py / verifier.py files that import `env_factory_v3.runtime`.
Every `env_factory_v3.X` resolves to the very same module object as `env_factory.X`, so exceptions such
as BusinessError raised by old sandbox code are the classes the runtime catches.
"""
import importlib
import importlib.abc
import importlib.util
import sys


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith('env_factory_v3.'):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module('env_factory' + spec.name[len('env_factory_v3'):])

    def exec_module(self, module):
        return None


if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
