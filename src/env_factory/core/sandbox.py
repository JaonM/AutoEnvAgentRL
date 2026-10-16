"""Static checks and loading for model-written business code (env.py, verifier.py).

This is defense in depth, not a security boundary: generated code must also run inside
the delivery container. The rules keep business code pure and deterministic: no I/O,
no network, no clock, no global randomness, no reflection tricks.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import uuid
from pathlib import Path

ALLOWED_MODULES = {
    'bisect', 'collections', 'copy', 'dataclasses', 'datetime', 'decimal', 'enum', 'fractions',
    'functools', 'heapq', 'itertools', 'json', 'math', 'operator', 're', 'statistics', 'string',
    'typing', '__future__', 'env_factory.runtime', 'env_factory_v3.runtime',
}
FORBIDDEN_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'globals', 'locals',
                   'vars', 'breakpoint', 'exit', 'quit', 'help', 'memoryview', 'setattr', 'delattr'}
ALLOWED_DUNDERS = {'__name__', '__doc__', '__init__', '__eq__', '__hash__', '__lt__', '__repr__',
                   '__post_init__', '__v3_write__'}


class SandboxError(ValueError):
    pass


def check_source(source, *, filename='<generated>'):
    """Return a list of rule violations (empty when the source is acceptable)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f'syntax error: {exc}']
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    problems.append(f'line {node.lineno}: import {alias.name} is not allowed')
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level or module not in ALLOWED_MODULES:
                problems.append(f'line {node.lineno}: from {"." * node.level}{module} import is not allowed')
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f'line {node.lineno}: {node.id} is not allowed')
        elif isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr not in ALLOWED_DUNDERS:
            problems.append(f'line {node.lineno}: attribute {node.attr} is not allowed')
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f'line {node.lineno}: global/nonlocal state is not allowed')
    return problems


def load_module(path, *, role):
    """Statically check then import a generated module under a unique name."""
    path = Path(path)
    source = path.read_text()
    problems = check_source(source, filename=str(path))
    if problems:
        raise SandboxError(f'{path.name}: ' + '; '.join(problems[:8]))
    name = f'env_factory_generated_{role}_{uuid.uuid4().hex[:12]}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    required = {'env': ('make_tools', 'reference'), 'verifier': ('expected',)}.get(role, ())
    missing = [attr for attr in required if not callable(getattr(module, attr, None))]
    if missing:
        raise SandboxError(f'{path.name}: missing {", ".join(missing)}')
    return module
