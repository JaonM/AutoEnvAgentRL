"""One recursive inventory for exported code and implementation fingerprints."""
from pathlib import Path
import hashlib

PACKAGE_ROOT = Path(__file__).parent


def package_files(*, root=None, suffixes=('.py', '.md')):
    root = Path(root) if root is not None else PACKAGE_ROOT
    return {path.relative_to(root).as_posix(): path for path in sorted(root.rglob('*'))
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts}


def package_fingerprints(*, root=None, suffixes=('.py', '.md')):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in package_files(root=root, suffixes=suffixes).items()}
