"""Content identities for frozen base weights and tokenizer, independent of path."""

import hashlib
import json
from pathlib import Path
from rl.checkpointing.checkpoint import sha256


def source_identity(root=None):
    """Hash all RL source files, including implementations below subpackages.

    Relative paths distinguish equal basenames in different packages. Older
    flat-layout manifests intentionally fail the existing strict resume check.
    """
    from rl.paths import RL_ROOT

    root = Path(root) if root is not None else RL_ROOT
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def model_identity(model):
    root = Path(model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(model))
    relevant = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix
        in {".safetensors", ".json", ".model", ".tiktoken", ".py", ".txt", ".jinja", ".jinja2"}
    )
    if not any(p.suffix == ".safetensors" for p in relevant):
        raise ValueError("model identity requires local safetensors weights")
    files = {str(p.relative_to(root)): sha256(p) for p in relevant}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return str(root.resolve()), identity, files
