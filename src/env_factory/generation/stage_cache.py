"""Opt-in, content-addressed stage checkpoints; no credentials or prompts stored."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


class StageCache:
    def __init__(self, root: str | Path | None, *, revision: str, model: str) -> None:
        self.root = Path(root) if root else None
        self.revision, self.model = revision, model

    def key(self, stage: str, system: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps([self.revision, self.model, stage, system, payload], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def read(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        try:
            envelope = json.loads((self.root / f"{key}.json").read_text(encoding="utf-8"))
            value = envelope["result"]
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            return value if isinstance(value, dict) and digest == envelope.get("digest") else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, key: str, stage: str, result: dict[str, Any]) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        envelope = {"stage": stage, "revision": self.revision, "model": self.model,
                    "result": result, "digest": digest}
        temporary = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.root / f"{key}.json")
        finally:
            temporary.unlink(missing_ok=True)
