"""Atomic JSON report writes."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
