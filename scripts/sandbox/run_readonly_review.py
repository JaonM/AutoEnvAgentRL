#!/usr/bin/env python3
"""Run a read-only Codex review with bounded transient-connection retries."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import time


TRANSIENT = re.compile(
    r"failed to initialize in-process app-server client|"
    r"connection (?:reset|refused|closed|error)|network (?:error|unavailable)|"
    r"timed out|timeout|HTTP (?:429|502|503|504)|rate limit|stream disconnected",
    re.IGNORECASE,
)
INFRA_EXIT = 75


def run_review(
    *, cwd: Path, prompt: str, response: Path, stdout: Path, stderr: Path,
    model: str = "", retries: int = 3, retry_delay: float = 1.0,
) -> int:
    command = ["codex", "exec", "--ephemeral", "--sandbox", "read-only",
               "--output-last-message", str(response)]
    if model:
        command += ["--model", model]
    command.append(prompt)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    for attempt in range(1, retries + 1):
        response.unlink(missing_ok=True)
        try:
            completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            output, error, code = completed.stdout, completed.stderr, completed.returncode
        except OSError as exc:
            output, error, code = "", str(exc), 1
        with stdout.open("a", encoding="utf-8") as handle:
            handle.write(f"review attempt {attempt}/{retries}\n{output}\n")
        with stderr.open("a", encoding="utf-8") as handle:
            handle.write(f"review attempt {attempt}/{retries}\n{error}\n")
        if code == 0 and response.is_file() and response.stat().st_size:
            return 0
        if not TRANSIENT.search(output + "\n" + error):
            return code or 65
        if attempt < retries:
            time.sleep(retry_delay * attempt)
    return INFRA_EXIT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    return run_review(
        cwd=args.cwd, prompt=args.prompt, response=args.response,
        stdout=args.stdout, stderr=args.stderr, model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
