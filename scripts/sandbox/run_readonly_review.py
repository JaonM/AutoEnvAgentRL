#!/usr/bin/env python3
"""Run a read-only Codex review with a shared deadline and streaming logs."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from env_factory.generation.codex_transport import transport_args

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
    timeout: float = 600.0,
) -> int:
    if (retries < 1 or not math.isfinite(timeout) or timeout <= 0
            or not math.isfinite(retry_delay) or retry_delay < 0):
        raise ValueError("Review retries/timeout must be positive; retry delay must be finite and nonnegative")
    command = ["codex", "exec", "--ephemeral", "--sandbox", "read-only",
               "--output-last-message", str(response)]
    command += transport_args()
    if model:
        command += ["--model", model]
    command.append(prompt)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    deadline = time.monotonic() + timeout
    for attempt in range(1, retries + 1):
        response.unlink(missing_ok=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return INFRA_EXIT
        with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
            out.write(f"review attempt {attempt}/{retries}\n")
            out.flush()
            err.write(f"review attempt {attempt}/{retries}; remaining_seconds={remaining:.3f}\n")
            err.flush()
            out_offset, err_offset = out.tell(), err.tell()
            process = None
            try:
                process = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=err,
                                           text=True, start_new_session=True)
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                err.write("Review total deadline exceeded\n")
                err.flush()
                return INFRA_EXIT
            except OSError as exc:
                err.write(str(exc) + "\n")
                err.flush()
                code = 1
            finally:
                if process is not None:
                    # Clean the entire CLI tree on timeout, cancellation and exit.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if code == 0 and response.is_file() and response.stat().st_size:
            return 0
        with stdout.open(encoding="utf-8") as handle:
            handle.seek(out_offset)
            output = handle.read()
        with stderr.open(encoding="utf-8") as handle:
            handle.seek(err_offset)
            error = handle.read()
        if not TRANSIENT.search(output + "\n" + error):
            return code or 65
        if attempt < retries:
            time.sleep(min(retry_delay * attempt, max(0, deadline - time.monotonic())))
    return INFRA_EXIT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cwd", "response", "stdout", "stderr"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=float, default=600,
                        help="Total review seconds including connection retries (default 600)")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run_review(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
