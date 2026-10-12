"""Isolated `codex exec` sessions for model roles.

Each session works in its own folder (workspace-write sandbox) and may run the self-checks.
At most MAX_SESSIONS run at once across the process: four concurrent sessions triggered
OpenAI 403 responses in the 2026-10-12 spike. A session that fails because the service
refused it (403/429/5xx, no model output) is an infrastructure error and is retried once
after a backoff; it is never treated as a business failure.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

MODEL = 'gpt-6.1-sol'
MAX_SESSIONS = 2
_SLOTS = threading.BoundedSemaphore(MAX_SESSIONS)
INFRA_MARKERS = ('403 Forbidden', '429', 'Too Many Requests', '502 Bad Gateway', '503', 'stream disconnected',
                 'No such file or directory (os error 2)')


class InfrastructureError(RuntimeError):
    pass


def _transport_args():
    try:
        from env_factory.generation.codex_transport import transport_args
        return transport_args()
    except Exception:
        return []


def command(folder, model=MODEL):
    folder = Path(folder).resolve()
    return ['codex', 'exec', '--ephemeral', '--sandbox', 'workspace-write', '--skip-git-repo-check',
            '--model', model, '--json', '--ignore-user-config', '-c', 'approval_policy="never"',
            '-c', 'web_search="disabled"', '-C', str(folder),
            '--output-last-message', str(folder / 'last_message.txt'), *_transport_args(), '-']


def _usage(events_path):
    usage = {'input_tokens': 0, 'output_tokens': 0}
    errors = []
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = event.get('usage') or {}
        for key in usage:
            if isinstance(found.get(key), int):
                usage[key] += found[key]
        item = event.get('item') or {}
        if item.get('type') == 'error' or event.get('type') == 'error':
            errors.append(str(item.get('message') or event.get('message'))[:300])
    return usage, errors


def classify(status, returncode, usage, refused):
    """A session the provider never served is infrastructure, not a business result: a refused or
    silent non-zero exit, or a timeout before the model produced a single token (network down)."""
    if status == 'exited' and returncode != 0 and (refused or usage['output_tokens'] == 0):
        return 'infrastructure_error'
    if status == 'timeout' and usage['output_tokens'] == 0:
        return 'infrastructure_error'
    return status


class CodexRunner:
    """Runs one role session; the pipeline injects a fake runner in tests."""

    def __init__(self, *, model=MODEL, retry_backoff=60):
        self.model, self.retry_backoff = model, retry_backoff

    def __call__(self, folder, prompt, *, timeout, role):
        folder = Path(folder).resolve()
        attempts = []
        for attempt in (1, 2):
            record = self._once(folder, prompt, timeout, attempt)
            attempts.append(record)
            if record['status'] != 'infrastructure_error':
                break
            if attempt == 1:
                time.sleep(self.retry_backoff)
        final = dict(attempts[-1], role=role, attempts=len(attempts))
        if final['status'] == 'infrastructure_error':
            raise InfrastructureError(f'{role}: {final.get("errors") or final.get("stderr_tail")}')
        return final

    def _once(self, folder, prompt, timeout, attempt):
        (folder / 'prompt.txt').write_text(prompt)
        events_path, stderr_path = folder / f'events-{attempt}.jsonl', folder / f'stderr-{attempt}.txt'
        with _SLOTS:
            started = time.monotonic()
            with open(events_path, 'w') as events, open(stderr_path, 'w') as errors:
                process = subprocess.Popen(command(folder, self.model), stdin=subprocess.PIPE, stdout=events,
                                           stderr=errors, cwd=folder, text=True, start_new_session=True)
                try:
                    process.communicate(prompt, timeout=timeout)
                    status = 'exited'
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    status = 'timeout'
            seconds = round(time.monotonic() - started, 1)
        usage, errors = _usage(events_path)
        stderr_tail = stderr_path.read_text()[-400:]
        refused = any(marker in text for text in errors + [stderr_tail] for marker in INFRA_MARKERS)
        status = classify(status, process.returncode, usage, refused)
        return {'status': status, 'returncode': process.returncode, 'seconds': seconds,
                'usage': usage, 'errors': errors[-3:], 'stderr_tail': stderr_tail}
