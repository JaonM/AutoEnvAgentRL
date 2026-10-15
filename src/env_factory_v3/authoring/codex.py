"""Isolated `codex exec` sessions for model roles.

Each session works in its own folder (workspace-write sandbox) and may run the self-checks.
At most MAX_SESSIONS run at once across the process: four concurrent sessions triggered
OpenAI 403 responses in the 2026-10-12 spike. A session that fails because the service
refused it (403/429/5xx, no model output) is an infrastructure error and is retried once
after a backoff; it is never treated as a business failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
import subprocess
import threading
import time
from pathlib import Path

MODEL = 'gpt-6.1-sol'
# Isolation (plan W6): a session works on a copy of its folder in a fresh temporary directory, so no sibling session
# (the other implementation, the bundle) sits next to it; afterwards its log is scanned for reads of the project's
# output tree or paths that climb out of its folder, and a violating session's new or changed files are discarded.
# ENVFACTORY_CODEX_ISOLATE=0 turns this off.
PROJECT = Path(__file__).resolve().parents[3]
ISOLATION_ROOT = Path(os.environ.get('ENVFACTORY_CODEX_ROOT', tempfile.gettempdir())) / 'envfactory-codex'
CLIMB = re.compile(r'(?<![\w.])\.\./')
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
    if status == 'timeout' and (usage['output_tokens'] == 0 or refused):  # e.g. rate-limited mid-session
        return 'infrastructure_error'
    return status


SIBLING = re.compile(r'session-[A-Za-z0-9_]{6,}')


def isolation_violations(events_path, project=PROJECT, own=None):
    """Commands or paths in a session's log that reach the project's output tree, climb out of its folder, or name
    another session's temporary directory (concurrent sessions share ISOLATION_ROOT)."""
    found = []
    output = str(Path(project) / 'output')
    for line in Path(events_path).read_text().splitlines() if Path(events_path).exists() else []:
        others = [name for name in SIBLING.findall(line) if name != own]
        if output in line or CLIMB.search(line) or others:
            found.append(line[:200])
    return found[:5]


def _fingerprint(folder):
    return {p.relative_to(folder): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(folder).rglob('*') if p.is_file()}


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
        isolate = os.environ.get('ENVFACTORY_CODEX_ISOLATE', '1') != '0'
        workdir, before = folder, _fingerprint(folder)
        if isolate:
            ISOLATION_ROOT.mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix='session-', dir=ISOLATION_ROOT))
            shutil.copytree(folder, workdir, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('events-*.jsonl', 'stderr-*.txt'))
        with _SLOTS:
            started = time.monotonic()
            with open(events_path, 'w') as events, open(stderr_path, 'w') as errors:
                process = subprocess.Popen(command(workdir, self.model), stdin=subprocess.PIPE, stdout=events,
                                           stderr=errors, cwd=workdir, text=True, start_new_session=True)
                try:
                    process.communicate(prompt, timeout=timeout)
                    status = 'exited'
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    status = 'timeout'
            seconds = round(time.monotonic() - started, 1)
        if isolate:
            shutil.copytree(workdir, folder, dirs_exist_ok=True)
            shutil.rmtree(workdir, ignore_errors=True)
        violations = isolation_violations(events_path, own=workdir.name if isolate else None)
        if violations:
            # The session looked where it must not: whatever it wrote cannot be trusted. Its new or changed files go,
            # so the role's checks fail and the pipeline retries or rejects it as for any failed session.
            after = _fingerprint(folder)
            for name, digest in after.items():
                if before.get(name) != digest and not str(name).startswith(('events-', 'stderr-', 'prompt.txt')):
                    (folder / name).unlink(missing_ok=True)
            (folder / 'ISOLATION_VIOLATION.txt').write_text('\n'.join(violations))
        usage, errors = _usage(events_path)
        stderr_tail = stderr_path.read_text()[-400:]
        refused = any(marker in text for text in errors + [stderr_tail] for marker in INFRA_MARKERS)
        status = classify(status, process.returncode, usage, refused)
        return {'status': status, 'returncode': process.returncode, 'seconds': seconds,
                'usage': usage, 'errors': errors[-3:], 'stderr_tail': stderr_tail}


class CodexChat:
    """A chat-shaped client backed by one codex session per call, for low-volume writing and checking roles (the
    agent policy manual and its coverage check): same .chat(messages) interface as core.llm.ChatClient, so a role
    can switch providers without changing its code. Each call is an independent session in its own folder; the
    answer is the file answer.md the session writes (its last message otherwise)."""

    ANSWER = ('\n\n(Work alone in this folder: write your complete answer, exactly in the format asked above, to '
              'answer.md, and nothing else. Do not read or write files outside this folder.)')

    def __init__(self, root, *, runner=None, role='chat', timeout=900):
        self.root, self.role, self.timeout = Path(root), role, timeout
        self.runner = runner or CodexRunner()
        self.model = f'codex:{MODEL}'
        self._count, self._lock = 0, threading.Lock()

    def chat(self, messages, tools=None, **parameters):
        with self._lock:
            self._count += 1
            folder = self.root / f'{self.role}-{os.getpid()}-{int(time.time())}-{self._count:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        self.runner(folder, str(messages[-1]['content']) + self.ANSWER, timeout=self.timeout, role=self.role)
        for name in ('answer.md', 'last_message.txt'):
            if (folder / name).exists() and (folder / name).read_text().strip():
                return {'content': (folder / name).read_text().strip()}
        return {'content': ''}
