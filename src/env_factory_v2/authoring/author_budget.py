"""Bounded author sessions: real draft progress extends idle time, never total time."""
import json
import subprocess
import time
from pathlib import Path


class ProgressBudget:
    def __init__(self, started, total, idle):
        self.started = self.last_progress = started
        self.total, self.idle = total, min(idle, total)
        self.seen = set()

    def observe(self, receipt, now):
        result = receipt.get('result', {})
        tool, identity = receipt.get('tool'), result.get('source_identity')
        # Reads, repeated checks, failed checks and toggling back to an old
        # version are not fresh progress. A new successful check is progress.
        kind = 'checked' if tool == 'check_contract' else 'draft'
        marker = (kind, identity)
        if (identity and result.get('passed') is True and
                tool in {'put_section', 'edit_contract', 'check_contract'} and marker not in self.seen):
            self.seen.add(marker)
            self.last_progress = now

    def remaining(self, now):
        return min(self.started + self.total, self.last_progress + self.idle) - now


def monitor(process, prompt, folder, *, total, idle, submitted):
    started = time.monotonic()
    budget = ProgressBudget(started, total, idle)
    receipt_path = Path(folder) / 'workbench/latest_operation.json'
    initial = Path(folder) / 'initial.json'
    if initial.exists():
        import hashlib
        from env_factory_v2.contracts.expressions import canonical
        budget.seen.add(('draft', hashlib.sha256(canonical(json.loads(initial.read_text())).encode()).hexdigest()))
    first = True
    while True:
        if submitted():
            return 'submitted'
        now = time.monotonic()
        try:
            budget.observe(json.loads(receipt_path.read_text()), now)
        except (OSError, ValueError):
            pass
        remaining = budget.remaining(now)
        if remaining <= 0:
            return 'total_timeout' if now >= started + total else 'idle_timeout'
        try:
            process.communicate(prompt if first else None, timeout=min(1, remaining))
            return 'exited'
        except subprocess.TimeoutExpired:
            first = False
