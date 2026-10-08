"""Observe a silent initial Codex invocation within its existing time budget."""
import hashlib
import json
import subprocess
import time


class NoAuthorActivity(subprocess.TimeoutExpired):
    pass


def workspace_signature(workspace):
    result = {}
    for name in ('blueprint.json', 'source.json', 'response.txt', 'request.json', 'AUTHORING.md'):
        path = workspace / name
        if path.is_symlink():
            result[name] = 'symlink'  # Do not follow an authored link.
        elif path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def has_activity(events):
    for line in events.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'turn.completed':
            return True
        if (str(event.get('type', '')).startswith('item.')
                and isinstance(event.get('item'), dict) and event['item'].get('type') != 'error'):
            return True
    return False


def communicate(process, prompt, *, budget, events, stdout, workspace, before, attempt, window=120):
    """Probe only when the caller selected a fresh blueprint startup watch.

    A probe is not a business repair and does not measure provider/network time.
    Once activity exists, keep the process and its original absolute deadline.
    """
    started = time.monotonic()
    try:
        process.communicate(prompt, timeout=window)
    except subprocess.TimeoutExpired:
        stdout.flush()
        cli_activity = has_activity(events)
        changed = workspace_signature(workspace) != before
        attempt['startup_check'] = {'window_seconds': window,
            'observed_at': time.time(), 'elapsed_seconds': time.monotonic() - started,
            'substantive_cli_activity': cli_activity, 'workspace_changed': changed,
            'scope': 'Observed CLI/workspace activity only; no provider-cause attribution.'}
        if not cli_activity and not changed:
            raise NoAuthorActivity('initial author invocation without substantive activity', window)
        process.communicate(timeout=max(0, budget - (time.monotonic() - started)))
