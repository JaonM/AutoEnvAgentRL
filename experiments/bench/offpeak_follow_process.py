"""Preserve an existing frozen generation process while enforcing off-peak windows.

This migration adapter pauses the whole producer tree, including Codex children.
New producers use per-request gates instead. It never rewrites frozen inputs.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from env_factory.production import offpeak


def processes():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,lstart=,command='], text=True)
    result = {}
    for line in rows.splitlines():
        parts = line.split(None, 7)
        if len(parts) == 8:
            result[int(parts[0])] = {'parent': int(parts[1]), 'born': ' '.join(parts[2:7]), 'command': parts[7]}
    return result


def tree(rows, root):
    ids = {root}
    while True:
        added = {pid for pid, row in rows.items() if row['parent'] in ids} - ids
        if not added:
            return ids
        ids.update(added)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    rows = processes();identity = rows[args.pid]
    if 'supplement_world_diversity.py' not in identity['command'] or str(args.producer.resolve()) not in identity['command']:
        raise ValueError('PID is not the specified producer')
    held = {};last_state = None;last_saved = 0
    while True:
        rows = processes()
        if rows.get(args.pid) != identity:
            save(args.status, {'phase': 'producer_exited', 'updated': datetime.now().astimezone().isoformat()})
            return
        peak = offpeak.is_peak()
        ids = tree(rows, args.pid)
        if peak:
            # Stop the parent before discovering/stopping children, preventing new launches.
            os.kill(args.pid, signal.SIGSTOP)
            rows = processes();ids = tree(rows, args.pid)
            for pid in ids:
                try:
                    os.kill(pid, signal.SIGSTOP);held[pid] = rows[pid]['born']
                except ProcessLookupError:
                    pass
        else:
            # Includes a producer suspended before this adapter was installed.
            held.setdefault(args.pid, identity['born'])
            for pid, born in sorted(held.items(), key=lambda item: item[0] == args.pid):
                if rows.get(pid, {}).get('born') == born:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            held.clear()
        phase = 'waiting_for_offpeak' if peak else 'running_offpeak'
        if phase != last_state or time.monotonic() - last_saved > 30:
            save(args.status, {'phase': phase, 'updated': datetime.now().astimezone().isoformat(),
                 'producer_pid': args.pid, 'producer': str(args.producer.resolve()),
                 'held_pids': sorted(held), 'seconds_to_switch': offpeak.seconds_to_switch(),
                 'scope': 'existing frozen producer and descendants; no dataset or scoring changes'})
            last_state = phase;last_saved = time.monotonic()
        time.sleep(min(2, offpeak.seconds_to_switch()))


if __name__ == '__main__':
    main()
