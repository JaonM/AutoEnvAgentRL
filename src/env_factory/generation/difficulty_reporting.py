"""Disjoint frozen-profile yield and latency, including failed planned samples."""
from datetime import datetime
import json
import math
import statistics

from .difficulty_profiles import PROFILES


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _seconds(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _distribution(values):
    return {'observed': len(values), 'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def summarize_difficulty(task_dirs, sandbox_root, *, requested, wall_seconds):
    rows = []
    directories = sorted({directory.resolve() for directory in task_dirs})
    for directory in directories:
        manifest = _read(directory / 'sample_manifest.json')
        task = _read(directory / 'task.json')
        assigned = manifest.get('difficulty_profile')
        # Old cohorts were not stratified; never retroactively assign a profile
        # from an author's free-text complexity label.
        profile = assigned if assigned in PROFILES else 'unassigned'
        build_path = directory.parent / 'sandbox' if directory.name == 'task' else sandbox_root / directory.name
        build = _read(build_path / 'pipeline_result.json')
        generation = _read(directory / 'pipeline_result.json')
        generated = bool(task) and (not manifest or manifest.get('status') == 'completed')
        terminal = manifest.get('status') == 'failed' or bool(generation.get('failure_code')) or 'passed' in build
        reserved = _timestamp(manifest.get('reserved_at'))
        ended = _timestamp(build.get('build_completed_at')) if generated else None
        if ended is None and terminal and not generated:
            ended = _timestamp(manifest.get('failed_at') or manifest.get('completed_at'))
        elapsed = _seconds(ended - reserved) if ended is not None and reserved is not None else None
        feedback = []
        if reserved is not None:
            for path in directory.rglob('execution_progress.jsonl'):
                try:
                    lines = path.read_text().splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    when = _seconds(event.get('timestamp'))
                    if (event.get('status') == 'actionable_feedback' and when is not None
                            and when >= reserved and (ended is None or when <= ended)):
                        feedback.append(when - reserved)
        rows.append({'task_directory': str(directory), 'profile': profile, 'generated': generated,
            'build_attempted': bool(build), 'terminal': terminal,
            'training_ready': generated and build.get('training_ready') is True,
            'infrastructure_failure': (build or generation).get('failure_code') == 'INFRA',
            'reservation_to_terminal_seconds': elapsed,
            'reservation_to_first_recorded_feedback_seconds': min(feedback) if feedback else None})
    untracked = max(0, requested - len(rows))
    valid_denominator = requested > 0 and len(rows) <= requested
    groups = {}
    for name in ['all', *sorted({row['profile'] for row in rows}), *(['untracked'] if untracked else [])]:
        selected = rows if name == 'all' else [row for row in rows if row['profile'] == name]
        count = requested if name == 'all' else untracked if name == 'untracked' else len(selected)
        ready = sum(row['training_ready'] for row in selected)
        groups[name] = {'requested': count, 'generated': sum(row['generated'] for row in selected),
            'build_attempted': sum(row['build_attempted'] for row in selected),
            'terminal': sum(row['terminal'] for row in selected), 'training_ready': ready,
            'pending_or_untracked': count - sum(row['terminal'] for row in selected) if valid_denominator else None,
            'infrastructure_failures': sum(row['infrastructure_failure'] for row in selected),
            'training_ready_yield': ready / count if valid_denominator and count else None,
            'ready_per_cohort_wall_hour': ready * 3600 / wall_seconds
                if valid_denominator and _seconds(wall_seconds) and wall_seconds > 0 else None,
            **{key: _distribution([row[key] for row in selected if row[key] is not None]) for key in (
                'reservation_to_terminal_seconds', 'reservation_to_first_recorded_feedback_seconds')}}
    return {'version': '1.0', 'scope': 'current_invocation', 'requested': requested,
        'tracked_requests': len(rows), 'untracked_requests': untracked,
        'planned_denominator_valid': valid_denominator, 'cohort_wall_seconds': _seconds(wall_seconds),
        'groups': groups, 'samples': rows,
        'note': 'Difficulty groups are disjoint. Unassigned historical tasks stay unassigned. '
                'Latencies include failures and queue time when timestamps exist; observed counts expose missing evidence. '
                'Feedback measures only recorded actionable validation findings. '
                'Profile throughput is its contribution per shared cohort wall hour, not isolated profile runtime.'}
