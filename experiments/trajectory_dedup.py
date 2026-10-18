"""Conservative, task-local trajectory deduplication before SFT augmentation.

Raw collection/evaluation logs are never edited. This is a lexical/structural filter,
not an LLM semantic judge: different arguments, results, dialogue branches, world
variants and failed-call recovery paths are retained. No industry tool names are used.

    python experiments/trajectory_dedup.py --teacher raw.jsonl --output candidates.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

VERSION = 1
DEFAULT_THRESHOLD = 0.85


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def task_key(episode):
    # Domain disambiguates external datasets that reuse numeric task IDs.
    return packed([episode.get('domain'), episode.get('id')])


def training_candidate(episode):
    return bool((episode.get('success') or episode.get('near_miss')) and episode.get('messages')
                and (episode.get('trajectory_policy') or {}).get('passed') is not False)


def _features(episode):
    aliases = {v: k for k, v in (episode.get('naming') or {}).items()
               if isinstance(v, str) and not k.startswith('_')}

    def names(text):
        return re.sub(r'\b[A-Za-z_][A-Za-z_0-9]*\b', lambda m: aliases.get(m[0], m[0]), text)

    def value(raw, parse=False):
        if parse and isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return raw
        if isinstance(raw, dict):
            return {('consent' if aliases.get(k) == 'consent' else k): value(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return [value(v) for v in raw]
        return raw

    trace, dialogue, call_ids = [], [], {}
    for message in episode.get('messages') or []:
        role = message.get('role')
        text = message.get('content') or ''
        if role == 'system':
            trace.append(['system', names(text)])
        elif role == 'tool':
            # IDs vary between runs; result order, content and error metadata matter.
            trace.append(['tool', aliases.get(message.get('name'), message.get('name')),
                          call_ids.get(message.get('tool_call_id'), message.get('tool_call_id')),
                          value(text, parse=True), message.get('tool_failure')])
        else:
            for c in message.get('tool_calls') or []:
                if c.get('id') is not None:
                    call_ids[c['id']] = len(call_ids)
            calls = [[aliases.get(c['function']['name'], c['function']['name']),
                      value(c['function'].get('arguments') or '{}', parse=True)]
                     for c in message.get('tool_calls') or []]
            # Preserve batching and mixed text/calls. Neither is a quality violation.
            trace.append([role, calls, bool(text.strip())])
            if text.strip():
                dialogue.append(' '.join(names(text).casefold().split()))
    context = [episode.get(k) for k in ('world', 'consent', 'language', 'platform_identity',
                                       'evaluation_identity')]
    context += [bool(episode.get('success')), bool(episode.get('near_miss'))]
    return packed([task_key(episode), context, trace]), dialogue


# Protect factual and polarity changes even when nearly all surrounding words match.
ANCHORS = re.compile(r'\b[\w#@.-]*\d[\w@./:#-]*\b|\b(?:no|not|never|cannot|can\'t|won\'t|'
                     r'don\'t|didn\'t|refuse|refused|decline|declined)\b|不|没|无|拒绝', re.I)


def dialogue_similarity(left, right):
    if len(left) != len(right):
        return 0.0
    scores = []
    for a, b in zip(left, right):
        if ANCHORS.findall(a) != ANCHORS.findall(b):
            return 0.0
        # Each turn must match; long boilerplate cannot drown out a changed reply.
        scores.append(SequenceMatcher(None, a, b, autojunk=False).ratio())
    return min(scores, default=1.0)


def _reference(episode):
    return {**{k: episode.get(k) for k in ('id', 'domain', 'seed', 'world', 'source_episode')},
            'sha256': hashlib.sha256(packed(episode).encode()).hexdigest()}


def _quality(episode):
    failures = 0
    for m in episode.get('messages') or []:
        if m.get('role') != 'tool':
            continue
        raw = m.get('content') or ''
        try:
            result = json.loads(raw)
        except (ValueError, TypeError):
            result = {}
        failures += bool(m.get('tool_failure') or (isinstance(result, dict) and result.get('ok') is False)
                         or (isinstance(raw, str) and raw.startswith('Error')))
    return (not bool(episode.get('success')),
            (episode.get('trajectory_policy') or {}).get('passed') is not True,
            -float(episode.get('reward') or 0), failures, len(episode.get('messages') or []),
            _reference(episode)['sha256'])


def deduplicate(episodes, threshold=DEFAULT_THRESHOLD, *, enabled=True, priority=None):
    """Return original retained records plus an audit. Never modify input records.

    Only compare against retained representatives (no transitive similarity chaining).
    Missing task IDs are kept separately. `priority` may supply a quality sort key.
    """
    if not 0 < threshold <= 1:
        raise ValueError('dedup threshold must be in (0, 1]')
    buckets, retained, dropped = collections.defaultdict(list), set(), []
    rank = (lambda e: (priority(e), _quality(e))) if priority else _quality
    for index in sorted(range(len(episodes)), key=lambda i: rank(episodes[i])):
        episode = episodes[index]
        key, dialogue = _features(episode)
        match = None
        if enabled and episode.get('id') is not None and episode.get('messages'):
            for other, other_dialogue in buckets[key]:
                score = dialogue_similarity(dialogue, other_dialogue)
                if score >= threshold:
                    match = other, score
                    break
        if match is None:
            buckets[key].append((index, dialogue))
            retained.add(index)
        else:
            other, score = match
            dropped.append({'dropped': _reference(episode), 'kept': _reference(episodes[other]),
                            'similarity': round(score, 6), 'reason': 'same_behavior_similar_dialogue'})
    kept = [e for i, e in enumerate(episodes) if i in retained]
    kept_counts = collections.Counter(map(task_key, kept))
    report = {'version': VERSION, 'enabled': enabled, 'threshold': threshold, 'input': len(episodes),
              'kept': len(kept), 'dropped': len(dropped), 'comparisons': dropped,
              'per_task': {key: {'input': count, 'kept': kept_counts[key]}
                           for key, count in sorted(collections.Counter(map(task_key, episodes)).items())}}
    return kept, report


def add_arguments(parser):
    parser.add_argument('--dedup-threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='minimum per-dialogue-turn similarity after exact behavior matching (default: .85)')
    parser.add_argument('--no-dedup', action='store_true', help='disable trajectory dedup for controlled comparisons')


def validation_keys(episodes, share):
    """Split already shuffled episodes by task, keeping all variants on one side."""
    keys = list(dict.fromkeys(map(task_key, episodes)))
    return set(keys[:max(1, round(len(keys) * share))])


def write_candidates(raw, output, threshold=DEFAULT_THRESHOLD, *, enabled=True):
    """Write a separate candidate file; downstream quality/balance gates still apply."""
    candidates = [e for e in raw if training_candidate(e)]
    kept, report = deduplicate(candidates, threshold, enabled=enabled)
    report['ineligible'] = len(raw) - len(candidates)
    output = Path(output)
    if output == output.with_suffix('.dedup.json'):
        raise ValueError('candidate output must differ from its .dedup.json audit path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in kept))
    output.with_suffix('.dedup.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    add_arguments(parser)
    args = parser.parse_args()
    output = Path(args.output)
    inputs = {Path(p).resolve() for p in args.teacher}
    if {output.resolve(), output.with_suffix('.dedup.json').resolve()} & inputs:
        parser.error('--output must not overwrite a raw teacher input')
    raw = [json.loads(line) for path in args.teacher for line in Path(path).read_text().splitlines() if line.strip()]
    report = write_candidates(raw, output, args.dedup_threshold, enabled=not args.no_dedup)
    print(json.dumps({k: v for k, v in report.items() if k not in ('comparisons', 'per_task')}))


if __name__ == '__main__':
    main()
