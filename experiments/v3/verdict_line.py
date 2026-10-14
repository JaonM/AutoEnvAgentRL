"""One log line for a dev_check verdict: python experiments/v3/verdict_line.py LABEL VERDICT.json"""
import json
import sys

KEYS = ('episodes', 'success', 'mean_reward', 'repeat_failures', 'step_limit', 'wide_turns', 'ungrounded_ids',
        'asks_before_lookup')
label, path = sys.argv[1], sys.argv[2]
report = json.load(open(path))
model = report.get('model', report)
print(f"{label}: passed={report.get('passed')} " + ' '.join(f'{k}={model.get(k)}' for k in KEYS)
      + (f" problems={report.get('problems')}" if report.get('problems') else ''))
