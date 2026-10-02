#!/usr/bin/env python3
"""Local GPU decode benchmark; no external API or environment calls."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3-4B-Instruct-2507-4bit')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='output/rl_runs/batch_decode_benchmark.json')
    args = parser.parse_args()
    if min(args.samples, args.tokens, args.repeats) < 1:
        parser.error('samples, tokens and repeats must be positive')
    from rl.model import Policy
    policy = Policy(args.model)
    messages = [[{'role':'user', 'content':f'Explain {i+2} ways to design a reliable computer program.'}]
                for i in range(args.samples)]
    def run(batched):
        decoder = policy.batch_decoder(greedy=True)
        started = time.perf_counter()
        results = {}
        if batched:
            for i, message in enumerate(messages):
                decoder.add(i, message, args.tokens, 4096)
            while decoder.requests:
                results.update(decoder.tick())
        else:
            # Same engine and statistics collection, with only one active row.
            for i, message in enumerate(messages):
                decoder.add(i, message, args.tokens, 4096)
                while decoder.requests:
                    results.update(decoder.tick())
        return time.perf_counter()-started, decoder, results
    run(False)
    run(True)
    times = {False:[], True:[]}
    for _ in range(args.repeats):
        for batched in (False, True):
            seconds, decoder, results = run(batched)
            times[batched].append(seconds)
    single, batch = (statistics.median(times[key]) for key in (False, True))
    result = dict(model=args.model, samples=args.samples, max_tokens=args.tokens, repeats=args.repeats,
                  sequential_seconds=single, batched_seconds=batch, speedup=single/batch,
                  batched_forward_calls=decoder.forward_calls, max_batch_size=decoder.max_batch_size,
                  generated_tokens=sum(len(r['tokens']) for r in results.values()),
                  scope='prefill + decode + behavior statistics; excludes environment and training')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
