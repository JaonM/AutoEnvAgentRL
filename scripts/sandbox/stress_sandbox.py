#!/usr/bin/env python3
"""Small Trainer-protocol load probe for a running sandbox."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from urllib.request import Request, urlopen


def request(base: str, token: str, index: int) -> tuple[int, float]:
    started = time.perf_counter()
    body = json.dumps({"episode_id": f"stress-{index}", "seed": index}).encode()
    req = Request(base.rstrip("/") + "/v1/reset", data=body, method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urlopen(req, timeout=15) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"reset returned {response.status}")
    return index, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: request(args.base_url, args.token, i), range(args.requests)))
    durations = [duration for _, duration in results]
    print(json.dumps({"requests": len(results), "concurrency": args.concurrency, "duration_ms": (time.perf_counter() - started) * 1000, "p50_ms": sorted(durations)[len(durations)//2], "max_ms": max(durations)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
