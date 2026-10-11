"""Bounded sample workers; only the controller mutates the published cohort."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_batch(samples, *, concurrency, execute, started, completed):
    if type(concurrency) is not int or concurrency not in (1, 2):
        raise ValueError('batch concurrency must be 1 or 2')
    pending = iter(samples)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        active = {}
        def launch():
            try:
                sample = next(pending)
            except StopIteration:
                return
            started(sample)
            active[executor.submit(execute, dict(sample))] = sample
        for _ in range(concurrency):
            launch()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                sample = active.pop(future)
                completed(sample, future.result())
            for _ in ready:
                launch()
