"""DeepSeek's off-peak hours: high-volume model work (policy gates, teacher collection, re-grading, quick evals,
RL rollouts) waits for them, at about half the price.

Peak is 09:00-12:00 and 14:00-18:00 Beijing time. Whether weekends are entirely off-peak is not settled in the
public reports, so they count like weekdays unless OFFPEAK_WEEKENDS=1. ENVFACTORY_OFFPEAK=0 turns waiting off.

    python -m env_factory_v3.production.offpeak --wait      # block until off-peak
    python -m env_factory_v3.production.offpeak --status    # "peak" or "off-peak", seconds to the next switch
"""
from __future__ import annotations

import argparse
import datetime
import os
import time

BEIJING = datetime.timezone(datetime.timedelta(hours=8))
PEAK = ((9, 12), (14, 18))  # Beijing hours, [start, end)


def enabled():
    return os.environ.get('ENVFACTORY_OFFPEAK', '1') != '0'


def is_peak(moment=None):
    moment = (moment or datetime.datetime.now(BEIJING)).astimezone(BEIJING)
    if moment.weekday() >= 5 and os.environ.get('OFFPEAK_WEEKENDS') == '1':
        return False
    return any(start <= moment.hour < end for start, end in PEAK)


def seconds_to_switch(moment=None):
    """Seconds until the current period (peak or off-peak) ends."""
    moment = (moment or datetime.datetime.now(BEIJING)).astimezone(BEIJING)
    peak = is_peak(moment)
    probe = moment.replace(second=0, microsecond=0)
    for minutes in range(1, 3 * 24 * 60):
        if is_peak(probe + datetime.timedelta(minutes=minutes)) != peak:
            return max(1, int((probe + datetime.timedelta(minutes=minutes) - moment).total_seconds()))
    return 3600


def wait(log=None):
    """Block while it is peak time (when enabled); returns the seconds waited."""
    waited = 0
    while enabled() and is_peak():
        pause = min(seconds_to_switch(), 900)
        if log and not waited:
            log(f'peak hours: waiting for off-peak (about {seconds_to_switch() // 60} min)')
        time.sleep(pause)
        waited += pause
    return waited


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.wait:
        wait(lambda text: print(text, flush=True))
    else:
        print(('peak' if is_peak() else 'off-peak'), seconds_to_switch(), flush=True)


if __name__ == '__main__':
    main()
