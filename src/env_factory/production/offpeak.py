"""DeepSeek's off-peak hours: high-volume model work (policy gates, teacher collection, re-grading, quick evals,
RL rollouts) waits for them, at about half the price.

Peak is 09:00-12:00 and 14:00-18:00 Beijing time on weekdays; Saturday and Sunday are off-peak all day.
ENVFACTORY_OFFPEAK=0 turns waiting off.

    python -m env_factory.production.offpeak --wait      # block until off-peak
    python -m env_factory.production.offpeak --status    # "peak" or "off-peak", seconds to the next switch
"""
from __future__ import annotations

import argparse
import datetime
import os
import time
from functools import wraps
from urllib.parse import urlparse

BEIJING = datetime.timezone(datetime.timedelta(hours=8))
PEAK = ((9, 12), (14, 18))  # Beijing hours, [start, end)


def enabled():
    return os.environ.get('ENVFACTORY_OFFPEAK', '1') != '0'


def is_peak(moment=None):
    moment = (moment or datetime.datetime.now(BEIJING)).astimezone(BEIJING)
    if moment.weekday() >= 5:  # weekends are off-peak around the clock
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


def scheduled_remote(base_url, model=''):
    """Identify DeepSeek calls, including a configured provider gateway; exclude local inference."""
    host = urlparse(base_url or '').hostname
    if host in ('localhost', '127.0.0.1', '::1'):
        return False
    return ('deepseek' in str(model).lower() or 'deepseek' in (host or '').lower())


def gate_completion(completion):
    """Check every tau2 completion, including calls after a tariff-window boundary."""
    @wraps(completion)
    def call(*args, **kwargs):
        model = kwargs.get('model', args[0] if args else '')
        if scheduled_remote(kwargs.get('api_base') or kwargs.get('base_url'), model):
            wait(lambda message: print(message, flush=True))
        return completion(*args, **kwargs)
    return call


def gate_chat_clients():
    """Opt-in batch transport scheduling, leaving sandbox semantics and identity unchanged.

    Gate each HTTP attempt, rather than a whole conversation or only its first
    call. An already submitted request may finish after the boundary.
    """
    from env_factory.core.llm import ChatClient
    if getattr(ChatClient, '_offpeak_transport', False):
        return
    initialize = ChatClient.__init__

    @wraps(initialize)
    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if scheduled_remote(self.base_url, self.model):
            opening = self.opener.open

            @wraps(opening)
            def open_request(*a, **k):
                wait(lambda message: print(message, flush=True))
                return opening(*a, **k)
            self.opener.open = open_request

    ChatClient.__init__ = init
    ChatClient._offpeak_transport = True


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
