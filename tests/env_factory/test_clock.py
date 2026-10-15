"""Advancing clock: explicit waits and hidden scheduled events."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_core import library_modules, library_request  # noqa: E402

from env_factory.core import spec  # noqa: E402
from env_factory.runtime import Episode  # noqa: E402
from env_factory.serving import agent_view, server


def with_event(request):
    request = copy.deepcopy(request)
    hold = next(h for h in request['tables']['holds'] if h['active'])
    request['tables']['scheduled_events'] = [
        {'id': 'E1', 'at': 30, 'table': 'holds', 'row_id': hold['id'], 'set': {'active': False}}]
    return request, hold['id']


class ClockTests(unittest.TestCase):
    def test_events_apply_when_the_clock_passes_and_stay_hidden(self):
        request, hold_id = with_event(library_request())
        env, _, _ = library_modules()
        seen = {}

        def peek(db, now):
            """Return the hold rows a tool sees."""
            seen['tables'] = sorted(db)
            return [h for h in db['holds'] if h['id'] == hold_id]

        tools = {**env.make_tools(), 'peek': peek}
        episode = Episode(tools, request['tables'], request['user_facts'], start=10)
        self.assertTrue(episode.call('peek')[0]['active'])
        self.assertNotIn('scheduled_events', seen['tables'])
        episode.wait(15)
        self.assertTrue(episode.call('peek')[0]['active'])
        episode.wait(5)
        self.assertFalse(episode.call('peek')[0]['active'])
        episode.submit({})
        final = episode.result()['final_state']
        self.assertEqual(final['scheduled_events'], request['tables']['scheduled_events'])

    def test_requests_validate_events(self):
        request, _ = with_event(library_request())
        self.assertEqual(spec.validate(request), [])
        request['tables']['scheduled_events'][0]['row_id'] = 'nope'
        self.assertTrue(spec.validate(request))
        self.assertIn('advances only when the agent waits', spec.brief(with_event(library_request())[0]))

    def test_wait_is_offered_only_when_the_clock_advances(self):
        env, verifier, schema = library_modules()
        fixed = server.Service(env, verifier, schema, library_request())
        names = lambda service: [t['function']['name'] for t in agent_view.tools(
            service.instructions, 'multi_step_agentic')]
        self.assertNotIn('v3_wait', names(fixed))
        episode = fixed.reset(world='original')['episode']
        self.assertEqual(fixed.step(episode, {'type': 'wait', 'seconds': 5})['kind'], 'invalid')
        with self.assertRaises(ValueError):  # in process too, so the author's self-check catches it
            Episode(env.make_tools(), library_request()['tables'], {}, start=10).wait(1)
        request, _ = with_event(library_request())
        advancing = server.Service(env, verifier, schema, request)
        self.assertIn('v3_wait', names(advancing))
        episode = advancing.reset(world='original')['episode']
        self.assertEqual(advancing.step(episode, {'type': 'wait', 'seconds': 5})['result'], {'now': 15})


class CalendarClockTests(unittest.TestCase):
    def test_a_calendar_now_is_valid_fixed_and_shown_as_a_date(self):
        from env_factory.core import spec
        from env_factory.runtime import Episode
        from env_factory.serving import agent_view
        request = {**library_request(), 'time': {'start': '2024-05-15T15:00', 'tick': 0}}
        self.assertFalse([p for p in spec.validate(request) if 'time' in p])
        bad = {**library_request(), 'time': {'start': '15 May', 'tick': 0}}
        self.assertTrue([p for p in spec.validate(bad) if 'time' in p])
        seen = []
        episode = Episode({'probe': lambda db, now: seen.append(now) or {}}, {}, {}, start='2024-05-15T15:00')
        episode.call('probe')
        episode.call('probe')
        self.assertEqual(seen, ['2024-05-15T15:00', '2024-05-15T15:00'])  # tools get the calendar now, fixed
        text = agent_view.clock_text(agent_view.TEXT['en'], '2024-05-15T15:00', 'en')
        self.assertEqual(text, 'Current time: 2024-05-15 15:00 (Wednesday)')

    def test_perturbed_worlds_move_dates_by_hours_or_days(self):
        from env_factory.core.worlds import perturb
        data = {'bookings': [{'id': f'b{i}', 'departs': '2024-05-31T23:00', 'booked_on': '2024-05-14'}
                             for i in range(3)]}
        moved = [perturb(data, seed) for seed in range(1, 30)]
        values = {row['departs'] for world in moved for row in world['bookings']} | \
            {row['booked_on'] for world in moved for row in world['bookings']}
        self.assertIn('2024-06-01T00:00', values)  # across a day and a month boundary
        self.assertTrue(any(v != '2024-05-14' and len(v) == 10 for v in values))


if __name__ == '__main__':
    unittest.main()
