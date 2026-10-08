"""Recognize the narrowly defined replay emitted by a fresh episode reset."""


def initial_replay_events(events):
    """Return comparable initialization events, or None for activity/invalid data."""
    if not isinstance(events, list) or len(events) > 1:
        return None
    if not events:
        return []
    event = events[0]
    if (not isinstance(event, dict) or set(event) - {'sequence', 'event', 'timestamp', 'payload', 'result'}
            or type(event.get('sequence')) is not int or event['sequence'] != 1
            or event.get('event') != 'user_protocol' or event.get('result') != {}):
        return None
    payload = event.get('payload')
    if not isinstance(payload, dict) or set(payload) != {'required_stages', 'script_id'}:
        return None
    stages = payload['required_stages']
    if (not isinstance(payload['script_id'], str) or not payload['script_id']
            or not isinstance(stages, list) or not all(isinstance(s, str) and s for s in stages)
            or len(set(stages)) != len(stages)):
        return None
    return [{key: value for key, value in event.items() if key != 'timestamp'}]
