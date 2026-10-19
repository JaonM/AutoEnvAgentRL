from datetime import datetime
from types import SimpleNamespace

from env_factory.production import offpeak


def moment(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=offpeak.BEIJING)


def test_configured_windows_and_boundaries():
    assert not offpeak.is_peak(moment(19, 8, 59))
    assert offpeak.is_peak(moment(19, 9))
    assert not offpeak.is_peak(moment(19, 12))
    assert offpeak.is_peak(moment(19, 14))
    assert not offpeak.is_peak(moment(19, 18))
    assert not offpeak.is_peak(moment(18, 10))
    assert offpeak.seconds_to_switch(moment(19, 11, 59)) == 60


def test_remote_completion_waits_each_call_but_local_inference_does_not(monkeypatch):
    calls = []
    monkeypatch.setattr(offpeak, 'wait', lambda log: calls.append('wait'))
    wrapped = offpeak.gate_completion(lambda **kwargs: calls.append('send'))
    wrapped(model='openai/deepseek-flash', api_base='https://gateway.example/v1')
    wrapped(model='openai/deepseek-flash', api_base='https://gateway.example/v1')
    wrapped(model='deepseek-local', api_base='http://127.0.0.1:9510/v1')
    assert calls == ['wait', 'send', 'wait', 'send', 'send']


def test_chat_transport_checks_each_http_attempt(monkeypatch):
    from env_factory.core import llm
    events = []
    class FakeClient:
        def __init__(self, model):
            self.base_url = 'https://gateway.example/v1';self.model = model
            self.opener = SimpleNamespace(open=lambda request: events.append('send'))
    monkeypatch.setattr(llm, 'ChatClient', FakeClient)
    monkeypatch.setattr(offpeak, 'wait', lambda log: events.append('wait'))
    offpeak.gate_chat_clients();offpeak.gate_chat_clients()
    client = FakeClient('deepseek-flash')
    client.opener.open(None);client.opener.open(None)
    FakeClient('another-model').opener.open(None)
    assert events == ['wait', 'send', 'wait', 'send', 'send']


def test_wait_resumes_when_window_opens(monkeypatch):
    states = iter([True, True, False]);pauses = []
    monkeypatch.setenv('ENVFACTORY_OFFPEAK', '1')
    monkeypatch.setattr(offpeak, 'is_peak', lambda: next(states))
    monkeypatch.setattr(offpeak, 'seconds_to_switch', lambda: 10)
    monkeypatch.setattr(offpeak.time, 'sleep', pauses.append)
    assert offpeak.wait() == 20
    assert pauses == [10, 10]
