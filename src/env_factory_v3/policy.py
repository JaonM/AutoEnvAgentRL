"""Policy gate: can a strong model, seeing only what a trained policy sees, solve the task?

An independent OpenAI-compatible model (not the one that wrote env.py and oracle.py) plays k
episodes through the bundle's HTTP service and the shared agent view, one world per episode in
the bundle's served order. A task no strong policy solves gives a learner no successful
trajectories to start from. Provider failures are infrastructure errors, never business results.

    python -m env_factory_v3 policy BUNDLE [--k 4] [--prefix LLM] [--user-mode natural]
    # player: <PREFIX>_API_KEY/BASE_URL/MODEL; natural user simulator: V3_USER_* (falls back to LLM_*)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from env_factory_v3 import agent_view, bundle, server
from env_factory_v3.llm import ChatClient, ProviderError, load_env_file  # noqa: F401  (re-exported)

RESULT = 'policy.json'


def _post(opener, url, body):
    # /reward of a rubric task waits for the judge model (with its retries), hence the long timeout.
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
                      headers={'Content-Type': 'application/json'})
    try:
        with opener.open(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code >= 500:  # judge or simulator provider failure inside the service
            raise ProviderError(f'service failed: HTTP {error.code}') from error
        return json.load(error)
    except (URLError, TimeoutError, OSError) as error:
        raise ProviderError(f'service unreachable: {type(error).__name__}') from error


LEFT = '（系统提示：用户已经结束了对话。请立即调用 v3_answer 提交最终结果；不提交，本次任务记为未完成。）'


def left(reply):
    """'warning' or 'closed' when the simulated user has left the conversation (see server.IDLE_LIMIT)."""
    if reply.get('kind') == 'finished':
        return 'closed'
    return (reply.get('result') or {}).get('ended')


def play(base_url, client, world, category, *, max_turns=30):
    """One episode on `world`; returns its grade and transcript."""
    opener = build_opener(ProxyHandler({}))
    reset = _post(opener, base_url + '/reset', {'world': world})
    episode, instructions = reset['episode'], reset['instructions']
    names = {tool['name'] for tool in instructions['tools']}
    tools = agent_view.tools(instructions, category)
    messages = agent_view.messages(instructions)
    answered, closed, calls, turns = False, False, 0, 0
    step = lambda action: _post(opener, base_url + '/step', {'episode': episode, 'action': action})
    try:
        while not answered and not closed and turns < max_turns:
            turns += 1
            reply = client.chat(messages, tools)
            message = {'role': 'assistant', 'content': reply.get('content') or ''}
            if reply.get('tool_calls'):
                message['tool_calls'] = reply['tool_calls']
            messages.append(message)
            if not reply.get('tool_calls'):
                if agent_view.speaks(instructions):
                    said = step({'type': 'chat', 'message': message['content']})
                    text = (said.get('result') or {}).get('reply') or said.get('error', '')
                    ended = left(said)
                    if ended == 'closed':
                        answered = (said.get('result') or {}).get('submitted') is True
                        break
                    messages.append({'role': 'user', 'content': f'{text}\n\n{LEFT}' if ended else text})
                    continue
                answer = agent_view.final_answer(reply.get('content'))
                step({'type': 'answer', 'answer': answer if answer is not None else {}})
                answered = True
                break
            for call in reply['tool_calls']:
                name = call['function']['name']
                try:
                    params = json.loads(call['function'].get('arguments') or '{}')
                    action = agent_view.action(name, params, names)
                except (ValueError, KeyError) as error:
                    result = {'ok': False, 'kind': 'invalid', 'error': str(error)}
                else:
                    if answered:
                        result = {'ok': False, 'kind': 'finished', 'error': 'episode already answered'}
                    elif action is None:
                        result, answered = step({'type': 'answer', 'answer': params}), True
                    else:
                        calls += action['type'] == 'tool'
                        result = step(action)
                        ended = left(result) if action['type'] == 'chat' else None
                        if ended == 'warning':
                            result = {**result, 'notice': LEFT}
                        elif ended == 'closed':
                            answered = (result.get('result') or {}).get('submitted') is True
                            closed = True
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', name), 'name': name,
                                 'content': json.dumps(result, ensure_ascii=False)})
        if not answered:
            # Same rule as training (rl V3Episode.finish): no final answer earns 0, whatever the state.
            return {'world': world, 'reward': 0.0, 'success': False, 'fields': None, 'state_ok': None,
                    'safety': None, 'turns': turns, 'tool_calls': calls, 'answered': False,
                    'reason': 'no_final_answer', 'messages': messages}
        grade = _post(opener, base_url + '/reward', {'episode': episode})
        return {'world': world, 'reward': grade.get('reward'), 'success': grade.get('success') is True,
                'fields': grade.get('fields'), 'state_ok': grade.get('state_ok'), 'safety': grade.get('safety'),
                'turns': turns, 'tool_calls': calls, 'answered': answered, 'messages': messages}
    finally:
        try:
            _post(opener, base_url + '/release', {'episode': episode})
        except ProviderError:
            pass


def gate(folder, client, *, k=4, parallel=4, max_turns=30, user_mode='structured', user_client=None):
    """k episodes on the bundle's served worlds; passes when at least one succeeds."""
    folder = Path(folder)
    loaded = bundle.load_bundle(folder)
    service = bundle.service(loaded, user_mode=user_mode, user_client=user_client)
    httpd = server.start(service)
    base_url = f'http://127.0.0.1:{httpd.server_address[1]}'
    worlds = [service.order[i % len(service.order)] for i in range(k)]

    def one(world):
        try:
            return play(base_url, client, world, loaded['request']['category'], max_turns=max_turns)
        except ProviderError as error:
            return {'world': world, 'infrastructure': str(error)}

    try:
        with ThreadPoolExecutor(parallel) as pool:
            episodes = list(pool.map(one, worlds))
    finally:
        httpd.shutdown()
        httpd.server_close()
    played = [e for e in episodes if 'infrastructure' not in e]
    successes = sum(e['success'] for e in played)
    report = {'model': client.model, 'user_mode': user_mode,
              'user_model': getattr(user_client, 'model', None), 'k': k, 'played': len(played), 'successes': successes,
              'success_rate': round(successes / len(played), 3) if played else None,
              'mean_reward': round(sum(e['reward'] or 0 for e in played) / len(played), 3) if played else None,
              'status': ('infrastructure' if not played else 'passed' if successes else 'unsolved'),
              'platform_identity': loaded['manifest']['platform_identity'], 'episodes': episodes}
    name = RESULT if user_mode == 'structured' else f'policy-{user_mode}.json'
    (folder / name).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
