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

from env_factory_v3.certify import bundle
from env_factory_v3.core.llm import ChatClient, ProviderError, load_env_file  # noqa: F401  (re-exported)
from env_factory_v3.serving import agent_view, server

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




def left(reply):
    """'warning' or 'closed' when the simulated user has left the conversation (see server.IDLE_LIMIT)."""
    if reply.get('kind') == 'finished':
        return 'closed'
    return (reply.get('result') or {}).get('ended')


def play(base_url, client, world, category, *, max_turns=30, answer_tool=True):
    """One episode on `world`; returns its grade and transcript."""
    opener = build_opener(ProxyHandler({}))
    reset = _post(opener, base_url + '/reset', {'world': world})
    episode, instructions = reset['episode'], reset['instructions']
    names = {tool['name'] for tool in instructions['tools']}
    tools = agent_view.tools(instructions, category, answer_tool=answer_tool)
    messages = agent_view.messages(instructions, answer_tool=answer_tool)
    natural = agent_view.speaks(instructions)
    notice = agent_view.left_notice(agent_view.language(instructions), 'v3_answer' if answer_tool else None)
    # Structured: the answer ends the episode. Free dialogue: the answer is only recorded; the
    # episode ends when the user leaves (or at max_turns) and is graded then (server._close).
    answered, closed, calls, turns = False, False, 0, 0
    step = lambda action: _post(opener, base_url + '/step', {'episode': episode, 'action': action})
    try:
        while not (answered and not natural) and not closed and turns < max_turns:
            turns += 1
            reply = client.chat(messages, tools)
            message = {'role': 'assistant', 'content': reply.get('content') or ''}
            if reply.get('tool_calls'):
                message['tool_calls'] = reply['tool_calls']
            messages.append(message)
            if not reply.get('tool_calls'):
                if natural:
                    said = step({'type': 'chat', 'message': message['content']})
                    text = (said.get('result') or {}).get('reply') or said.get('error', '')
                    ended = left(said)
                    if ended == 'closed':
                        closed = True
                        break
                    messages.append({'role': 'user', 'content': f'{text}\n\n{notice}' if ended else text})
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
                    if (answered and not natural) or closed:
                        result = {'ok': False, 'kind': 'finished', 'error': 'episode already ended'}
                    elif action is None:
                        result, answered = step({'type': 'answer', 'answer': params}), True
                    else:
                        calls += action['type'] == 'tool'
                        result = step(action)
                        ended = left(result) if action['type'] == 'chat' else None
                        if ended == 'warning':
                            result = {**result, 'notice': notice}
                        elif ended == 'closed':
                            closed = True
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', name), 'name': name,
                                 'content': json.dumps(result, ensure_ascii=False)})
        if not answered and not natural:
            # Same rule as training (rl V3Episode.finish): no final answer earns 0, whatever the state.
            return {'world': world, 'reward': 0.0, 'success': False, 'fields': None, 'state_ok': None,
                    'safety': None, 'turns': turns, 'tool_calls': calls, 'answered': False,
                    'reason': 'no_final_answer', 'messages': messages}
        grade = _post(opener, base_url + '/reward', {'episode': episode})
        return {'world': world, 'reward': grade.get('reward'), 'success': grade.get('success') is True,
                'fields': grade.get('fields'), 'state_ok': grade.get('state_ok'), 'safety': grade.get('safety'),
                'turns': turns, 'tool_calls': calls, 'answered': answered, 'answer': grade.get('answer'),
                'outcome': grade.get('outcome'), 'valid': grade.get('valid', True),
                'simulator_audit': grade.get('simulator_audit'), 'messages': messages}
    finally:
        try:
            _post(opener, base_url + '/release', {'episode': episode})
        except ProviderError:
            pass


SAY = """你是一位客服。请把下面的处理结果用平实的{language}完整地告诉用户：每个数字和编号都要说到，金额按字段说明换算成元（英文用美元）来说，
结论性的选项用日常说法表达。只输出你对用户说的话。

答案格式：
{schema}

处理结果：
{answer}"""


def extraction_probe(loaded, service, client):
    """Can this task's answer be delivered by talking? A model says the expected answer to the user
    the way a good agent would; the service's extractor must read every graded field back exactly."""
    from env_factory_v3.core import answers
    request = loaded['request']
    expected = loaded['oracle'].expected(request['tables'], request['user_facts'], request['time']['start'])['answer']
    speech = (client.chat([{'role': 'user', 'content': SAY.format(
        language='英文' if request.get('language') == 'en' else '中文',
        schema=json.dumps(loaded['schema']['fields'], ensure_ascii=False), answer=json.dumps(
            expected, ensure_ascii=False))}], temperature=0).get('content') or '').strip()
    extracted = service.extractor().extract(loaded['schema'], [speech])
    fields = answers.compare(loaded['schema'], expected, extracted)
    return {'passed': all(fields.values()), 'fields': fields, 'speech': speech[:600]}


def gate(folder, client, *, k=4, parallel=4, max_turns=30, user_mode='structured', user_client=None,
         answer_tool='never'):
    """k episodes on the bundle's served worlds; passes when at least one succeeds."""
    folder = Path(folder)
    loaded = bundle.load_bundle(folder)
    service = bundle.service(loaded, user_mode=user_mode, user_client=user_client)
    httpd = server.start(service)
    base_url = f'http://127.0.0.1:{httpd.server_address[1]}'
    worlds = [service.order[i % len(service.order)] for i in range(k)]
    # Free dialogue offers no answer tool by default (delivery by talking, as in training);
    # 'random' alternates episodes with and without it, 'always' offers it in every episode.
    tool_plan = [user_mode != 'natural' or answer_tool == 'always' or (answer_tool == 'random' and i % 2 == 0)
                 for i in range(k)]

    def one(job):
        world, answer_tool = job
        try:
            return {**play(base_url, client, world, loaded['request']['category'], max_turns=max_turns,
                           answer_tool=answer_tool), 'answer_tool': answer_tool}
        except ProviderError as error:
            return {'world': world, 'infrastructure': str(error)}

    try:
        with ThreadPoolExecutor(parallel) as pool:
            episodes = list(pool.map(one, zip(worlds, tool_plan)))
    finally:
        httpd.shutdown()
        httpd.server_close()
    played = [e for e in episodes if 'infrastructure' not in e]
    successes = sum(e['success'] for e in played)
    # Without an answer tool a business answer is read from speech: it must be readable at all.
    probe = None
    if user_mode == 'natural' and loaded['oracle'] is not None and not all(tool_plan):
        try:
            probe = extraction_probe(loaded, service, client)
        except ProviderError as error:
            probe = {'passed': False, 'infrastructure': str(error)}
    report = {'model': client.model, 'user_mode': user_mode,
              'user_model': getattr(user_client, 'model', None), 'k': k, 'played': len(played), 'successes': successes,
              'success_rate': round(successes / len(played), 3) if played else None,
              'simulator_faults': sum(e.get('valid') is False for e in played),
              'mean_reward': round(sum(e['reward'] or 0 for e in played) / len(played), 3) if played else None,
              'status': ('infrastructure' if not played or (probe or {}).get('infrastructure') else
                         'unextractable' if probe and not probe['passed'] else 'passed' if successes else 'unsolved'),
              'extraction_probe': probe,
              'platform_identity': loaded['manifest']['platform_identity'], 'episodes': episodes}
    name = RESULT if user_mode == 'structured' else f'policy-{user_mode}.json'
    (folder / name).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
