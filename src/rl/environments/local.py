"""Real sandbox episodes; policy inputs never include trainer-only state/rewards."""
from __future__ import annotations
import copy
import base64
import sqlite3
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from rl.errors import InfrastructureError
from rl.environments.system_prompt import render_system_prompt


def verify_training_sandbox(root):
    root = Path(root).resolve()
    status = json.loads((root / 'status.json').read_text())
    result = json.loads((root / 'pipeline_result.json').read_text())
    if status.get('training_ready') is not True or result.get('training_ready') is not True:
        raise ValueError('sandbox has not passed final training qualification')
    for name, digest in status.get('artifact_hashes', {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'sandbox artifact changed after qualification: {name}')
    if not status.get('artifact_hashes'):
        raise ValueError('missing artifact identity evidence')
    return root


class SandboxEpisode:
    def __init__(self, root):
        self.root = verify_training_sandbox(root)
        self.task = json.loads((self.root / 'task.json').read_text())
        self.headers = {'Authorization': 'Bearer ' + os.environ['SANDBOX_TRAINER_API_KEY']}
        self._closed = False
        self.temporary = tempfile.TemporaryDirectory(prefix='envfactory-rl-episode-')
        self.app = None
        self._modules_before = set(sys.modules)
        names = {p.stem for p in self.root.glob('*.py')}
        self._displaced = {name: module for name, module in list(sys.modules.items())
                           if name.split('.')[0] in names}
        for name in self._displaced:
            del sys.modules[name]
        sys.path.insert(0, str(self.root))
        try:
            spec = importlib.util.spec_from_file_location('rl_sandbox_' + uuid.uuid4().hex, self.root / 'app.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.app = module.create_app(db_path=Path(self.temporary.name) / 'episode.sqlite3')
        except Exception:
            self.close()
            raise
        self.trace = []

    def request(self, method, path, body=None):
        started = time.monotonic()
        status, value, _ = self.app.handle(method, path, body, self.headers)
        self.trace.append({'path': path, 'request': copy.deepcopy(body), 'status': status,
                           'response': copy.deepcopy(value), 'seconds': time.monotonic() - started})
        from rl.environments.tool_transport import is_authored_tool_fault
        if status >= 500 and not is_authored_tool_fault(self.task, path, status, value):
            raise InfrastructureError(f'environment endpoint failed: {path} ({status})')
        if status >= 400 and not path.startswith('/v1/tools/'):
            raise RuntimeError(f'environment endpoint failed: {path} ({status})')
        return status, value

    def get_current_time(self) -> str | None:
        """Reserved sandbox time API hook, shared by local and HTTP episodes.

        TODO: call the sandbox time endpoint once its contract exists, returning
        Asia/Shanghai time as YYYY-MM-DD HH:mm:ss. Never use the host clock.
        """
        return None

    def reset(self, seed):
        self.trace = []
        reset = self.request('POST', '/v1/reset', {'episode_id': uuid.uuid4().hex, 'seed': seed,
                                                   'reward_mode': 'episode_end'})[1]
        if reset.get('reward_mode') != 'episode_end':
            raise RuntimeError('sandbox does not support episode_end rewards; rebuild its runtime')
        self.tools = self.request('GET', '/v1/tools')[1]['tools']
        self.names = {tool['function']['name'] for tool in self.tools}
        public = self.task.get('public_input', {})
        from .user_interaction import interaction_tool, policy_public_input
        self.user_interaction_tool = interaction_tool(public, self.tools)
        if self.user_interaction_tool is not None:
            self.tools = [*self.tools, self.user_interaction_tool]
            public = policy_public_input(public, self.user_interaction_tool)
        user = public.get('initial_user_message') or self.task['task']
        if public.get('materials'):
            user += '\nPublic materials: ' + json.dumps(public['materials'], ensure_ascii=False)
        self.conversation = [{'role': 'user', 'content': user}]
        self.messages = [{'role': 'system', 'content': render_system_prompt(self.get_current_time())},
            *self.conversation]
        # Observation endpoint is policy-visible by the sandbox contract.
        observation = copy.deepcopy(self.request('GET', '/v1/observation')[1])
        if self.user_interaction_tool is not None:
            visible = observation.get('public_observation', {})
            if isinstance(visible, dict) and isinstance(visible.get('task_input'), dict):
                visible['task_input'] = policy_public_input(visible['task_input'], self.user_interaction_tool)
        self.messages.append({'role': 'user', 'content': json.dumps(
            {'observation': observation}, ensure_ascii=False)})
        self.reward = None
        self.terminated = False
        return self.messages

    def step(self, action):
        if self.reward is not None:
            raise RuntimeError('episode already scored')
        message = {'role': 'assistant', 'content': action} if isinstance(action, str) else copy.deepcopy(action)
        try:
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                raise ValueError('expected an assistant message')
            if error := message.pop('protocol_error', None):
                raise ValueError(error)
            message['content'] = message.get('content') or ''
            if not isinstance(message['content'], str):
                raise ValueError('assistant content must be text')
            calls = message.get('tool_calls', [])
            if not isinstance(calls, list):
                raise ValueError('tool_calls must be a list')
            arguments = []
            ids = {m['tool_call_id'] for m in self.messages if m.get('role') == 'tool'}
            for index, call in enumerate(calls):
                if not isinstance(call, dict) or call.get('type') != 'function':
                    raise ValueError('expected a function tool call')
                function = call.get('function')
                if not isinstance(function, dict) or not isinstance(function.get('name'), str) or not function['name']:
                    raise ValueError('tool call requires a function name')
                params = function.get('arguments')
                params = json.loads(params) if isinstance(params, str) else params
                if not isinstance(params, dict):
                    raise ValueError('tool arguments must be an object')
                arguments.append(params)
                function['arguments'] = json.dumps(params, ensure_ascii=False)
                call.setdefault('id', f'call_{len(self.messages)}_{index}')
                if not isinstance(call['id'], str) or not call['id'] or call['id'] in ids:
                    raise ValueError('tool call IDs must be nonempty and unique')
                ids.add(call['id'])
            if not calls and not message['content'].strip():
                raise ValueError('empty assistant response')
        except (ValueError, TypeError) as error:
            content = message.get('content', '') if isinstance(message, dict) else str(message)
            self.messages.append({'role': 'assistant', 'content': content})
            visible = {'protocol_error': str(error), 'instruction':
                       '使用完整的原生工具调用格式，或直接回复用户。'}
            self.messages.append({'role': 'user', 'content': json.dumps(visible, ensure_ascii=False)})
        else:
            self.messages.append(message)
            # Application failures must escape; they are not policy JSON errors.
            if calls:
                for call, params in zip(calls, arguments):
                    name = call['function']['name']
                    interaction = getattr(self, 'user_interaction_tool', None)
                    if getattr(self, 'terminated', False):
                        status, value = 409, {'error': 'episode already terminated'}
                    elif interaction is not None and name == interaction['function']['name']:
                        from .user_interaction import interaction_content
                        from env_factory.sandbox_runtime import SandboxError
                        try:
                            content = interaction_content(params, interaction['function']['parameters'])
                        except SandboxError as error:
                            status, value = 400, {'error': str(error)}
                        else:
                            user = self._user_turn(content)
                            status, value = 200, {'user_query': user['user_query'],
                                                'should_end': self.terminated}
                    elif name in self.names:
                        previous_headers = getattr(self, 'headers', {})
                        self.headers = {**previous_headers, 'Idempotency-Key': call['id']}
                        try:
                            status, value = self.request('POST', '/v1/tools/' + name, params)
                        finally:
                            self.headers = previous_headers
                    else:
                        status, value = 400, {'error': 'unknown tool', 'name': name}
                    self.messages.append({'role': 'tool', 'name': name, 'tool_call_id': call['id'],
                        'content': json.dumps({'status': status, 'tool_result': value}, ensure_ascii=False)})
            else:
                user = self._user_turn(message['content'])
                self.messages.append({'role': 'user', 'content': user['user_query']})
        return 0.0, self.terminated

    def _user_turn(self, content):
        self.request('POST', '/v1/agent_response', {'content': content})
        self.conversation.append({'role': 'assistant', 'content': content})
        user = self.request('POST', '/v1/user_simulator', {'messages': self.conversation})[1]
        self.conversation.append({'role': 'user', 'content': user['user_query']})
        self.terminated = bool(user.get('should_end'))
        return user

    def finish(self):
        if self.reward is None:
            reward = float(self.request('GET', '/v1/reward')[1]['reward'])
            if not math.isfinite(reward):
                raise ValueError('non-finite environment reward')
            self.reward = reward
        replay = self.request('GET', '/v1/replay')[1]
        if any(e.get('payload', {}).get('used_fallback') for e in replay.get('events', [])):
            raise InfrastructureError('sandbox runtime used fallback')
        return {'final_reward': self.reward, 'terminated': self.terminated, 'trace': self.trace}

    def snapshot(self):
        """Snapshot the committed local sandbox plus policy-visible conversation."""
        database = sqlite3.connect(self.app.episode_store.db_path)
        memory = sqlite3.connect(':memory:')
        try:
            database.backup(memory)
            payload = base64.b64encode(memory.serialize()).decode('ascii')
        finally:
            memory.close()
            database.close()
        return {'database': payload, 'user_interaction_tool': copy.deepcopy(getattr(self, 'user_interaction_tool', None)),
                **{name: copy.deepcopy(getattr(self, name)) for name in
            ('messages', 'conversation', 'reward', 'terminated', 'trace', 'tools')}, 'names': sorted(self.names)}

    def restore(self, state):
        memory = sqlite3.connect(':memory:')
        database = sqlite3.connect(self.app.episode_store.db_path)
        try:
            memory.deserialize(base64.b64decode(state['database']))
            memory.backup(database)
        finally:
            memory.close()
            database.close()
        for name in ('messages', 'conversation', 'reward', 'terminated', 'trace', 'tools'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.names = set(state['names'])
        self._restore_interaction_tool(state)

    def _restore_interaction_tool(self, state):
        self.user_interaction_tool = copy.deepcopy(state.get('user_interaction_tool'))
        if 'user_interaction_tool' not in state:
            from .user_interaction import interaction_tool
            business = [tool for tool in self.tools if tool['function']['name'] in self.names]
            expected = interaction_tool(self.task.get('public_input', {}), business)
            if expected is not None:
                self.user_interaction_tool = next((copy.deepcopy(tool) for tool in self.tools
                    if tool['function']['name'] == expected['function']['name']), None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self.app, 'close', None)
            if callable(close):
                close()
        finally:
            self.app = None
            for name, module in list(sys.modules.items()):
                filename = getattr(module, '__file__', None)
                if filename and Path(filename).resolve().is_relative_to(self.root):
                    if name not in self._modules_before or name in self._displaced:
                        del sys.modules[name]
            sys.modules.update(self._displaced)
            if str(self.root) in sys.path:
                sys.path.remove(str(self.root))
            self.temporary.cleanup()
