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

from .errors import InfrastructureError


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
        if status >= 500:
            raise InfrastructureError(f'environment endpoint failed: {path} ({status})')
        if status >= 400 and not path.startswith('/v1/tools/'):
            raise RuntimeError(f'environment endpoint failed: {path} ({status})')
        return status, value

    def reset(self, seed):
        self.trace = []
        reset = self.request('POST', '/v1/reset', {'episode_id': uuid.uuid4().hex, 'seed': seed,
                                                   'reward_mode': 'episode_end'})[1]
        if reset.get('reward_mode') != 'episode_end':
            raise RuntimeError('sandbox does not support episode_end rewards; rebuild its runtime')
        tools = self.request('GET', '/v1/tools')[1]['tools']
        self.names = {tool['function']['name'] for tool in tools}
        public = self.task.get('public_input', {})
        user = public.get('initial_user_message') or self.task['task']
        if public.get('materials'):
            user += '\nPublic materials: ' + json.dumps(public['materials'], ensure_ascii=False)
        self.conversation = [{'role': 'user', 'content': user}]
        self.messages = [{'role': 'system', 'content': (
            '你是通过工具完成任务的 agent。每轮必须输出且只输出一个合法 JSON 对象。'
            '禁止在 JSON 外输出文字、算式或 Markdown。用户要求的答案格式只约束 content 字段，'
            '不能省略外层 JSON。调用工具的格式：'
            '{"kind":"tool","name":"工具名称","arguments":{}}。'
            '给用户回复的格式：{"kind":"respond","content":"最终答案或澄清问题"}。'
            '需要计算时，在同一个 JSON 中增加 calculation 字段，按步骤计算，再把答案写入 content。'
            '例如求 12 与 7 的和：'
            '{"calculation":"12+7=19","kind":"respond","content":"总数：19"}。'
            '计算必须使用本次工具返回的实际数据，不得编造。'
            '工具定义：' + json.dumps(tools, ensure_ascii=False))},
            *self.conversation]
        # Observation endpoint is policy-visible by the sandbox contract.
        self.messages.append({'role': 'user', 'content': json.dumps(
            {'observation': self.request('GET', '/v1/observation')[1]}, ensure_ascii=False)})
        self.reward = None
        self.terminated = False
        return self.messages

    def step(self, text):
        if self.reward is not None:
            raise RuntimeError('episode already scored')
        self.messages.append({'role': 'assistant', 'content': text})
        try:
            action = json.loads(text)
            if not isinstance(action, dict):
                raise ValueError('action must be JSON object')
            if action.get('kind') == 'tool':
                if action.get('name') not in self.names or not isinstance(action.get('arguments'), dict):
                    raise ValueError('unknown tool or invalid arguments')
            elif action.get('kind') != 'respond' or not isinstance(action.get('content'), str) or not action['content'].strip():
                raise ValueError('expected tool or respond')
        except (ValueError, TypeError) as error:
            visible = {'protocol_error': str(error), 'instruction':
                       '只输出一个 JSON 对象。回复必须包装为 {"kind":"respond","content":"你的回答"}。'}
        else:
            # Application failures must escape; they are not policy JSON errors.
            if action['kind'] == 'tool':
                status, value = self.request('POST', '/v1/tools/' + action['name'], action['arguments'])
                visible = {'status': status, 'tool_result': value}
            else:
                self.request('POST', '/v1/agent_response', {'content': action['content']})
                self.conversation.append({'role': 'assistant', 'content': action['content']})
                user = self.request('POST', '/v1/user_simulator', {'messages': self.conversation})[1]
                self.conversation.append({'role': 'user', 'content': user['user_query']})
                visible = {'user_query': user['user_query']}
                self.terminated = bool(user.get('should_end'))
        self.messages.append({'role': 'user', 'content': json.dumps(visible, ensure_ascii=False)})
        return 0.0, self.terminated

    def finish(self):
        if self.reward is None:
            reward = float(self.request('GET', '/v1/reward')[1]['reward'])
            if not math.isfinite(reward):
                raise ValueError('non-finite environment reward')
            self.reward = reward
        replay = self.request('GET', '/v1/replay')[1]
        if any(e.get('payload', {}).get('used_fallback') for e in replay.get('events', [])):
            raise ValueError('sandbox runtime used fallback')
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
        return {'database': payload, **{name: copy.deepcopy(getattr(self, name)) for name in
            ('messages', 'conversation', 'reward', 'terminated', 'trace')}, 'names': sorted(self.names)}

    def restore(self, state):
        memory = sqlite3.connect(':memory:')
        database = sqlite3.connect(self.app.episode_store.db_path)
        try:
            memory.deserialize(base64.b64decode(state['database']))
            memory.backup(database)
        finally:
            memory.close()
            database.close()
        for name in ('messages', 'conversation', 'reward', 'terminated', 'trace'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.names = set(state['names'])

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
