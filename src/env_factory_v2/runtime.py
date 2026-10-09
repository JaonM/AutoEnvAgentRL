"""In-memory transactional world with virtual time and provenance-bound tools."""
from __future__ import annotations

import copy
import hashlib
import secrets
import re
from datetime import datetime, timedelta

from .contract import compile_contract, require, validate_data, validate_value
from .expressions import ContractError, canonical, evaluate, path, typed_equal
from .user_simulator import UserSimulator


class Sandbox:
    def __init__(self, contract):
        self.compiled = compile_contract(contract['source'] if 'source' in contract else contract)
        self.source = self.compiled['source']
        self.tools = {tool['id']: tool for tool in self.source['tools']}
        self.rules = {rule['id']: rule for rule in self.source['rules']}
        self.reset()

    def reset(self, mutations=()):
        self.data = {table['id']: copy.deepcopy(table['rows']) for table in self.source['tables']}
        for mutation in mutations:
            require(set(mutation) == {'table', 'key', 'column', 'value'}, 'invalid data mutation')
            table = next((t for t in self.source['tables'] if t['id'] == mutation['table']), None)
            require(table is not None and mutation['column'] in table['schema']['properties'], 'unknown mutation field')
            row = self._row(self.data, table['id'], mutation['key'])
            row[mutation['column']] = copy.deepcopy(mutation['value'])
        validate_data(self.source, self.data)
        self.initial = copy.deepcopy(self.data)
        self.user = UserSimulator(self.source)
        self.observations = {}
        self.evidence = {}
        self.trace = []
        self.ticks = 0
        self.seconds = 0
        self.version = 0
        self.cursors = {}
        self.applied_events = set()
        self.violations = []
        self.closed = False
        self.business_attempts = 0
        self._events()
        return {'episode': secrets.token_hex(16), 'public': copy.deepcopy(self.compiled['public'])}

    def time(self):
        base = datetime.fromisoformat(self.source['time']['base'])
        return {'datetime': (base + timedelta(seconds=self.seconds)).isoformat(),
                'tick': self.ticks, 'elapsed_seconds': self.seconds}

    def context(self, *, args=None, data=None, row=None, private=False):
        return {'data': self.data if data is None else data, 'initial': self.initial,
                'args': args or {}, 'row': row or {}, 'evidence': self.evidence,
                'user': {**self.user.facts, **self.user.disclosed} if private else self.user.disclosed,
                'user_state': self.user.snapshot(),
                'time': self.time(), '_rules': self.rules}

    def _row(self, data, table_id, key):
        table = next(t for t in self.source['tables'] if t['id'] == table_id)
        rows = [row for row in data[table_id] if canonical(row[table['primary_key']]) == canonical(key)]
        require(len(rows) == 1, 'write/mutation target must identify exactly one row')
        return rows[0]

    def _events(self):
        candidate = copy.deepcopy(self.data)
        due = [(i, e) for i, e in enumerate(self.source.get('events', []))
               if i not in self.applied_events and e['at_seconds'] <= self.seconds]
        for i, event in sorted(due, key=lambda pair: (pair[1]['at_seconds'], pair[0])):
            self._row(candidate, event['table'], event['key']).update(copy.deepcopy(event['set']))
        if due:
            validate_data(self.source, candidate)
            self.data = candidate
            self.applied_events.update(i for i, _ in due)
            self.version += 1

    def _advance(self, seconds=None):
        require(not self.closed, 'episode already finalized')
        require(self.ticks < self.source['limits']['max_actions'], 'action budget exhausted')
        self.ticks += 1
        self.seconds += self.source['time']['seconds_per_action'] if seconds is None else seconds
        self._events()

    def resolve(self, value):
        if isinstance(value, dict) and set(value) == {'$result', 'path'}:
            require(value['$result'] in self.observations, 'unknown observed result')
            return path(self.observations[value['$result']]['result'], value['path'])
        if isinstance(value, dict) and set(value) == {'$user', 'path'}:
            require(value['$user'] in self.user.completed, 'undisclosed user response')
            return path(self.user.completed[value['$user']], value['path'])
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        return copy.deepcopy(value)

    def interact(self, stage, *, question, proposal=None, source_id=None):
        self._advance()
        require(isinstance(question, str) and question.strip(), 'a user question/proposal is required')
        resolved = self.resolve(proposal) if proposal is not None else None
        user_stage = self.user.stages.get(stage, {})
        if user_stage.get('kind') == 'choice' and self.tools[user_stage['source_tool']].get('complete_retrieval'):
            from .reward import complete_retrieval
            observation = self.observations.get(source_id)
            require(observation is not None and observation['tool'] == user_stage['source_tool'],
                    'choice must use an actual observed menu')
            menu = observation['result']
            require(menu['state_version'] == self.version
                    and len(menu['items']) == menu['total']
                    and menu['next_cursor'] is None
                    and complete_retrieval(self, user_stage['source_tool'],
                        query_identity=menu['query_identity'], state_version=self.version),
                    'choice requires the complete current observed menu before selection')
        result = self.user.respond(stage, observations=self.observations, tools=self.tools,
                                   proposal=resolved, source_id=source_id)
        self.trace.append({'kind': 'user', 'stage': stage, 'question': question, 'proposal': resolved,
                           'result': copy.deepcopy(result), 'time': self.time()})
        return result

    def wait(self, seconds):
        require(type(seconds) in {int, float} and 0 <= seconds <= 86400, 'invalid virtual wait')
        self._advance(seconds)
        self.trace.append({'kind': 'wait', 'seconds': seconds, 'time': self.time()})
        return self.time()

    def revoke(self, stage):
        self._advance()
        self.user.revoke(stage)
        self.trace.append({'kind': 'revoke', 'stage': stage, 'time': self.time()})

    def call(self, tool_id, arguments, *, call_id, cursor=None):
        self._advance()
        self.business_attempts += 1
        if self.source['category'] == 'direct_response':
            self.violations.append('unnecessary_business_tool')
        require(isinstance(call_id, str) and call_id and '.' not in call_id
                and call_id not in self.observations, 'unique call id required')
        require(tool_id in self.tools, 'unknown business tool')
        tool = self.tools[tool_id]
        resolved = self.resolve(arguments)
        record = {'kind': 'tool', 'id': call_id, 'tool': tool_id, 'arguments': copy.deepcopy(resolved), 'time': self.time()}
        try:
            validate_value(resolved, tool['parameters'], tool_id)
            for argument, fact in tool.get('user_bindings', {}).items():
                require(fact in self.user.disclosed and typed_equal(resolved[argument], self.user.disclosed[fact]),
                        'tool argument must use the actual disclosed user value')
            for binding in tool.get('result_bindings', []):
                value = arguments.get(binding['argument'])
                providers = binding['tool'] if isinstance(binding['tool'], list) else [binding['tool']]
                require(isinstance(value, dict) and set(value) == {'$result', 'path'}
                        and value['$result'] in self.observations
                        and self.observations[value['$result']]['tool'] in providers
                        and re.fullmatch(re.escape(binding['path']).replace(r'\*', r'\d+'), value['path']),
                        'required observed-result dependency missing')
                if binding.get('recency') == 'latest':
                    latest = next((turn['id'] for turn in reversed(self.trace)
                                   if turn.get('kind') == 'tool' and turn.get('status') == 'ok'), None)
                    require(value['$result'] == latest,
                            'required observed-result dependency missing: latest successful business result required')
            if tool['kind'] == 'query':
                result = self._query(tool, resolved, cursor)
            else:
                require(cursor is None, 'writes cannot use cursors')
                try:
                    self.user.authorize(tool['authorization'], tool_id, resolved)
                except ContractError:
                    self.violations.append('unauthorized_write')
                    raise
                result = self._transaction(tool, resolved)
                self.user.consume(tool['authorization'])
            record.update(status='ok', result=copy.deepcopy(result))
            self.observations[call_id] = {'tool': tool_id, 'result': copy.deepcopy(result), 'arguments': copy.deepcopy(resolved)}
            self.evidence[call_id] = copy.deepcopy(result)
            return result
        except ContractError as exc:
            record.update(status='error', error=str(exc))
            raise
        finally:
            self.trace.append(record)

    def _query(self, tool, args, cursor):
        signature = hashlib.sha256(canonical([tool['id'], args]).encode()).hexdigest()
        offset = 0
        if cursor is not None:
            require(cursor in self.cursors, 'unknown/stale cursor')
            saved = self.cursors[cursor]
            require(saved['signature'] == signature and saved['version'] == self.version, 'cursor query/state mismatch')
            offset = saved['offset']
        rows = []
        for row in self.data[tool['table']]:
            conditions = [evaluate(self.rules[r]['predicate'], self.context(args=args, row=row)) for r in tool['rule_ids']]
            require(all(type(item) is bool for item in conditions), 'business rule must produce boolean')
            if all(conditions):
                rows.append(copy.deepcopy(row))
        for order in reversed(tool['order_by']):
            field = order['field']
            present = [row for row in rows if row[field] is not None]
            absent = [row for row in rows if row[field] is None]
            present.sort(key=lambda row: row[field], reverse=order['direction'] == 'desc')
            rows = present + absent
        page = rows[offset:offset + tool['page_size']]
        next_cursor = None
        if offset + len(page) < len(rows):
            next_cursor = secrets.token_hex(24)
            self.cursors[next_cursor] = {'signature': signature, 'version': self.version, 'offset': offset + len(page)}
        return {'items': [{key: row[key] for key in tool['columns']} for row in page],
                'total': len(rows), 'offset': offset, 'next_cursor': next_cursor,
                'query_identity': signature, 'state_version': self.version}

    def _transaction(self, tool, args):
        before = copy.deepcopy(self.data)
        candidate = copy.deepcopy(self.data)
        changes = []
        for operation in tool['operations']:
            key = evaluate(operation['key'], self.context(args=args, data=candidate))
            row = self._row(candidate, operation['table'], key)
            original_row = self._row(before, operation['table'], key)
            precondition_context = self.context(args=args, data=before, row=original_row)
            checks = [evaluate(self.rules[r]['predicate'], precondition_context) for r in tool['rule_ids']]
            failed = [rule for rule, check in zip(tool['rule_ids'], checks) if type(check) is not bool or not check]
            require(not failed, 'transaction business precondition failed: ' + ', '.join(failed))
            context = self.context(args=args, data=candidate, row=row)
            updates = {column: evaluate(expression, context) for column, expression in operation['set'].items()}
            row.update(updates)
            changes.append({'table': operation['table'], 'key': key, 'set': copy.deepcopy(updates)})
        validate_data(self.source, candidate)
        for invariant in self.source.get('invariants', []):
            require(evaluate(invariant['predicate'], self.context(data=candidate)) is True,
                    'transaction invariant violated: ' + invariant['rule_id'])
        self.data = candidate
        self.version += 1
        return {'committed': True, 'changes': changes}

    def traverse(self, tool_id, arguments, *, call_id):
        require(call_id not in self.observations, 'unique traversal id required')
        items, cursor, identity, version, total = [], None, None, None, None
        pages = []
        while True:
            page_id = f'{call_id}_page{len(pages)}'
            page = self.call(tool_id, arguments, call_id=page_id, cursor=cursor)
            if identity is None:
                identity, version, total = page['query_identity'], page['state_version'], page['total']
            require((identity, version, total) == (page['query_identity'], page['state_version'], page['total']), 'unstable pagination')
            items.extend(page['items'])
            pages.append(page_id)
            cursor = page['next_cursor']
            if cursor is None:
                break
        return self.collect(pages, call_id=call_id)

    def collect(self, page_ids, *, call_id):
        """Assemble only real, complete pages; no new business reads or invented rows."""
        self._advance()
        require(isinstance(call_id, str) and call_id and '.' not in call_id
                and call_id not in self.observations, 'unique collection id required')
        require(isinstance(page_ids, list) and page_ids and len(set(page_ids)) == len(page_ids), 'unique observed pages required')
        require(all(identity in self.observations for identity in page_ids), 'unknown page observation')
        observations = [self.observations[identity] for identity in page_ids]
        pages = [observation['result'] for observation in observations]
        tool = observations[0]['tool']
        require(self.tools[tool]['kind'] == 'query' and all(observation['tool'] == tool for observation in observations), 'collection requires one query tool')
        identity, version, total = (pages[0][key] for key in ('query_identity', 'state_version', 'total'))
        offset, items = 0, []
        for i, page in enumerate(pages):
            require('offset' in page and page['offset'] == offset
                    and (page['query_identity'], page['state_version'], page['total']) == (identity, version, total),
                    'collection must contain contiguous pages of one unchanged query')
            require((page['next_cursor'] is None) == (i == len(pages) - 1), 'collection must reach actual end')
            items.extend(page['items'])
            offset += len(page['items'])
        require(offset == total, 'incomplete page collection')
        result = {'items': items, 'total': total, 'next_cursor': None, 'query_identity': identity, 'pages': page_ids, 'state_version': version}
        self.observations[call_id] = {'tool': tool, 'arguments': copy.deepcopy(observations[0]['arguments']), 'result': copy.deepcopy(result)}
        self.evidence[call_id] = copy.deepcopy(result)
        self.trace.append({'kind': 'collect', 'id': call_id, 'pages': list(page_ids), 'time': self.time()})
        return result
