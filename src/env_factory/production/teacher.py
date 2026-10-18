"""Bounded teacher draft rejection before execution; never rewrite an executed trajectory.

This is a demonstration-generation strategy, not an unassisted model evaluation.
All rejected drafts and sampling counts remain available as separate evidence.
"""
from __future__ import annotations

import copy
import threading

from env_factory.certify import trajectory


class TeacherProtocolError(RuntimeError):
    """The teacher exhausted its draft budget; not an infrastructure or environment fault."""


class ConstrainedTeacher:
    def __init__(self, client, rules, *, attempts=3):
        if type(attempts) is not int or attempts < 1:
            raise ValueError('attempts must be a positive integer')
        self.client, self.rules, self.attempts = client, copy.deepcopy(rules), attempts
        self.model = client.model
        self.requests, self.drafts, self.rejected = 0, 0, []
        self.lock = threading.Lock()

    def chat(self, messages, tools=None, **parameters):
        with self.lock:
            self.requests += 1
            request_id = self.requests
        view = messages
        for attempt in range(1, self.attempts + 1):
            reply = self.client.chat(view, tools, **parameters)
            with self.lock:
                self.drafts += 1
            audit = trajectory.audit([{**reply, 'role': 'assistant'}], self.rules)
            if audit['passed']:
                return reply
            with self.lock:
                self.rejected.append({'request': request_id, 'attempt': attempt,
                                      'reply': copy.deepcopy(reply), 'violations': audit['violations']})
            # This feedback is teacher-only. The invalid draft has never reached the environment,
            # and neither it nor this instruction is inserted into the student's conversation.
            view = trajectory.provider_view(messages, self.rules)
            view = [{**view[0], 'content': view[0]['content'] +
                     '\nYour previous draft was rejected BEFORE any tool was executed because it violated '
                     'the declared dialogue policy. Generate a fresh draft following the reminders above '
                     'exactly. No action from the rejected draft has happened.'}, *view[1:]]
        raise TeacherProtocolError(f'teacher exceeded {self.attempts} drafts for request {request_id}')

    def report(self):
        with self.lock:
            return {'method': 'bounded_pre_execution_draft_rejection', 'max_drafts_per_request': self.attempts,
                    'requests': self.requests, 'drafts': self.drafts, 'rejected_drafts': len(self.rejected),
                    'rejections': copy.deepcopy(self.rejected),
                    'interpretation': 'Generation efficiency, not an unassisted policy success rate.'}
