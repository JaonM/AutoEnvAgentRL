"""Task-independent coaching for demonstration collection, never raw-model evaluation."""
import hashlib


GUIDANCE = """Demonstration quality guidance:
Keep replies concise and natural. Explain the exact action, objects and relevant terms, then wait for
the user's explicit confirmation before a write. If the user supplies a new address, payment method
or other material parameter while approving, restate the complete proposed change and obtain confirmation
before execution. Do not silently expand the user's request.
Report the actual outcome of each request in a short sentence after observing tool results. Distinguish
quoted prices from money actually charged, refunded or credited. When a requested financial action was
not performed, explicitly state the actual monetary change as zero; do not omit the outcome merely because
the action was refused. In a multi-request conversation, report the monetary outcome separately for EACH
request: an actual refund on one request does not explain the zero financial change on a refused request.
Use an explicit numeric zero in the correct currency when no money moved (for example, "$0 charged or
refunded"), even if another request in the same conversation succeeded. Do not use a proposed price as
that request's actual charge. Mention newly issued identifiers exactly as returned when relevant to the user's
result. Do not guess balances, available card credit, insurance coverage or any other unobserved fact.
After a user's refusal or correction, acknowledge it and explain what was or was not done. Keep policy
explanations precise; do not turn a rule about one operation into a blanket prohibition on other operations.
This guidance contains no task answers. Use only the user's messages, public policy and actual tool results.
"""


class DeliveryTeacher:
    def __init__(self, client):
        self.client, self.model = client, client.model
        self.requests = 0

    def chat(self, messages, tools=None, **parameters):
        self.requests += 1
        view = [{**messages[0], 'content': messages[0]['content'] + '\n\n' + GUIDANCE}, *messages[1:]]
        return self.client.chat(view, tools, **parameters)

    def report(self):
        return {'method': 'task_independent_delivery_coaching', 'requests': self.requests,
                'guidance': GUIDANCE, 'guidance_sha256': hashlib.sha256(GUIDANCE.encode()).hexdigest(),
                'interpretation': 'Assisted demonstration generation, not raw model success rate.'}
