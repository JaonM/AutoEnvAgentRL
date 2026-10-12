"""The answer an agent delivered by talking: extracted from its speech when it recorded none.

In free dialogue the agent need not call the answer tool. When the conversation ends without a
recorded answer, a model (V3_JUDGE_*, falling back to LLM_*; temperature 0; cached by content)
reads everything the agent said to the user (the whole conversation, in order; when it states a
result more than once the last statement counts, and restated inputs or rules are not results)
and fills the answer schema; it never sees tool
results, so nothing the agent did not say can be credited. Every extracted number, id or other
free string must then appear in that speech (numerals.said for numbers, so 36000 fen said as
"360 元" counts); a value that does not is replaced by null. Enums and booleans are taken as the
model read them. The result is graded exactly like a recorded answer.
"""
from __future__ import annotations

import copy
import hashlib
import json

from env_factory_v3.core.numerals import said

PROMPT = """下面是客服在一次服务中对用户说过的全部话。请按答案格式，把客服明确告诉用户的处理结果填成 JSON。

答案格式（每个字段的类型和含义）：
{schema}

客服说过的话：
{speech}

要求：只依据客服说过的话，只提取客服给用户的处理结果；复述用户提供的信息、规则、价目或中间计算过程都不算结果。同一项前后说法不一致时，以最后一次为准。没说到的字段填 null，不要推测、补全或自己计算。数值按字段说明的单位换算（例如字段要求“分”/cents，客服说“360元”或“$360”就填 36000）。对话可能是中文或英文。枚举字段只能从给定选项中选，选不出就填 null。数组和对象按格式逐项填写，客服没说到的元素不要编造。
只输出一个 JSON 对象，键与答案格式的字段完全一致。"""


def speech_of(trace):
    """What the agent said to the user, in order (plain messages and approval requests)."""
    return [e['message'] for e in trace if e.get('kind') == 'chat' and isinstance(e.get('message'), str)]


def verify(value, spec, text):
    """The value with every number or free string that was not actually said replaced by None."""
    if value is None or not isinstance(spec, dict):
        return value
    kind = spec.get('type')
    if kind in ('integer', 'number') and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if said(text, value) else None
    if kind == 'string' and isinstance(value, str) and 'enum' not in spec:
        return value if value and value in text else None
    if kind == 'array' and isinstance(value, list):
        return [verify(item, spec.get('items'), text) for item in value]
    if kind == 'object' and isinstance(value, dict) and spec.get('properties'):
        return {key: verify(value.get(key), sub, text) for key, sub in spec['properties'].items()}
    return value


class Extractor:
    def __init__(self, client):
        self.client, self.cache = client, {}

    def extract(self, schema, speech):
        """{field: value} for the schema's fields (None when not stated), or None without speech."""
        text = '\n'.join(m for m in speech if m.strip())
        if not text:
            return None
        fields = schema['fields']
        prompt = PROMPT.format(schema=json.dumps(fields, ensure_ascii=False, indent=1), speech=text)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.cache:
            from env_factory_v3.core.llm import ProviderError
            for _ in range(2):
                content = self.client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
                try:
                    raw = json.loads(content[content.index('{'):content.rindex('}') + 1])
                    if isinstance(raw, dict):
                        break
                except ValueError:
                    continue
            else:
                raise ProviderError('extractor returned no usable answer')
            self.cache[key] = raw
        raw = copy.deepcopy(self.cache[key])
        return {name: verify(raw.get(name), spec, text) for name, spec in fields.items()}
