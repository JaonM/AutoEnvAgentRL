"""The answer an agent delivered by talking: extracted from its speech when it recorded none.

In free dialogue the agent need not call the answer tool. When the conversation ends without a
recorded answer, a model (JUDGE_*, falling back to LLM_*; temperature 0; cached by content)
reads everything the agent said to the user (the whole conversation, in order; when it states a
result more than once the last statement counts, and restated inputs or rules are not results)
and fills the answer schema; it never sees tool
results, so nothing the agent did not say can be credited. Every extracted number, id or other
free string must then appear in that speech (numerals.said for numbers, so 36000 fen said as
"360 元" counts); a value that does not is replaced by null. Enums and booleans are taken as the
model read them. The result is graded exactly like a recorded answer.

Records: an agent names records the way a person does ("the Basic plan"), not by id. The records its
tool results returned (id plus a few naming strings) are listed for the model, which may fill an id
for a record the agent named; such an id counts as said when one of that record's names was said.
Without this, a record id the user never heard could only be skipped, so naming the wrong record
went ungraded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata

from env_factory.core.numerals import said

PROMPT = """下面是客服在一次服务中对用户说过的全部话。请按答案格式，把客服明确告诉用户的处理结果填成 JSON。

答案格式（每个字段的类型和含义）：
{schema}

客服说过的话：
{speech}

{records}要求：只依据客服说过的话，只提取客服给用户的处理结果；复述用户提供的信息、规则、价目或中间计算过程都不算结果。同一项前后说法不一致时，以最后一次为准。没说到的字段填 null，不要推测、补全或自己计算。数值按字段说明的单位换算（例如字段要求“分”/cents，客服说“360元”或“$360”就填 36000）。对话可能是中文或英文。枚举字段只能从给定选项中选，选不出就填 null。数组和对象按格式逐项填写，客服没说到的元素不要编造。
只输出一个 JSON 对象，键与答案格式的字段完全一致。"""


def speech_of(trace):
    """What the agent said to the user, in order (plain messages and approval requests)."""
    return [e['message'] for e in trace if e.get('kind') == 'chat' and isinstance(e.get('message'), str)]


MAX_RECORDS, NAMES_PER_RECORD = 120, 3


def records_of(trace):
    """{record id: [naming strings]} from the rows the agent's tool calls returned."""
    found = {}

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('id'), str):
                names = [v for k, v in value.items() if k != 'id' and isinstance(v, str) and len(v) >= 3
                         and not v.replace('-', '').replace('_', '').isdigit()][:NAMES_PER_RECORD]
                if names:
                    found.setdefault(value['id'], names)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for entry in trace:
        if entry.get('kind') == 'tool' and entry.get('status') == 'ok':
            walk(entry.get('result'))
    return dict(list(found.items())[:MAX_RECORDS])


MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december')
ISO_DAY = re.compile(r'^(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$')


def date_said(text, value):
    """Whether the calendar day of an ISO date or time was said in any usual form: the agent says "May 31, 2024"
    or "5月31日" and the extractor writes the schema's "2024-05-31T18:00", which is not in the text verbatim (pool3,
    the first calendar pool, lost every date field this way). The time of day may come from the policy (a deadline's
    18:00 cutoff), so only the day must have been said."""
    found = ISO_DAY.match(value.strip())
    if not found:
        return False
    year, month, day = (int(x) for x in found.groups())
    if not 1 <= month <= 12:
        return False
    name = MONTHS[month - 1]
    short = (name[:3], 'sept') if name == 'september' else (name[:3],)
    names = '(?:' + '|'.join([name] + [rf'{x}\.?' for x in short if x != name]) + ')'
    patterns = (rf'{year}-0?{month}-0?{day}(?!\d)',
                rf'\b{names}\s+0?{day}(?:st|nd|rd|th)?(?!\d)',
                rf'(?<!\d)0?{day}(?:st|nd|rd|th)?\s+(?:of\s+)?{names}(?![a-z])',
                rf'(?<![\d/])0?{month}/0?{day}(?:/(?:{year}|{year % 100:02d}))?(?![\d/])',
                rf'(?<!\d){year}[./]0?{month}[./]0?{day}(?!\d)',
                rf'(?<!\d)0?{month}\s*月\s*0?{day}\s*[日号]')
    if name == 'may':  # the month only when capitalised, not the verb ("you may 3 times")
        return any(re.search(pattern.replace('may', 'May'), text) for pattern in patterns)
    lowered = text.casefold()
    return any(re.search(pattern, lowered) for pattern in patterns)


def text_said(text, value):
    """A string said up to typography: case, full-width forms and spaces (Chinese text often spaces digits: the
    agent says "尾号 1028" for "尾号1028"), but not inside a longer word ("MAN" is not said by "Manchester")."""
    key = re.sub(r'\s+', '', unicodedata.normalize('NFKC', value).casefold())
    if not key:
        return False
    body = r'\s*'.join(re.escape(ch) for ch in key)
    before = r'(?<![a-z0-9])' if key[0].isascii() and key[0].isalnum() else ''
    after = r'(?![a-z0-9])' if key[-1].isascii() and key[-1].isalnum() else ''
    return bool(re.search(before + body + after, unicodedata.normalize('NFKC', text).casefold()))


CLOCK = re.compile(r'^(\d{1,2}):(\d{2})$')


def time_said(text, value):
    """Whether a clock time "HH:MM" was said as "6 pm", "6:30 PM", "18点", "下午6点半" or "晚上6点30分"."""
    found = CLOCK.match(value.strip())
    if not found:
        return False
    hour, minute = int(found[1]), int(found[2])
    if hour > 23 or minute > 59:
        return False
    twelve, suffix = hour % 12 or 12, ('p' if hour >= 12 else 'a')
    lowered = text.casefold()
    english = rf'(?<![\d:]){twelve}{"" if minute == 0 else f":{minute:02d}"}(?::00)?\s*{suffix}\.?\s*m\b'
    if minute == 0:
        english = rf'(?<![\d:]){twelve}(?::00)?\s*{suffix}\.?\s*m\b'
    if re.search(english, lowered):
        return True
    period = '(?:下午|晚上|傍晚|中午)' if hour >= 12 else '(?:上午|早上|凌晨|早晨)'
    minutes = '(?:整)?' if minute == 0 else ('(?:半|30分)' if minute == 30 else f'0?{minute}分')
    # a bare hour below 12 ("6点") is the morning only when no afternoon word stands before it ("下午6点" is 18:00)
    bare = rf'(?<!\d){hour}' if hour >= 12 else rf'(?<!下午)(?<!晚上)(?<!傍晚)(?<!中午)(?<!\d){hour}'
    chinese = rf'(?:{bare}|{period}\s*{twelve})\s*点\s*{minutes}'
    return bool(re.search(chinese + (r'(?!\s*\d)' if minute == 0 else ''), text))


def verify(value, spec, text, records=None):
    """The value with every number or free string that was not actually said replaced by None; a record id
    counts as said when one of the record's names was (see Records above)."""
    if value is None or not isinstance(spec, dict):
        return value
    kind = spec.get('type')
    if kind in ('integer', 'number') and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if said(text, value) else None
    if kind == 'string' and isinstance(value, str) and 'enum' not in spec:
        if value and (value in text or text_said(text, value) or date_said(text, value) or time_said(text, value)):
            return value
        lowered = text.casefold()
        named = any(name.casefold() in lowered for name in (records or {}).get(value, []))
        return value if value and named else None
    if kind == 'array' and isinstance(value, list):
        return [verify(item, spec.get('items'), text, records) for item in value]
    if kind == 'object' and isinstance(value, dict) and spec.get('properties'):
        return {key: verify(value.get(key), sub, text, records) for key, sub in spec['properties'].items()}
    return value


class Extractor:
    def __init__(self, client):
        self.client, self.cache = client, {}

    def extract(self, schema, speech, records=None):
        """{field: value} for the schema's fields (None when not stated), or None without speech. `records`
        (records_of the episode's trace) lets the model fill ids of records the agent named."""
        text = '\n'.join(m for m in speech if m.strip())
        if not text:
            return None
        fields = schema['fields']
        listing = ('客服查到的记录（客服用名称或描述指代某条记录时，相应字段填这里的 id；没指代就不要填）：\n'
                   + json.dumps(records, ensure_ascii=False) + '\n\n') if records else ''
        prompt = PROMPT.format(schema=json.dumps(fields, ensure_ascii=False, indent=1), speech=text, records=listing)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.cache:
            from env_factory.core.llm import ProviderError
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
        return {name: verify(raw.get(name), spec, text, records) for name, spec in fields.items()}
