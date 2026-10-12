"""How real users open a conversation: deterministic red flags and speaking personas.

Shared by the request generator (checks request) and the opening rewriter. A red flag is wording
a real user would not say: amounts in 分, transaction semantics or rule restatements, dictating
the platform's confirmation protocol (the platform asks before every write anyway), and virtual
clock values ("当前时间是10", "第10天").
"""
from __future__ import annotations

import hashlib
import re

PHRASES = ('多少分', '别影响', '不影响已', '原子', '回滚', '都别扣', '一批都别', '一次办好', '有冲突的别',
           '有冲突的就不', '指定的', '操作前', '分别让我确认')
PATTERNS = (
    (r'\d+\s*分(?![钟之])', '金额用“分”（用户说“元”）'),
    (r'先(告诉|跟|给|让|发)我.{0,12}(确认|看|过目)', '交代确认流程'),
    (r'等我(确认|点头|同意|回复)', '交代确认流程'),
    (r'我(确认|点头|同意)(了|后|之后|完)(你)?再', '交代确认流程'),
    (r'确认(后|之后|完)(你)?再', '交代确认流程'),
    (r'每(一)?(步|件|条|项|笔|个|次)[^，。；]{0,10}(之?前)?(都)?先', '交代确认流程'),
    (r'当前时间(是|为)?\s*\d+', '虚拟时间'),
    (r'第\s*\d+\s*(天|日)(?![内])', '虚拟时间'),
    (r'(时刻|时间点|时间)(是|为)?\s*\d+(?!\s*(分钟|小时|天|点|号|元|块))', '虚拟时间'),
)

PERSONAS = (
    ('话少直接', '一两句话，开门见山，不寒暄'),
    ('着急', '语气有点急，比如“赶时间”“快帮我看下”'),
    ('带点背景', '先用半句话说明为什么要办这件事，再提要求'),
    ('客气', '礼貌但不啰嗦，可以用“请问”“想咨询下”'),
    ('口语随意', '口语化，带语气词，比如“哈”“呗”“哎”'),
)


def red_flags(text):
    flags = [phrase for phrase in PHRASES if phrase in text]
    flags += [label for pattern, label in PATTERNS if re.search(pattern, text)]
    return sorted(set(flags))


def persona(key):
    """A fixed persona per task id, so rewrites vary across tasks but are reproducible."""
    return PERSONAS[int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % len(PERSONAS)]


def diversity(texts):
    """Share of openings that start with the stock '帮我'/'麻烦'/'请', and distinct first two characters."""
    stock = sum(1 for t in texts if t.strip().startswith(('帮我', '麻烦', '请帮')))
    return {'stock_start_share': round(stock / max(1, len(texts)), 3),
            'distinct_first_two_chars': len({t.strip()[:2] for t in texts})}
