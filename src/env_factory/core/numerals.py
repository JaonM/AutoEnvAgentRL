"""How a number may be said in Chinese conversation: digits, Chinese numerals, and amounts in cents
said in 元. Used to check that a generated opening states every number of its scene, and that a
value extracted from an agent's speech was really said (extract.py).
"""
from __future__ import annotations

import re

DIGITS = '零一二三四五六七八九'


def _under_10k(n):
    parts, zero = [], False
    for value, unit in ((1000, '千'), (100, '百'), (10, '十'), (1, '')):
        digit = n // value % 10
        if digit:
            parts.append(('零' if zero else '') + DIGITS[digit] + unit)
            zero = False
        elif parts:
            zero = True
    return ''.join(parts)


def chinese(n):
    """Chinese readings of a non-negative integer below 10^8: 12 -> 十二, 2 -> 二/两, 180 -> 一百八十/一百八."""
    if not 0 <= n < 10 ** 8:
        return set()
    if n == 0:
        return {'零'}
    high, low = divmod(n, 10000)
    text = (_under_10k(high) + '万' + ('零' if 0 < low < 1000 else '') if high else '') + _under_10k(low)
    if text.startswith('一十'):
        text = text[1:]
    forms = {text}
    if re.search(r'[一二三四五六七八九][百千万][一二三四五六七八九][十百千]$', text):
        forms.add(text[:-1])                       # 一百八十 -> 一百八, 一千五百 -> 一千五
    forms |= {'两' + f[1:] for f in forms if f.startswith('二') and (n < 10 or not f.startswith('二十'))}
    return forms


def spoken(number):
    """Ways a user may say a number: as is, in Chinese numerals (两, 十二), or (cents) in 元 such as
    1.8 / 1块8 / 一块八."""
    forms = {f'{number:g}'}
    if isinstance(number, int) and not isinstance(number, bool):
        forms |= chinese(number)
        yuan = number / 100
        forms |= {f'{yuan:g}', f'{yuan:.2f}', f'{yuan:,.2f}', f'{number:,}'}  # 3.5, 3.50, 1,234.50, 12,345
        whole, cents = divmod(number, 100)
        if not cents:
            forms |= chinese(whole)
        elif cents % 10 == 0:
            forms |= {f'{w}块{DIGITS[cents // 10]}' for w in chinese(whole)} | {f'{whole}块{cents // 10}'}
    return forms


ENGLISH = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
           'seventeen eighteen nineteen twenty').split()
TENS = {30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}


def english(number):
    """English words for small whole numbers ("One offer fits", "three days late")."""
    if not (isinstance(number, int) and not isinstance(number, bool)):
        return set()
    if 0 <= number < len(ENGLISH):
        return {ENGLISH[number]}
    return {TENS[number]} if number in TENS else set()


def said(text, number):
    """True when `text` states `number` in one of its spoken forms, not as part of a longer number: digits,
    Chinese numerals, cents in 元/dollars, English words for small numbers, a share below 1 as a percentage
    (0.15 as "15%"), and a negative amount by its magnitude (the sign is said in words: "you owe $12.50")."""
    if isinstance(number, (int, float)) and not isinstance(number, bool) and number < 0 and said(text, -number):
        return True
    if isinstance(number, float) and 0 < number < 1 and re.search(
            rf'(?<![\d.]){re.escape(f"{number * 100:g}")}\s*(?:%|percent|个百分点)', text, re.I):
        return True
    lowered = text.casefold()
    if any(re.search(rf'\b{word}\b', lowered) for word in english(number)):
        return True
    for form in spoken(number):
        digits = form[0].isdigit()
        before, after = (r'(?<![\d.])', r'(?![\d])') if digits else (r'(?<![零一二两三四五六七八九十百千万])',
                                                                          r'(?![零一二两三四五六七八九十百千万])')
        if re.search(before + re.escape(form) + after, text):
            return True
    return False
