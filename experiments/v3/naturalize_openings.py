"""Rewrite frozen task openings into how a real user talks, with an independent review.

  .venv/bin/python experiments/v3/naturalize_openings.py FROZEN_DIR OUT_DIR [IDS]

The task itself must not change: same goal, same items and order, every number and id the
task depends on, and everything the user wants to be told. What goes: rule restatements and
transaction semantics a user would not say (those live in the business rules), field names,
math-problem phrasing. A separate reviewer session scores naturalness (1-5) and checks task
equivalence; a rewrite is accepted only when natural >= 4, equivalent, nothing lost or added,
no rule restatement and no deterministic red flag (amounts in 分, transaction-semantics phrases) (up to 3 attempts, the reviewer's issues fed back). The originals get
the same review for a before/after comparison. OUT_DIR is a copy of FROZEN_DIR whose manifest
`raw_request` values are replaced; naturalization.json records every attempt and review.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory_v3.llm import ChatClient, load_env_file  # noqa: E402

REWRITE = """把下面这条“用户开场请求”改写成真实用户在客服/助手对话里会说的话。

任务类别：{category}
业务规则（系统一方的规则；用户不知道也不会复述）：{rules}
原始开场请求：{request}
{feedback}
必须做到：
1. 任务不变：同样的目标、对象和顺序；任务依赖的每个数字和编号都保留；用户想最后被告知的内容保留；不新增信息。
2. 删除用户不会说的话：
   - 事务/系统语义，例如“办不成的别影响已经办好的”“有冲突的别改”“不够就一批都别扣”“确认后一次办好”“原子”“回滚”——这些已由业务规则保证，直接删掉；
   - 规则复述和术语腔，例如“按先到期先出分配”可以说成“先用快过期的”；
   - 字段名、“指定的那个”这类空泛指代（改成“我那个包裹”“我那条预约”这种口语）。
3. 金额一律按元说（1元=100分，换算必须准确，如180分=1.8元），问价时问“多少钱”，不要说“分”。
4. 纯计算类（无工具可查）必须把计算要用的数字说出来，但要像用户引用自己看到的价目表/单据，例如“价目表上写首小时3块、之后每半小时1块2、一天最多15块”。
5. 口语、简短、自然；一到三句话。

示例
原始：帮我按顺序寄这两件包裹，每件先告诉我选哪家、多少钱，再问我确认；办不成的别影响已经寄好的，最后告诉我寄出了哪些包裹和钱包余额。
改写：我有两个包裹要寄，按顺序来吧。每个寄之前跟我说下走哪家、多少钱，我点头了再寄。弄完告诉我寄出去了哪几个，钱包还剩多少。

只输出改写后的话，不要解释。"""

REVIEW = """你是任务数据审核员。判断一条用户开场请求像不像真实用户说的话，并核对它和原始版本是否是同一个任务。

业务规则：{rules}
原始开场请求：{original}
待审开场请求：{candidate}

自然度评分标准（严格执行）：
5 = 完全是真人口语：只说自己的目的、自己知道的信息和想知道的结果。
4 = 基本自然，有轻微书面腔。
3 = 能看出是在替系统交代流程，或用“分”报价、用术语腔。
2 = 明显在复述业务规则或事务语义（如“办不成的别影响已办好的”“不够就都别扣”），或像应用题。
1 = 像系统说明或接口文档。
锚定样例：
- “帮我按顺序寄这两件包裹……办不成的别影响已经寄好的……” → 2
- “我这次停车181分钟，首小时收费300分，之后每半小时120分……停车费是多少分？” → 2
- “我有两个包裹要寄，按顺序来。每个寄之前跟我说下走哪家、多少钱，我点头了再寄。” → 5

只输出 JSON：{{"natural": 1-5 的整数,
"equivalent": true/false（目标、对象、顺序、任务依赖的数字编号、用户想被告知的内容都一致；金额换算成元且换算正确也算一致）,
"missing": [原始版本里有而待审版本丢失的任务信息；被删掉的事务语义/规则复述不算丢失],
"added": [待审版本新增的、原始版本没有的任务信息],
"rule_restatement": true/false,
"issues": "一句话说明主要问题，没有则为空"}}"""

# Deterministic red flags: any hit rejects a rewrite whatever the reviewer says.
RED_FLAGS = ('分？', '多少分', '别影响', '不影响已', '原子', '回滚', '都别扣', '一批都别', '一次办好', '有冲突的别',
             '有冲突的就不', '指定的')


def red_flags(text):
    flags = [flag for flag in RED_FLAGS if flag in text]
    if re.search(r'\d+\s*分(?![钟之])', text):
        flags.append('金额用“分”')
    return flags


def ask_json(client, prompt):
    text = (client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0]
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def review(client, rules, original, candidate):
    return ask_json(client, REVIEW.format(rules=rules, original=original, candidate=candidate))


def accepted(verdict, candidate=''):
    return (not red_flags(candidate) and verdict.get('natural', 0) >= 4 and verdict.get('equivalent') is True and not verdict.get('missing')
            and not verdict.get('added') and verdict.get('rule_restatement') is False)


def naturalize(writer, reviewer, sample, rules, attempts=3):
    original, feedback, history = sample['raw_request'], '', []
    for _ in range(attempts):
        candidate = (writer.chat([{'role': 'user', 'content': REWRITE.format(
            category=sample['category'], rules=rules, request=original, feedback=feedback)}],
            temperature=0.7).get('content') or '').strip()
        verdict = review(reviewer, rules, original, candidate)
        verdict['red_flags'] = red_flags(candidate)
        history.append({'candidate': candidate, 'review': verdict})
        if accepted(verdict, candidate):
            return candidate, history
        feedback = (f'上一次改写：{candidate}\n审核意见：{verdict.get("issues")}；丢失：{verdict.get("missing")}；'
                    f'新增：{verdict.get("added")}；复述规则：{verdict.get("rule_restatement")}；'
                    f'禁用说法：{verdict["red_flags"]}。请修正。\n')
    return None, history


def main(frozen, out, only=None):
    load_env_file(ROOT / '.env')
    frozen, out = Path(frozen), Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new output folder')
    shutil.copytree(frozen, out)
    writer, reviewer = ChatClient('LLM'), ChatClient('V3_REVIEW')
    manifest = json.loads((out / 'manifest.json').read_text())
    report = []
    for sample in manifest['samples']:
        if only and sample['id'] not in only:
            continue
        rules = json.loads((out / f'{sample["id"]:02d}' / 'business.json').read_text())['policy']
        before = review(reviewer, rules, sample['raw_request'], sample['raw_request'])
        before['red_flags'] = red_flags(sample['raw_request'])
        rewritten, history = naturalize(writer, reviewer, sample, rules)
        row = {'id': sample['id'], 'category': sample['category'], 'original': sample['raw_request'],
               'original_review': before, 'rewritten': rewritten, 'attempts': history}
        if rewritten:
            sample['original_raw_request'], sample['raw_request'] = sample['raw_request'], rewritten
        report.append(row)
        print(f"{sample['id']:02d} natural {before.get('natural')} -> "
              f"{history[-1]['review'].get('natural') if rewritten else 'kept'} | {rewritten or sample['raw_request']}",
              flush=True)
    manifest['naturalized'] = {'writer': writer.model, 'reviewer': reviewer.model, 'from': str(frozen)}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    kept = [r for r in report if r['rewritten']]
    summary = {'samples': len(report), 'rewritten': len(kept),
               'natural_before': round(sum(r['original_review'].get('natural', 0) for r in report) / len(report), 2),
               'natural_after': round(sum(r['attempts'][-1]['review'].get('natural', 0) if r['rewritten']
                                          else r['original_review'].get('natural', 0) for r in report) / len(report), 2),
               'rows': report}
    (out / 'naturalization.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3].split(',')} if len(sys.argv) > 3 else None)
