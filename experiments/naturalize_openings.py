"""Rewrite task openings into how a real user talks, with an independent review.

  .venv/bin/python experiments/naturalize_openings.py SOURCE OUT [IDS]

SOURCE is a frozen V2 dataset folder (manifest.json `raw_request`) or a folder of generated request
.json files (`user_request`); OUT is a new folder with the openings replaced (frozen: a copy of the
dataset; requests: rewritten request files). The task must not change: same goal, items and
order, every number and id it depends on, and what the user wants to be told.

What a real user does not say, and goes:
1. the platform's confirmation protocol ("每件先告诉我……等我确认再……"): the platform asks before
   every write anyway; caring about money is fine ("别乱扣钱"), dictating steps is not;
2. virtual clock values ("当前时间是10", "第10天"): said relatively instead ("还有20分钟开诊");
3. math-problem listings in direct tasks: the numbers stay, said as one would read a receipt;
4. stock sameness: each task gets a fixed persona (env_factory.production.openings.PERSONAS), and not
   every opening starts with "帮我"/"麻烦".
Plus the earlier rules: amounts in 元, no transaction semantics or rule restatements, no field names.

A rewrite is accepted when the reviewer scores it natural >= 4, equivalent, nothing lost or added,
no rule restatement, and no deterministic red flag (env_factory.production.openings.red_flags); up to 3
attempts with the reviewer's issues fed back. naturalization.json records every attempt, the
before/after reviews and diversity metrics.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory.core.llm import ChatClient, load_env_file  # noqa: E402
from env_factory.production.openings import diversity, persona, red_flags  # noqa: E402

REWRITE = """把下面这条“用户开场请求”改写成真实用户在客服/助手对话里会说的话。

任务类别：{category}
业务规则（系统一方的规则；用户不知道也不会复述）：{rules}
原始开场请求：{request}
这位用户的说话风格：{persona}（{persona_hint}）
{feedback}
必须做到：
1. 任务不变：同样的目标、对象和顺序；任务依赖的每个数字和编号都保留；用户想知道的结果保留（可以随口说，如“弄完跟我说下还剩多少钱”）；不新增信息。
2. 不要交代确认流程：平台会在每次扣钱/写入前自动征求用户同意，所以删掉“每件先告诉我……等我确认再……”“确认后再……”“先给我看看再办”这类步骤要求；可以表达关心（如“别乱扣钱”“价格合适就办”），但不要规定步骤。
3. 不要出现虚拟时间编号（如“当前时间是10”“第10天”“时间30”）：改成相对说法，单位按业务规则（如“还有20分钟开诊”“两天后到期”“十天前买的”）。
4. 纯计算类（没有工具可查）必须把计算要用的数字都说出来，但像随口念单据：一两句连着说，不要用“甲、乙、丙”逐条罗列，不要写成题目。
5. 删除用户不会说的话：事务/系统语义（“办不成的别影响已办好的”“原子”“回滚”）、规则复述、术语腔、字段名、“指定的那个”。
6. 金额一律按元说（1元=100分，换算准确），问价问“多少钱”。
7. 按上面的说话风格来写；不要总以“帮我”“麻烦”开头；一到三句话。

只输出改写后的话，不要解释。"""

REVIEW = """你是任务数据审核员。判断一条用户开场请求像不像真实用户说的话，并核对它和原始版本是否是同一个任务。

业务规则：{rules}
原始开场请求：{original}
待审开场请求：{candidate}

自然度评分标准（严格执行）：
5 = 完全是真人口语：只说自己的目的、自己知道的信息和想知道的结果。
4 = 基本自然，有轻微书面腔。
3 = 在替系统交代流程（如“每件先告诉我……等我确认再……”），或出现虚拟时间编号（“当前时间是10”“第10天”），或像在逐条罗列题目数据，或用“分”报价、术语腔。
2 = 明显在复述业务规则或事务语义（如“办不成的别影响已办好的”“不够就都别扣”），或像应用题。
1 = 像系统说明或接口文档。
锚定样例：
- “这两个包裹按顺序寄。每件寄之前先告诉我走哪家、多少钱，等我确认再寄。” → 3
- “我想算下这四笔预约现在取消能退多少：当前时间是10，预约甲开诊时间30、实付80元……” → 2
- “有两个包裹要寄，先寄前面那个，挑便宜的走就行，弄完跟我说下钱包还剩多少。” → 5

只输出 JSON：{{"natural": 1-5 的整数,
"equivalent": true/false（目标、对象、顺序、任务依赖的数字编号、用户想知道的结果都一致；金额换算成元、虚拟时间换成正确的相对说法也算一致）,
"missing": [原始版本里有而待审版本丢失的任务信息；删掉的确认流程、事务语义、规则复述不算丢失],
"added": [待审版本新增的、原始版本没有的任务信息],
"rule_restatement": true/false,
"issues": "一句话说明主要问题，没有则为空"}}"""


def ask_json(client, prompt):
    text = (client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or '').strip()
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def review(client, rules, original, candidate):
    verdict = ask_json(client, REVIEW.format(rules=rules, original=original, candidate=candidate))
    verdict['red_flags'] = red_flags(candidate)
    return verdict


def accepted(verdict):
    return (not verdict['red_flags'] and verdict.get('natural', 0) >= 4 and verdict.get('equivalent') is True
            and not verdict.get('missing') and not verdict.get('added') and verdict.get('rule_restatement') is False)


def naturalize(writer, reviewer, key, category, original, rules, attempts=3):
    name, hint = persona(key)
    feedback, history = '', []
    for _ in range(attempts):
        candidate = (writer.chat([{'role': 'user', 'content': REWRITE.format(
            category=category, rules=rules, request=original, persona=name, persona_hint=hint,
            feedback=feedback)}], temperature=0.8).get('content') or '').strip()
        verdict = review(reviewer, rules, original, candidate)
        history.append({'candidate': candidate, 'review': verdict})
        if accepted(verdict):
            return candidate, name, history
        feedback = (f'上一次改写：{candidate}\n审核意见：{verdict.get("issues")}；丢失：{verdict.get("missing")}；'
                    f'新增：{verdict.get("added")}；复述规则：{verdict.get("rule_restatement")}；'
                    f'禁用说法：{verdict["red_flags"]}。请修正。\n')
    return None, name, history


def items(source, out):
    """(key, category, opening, rules, write_back) for each task in a frozen dataset or requests folder."""
    if (source / 'manifest.json').exists():
        shutil.copytree(source, out)
        manifest = json.loads((out / 'manifest.json').read_text())

        def save(sample, text):
            sample['original_raw_request'], sample['raw_request'] = sample['raw_request'], text
            (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        for sample in manifest['samples']:
            rules = json.loads((out / f'{sample["id"]:02d}' / 'business.json').read_text())['policy']
            yield sample['id'], sample['category'], sample['raw_request'], rules, (lambda t, s=sample: save(s, t))
    else:
        out.mkdir(parents=True)
        for path in sorted(source.glob('*.json')):
            request = json.loads(path.read_text())

            def save(text, request=request, path=path):
                request['original_user_request'], request['user_request'] = request['user_request'], text
                (out / path.name).write_text(json.dumps(request, ensure_ascii=False, indent=1))
            shutil.copy(path, out / path.name)  # unchanged copy until a rewrite is accepted
            yield request['id'], request['category'], request['user_request'], request['business_rules'], save


def main(source, out, only=None):
    load_env_file(ROOT / '.env')
    source, out = Path(source), Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new output folder')
    writer, reviewer = ChatClient('LLM'), ChatClient('V3_REVIEW')
    report = []
    for key, category, original, rules, save in items(source, out):
        if only and str(key) not in only:
            continue
        before = review(reviewer, rules, original, original)
        rewritten, name, history = naturalize(writer, reviewer, key, category, original, rules)
        if rewritten:
            save(rewritten)
        report.append({'id': key, 'category': category, 'persona': name, 'original': original,
                       'original_review': before, 'rewritten': rewritten, 'attempts': history})
        print(f"{key} [{name}] natural {before.get('natural')}{before['red_flags'] or ''} -> "
              f"{history[-1]['review'].get('natural') if rewritten else 'kept'} | {rewritten or original}", flush=True)
    final = [r['rewritten'] or r['original'] for r in report]
    after = [r['attempts'][-1]['review'] if r['rewritten'] else r['original_review'] for r in report]
    summary = {'source': str(source), 'writer': writer.model, 'reviewer': reviewer.model, 'samples': len(report),
               'rewritten': sum(bool(r['rewritten']) for r in report),
               'natural_before': round(sum(r['original_review'].get('natural', 0) for r in report) / len(report), 2),
               'natural_after': round(sum(v.get('natural', 0) for v in after) / len(report), 2),
               'red_flagged_before': sum(bool(r['original_review']['red_flags']) for r in report),
               'red_flagged_after': sum(bool(red_flags(t)) for t in final),
               'diversity_before': diversity([r['original'] for r in report]), 'diversity_after': diversity(final),
               'rows': report}
    (out / 'naturalization.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None)
