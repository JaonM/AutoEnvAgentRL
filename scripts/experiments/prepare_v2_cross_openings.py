"""Native natural openings from user scenes, without acceptance or tool contracts."""
import argparse
import json
from pathlib import Path
from env_factory_v2.native.code_agent import NativeCodeAgent, object_schema
from v2_preparation_pack import load_pack, identity, user_scenes, exact_ids, business_review_input


def scenes():
    outcomes = [
        ('还书晚了，想知道要交多少钱', {'晚还天数': 3, '免罚天数': 1, '每天罚金元': 2, '单本上限元': 8}),
        ('借的书还没看完，想知道能否续借', {}),
        ('预约的书还没拿到，想知道排位和前面有效预约人数', {}),
        ('两本书想续借，每本分别确认，不能续的查候补情况，保留已办好的那本', {}),
        ('公司通勤变了，预约的书想换取书点，先看可选地方，确认再改，失败保留原预约', {}),
        ('买东西只按整箱卖，想知道最少订几箱及多出几件', {'需求件数': 23, '每箱件数': 6, '最低箱数': 2}),
        ('准备使用手头报价买东西，想知道现在是否有效及最低购买量', {}),
        ('一家公司可能供不齐，想比较供货组合和最低总费用', {}),
        ('分两家公司购买，分别确认数量费用，第一笔后再确认第二笔，失败保留第一笔并说明缺口', {}),
        ('报价很多，想查全符合交期质量的报价，选满足需求的最便宜组合，确认后购买', {}),
        ('提前结束订阅，想知道按天能退多少钱', {'账期天数': 30, '实付元': 90, '已用天数': 11}),
        ('曾申请换套餐，想知道那次申请什么时候生效及现在使用哪个套餐', {}),
        ('月中换过套餐，想核对本期费用及各时间段怎么算', {}),
        ('想调整套餐，比较不同生效日期费用，日期可能更正，只按最终确认日期办理', {}),
        ('想取消两项订阅，分别算退款并确认，一项办好后另一项不能退也别恢复第一项', {}),
        ('想知道能不能读、能不能写，读写分别判断，拒绝优先',
         {'上层目录': '允许读', '子目录': '拒绝读', '文件': '允许写'}),
        ('想打开文件，想知道是否有权限，若不能需知道阻拦原因', {}),
        ('同目录两个文件一个能改另一个不能，想查清有效授权和差别', {}),
        ('请同事临时编辑两个文件，先检查限制，给出可行权限及截止时间，确认再办', {}),
        ('项目目录层级很深，想撤同事临时编辑权限，先查清来源，分别确认，办后检查是否仍能编辑', {}),
    ]
    return [{'id': i, 'situation_and_desired_outcome': outcome, 'known_information': known}
            for i, (outcome, known) in enumerate(outcomes, 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-generated', type=Path)
    parser.add_argument('--business-review', action='store_true')
    parser.add_argument('--case-pack', type=Path, help='Explicit independent cases, scenes, and historical rules; defaults to historical corpus')
    args = parser.parse_args()
    pack = load_pack(args.case_pack)
    ids = exact_ids(pack['cases'])
    count = len(ids)
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    payload = {'scenes': user_scenes(pack)}
    save('preparation_identity.json', identity(pack))
    save('user_scenes.json', payload)
    agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'native-generation')
    if args.review_generated:
        if json.loads((args.review_generated / 'preparation_identity.json').read_text()) != identity(pack):
            raise ValueError('Generated openings refer to a different preparation pack')
        original = json.loads((args.review_generated / 'generated_openings.json').read_text())['response']['openings']
        if args.business_review:
            review_input = business_review_input(pack, original)
            save('review_input.json', review_input)
            item = object_schema({'id': {'type': 'integer'}, 'genuine_rule_or_composition_novelty': {'type': 'boolean'},
                'public_demand_consistent': {'type': 'boolean'}, 'public_policy_sufficient': {'type': 'boolean'},
                'new_dimensions': {'type': 'array', 'items': {'type': 'string'}},
                'remaining_ambiguities': {'type': 'array', 'items': {'type': 'string'}}, 'reason': {'type': 'string'}})
            result, provenance = agent.ask('Independently audit each new case against historical rules. '
                'A different business name alone is not novelty; identify an actual new rule or composition. '
                'Generic primitives may overlap. Check user demand consistency and whether PUBLIC policy '
                'specifies enough business meaning without contradictory or hidden requirements. '
                'Flag missing units, rounding, time boundaries, authorization scope, and unresolved precedence. '
                'Private user information is available through clarification, not in the initial opening; '
                'do not require users to disclose all IDs, choices, or dates up front. Virtual time is '
                'obtained from the platform clock during execution; lack of a frozen clock value is not '
                'a policy omission. Do flag business rules that are not publicly stated. '
                'Do not infer absent policies or repair inputs. No tools.', review_input,
                object_schema({'reviews': {'type': 'array', 'items': item}}))
            save('business_review.json', {'response': result, 'provenance': provenance})
            exact_ids(result['reviews'], ids)
            passed = sum(all(row[k] is True for k in ('genuine_rule_or_composition_novelty',
                'public_demand_consistent', 'public_policy_sufficient')) for row in result['reviews'])
            print(f'Business preparation review: {passed}/{count} passed; not measured yield', flush=True)
            return
        exact_ids(original, ids)
        review_input = {'scenes': user_scenes(pack), 'openings': original}
        save('review_input.json', review_input)
        review_item = object_schema({'id': {'type': 'integer'}, 'natural': {'type': 'boolean'},
            'faithful': {'type': 'boolean'}, 'no_invented_facts_or_steps': {'type': 'boolean'},
            'reason': {'type': 'string'}})
        result, provenance = agent.ask('Independently review every Chinese user opening against its scene. '
            'Check ordinary user voice, preservation of all known information and desired outcomes, '
            'and absence of invented facts or platform execution steps. Business preferences explicitly '
            'in a scene are allowed. Do not rewrite openings or compute answers. No tools.', review_input,
            object_schema({'reviews': {'type': 'array', 'items': review_item}}))
        save('independent_review.json', {'response': result, 'provenance': provenance})
        exact_ids(result['reviews'], ids)
        passed = sum(all(r[key] is True for key in ('natural', 'faithful', 'no_invented_facts_or_steps'))
                     for r in result['reviews'])
        print(f'Independent opening review: {passed}/{count} passed; not factory yield', flush=True)
        return
    item = object_schema({'id': {'type': 'integer'}, 'user_request': {'type': 'string'}})
    result, provenance = agent.ask('Write one natural Chinese opening per scene as an ordinary user. '
        'Preserve all supplied known information and desired outcomes. Do not compute answers, '
        'invent facts, name tools, give implementation steps, or add business rules. No tools.',
        payload, object_schema({'openings': {'type': 'array', 'items': item}}))
    save('generated_openings.json', {'response': result, 'provenance': provenance})
    exact_ids(result['openings'], ids)
    print(f'Generated {count} native openings; independent review and freeze still pending', flush=True)


if __name__ == '__main__':
    main()
