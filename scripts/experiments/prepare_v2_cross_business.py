"""Unfrozen cross-business preparation; never substitute these checks for yield."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path


BLUEPRINT = Path(__file__).resolve().parents[2] / 'experiments/v2/cross_business_20261011_case_blueprint.json'


def library_data():
    return {
        'books': [{'id': 'B2', 'renewal_limit': 1, 'renewal_days': 3},
                  {'id': 'B1', 'renewal_limit': 2, 'renewal_days': 7}],
        'loans': [{'id': 'L1', 'reader_id': 'R1', 'book_id': 'B1', 'active': True, 'due_day': 10, 'renewals': 0},
                  {'id': 'L2', 'reader_id': 'R1', 'book_id': 'B2', 'active': True, 'due_day': 10, 'renewals': 0},
                  {'id': 'L3', 'reader_id': 'R2', 'book_id': 'B1', 'active': True, 'due_day': 9, 'renewals': 1}],
        'holds': [{'id': 'H1', 'reader_id': 'R1', 'book_id': 'B1', 'active': True, 'requested_day': 2, 'copy_id': 'K0'},
                  {'id': 'H2', 'reader_id': 'R2', 'book_id': 'B2', 'active': True, 'requested_day': 1, 'copy_id': None},
                  {'id': 'H3', 'reader_id': 'R1', 'book_id': 'B2', 'active': True, 'requested_day': 1, 'copy_id': None},
                  {'id': 'H4', 'reader_id': 'R2', 'book_id': 'B2', 'active': False, 'requested_day': 0, 'copy_id': None},
                  {'id': 'H5', 'reader_id': 'R1', 'book_id': 'B1', 'active': True, 'requested_day': 3, 'copy_id': 'K3'}],
        'copies': [{'id': 'K0', 'book_id': 'B1', 'site_id': 'S1', 'reserved_by': 'H1'},
                   {'id': 'K1', 'book_id': 'B2', 'site_id': 'S2', 'reserved_by': None},
                   {'id': 'K2', 'book_id': 'B1', 'site_id': 'S2', 'reserved_by': None},
                   {'id': 'K3', 'book_id': 'B1', 'site_id': 'S2', 'reserved_by': 'H5'},
                   {'id': 'K4', 'book_id': 'B1', 'site_id': 'S3', 'reserved_by': None}],
        'sites': [{'id': 'S1', 'name': '家附近', 'supports_holds': True},
                  {'id': 'S2', 'name': '公司附近', 'supports_holds': True},
                  {'id': 'S3', 'name': '自助还书点', 'supports_holds': False}],
    }


def library_cases():
    blueprints = json.loads(BLUEPRINT.read_text())['cases'][:5]
    details = [
        ('library_fine', {}, dict(overdue_days=3, grace_days=1, daily_cents=200, cap_cents=800),
         '', []),
        ('library_renewal_eligibility', dict(loan_id='L1', reader_id='R1'), dict(current_day=10),
         '本实验业务日固定为第10日，到期日当天未逾期；只处理本人有效借阅。次数上限来自借阅实际关联书目。', ['books', 'loans', 'holds']),
        ('library_queue', dict(hold_id='H3', reader_id='R1'), {},
         '按用户指定的本人预约编号查询；允许读取同书目其他读者有效预约以计算排位。无指定本人有效预约时排位与前人数均为空。', ['holds']),
        ('library_partial_renewal', dict(loan_ids=['L1', 'L2'], reader_id='R1', approve_first=True, approve_second=True), dict(current_day=10),
         '本实验业务日固定为第10日；仅本人有效借阅可续借，到期当天未逾期，次数未达实际关联书目上限且无其他读者有效预约。新到期日为原到期日加该书renewal_days，次数加1。对续不了的书报告同书目有效预约人数，保持预约不变。', ['books', 'loans', 'holds']),
        ('library_pickup_change', dict(hold_id='H1', reader_id='R1', copy_choice='K2', approve_change=True), {},
         '仅本人有效预约可变更，原副本必须由该预约占用；未分配副本的预约不能使用本变更业务首次分配副本。新副本必须同书目且未占用，不允许选择原副本。展示所有符合条件的取书点。', ['holds', 'copies', 'sites']),
    ]
    result = []
    for blueprint, (kind, facts, params, extra, tables) in zip(blueprints, details):
        data = library_data()
        result.append({
            'id': blueprint['id'], 'business_family': blueprint['business_family'],
            'stratum': blueprint['stratum'], 'title': blueprint['title'],
            'raw_request': blueprint['draft_user_request'],
            'business': {'provenance': 'operator-authored synthetic heldout preparation; not frozen',
                         'policy': blueprint['public_business_policy'] + extra,
                         'tables': [{'id': name, 'rows': copy.deepcopy(data[name])} for name in tables]},
            'user': {'facts': facts},
            'acceptance': {'kind': kind, 'params': params, 'observations': {}, 'mutation': None},
        })
    return result


def procurement_cases():
    blueprints = json.loads(BLUEPRINT.read_text())['cases'][5:10]
    def quote(key, pack, minimum, price, stock):
        return dict(id=key, pack=pack, minimum_boxes=minimum, box_cents=price, stock=stock,
                    starts_at=0, ends_at=86400, delivery_day=12, quality=4)
    basic = [quote('Q1', 6, 2, 100, 24), quote('Q2', 5, 1, 110, 15)]
    large = [quote(f'Q{i:04d}', 6, 1, 50, 6) for i in range(240)]
    for index, row in enumerate(large):
        if index % 3 == 0:
            row['ends_at'] = 0
        elif index % 3 == 1:
            row['delivery_day'] = 20
        else:
            row['quality'] = 2
    large.extend([quote('V1', 6, 2, 100, 12), quote('V2', 5, 2, 80, 15)])
    details = [
        ('procurement_minimum', [], {}, dict(demand=23, pack=6, minimum_boxes=2)),
        ('procurement_quote', basic, dict(quote_id='Q1'), {}),
        ('procurement_combination', basic, dict(demand=23), {}),
        ('procurement_partial', basic, dict(demand=23, purchase_plans=[[['Q1', 2]], [['Q2', 3]]],
            approvals=[True, True], order_ids=['O1', 'O2']), {}),
        ('procurement_large', large, dict(demand=23, deadline=12, minimum_quality=4,
            authorized_plan=[['V1', 2], ['V2', 3]], approve_purchase=True, order_id='O1'), {}),
    ]
    result = []
    for blueprint, (kind, quotes, facts, params) in zip(blueprints, details):
        tables = [{'id': 'quotes', 'rows': copy.deepcopy(quotes)}] if quotes else []
        if blueprint['stratum'] == 'complex':
            tables += [{'id': 'budgets', 'rows': [{'id': 'P1', 'remaining_cents':
                600 if kind == 'procurement_large' else 400}]},
                       {'id': 'orders', 'rows': []}]
        policy = blueprint['public_business_policy']
        if quotes:
            policy += ('starts_at和ends_at使用虚拟累计秒数，交期使用业务日编号；'
                       '报价起止边界均包含，使用统一虚拟时间接口判断，过期报价不参与任何组合。'
                       'box_cents为每箱价格（分），stock为可供件数。所有报价均对应本次同一商品。')
        if blueprint['stratum'] == 'complex':
            policy += ('用户有权操作预算P1；订单包含id、cost_cents及lines，每条明细包含quote_id与quantity件数；'
                       '订单编号由用户指定，已有同编号订单时不得重复提交。')
        if kind == 'procurement_partial':
            policy += '提交前展示每笔具体报价编号、箱数和费用，授权只适用于展示的该笔方案。订单编号由用户选择。'
        if kind == 'procurement_large':
            policy += ('提交前展示具体报价编号、箱数和总费用，授权绑定该方案、报价条件与预算快照；'
                       '相关报价价格、库存或预算改变即不得使用旧授权提交。费用相同优先超购少；等价最优组合均可。')
        result.append({'id': blueprint['id'], 'business_family': blueprint['business_family'],
            'stratum': blueprint['stratum'], 'title': blueprint['title'],
            'raw_request': blueprint['draft_user_request'],
            'business': {'provenance': 'operator-authored synthetic heldout preparation; not frozen',
                         'policy': policy, 'tables': tables},
            'user': {'facts': facts},
            'acceptance': {'kind': kind, 'params': params, 'observations': {}, 'mutation': None},
            'integration_status': 'requires runtime clock and authorization evidence adapter'})
    return result


def subscription_cases():
    blueprints = json.loads(BLUEPRINT.read_text())['cases'][10:15]
    data = {
        'subscriptions': [dict(id='S1', reader_id='R1', base_plan_id='A', active=True,
            paid_cents=9000, period_days=30, refunded_cents=0),
            dict(id='S2', reader_id='R1', base_plan_id='A', active=True,
            paid_cents=9000, period_days=30, refunded_cents=9000)],
        'changes': [dict(id='C1', subscription_id='S1', plan_id='B', effective_day=10, cancelled=False),
            dict(id='C2', subscription_id='S1', plan_id='A', effective_day=35, cancelled=False),
            dict(id='C3', subscription_id='S1', plan_id='A', effective_day=20, cancelled=True),
            dict(id='C4', subscription_id='S2', plan_id='A', effective_day=0, cancelled=False)],
        'plans': [dict(id='A', period_cents=1000, period_days=30), dict(id='B', period_cents=2000, period_days=30)],
        'bills': [dict(id='I1', subscription_id='S1', amount_cents=1666, period_start=0, period_end=30)],
        'refunds': [dict(id='F0', subscription_id='S2', amount_cents=9000)],
    }
    details = [
        ('subscription_refund', {}, dict(paid_cents=9000, period_days=30, used_days=11), []),
        ('subscription_current', dict(subscription_id='S1', reader_id='R1', change_id='C2'),
         dict(current_day=20), ['subscriptions', 'changes']),
        ('subscription_bill', dict(subscription_id='S1', reader_id='R1'), dict(start=0, end=30),
         ['subscriptions', 'changes', 'plans']),
        ('subscription_correction', dict(subscription_id='S1', reader_id='R1', change_id='C1', plan_id='B',
            comparison_days=[10, 15], final_day=15, authorized_day=15, authorized_cents=1500,
            approve_change=True, bill_id='I1'), dict(start=0, end=30), ['subscriptions', 'changes', 'plans', 'bills']),
        ('subscription_cancel', dict(subscription_ids=['S1', 'S2'], reader_id='R1', used_days=[11, 11],
            approvals=[True, True], refund_ids=['F1', 'F2']), {}, ['subscriptions', 'refunds']),
    ]
    result = []
    for blueprint, (kind, facts, params, tables) in zip(blueprints, details):
        policy = blueprint['public_business_policy'] + ('账期包含起始日、不包含结束日；生效日当天使用新套餐。'
            '所有金额以分表示，分段费用精确累加后统一向下取整为整数分；退款也向下取整。仅处理本人订阅。'
            'cancelled为真的变更不生效，同一订阅同一生效日不得存在多条未撤销变更。')
        if kind == 'subscription_current':
            policy += '本实验业务日固定为第20日；报告指定变更的生效日及当前套餐。'
        if kind == 'subscription_bill':
            policy += ('本次核对账期从第0日开始，到第30日结束（不含第30日）。'
                       '列出各计费区间的起止日和套餐，以及总费用。')
        if kind == 'subscription_correction':
            policy += ('授权绑定最终日期和对应总费用；更新既有变更及账单，不新增重复账单。'
                       '本业务仅调整指定账期[start,end)内的生效日期，跨出该账期应拒绝并保持状态不变。')
        if kind == 'subscription_cancel':
            policy += ('退款为实付乘未用天数除账期天数，向下取整到分，并以实付减历史退款余额为上限。'
                       '首次退款允许创建记录；每笔记录包含id、subscription_id、amount_cents，累计与记录合计必须一致。'
                       '无可退余额时不取消、不新增退款记录。逐项报告可退金额及实际退款结果。')
        result.append({'id': blueprint['id'], 'business_family': blueprint['business_family'],
            'stratum': blueprint['stratum'], 'title': blueprint['title'], 'raw_request': blueprint['draft_user_request'],
            'business': {'provenance': 'operator-authored synthetic heldout preparation; not frozen',
                'policy': policy, 'tables': [{'id': key, 'rows': copy.deepcopy(data[key])} for key in tables]},
            'user': {'facts': facts}, 'acceptance': {'kind': kind, 'params': params, 'observations': {}, 'mutation': None}})
    return result


def access_cases():
    blueprints = json.loads(BLUEPRINT.read_text())['cases'][15:20]
    def grant(identity, node, principal='R1', kind='user', permission='write', effect='allow', active=True):
        return dict(id=identity, node_id=node, principal_id=principal, principal_kind=kind,
            permission=permission, effect=effect, active=active, starts_at=0, expires_at=None)
    base = {'nodes': [dict(id='N0', parent_id=None), dict(id='N1', parent_id='N0'),
                     dict(id='F1', parent_id='N1'), dict(id='F2', parent_id='N1')],
        'grants': [grant('G1', 'N0', 'T1', 'group'), grant('G2', 'F2', effect='deny'),
                   grant('G3', 'F1', permission='read'), grant('G4', 'N1', permission='read', effect='deny'),
                   grant('T0', 'F1', 'R2', active=False), grant('T1', 'F2', 'R2', active=False),
                   grant('D1', 'N1', 'R2', effect='deny')],
        'memberships': [dict(id='M1', user_id='R1', group_id='T1', active=True, starts_at=0, expires_at=100)]}
    details = [
        ('access_direct', {}, {'effects': [{'permission': 'read', 'effect': 'allow'},
            {'permission': 'read', 'effect': 'deny'}, {'permission': 'write', 'effect': 'allow'}]}),
        ('access_query', dict(node_id='F1', user_id='R1'), {}),
        ('access_compare', dict(node_ids=['F1', 'F2'], user_id='R1'), {}),
        ('access_temporary', dict(node_ids=['F1', 'F2'], grant_ids=['T0', 'T1'], user_id='R2',
            expires_at=50, approve_grants=True), {}),
        ('access_revoke', dict(node_id='F1', user_id='R1', grant_ids=['G1', 'G5'], approvals=[True, False]), {}),
    ]
    result = []
    for blueprint, (kind, facts, params) in zip(blueprints, details):
        data = copy.deepcopy(base)
        if kind == 'access_temporary':
            # One target is feasible, the other has a file-level denial.
            next(g for g in data['grants'] if g['id'] == 'D1')['node_id'] = 'F2'
        if kind == 'access_revoke':
            data['nodes'] = [dict(id=f'N{i}', parent_id=f'N{i-1}' if i else None) for i in range(120)]
            data['nodes'] += [dict(id='F1', parent_id='N119'), dict(id='F2', parent_id='N119')]
            data['grants'].append(grant('G5', 'N80'))
        policy = blueprint['public_business_policy'] + ('授权和组成员关系均在active为真、虚拟累计秒数不小于starts_at、'
            '且小于expires_at时有效；expires_at为空表示永久有效。权限沿实际parent_id继承，读写独立，拒绝优先，默认拒绝。')
        if kind in ('access_query', 'access_compare'):
            policy += '报告有效允许与拒绝的记录编号，并说明无有效允许或显式拒绝的原因。'
        if kind == 'access_temporary':
            policy += ('本实验请求者是有权维护所指定记录的项目管理员；任何有效拒绝（包含文件级拒绝）均阻止临时授权。'
                       '启用既有记录时starts_at设为实际提交时的虚拟累计秒数，expires_at设为用户精确确认的截止时间。'
                       '两个对象分别检查，允许只授权可行对象；报告当前和截止时点的实际编辑权限。')
        if kind == 'access_revoke':
            policy += ('本实验请求者是有权撤销所指定个人或共享组授权记录的项目管理员，撤销共享组授权会影响该组全部成员，须在确认前告知。'
                       '授权仅针对明确确认的记录编号；报告原允许来源、剩余来源和最终编辑权限。')
        result.append({'id': blueprint['id'], 'business_family': blueprint['business_family'],
            'stratum': blueprint['stratum'], 'title': blueprint['title'], 'raw_request': blueprint['draft_user_request'],
            'business': {'provenance': 'operator-authored synthetic heldout preparation; not frozen',
                'policy': policy, 'tables': [] if kind == 'access_direct' else
                [{'id': key, 'rows': rows} for key, rows in data.items()]},
            'user': {'facts': facts}, 'acceptance': {'kind': kind, 'params': params, 'observations': {}, 'mutation': None},
            'integration_status': 'requires runtime clock and authorization evidence adapter'})
    return result


def cases():
    from env_factory_v2.business_graph import validate_relations
    rows = library_cases() + procurement_cases() + subscription_cases() + access_cases()
    # Explicit operator declarations, never inferred from similar field names.
    declarations = [
        ('loans', 'book_id', 'books', False), ('holds', 'book_id', 'books', False),
        ('holds', 'copy_id', 'copies', True), ('copies', 'reserved_by', 'holds', True),
        ('copies', 'site_id', 'sites', False), ('copies', 'book_id', 'books', False),
        ('changes', 'subscription_id', 'subscriptions', False),
        ('changes', 'plan_id', 'plans', False), ('subscriptions', 'base_plan_id', 'plans', False),
        ('bills', 'subscription_id', 'subscriptions', False),
        ('refunds', 'subscription_id', 'subscriptions', False),
        ('nodes', 'parent_id', 'nodes', True), ('grants', 'node_id', 'nodes', False),
    ]
    for row in rows:
        business = row['business']
        tables = {table['id'] for table in business['tables']}
        relations = []
        for source, column, target, nullable in declarations:
            if source not in tables or target not in tables:
                continue
            meaning = f'业务关联：{source}.{column} 指向 {target}.id' + (
                '，允许未关联的空值。' if nullable else '，必须关联已有记录。')
            business['policy'] += '\n' + meaning
            relations.append(dict(id=source + '_' + column, source_table=source,
                source_column=column, target_table=target, target_column='id', nullable=nullable,
                policy_text=meaning))
        business['relations'] = relations
        validate_relations(business)
    return rows


def export_preparation(destination):
    """Materialize reviewable source assets. This is deliberately not a freeze."""
    rows = cases()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 1, 'status': 'preparation_only_not_measurement_ready',
        'scope': 'synthetic cross-business heldout preparation',
        'mix': {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8},
        'measurement_ready': False, 'samples': [],
        'remaining': ['runtime clock and exact authorization evidence integration',
                      'public provenance and actual novelty audit',
                      'native natural openings and profile review',
                      'isolated knowledge graph installation and readback',
                      'full corpus and fixed protocol freeze']}
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        folder = destination / f"{row['id']:02d}"
        folder.mkdir()
        for name in ('business', 'user', 'acceptance'):
            save(folder / f'{name}.json', row[name])
        save(folder / 'draft_request.json', {'draft_user_request': row['raw_request'],
            'provenance': 'operator blueprint; not native natural opening'})
        manifest['samples'].append({key: row[key] for key in ('id', 'business_family', 'stratum', 'title')})
        manifest['samples'][-1]['assets'] = {name: f"{row['id']:02d}/{name}.json"
                                           for name in ('business', 'user', 'acceptance', 'draft_request')}
    save(destination / 'manifest.json', manifest)
    save(destination / 'preparation_checksums.json', {'is_measurement_freeze': False,
        'files': {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(destination.rglob('*.json'))}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    export_preparation(parser.parse_args().output)
