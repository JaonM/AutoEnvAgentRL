"""Unfrozen cross-business preparation; never substitute these checks for yield."""
from __future__ import annotations

import copy
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
         '只查询本人有效预约；无本人有效预约时排位与前人数均为空。', ['holds']),
        ('library_partial_renewal', dict(loan_ids=['L1', 'L2'], reader_id='R1', approve_first=True, approve_second=True), dict(current_day=10),
         '本实验业务日固定为第10日；仅本人有效借阅可续借，到期当天未逾期，次数未达实际关联书目上限且无其他读者有效预约。新到期日为原到期日加该书renewal_days，次数加1。对续不了的书报告同书目有效预约人数，保持预约不变。', ['books', 'loans', 'holds']),
        ('library_pickup_change', dict(hold_id='H1', reader_id='R1', copy_choice='K2', approve_change=True), {},
         '仅本人有效预约可变更，原副本必须由该预约占用；新副本必须同书目且未占用，不允许选择原副本。展示所有符合条件的取书点。', ['holds', 'copies', 'sites']),
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
                       'box_cents为每箱价格（分），stock为可供件数。所有报价均对应本次同一商品。')
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
