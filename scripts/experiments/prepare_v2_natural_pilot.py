"""Prepare operator-designed natural requests before any model generation.

This is synthetic business evidence, not production-user sampling. Acceptance
files are separate and never passed to the author or rollout policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def table(name, rows=None, generator=None):
    return {'id': name, **({'rows': rows} if rows is not None else {'generator': generator})}


def cases():
    result = []
    def add(stratum, title, request, policy, tables=(), facts=None, kind=None, params=None, observations=None, mutation=None):
        result.append({'stratum': stratum, 'title': title, 'raw_request': request,
            'business': {'provenance': 'operator-authored synthetic business dossier; frozen before generation',
                         'policy': policy, 'tables': list(tables)},
            'user': {'facts': facts or {}},
            'acceptance': {'kind': kind, 'params': params or {},
                           'observations': observations or {}, 'mutation': mutation}})
    add('direct', '办公室电费', '办公室暖风机1100瓦，每天开3小时，照明65瓦开5小时。电价每度0.9元，帮我算算一天用了几度电、花多少钱。',
        '1度电=1千瓦时，按实际功率乘时长计算，费用不另加服务费。', kind='energy', params={'watts': [1100, 65], 'hours': [3, 5], 'price': 0.9})
    add('direct', '优惠后的实付', '我买了128元和96元的两件东西，有一张15元券，按店里的会员优惠最后要付多少？也告诉我总共省了多少。',
        '会员订单原价满200元打九折，然后减15元券；不满200元只减券。实付最低0元。',
        kind='discount', params={'prices': [128, 96], 'threshold': 200, 'rate': 0.9, 'coupon': 15})
    add('direct', '会议室计费', '我今天9点20分到12点05分用了会议室，帮我算使用时长和费用。',
        '会议室每小时24元，按实际分钟比例计费，不向上取整。', kind='duration', params={'start_minutes': 560, 'end_minutes': 725, 'hourly_rate': 24})
    add('direct', '讲师候选排序', '这四位讲师，甲评分4.8报价600，乙评分4.9报价800，丙评分4.9报价700，丁评分4.8报价600。按我们的选人规则列出前三位和他们的总报价。',
        '优先评分高，再选报价低；两者相同按姓名升序。姓名排序约定甲、乙、丙、丁对应编号A、B、C、D，最后以编号升序打破平局。',
        kind='ranking', params={'candidates': [{'id': 'A', 'name': '甲', 'rating': 4.8, 'price': 600}, {'id': 'B', 'name': '乙', 'rating': 4.9, 'price': 800}, {'id': 'C', 'name': '丙', 'rating': 4.9, 'price': 700}, {'id': 'D', 'name': '丁', 'rating': 4.8, 'price': 600}]})
    add('simple', '维修单进度', '帮我看看我的维修单现在到哪一步了，预计哪天修好？',
        '按用户提供的维修单编号精确查询当前状态和预计完成日期；标题和姓名不作为身份依据。',
        [table('tickets', [{'id': f'T{i:02d}', 'title': '屏幕维修', 'status': 'waiting' if i != 7 else 'repairing', 'eta': '2026-10-13' if i != 7 else '2026-10-12'} for i in range(1, 9)])],
        {'ticket_id': 'T07'}, 'ticket', observations={'tickets': ['T07']}, mutation={'table': 'tickets', 'key': 'T07', 'column': 'status', 'value': 'completed'})
    add('simple', '会员积分', '查一下我的会员卡还有多少积分，现在能不能使用？',
        '按会员卡编号查询points与enabled，0积分和false启用状态都是有效结果。',
        [table('members', [{'id': f'M{i:02d}', 'points': 0 if i == 3 else i * 10, 'enabled': i != 3} for i in range(1, 7)])],
        {'member_id': 'M03'}, 'member', observations={'members': ['M03']}, mutation={'table': 'members', 'key': 'M03', 'column': 'points', 'value': 25})
    add('simple', '有货鞋款', '看看有我尺码、现在还能买的鞋有哪些，给我款号就行。',
        '只选size等于用户尺码、active=true、stock>0的鞋款；款名不影响资格，给出全部款号。',
        [table('shoes', [{'id': f'S{i}', 'size': 39 if i % 2 else 40, 'active': i != 5, 'stock': 0 if i == 7 else i, 'name': '日常跑鞋'} for i in range(1, 11)])],
        {'size': 39}, 'shoes', observations={'shoes': ['S1', 'S3', 'S9']}, mutation={'table': 'shoes', 'key': 'S3', 'column': 'stock', 'value': 0})
    add('simple', '待报销票据', '帮我找出我还没报销的票据，列一下编号，一共多少钱？',
        '按用户employee_id筛选paid=false且cancelled=false的票据；返回完整编号集合与金额合计。',
        [table('receipts', [{'id': f'R{i}', 'employee_id': 'E2' if i % 2 else 'E1', 'paid': i == 3, 'cancelled': i == 5, 'amount': i * 17} for i in range(1, 13)])],
        {'employee_id': 'E2'}, 'receipts', observations={'receipts': ['R1', 'R7', 'R9', 'R11']}, mutation={'table': 'receipts', 'key': 'R7', 'column': 'paid', 'value': True})
    add('ordinary', '订单配送追踪', '我的订单现在配送到哪儿了？预计什么时候到？',
        '按订单编号读取其shipment_id，再查询该实际配送记录的status和eta；同名收件人不能串单。',
        [table('orders', [{'id': 'O1', 'shipment_id': 'P2', 'recipient': '小林'}, {'id': 'O2', 'shipment_id': 'P1', 'recipient': '小林'}]), table('shipments', [{'id': 'P1', 'status': 'pending', 'eta': '2026-10-16'}, {'id': 'P2', 'status': 'out_for_delivery', 'eta': '2026-10-11'}])],
        {'order_id': 'O1'}, 'shipment', observations={'orders': ['O1'], 'shipments': ['P2']}, mutation={'table': 'shipments', 'key': 'P2', 'column': 'status', 'value': 'delivered'})
    add('ordinary', '设备保修费用', '这台设备维修还能走保修吗？如果不能，这次检修按对应型号要收多少？',
        '设备age_days<=实际关联型号warranty_days时保修且费用0，否则费用是该型号inspection_fee。先查设备，再查其model_id对应型号。',
        [table('devices', [{'id': 'D1', 'model_id': 'K2', 'age_days': 365}, {'id': 'D2', 'model_id': 'K1', 'age_days': 200}]), table('models', [{'id': 'K1', 'warranty_days': 180, 'inspection_fee': 80}, {'id': 'K2', 'warranty_days': 365, 'inspection_fee': 120}])],
        {'device_id': 'D1'}, 'warranty', observations={'devices': ['D1'], 'models': ['K2']}, mutation={'table': 'devices', 'key': 'D1', 'column': 'age_days', 'value': 366})
    add('ordinary', '部门在职人员', '帮我查一下这个部门现在有哪些在职同事，给我员工编号和人数。',
        '先按用户部门名称确定真实部门编号，再查该部门active=true的全部员工；员工姓名不影响身份。',
        [table('departments', [{'id': 'DEP1', 'name': '研发'}, {'id': 'DEP2', 'name': '运营'}]), table('employees', [{'id': f'E{i}', 'department_id': 'DEP1' if i % 2 else 'DEP2', 'active': i != 5, 'name': '小林'} for i in range(1, 11)])],
        {'department_name': '研发'}, 'employees', observations={'departments': ['DEP1'], 'employees': ['E1', 'E3', 'E7', 'E9']}, mutation={'table': 'employees', 'key': 'E7', 'column': 'active', 'value': False})
    add('ordinary', '取消健身预约', '帮我取消这次健身预约。先告诉我会改什么，我确认后再处理，处理完告诉我预约和时段的状态。',
        '读取预约及其slot_id实际时段。授权后同时把预约active和对应时段reserved设为false，保留其他记录；拒绝时不修改，提交后查询核验。',
        [table('reservations', [{'id': 'B1', 'slot_id': 'L2', 'active': True}, {'id': 'B2', 'slot_id': 'L1', 'active': True}]), table('slots', [{'id': 'L1', 'reserved': True}, {'id': 'L2', 'reserved': True}])],
        {'reservation_id': 'B1', 'approve_consent': True}, 'cancel', observations={'reservations': ['B1'], 'slots': ['L2']})
    add('complex', '审批链追踪', '帮我从这份申请往下查，列出经过的审批环节顺序、累计耗时和最后负责人的名字，别跳过中间环节。',
        '申请关联entry_id；审批环节按实际next_id连接，null表示结束。最后环节owner_id关联人员。累计耗时为链上duration之和。',
        [table('applications', [{'id': 'A1', 'entry_id': 'N0'}]), table('nodes', [{'id': f'N{i}', 'next_id': f'N{i + 1}' if i < 9 else None, 'duration': i + 2, 'owner_id': 'U1' if i < 9 else 'U2'} for i in range(10)]), table('people', [{'id': 'U1', 'name': '陈宁'}, {'id': 'U2', 'name': '周青'}])],
        {'application_id': 'A1'}, 'chain', observations={'applications': ['A1'], 'nodes': [f'N{i}' for i in range(10)], 'people': ['U2']}, mutation={'table': 'nodes', 'key': 'N4', 'column': 'duration', 'value': 40})
    add('complex', '项目大额报销统计', '帮我把这个项目已批准、金额超过4900元的报销全部查完，告诉我有多少笔、总金额多少，别只算第一页。',
        '先按项目名称查project_id，再筛选该项目status=approved且amount>4900的报销。账本有6000条，每次查询最多64条。',
        [table('projects', [{'id': 'P1', 'name': '远山'}, {'id': 'P2', 'name': '近水'}]), table('expenses', generator={'count': 6000, 'seed': 0, 'columns': {'id': {'kind': 'sequence', 'start': 1, 'step': 1, 'prefix': 'X'}, 'project_id': {'kind': 'cycle', 'values': ['P1', 'P2']}, 'status': {'kind': 'cycle', 'values': ['approved', 'pending', 'denied']}, 'amount': {'kind': 'sequence', 'start': 1, 'step': 1}}})],
        {'project_name': '远山'}, 'expenses', mutation={'table': 'expenses', 'key': 'X4903', 'column': 'status', 'value': 'pending'})
    add('complex', '仓库补货资源统计', '查一下这个仓库还能拿出至少6件的SKU，一共多少种、合计多少件？数据多也请全部核对。',
        '先按仓库名称查真实warehouse_id，筛选该仓库stock>=6的全部库存。账本8000条，每次最多128条，0库存不是缺失。',
        [table('warehouses', [{'id': 'W1', 'name': '北库'}, {'id': 'W2', 'name': '南库'}]), table('inventory', generator={'count': 8000, 'seed': 0, 'columns': {'id': {'kind': 'sequence', 'start': 1, 'step': 1, 'prefix': 'SKU'}, 'warehouse_id': {'kind': 'cycle', 'values': ['W1', 'W2']}, 'stock': {'kind': 'cycle', 'values': [0, 3, 6, 9, 12]}}})],
        {'warehouse_name': '北库'}, 'inventory', mutation={'table': 'inventory', 'key': 'SKU3', 'column': 'stock', 'value': 0})
    add('complex', '换设备并调整维护', '我想把当前使用的设备换成兼容且有空余容量的那台。先给我可选设备，我选好并确认改动后再换，也把维护安排一起调整好。',
        '分配关联旧设备；候选需class_code匹配且capacity>0，完整展示。用户选新设备后读取新旧维护记录。一次更换同时旧capacity+1、新capacity-1、分配device_id改为新设备、旧维护scheduled=false、新维护scheduled=true；容量非负且总量守恒。提交后核验，拒绝不写入。',
        [table('assignments', [{'id': 'AS1', 'device_id': 'DV0', 'class_code': 'C'}]), table('equipment', [{'id': f'DV{i}', 'class_code': 'C', 'capacity': i + 1} for i in range(5)]), table('maintenance', [{'id': f'MN{i}', 'device_id': f'DV{i}', 'scheduled': i == 0} for i in range(5)])],
        {'assignment_id': 'AS1', 'device_choice': 'DV3', 'approve_consent': True}, 'replace', observations={'assignments': ['AS1'], 'equipment': ['DV0', 'DV1', 'DV2', 'DV3', 'DV4'], 'maintenance': ['MN0', 'MN3']})
    add('complex', '跨仓调拨', '帮我从源仓给目标仓调拨12件这个SKU，先核实库存并让我确认。调完更新这张调拨单，再告诉我两边库存和单据状态。',
        '调拨单关联source_id和target_id。两个库存记录必须同SKU，源stock>=12。授权后一笔事务源stock-12、目标stock+12、调拨单completed=true；库存守恒且非负，失败或拒绝不写入，提交后核验。',
        [table('transfers', [{'id': 'TR1', 'source_id': 'ST1', 'target_id': 'ST2', 'completed': False}]), table('stocks', [{'id': 'ST1', 'sku': 'K8', 'stock': 30}, {'id': 'ST2', 'sku': 'K8', 'stock': 5}, {'id': 'ST3', 'sku': 'K9', 'stock': 50}])],
        {'transfer_id': 'TR1', 'approve_consent': True}, 'transfer', observations={'transfers': ['TR1'], 'stocks': ['ST1', 'ST2']})
    add('complex', '开放后登记活动', '帮我盯一下这个活动，开放后重新确认有名额，再问我是否登记。确认后办好，告诉我有没有成功、还剩几个名额。',
        '活动关联登记记录。虚拟elapsed_seconds>=15才开放，开放前查询不显示可登记名额；每动作0秒，等待显式推进。开放后重新查，用户授权后原子activity.remaining-1、registration.enrolled=true；名额不足或拒绝不写入。基准2026-10-10T00:00:00+00:00。',
        [table('activities', [{'id': 'EV1', 'registration_id': 'RG1', 'remaining': 3}]), table('registrations', [{'id': 'RG1', 'activity_id': 'EV1', 'enrolled': False}])],
        {'activity_id': 'EV1', 'approve_consent': True}, 'enroll', observations={'activities': ['EV1'], 'registrations': ['RG1']})
    add('complex', '预订前更正区域', '帮我订个场地，先问我的区域偏好，再把可选场地给我选。提交前再跟我确认一次区域，如果我改了主意，就按新区域重新选，别沿用旧方案。',
        '区域更正使旧选择与授权失效。每区域六个候选，完整展示available=true的菜单。用户最终选中场地并授权后，一笔事务venue.available=false、booking.venue_id改为该场地。只影响最终场地，拒绝不写入，提交后核验。',
        [table('venues', [{'id': f'V{i}', 'region': 'east' if i <= 6 else 'west', 'available': True} for i in range(1, 13)]), table('bookings', [{'id': 'BK1', 'venue_id': None}])],
        {'region': 'east', 'region_new': 'west', 'venue_choice': 'V8', 'booking_id': 'BK1', 'approve_initial': True, 'approve_consent': True}, 'correction', observations={'venues': [f'V{i}' for i in range(1, 13)], 'bookings': ['BK1']})
    add('complex', '逐项开启服务', '帮我依次开启这两个服务，每开一个都先让我确认。第一个办完请重新查额度再办第二个；如果第二个额度不够，保留第一个的结果，别一起撤销。最后告诉我两个服务的状态和剩余额度。',
        '用户quota初始3，服务顺序F1再F2。每服务耗1单位，每笔分别原子更新用户quota-1、对应feature.active=true、对应既有log.completed=true。first和second是不同授权阶段，不可复用授权。第一笔后、第二笔前查询当前quota和功能状态。额度不足第二笔不写入但第一笔保留。',
        [table('accounts', [{'id': 'AC1', 'quota': 3}]), table('features', [{'id': 'F1', 'account_id': 'AC1', 'active': False, 'log_id': 'LG1'}, {'id': 'F2', 'account_id': 'AC1', 'active': False, 'log_id': 'LG2'}]), table('logs', [{'id': 'LG1', 'feature_id': 'F1', 'completed': False}, {'id': 'LG2', 'feature_id': 'F2', 'completed': False}])],
        {'account_id': 'AC1', 'approve_first': True, 'approve_second': True}, 'sequential', observations={'accounts': ['AC1'], 'features': ['F1', 'F2'], 'logs': ['LG1', 'LG2']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, case in enumerate(cases(), 1):
        folder = root / f'{i:02d}'
        folder.mkdir()
        for name in ('business', 'user', 'acceptance'):
            write(folder / f'{name}.json', case[name])
        rows.append({'id': i, 'stratum': case['stratum'], 'title': case['title'],
            'category': {'direct': 'direct_response', 'simple': 'simple_agentic'}.get(case['stratum'], 'multi_step_agentic'),
            'raw_request': case['raw_request'], 'business': f'{i:02d}/business.json',
            'user': f'{i:02d}/user.json', 'acceptance': f'{i:02d}/acceptance.json'})
    write(root / 'manifest.json', {'version': 1, 'scope': 'synthetic_natural_demand_end_to_end_pilot',
        'sampling': 'operator-authored; no production-user or double-blind population claim',
        'mix': {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}, 'samples': rows})
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.json'))}
    write(root / 'frozen.json', {'files': hashes})
    print(json.dumps({'samples': len(rows), 'output': str(root)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
