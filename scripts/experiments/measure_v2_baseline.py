"""Frozen 20-request V2 baseline; every planned sample remains in denominator."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from env_factory_v2.pipeline import implementation_identity


REQUESTS = [
    ('direct_response', 'direct', '税费计算', '公开给出商品净价为 240 元、税率为 0.075。回答税额和含税总价，分别为 18 和 258 元；无工具、无私有信息。反例包括把税率当作 7.5 或漏加税额。'),
    ('direct_response', 'direct', '日期与单位', '公开给出项目持续 9 周，每周 5 个工作日，每日 6 小时。回答总工作日和总工时，分别为 45 和 270；无工具。不得把自然日计入工作日。'),
    ('direct_response', 'direct', '集合与排序', '公开给出编号 [7,2,7,4,2,9]。去重后按升序回答编号 [2,4,7,9] 和数量 4；无工具。反例包含保留重复和错误顺序。'),
    ('direct_response', 'direct', '分段收费', '公开规则：前 3 公里固定 12 元，超过部分每公里 2 元，不足公里也按实际距离计费。距离 6.5 公里。回答超出公里数 3.5 和费用 19 元；无工具。验证分段边界逻辑。'),
    ('simple_agentic', 'simple', '预约查询', '用户私有事实是预约编号。先询问编号，再用唯一查询工具一次查出匹配预约的日期和状态。至少 6 个唯一编号，包含同名用户。回答来自实际记录；显示姓名不参与身份匹配。'),
    ('simple_agentic', 'simple', '商品筛选', '用户私有事实是所需尺码。先询问尺码，再一次调用唯一查询工具获取全部库存大于零且尺码匹配的商品，回答完整编号集合。至少 8 条数据，包含缺货、错尺码和不同商品显示名，页大小足够一次返回。'),
    ('simple_agentic', 'simple', '工单状态', '用户私有事实是工单编号。先询问编号，再一次调用唯一查询工具得到工单优先级和状态。至少 6 条数据；不得通过标题模糊匹配。变更实际状态必须改变正确答案，旧状态不得满分。'),
    ('simple_agentic', 'simple', '仓库库存', '用户私有事实是仓库编号。先询问编号，再一次调用唯一查询工具列出该仓库库存大于零的全部 SKU 及库存合计。至少 8 条数据、两个仓库，包含零库存。页大小足够一次返回；其他仓库库存不得计入。'),
    ('multi_step_agentic', 'ordinary', '订单与配送', '先获取订单，再用实际订单结果中的配送编号查询配送状态，至少两个工具并强制结果引用。两个订单即使收件人同名也不可串单。回答当前订单编号、配送状态和预计日期，包含缺失或错误依赖的失败轨迹。'),
    ('multi_step_agentic', 'ordinary', '候选选课', '先查询实际可选课程菜单，用户从已观察菜单中选择，再用真实选择查询课程时段。至少两个工具，后续参数必须绑定实际结果和用户选择。私有偏好不存在时采用 first_available，答案随实际选择变化，不能继续使用原偏好。'),
    ('multi_step_agentic', 'ordinary', '部门人员汇总', '先查询部门确定其真实编号，再引用该结果查询所有在职员工并回答完整员工编号集合和总人数。至少两个部门、有重名显示名、离职记录；部门名称不能替代实际编号绑定。'),
    ('multi_step_agentic', 'ordinary', '授权账户转账', '先读取源账户与目标账户余额，再引用实际读结果请求精确转账授权，用事务工具将 25 单位从源账户转到目标账户。至少两个工具。真实余额变化、守恒、不得透支均必须验证；拒绝时不写入也能正确完成。'),
    ('multi_step_agentic', 'complex', '长链路线规划', '建立至少 10 层连续节点依赖。先查起点，后续每步只能使用上一真实结果中的 next_id，至少两个工具且逐层实际引用，禁止凭静态编号跳步。最终回答路径顺序及成本合计。覆盖中间节点变化、错误引用及缺失一层；数据不能退化成两个节点。'),
    ('multi_step_agentic', 'complex', '长链审批追踪', '建立至少 8 层审批记录链。先查申请入口，再依据每层真实 next_id 逐层查询，最后查询末层所引用的负责人。至少两个工具，回答完整链和最终负责人；同名负责人不能作为身份约束。验证中间关系变更和跳层答案。'),
    ('multi_step_agentic', 'complex', '大规模双仓检索', '先查目标仓库真实编号，再检索 10000 条由种子计划生成的库存记录，至少两个工具。页大小不超过 100，目标仓库且库存大于零的合格记录必须跨至少 3 页。回答完整合格 SKU 集合、数量和库存总和，必须实际完整分页；改动非首尾记录应影响奖励目标。'),
    ('multi_step_agentic', 'complex', '大规模票据检索', '先获取项目真实编号，再在 6000 条种子生成票据中查询该项目金额大于 100 且状态为 approved 的全部票据，页大小不超过 64，结果至少 3 页。回答完整票据编号和总金额。不同项目及未批准票据不得计入；漏中间页不能满分。'),
    ('multi_step_agentic', 'complex', '跨表订单结算', '先读取订单、库存、账户，引用真实订单结果进行精确授权后一次原子事务更新至少两个表、至少 3 个对象：扣订单商品库存、扣买方余额、增加卖方余额，并将订单标记已结算。必须余额与库存不为负、资金守恒；任何一项失败全量回滚。验证拒绝、余额不足、库存不足及正确实际状态。'),
    ('multi_step_agentic', 'complex', '多账户批量分配', '先读取资金池与目标账户记录，引用真实结果请求对完整分配方案的精确授权。一次事务扣资金池 60、给三个不同账户分别增加 10、20、30，并更新独立分配记录表。至少两个表、5 个对象真实变化；守恒和全量回滚。拒绝后保持原状态，错误金额或错误账户方案不得沿用授权。'),
    ('multi_step_agentic', 'complex', '采购选择与付款', '先获取至少 6 个实际供应报价，用户选择报价，接着引用真实报价查询供应商付款账户，再请求精确授权并原子更新采购记录和双方余额。至少 3 个业务工具。私有偏好报价缺席时选实际首项；实付金额必须来自实际选择，拒绝分支不写入。至少两个表、3 个对象更新。'),
    ('multi_step_agentic', 'complex', '定时开放报名', '先获取活动真实编号，再通过引用查询名额。虚拟基准时间固定，报名开放在 elapsed_seconds=20；开放前查询无可报名名额，等待后重新观察再精确授权登记。事务必须更新活动剩余名额和报名记录两个表，禁止沿用开放前旧观察。回答实际报名状态和剩余名额；验证时间边界、未授权和名额不足回滚。'),
]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--sample-timeout', type=int, default=1200)
    parser.add_argument('--base-image', default='docker.m.daocloud.io/library/python:3.14-slim')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = implementation_identity()
    started = time.monotonic()
    samples = []
    for index, (category, stratum, title, goal) in enumerate(REQUESTS, 1):
        request = {'category': category, 'goal': goal, 'title': title,
                   'requirements': '用当前 V2 明确支持的能力实现上述完整业务，不可降低规模或省略义务。任务说明面向执行者，只披露公开信息；私有事实通过 User Simulator 获取。生成有效成功和失败轨迹及变更数据反例。'}
        path = root / f'request-{index:02d}.json'
        write(path, request)
        samples.append({'id': index, 'category': category, 'stratum': stratum, 'title': title,
                        'request': str(path), 'request_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'status': 'pending', 'qualified': False})
    protocol = {'version': 1, 'scope': 'fixed_requests_fresh_authoring_not_population_yield',
                'planned': 20, 'mix': {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8},
                'model': 'gpt-6.1-sol', 'concurrency': 1, 'repairs': 1, 'docker': True,
                'sample_timeout_seconds': args.sample_timeout, 'platform_identity': frozen,
                'bridge_timeout_seconds': json.loads(args.connection.read_text()).get('timeout_seconds', 180),
                'base_image': args.base_image, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'policy': 'No platform changes within round. All failures remain in planned denominator. No terminal sample reruns.'}
    write(root / 'protocol.json', protocol)
    def publish():
        terminal = sum(s['status'] in {'passed', 'failed'} for s in samples)
        good = sum(s['qualified'] for s in samples)
        groups = {}
        for name, size in protocol['mix'].items():
            selected = [s for s in samples if s['stratum'] == name]
            done = sum(s['status'] in {'passed', 'failed'} for s in selected)
            count = sum(s['qualified'] for s in selected)
            groups[name] = {'planned': size, 'terminal': done, 'qualified': count,
                            'yield': count / size if done == size else None}
        write(root / 'baseline.json', {'planned': 20, 'terminal': terminal, 'qualified': good,
              'end_to_end_yield': good / 20 if terminal == 20 else None, 'groups': groups,
              'wall_seconds': time.monotonic() - started, 'samples': samples})
    publish()
    for sample in samples:
        if implementation_identity() != frozen:
            write(root / 'blocked.json', {'reason': 'platform_changed', 'sample': sample['id']})
            raise RuntimeError('Platform changed; do not mix this round with another version')
        sample.update(status='running')
        publish()
        output = root / f'sample-{sample["id"]:02d}'
        command = [sys.executable, '-m', 'env_factory_v2', 'generate', '--request', sample['request'],
                   '--connection', str(args.connection.resolve()), '--output', str(output),
                   '--repairs', '1', '--base-image', args.base_image]
        tick = time.monotonic()
        with (root / f'sample-{sample["id"]:02d}.log').open('w') as stream:
            try:
                result = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=args.sample_timeout)
                sample['exit_code'] = result.returncode
            except subprocess.TimeoutExpired:
                sample.update(exit_code=None, failure_class='timeout')
        sample['seconds'] = time.monotonic() - tick
        if (output / 'progress.json').exists():
            progress = json.loads((output / 'progress.json').read_text())
            sample['qualified'] = progress.get('qualified') is True and sample.get('exit_code') == 0
            sample['attempts'] = progress.get('attempts', [])
            sample['progress'] = str(output / 'progress.json')
            if not sample['qualified']:
                last = sample['attempts'][-1] if sample['attempts'] else {}
                sample.setdefault('failure_class', last.get('status', progress.get('status', 'process_failed')))
                sample['failed_phase'] = last.get('failed_phase')
                sample['error'] = last.get('error')
        sample['status'] = 'passed' if sample['qualified'] else 'failed'
        publish()
        print(json.dumps({k: sample.get(k) for k in ('id', 'title', 'status', 'seconds', 'failed_phase', 'error')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
