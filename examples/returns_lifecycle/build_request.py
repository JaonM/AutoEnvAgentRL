"""Rebuild the checked-in split request: python examples/returns_lifecycle/build_request.py."""
from pathlib import Path
import copy
from env_factory.core import spec


def path(name):
    return {'path': name}


def op(name, *args):
    return {'op': name, 'args': list(args)}


def request():
    transitions = {}
    def transition(name, actor, source, target, *, emit=(), guard=True, effects=()):
        transitions[name] = {'machine': 'return', 'actor': actor, 'from': source, 'to': target,
                             'guard': guard, 'effects': list(effects), 'emit': list(emit)}
    def event(name, delay, when=True):
        return {'transition': name, 'entity': path('entity'), 'delay': delay, 'when': when}
    transition('apply', 'agent', ['none'], 'requested', guard=op('gt', path('row.amount'), 0),
               emit=[event('handover', path('params.ship_delay'), path('params.will_ship'))])
    transition('handover', 'customer', ['requested'], 'in_transit',
               emit=[event('receive', 20)])
    transition('receive', 'carrier', ['in_transit'], 'received',
               emit=[event('accept', 10, path('params.inspection_pass')),
                     event('reject', 10, op('not', path('params.inspection_pass')))])
    transition('accept', 'warehouse', ['received'], 'accepted')
    transition('reject', 'warehouse', ['received'], 'rejected')
    transition('refund', 'agent', ['accepted'], 'refund_pending',
               emit=[event('settle', 10, path('params.payment_success')),
                     event('payment_failure', 10, op('not', path('params.payment_success')))])
    transition('settle', 'payment', ['refund_pending'], 'refunded',
               effects=[{'table': 'returns', 'entity': path('entity'), 'set': {'refund': path('row.amount')}}])
    transition('payment_failure', 'payment', ['refund_pending'], 'refund_failed')
    transition('cancel', 'agent', ['requested'], 'cancelled')
    schema = {'version': 1, 'machines': {'return': {'table': 'returns', 'field': 'status', 'initial': 'none',
              'states': ['none', 'requested', 'in_transit', 'received', 'accepted', 'rejected',
                         'refund_pending', 'refunded', 'refund_failed', 'cancelled']}},
              'transitions': transitions, 'scopes': ['request_only', 'full'],
              'parameters': {'will_ship': {'enum': [True, False]}, 'ship_delay': {'min': 0, 'max': 30},
                             'inspection_pass': {'enum': [True, False]}, 'payment_success': {'enum': [True, False]}},
              'tools': {name: {'transition': action, 'entity_arg': 'return_id', 'key_arg': 'request_id'}
                        for name, action in [('request_return', 'apply'), ('request_refund', 'refund'), ('cancel_return', 'cancel')]},
              'invariants': [{'name': 'positive purchased amount', 'table': 'returns', 'assert': op('gt', path('row.amount'), 0)},
                             {'name': 'refund bounded by paid amount', 'table': 'returns',
                              'assert': op('all', op('ge', path('row.refund'), 0), op('le', path('row.refund'), path('row.amount')))},
                             {'name': 'money only after settlement', 'table': 'returns',
                              'assert': op('any', op('eq', path('row.refund'), 0), op('eq', path('row.status'), 'refunded'))}]}
    scenario = {'version': 1, 'scope': 'full', 'parameters': {'will_ship': True, 'ship_delay': 10,
                  'inspection_pass': True, 'payment_success': True}, 'history': []}
    data = {'returns': [{'id': 'R1', 'status': 'none', 'amount': 1000, 'refund': 0},
                        {'id': 'R2', 'status': 'none', 'amount': 2000, 'refund': 0}]}
    def check(status, refund=0):
        return [{'name': 'independent expected status', 'assert': op('eq', path('after.returns.0.status'), status)},
                {'name': 'independent expected refund', 'assert': op('eq', path('after.returns.0.refund'), refund)},
                {'name': 'other customer unchanged', 'assert': op('eq', path('after.returns.1'), path('initial.returns.1'))}]
    def call(tool, status=None, key='K1', error=None, refund=0):
        item = {'tool': tool, 'args': {'return_id': 'R1', 'request_id': key}}
        if error: item['error'] = error
        else: item['checks'] = check(status, refund)
        return item
    def wait(duration, status, refund=0):
        return {'wait': duration, 'checks': check(status, refund)}
    cases = [
        {'id': 'causal_success_and_retry', 'transitions': ['apply','handover','receive','accept','refund','settle'],
         'steps': [call('request_refund', error='precondition'), call('request_return', 'requested'),
                   call('request_return', 'requested'), wait(9, 'requested'), wait(1, 'in_transit'),
                   wait(20, 'received'), call('request_refund', key='too-early', error='precondition'),
                   wait(10, 'accepted'), call('request_refund', 'refund_pending', key='pay'), wait(10, 'refunded', 1000)]},
        {'id': 'cancel_discards_handover', 'transitions': ['apply','cancel'],
         'steps': [call('request_return', 'requested'), call('cancel_return','cancelled',key='cancel'), wait(100,'cancelled')]},
        {'id': 'unshipped_never_arrives', 'transitions': ['apply'], 'parameters': {'will_ship': False},
         'steps': [call('request_return','requested'), wait(100,'requested'), call('request_refund',key='pay',error='precondition')]},
        {'id': 'inspection_rejected', 'transitions': ['apply','handover','receive','reject'], 'parameters': {'inspection_pass': False},
         'steps': [call('request_return','requested'), wait(40,'rejected'), call('request_refund',key='pay',error='precondition')]},
        {'id': 'payment_failed', 'transitions': ['apply','handover','receive','accept','refund','payment_failure'],
         'parameters': {'payment_success': False}, 'steps': [call('request_return','requested'), wait(40,'accepted'),
                     call('request_refund','refund_pending',key='pay'),wait(10,'refund_failed')]},
    ]
    for case in cases:
        case['data'],case['now'],case['scenario'] = copy.deepcopy(data),0,copy.deepcopy(scenario)
        case['scenario']['parameters'].update(case.pop('parameters',{}))
    return {'id':'returns-lifecycle-v1','request_kind':'task','family':'retail_returns',
            'category':'multi_step_agentic','title':'退货物流与退款闭环','language':'zh',
            'user_request':'我想退 R1 对应的商品，帮我申请退货，并跟进到退款结果。',
            'business_rules':'本示例入口已完成身份认证，两笔单据属于同一客户。金额单位为分，原始商品金额必须为正。先获用户同意提交申请；未实际交运不得签收，'
                '物流签收后仓库验收，通过后才能请求退款。申请、取消和退款需用户同意。只允许交运前取消。'
                '仓库拒收不退款；支付失败不入账。申请的商品金额必须大于零。交运后20秒物流签收，'
                '再10秒完成验收，退款请求后10秒支付完成或失败。外部动作按情景发生，聊天声称不构成交运。'
                '最多跟进100秒；仍未交运时告知等待客户寄回。request_only只办理申请，full跟进退款或拒收/失败。'
                '退款到账金额等于商品金额；其它客户单据不能改变。相同幂等键和参数的重试返回原回执，不重复执行。',
            'time':{'start':0,'tick':0},'tables':data,
            'user_facts':{'return_id':'R1','scope':'full','approve_return':True,'approve_refund':True},
            'lifecycle_schema':schema,'scenario':scenario,'lifecycle_contract':{'version':1,'cases':cases}}


if __name__ == '__main__':
    spec.write(request(), Path(__file__).parent / 'request.json')
