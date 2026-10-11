import copy
import pytest
from env_factory_v2.examples import transaction,lit,ref
from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.contracts.transaction_targets import project_transaction


def fixture(count=576):
    s=transaction();s['invariants']=[]
    s['tables'][0]['rows'] += [{'id':f'X{i}', 'balance':i} for i in range(count-2)]
    s['tools'][1]['operations']=[{'table':'accounts','for_each':ref('data.accounts'),
        'key':ref('item.id'),'set':{'balance':{'add':[ref('row.balance'),lit(1)]}}}]
    # Preserve argument dependencies and consent; this fixture isolates iteration.
    s['rules'][0]['predicate']={'and':[{'eq':[ref('args.amount'),lit(20)]},
        {'eq':[ref('args.from_id'),lit('A')]},{'eq':[ref('args.to_id'),lit('B')]}]}
    return s


def ready(s):
    w=Sandbox(s);args={'from_id':'A','to_id':'B','amount':20}
    w.interact('approve',question='Confirm?',proposal={'tool':'transfer','arguments':args})
    return w,args


def test_large_collection_atomic_commit_and_projection_agree():
    w,args=ready(fixture());before=copy.deepcopy(w.data)
    projected,eligible=project_transaction(w.tools['transfer'],w.source,before,w.context(args=args),check_eligibility=True)
    assert eligible
    w._transaction(w.tools['transfer'],args)
    assert w.data==projected
    assert len(w.data['accounts'])==576
    assert [r['balance'] for r in w.data['accounts']]==[r['balance']+1 for r in before['accounts']]
    assert w.user.states['approve']=='consent.consumed'
    with pytest.raises(ContractError):w._transaction(w.tools['transfer'],args)
    assert w.data==projected


@pytest.mark.parametrize('items,error',[
    (['A','missing'],'exactly one'),(['A','A'],'duplicate target'),
    (list(range(4097)),'expanded operation limit'),([], 'no applicable operations')])
def test_collection_failure_preserves_whole_world_and_consent(items,error):
    s=fixture(2);s['tools'][1]['operations'][0].update(for_each=lit(items),key=ref('item'))
    w,args=ready(s);before=copy.deepcopy(w.data);user=copy.deepcopy(vars(w.user))
    with pytest.raises(ContractError,match=error):w._transaction(w.tools['transfer'],args)
    assert w.data==before and vars(w.user)==user


def test_collection_requires_real_authorization():
    w=Sandbox(fixture(2));before=copy.deepcopy(w.data)
    with pytest.raises(ContractError,match='authorization'):
        w._transaction(w.tools['transfer'],{'from_id':'A','to_id':'B','amount':20})
    assert w.data==before


def test_item_binding_is_scoped_to_collection_operation():
    s=fixture(2);del s['tools'][1]['operations'][0]['for_each']
    with pytest.raises(ContractError,match='forbidden expression reference'):compile_contract(s)


def test_collection_and_second_table_commit_as_one_atomic_unit():
    s=fixture(70)
    ledger=copy.deepcopy(s['tables'][0]);ledger['id']='ledger';ledger['rows']=[{'id':'L','balance':0}]
    s['tables'].append(ledger)
    s['tools'][1]['operations'].append({'table':'ledger','key':lit('L'),'set':{'balance':lit(70)}})
    w,args=ready(s);w._transaction(w.tools['transfer'],args)
    assert w.data['ledger'][0]['balance']==70
    assert len(w.data['accounts'])==70
    s['tools'][1]['operations'][-1]['key']=lit('absent')
    w,args=ready(s);before=copy.deepcopy(w.data)
    with pytest.raises(ContractError):w._transaction(w.tools['transfer'],args)
    assert w.data==before and w.user.states['approve']=='consent.authorized'
