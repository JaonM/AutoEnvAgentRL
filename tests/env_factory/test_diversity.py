from env_factory.diversity import features, observed_obstacles, repetition


def example(uid, name, amount, count=1, reason=None):
    req={'tables':{'users':[{'user_id':uid,'first_name':name}]},
         'user_facts':{'goals':[{'process':'change','params':{'user_id':uid,'amount':amount,'count':count}}]},
         'user_scene':{'goal':f'{name} wants a change for ${amount}.'}}
    ref={'reference_answer':{'goal_1':{'status':'completed' if reason is None else 'refused','reason':reason}},
         'reference_trace':[{'kind':'tool','tool':'update','args':{'user_id':uid,'amount':amount,'count':count},'status':'ok'}]}
    return features(req,ref,{'messages':[]})


def test_names_ids_amounts_do_not_create_novelty():
    a,b=example('U123','Alice',20),example('V789','Bob',99)
    assert a['task_key']==b['task_key']
    assert a['task_text']==b['task_text']


def test_policy_count_and_refusal_remain_distinct():
    a=example('U1','Alice',10)
    assert a['task_key']!=example('U1','Alice',10,count=2)['task_key']
    assert a['task_key']!=example('U1','Alice',10,reason='limit_exceeded')['task_key']


def test_wrong_first_requires_ordered_actual_witness():
    req={'user_facts':{'record_id':'RIGHT'},'user_script':{'wrong_first':{'record_id':'WRONG'}}}
    assert observed_obstacles(req,{'messages':[{'role':'user','content':'RIGHT'}]})=={}
    e={'messages':[{'role':'user','content':'WRONG'},{'role':'assistant','content':'Please check'},
                   {'role':'user','content':'RIGHT'}]}
    assert observed_obstacles(req,e)['wrong_first']==[0,2]


def test_rate_is_excess_not_all_cluster_members():
    r=repetition([{'id':'a','k':1},{'id':'b','k':1},{'id':'c','k':2}],lambda r:r['k'])
    assert r['excess']==1 and r['in_repeated_clusters']==2


def test_identity_relationships_are_preserved():
    req={'tables':{'rows':[{'record_id':'A'},{'record_id':'B'}]},'user_facts':{'goals':[]}}
    from env_factory.diversity import Normalizer
    n=Normalizer(req)
    assert n.value(['A','A'])!=n.value(['A','B'])


def test_policy_context_keeps_eligibility_boundaries():
    from env_factory.diversity_context import signature
    from env_factory.diversity import Normalizer,digest
    req={'time':{'start':'2026-01-02T12:00:00'}}
    config={'lookup':[{'path':'created_at','transform':'age_hours_bucket','threshold':24}]}
    def key(when):
        ref={'reference_trace':[{'kind':'tool','tool':'lookup','result':{'created_at':when}}]}
        return signature(req,ref,{'task':{}},Normalizer(req),config,digest)[0]
    assert key('2026-01-01T12:00:00')!=key('2026-01-01T11:59:59')
    assert key('2026-01-02T10:00:00')==key('2026-01-02T11:00:00')


def test_wrong_id_obstacle_cannot_depend_only_on_optional_question():
    from env_factory.diversity import interaction_preconditions
    r={'user_facts':{'order_id':'RIGHT'},'user_script':{'wrong_first':{'order_id':'WRONG'}},
       'user_request':'Please change my order.'}
    assert interaction_preconditions(r,'wrong_first')
    r['user_request']='Please change order WRONG.'
    assert not interaction_preconditions(r,'wrong_first')
    r['user_request']='Please change order WRONG, actually RIGHT.'
    assert interaction_preconditions(r,'wrong_first')


def test_flight_numbers_and_fares_are_masked_but_counts_are_not():
    from env_factory.diversity import Normalizer
    a=Normalizer({'user_facts':{'flight_number':'AV123'}})
    b=Normalizer({'user_facts':{'flight_number':'ZX789'}})
    assert a.value({'flight_number':'AV123','fare':99,'passenger_count':2})==b.value({'flight_number':'ZX789','fare':150,'passenger_count':2})
    assert a.value({'passenger_count':2})!=b.value({'passenger_count':3})


def test_decline_witness_handles_typographic_apostrophes_and_no_comma():
    req={'user_facts':{'approve_update_record':False}}
    for text in ['No, please keep it unchanged.', 'Please don’t add that.', "Don't proceed."]:
        assert 'declined_approval' in observed_obstacles(req,{'messages':[{'role':'user','content':text}]})
    assert 'declined_approval' not in observed_obstacles(req,{'messages':[{'role':'user','content':'No problem, I approve.'}]})


def test_failed_candidate_releases_its_reservation_for_retry():
    from env_factory.diversity import AdmissionLedger
    ledger=AdmissionLedger([],{'train':3,'dev':1})
    assert ledger.reserve('first','dev','key') is None
    assert ledger.reserve('parallel','dev','key')=='structure_cap'
    ledger.finish('first',accepted=False)
    assert ledger.reserve('retry','dev','key') is None
    ledger.finish('retry',accepted=True)
    assert ledger.reserve('excess','dev','key')=='structure_cap'


def test_successful_train_cap_is_three_and_cross_split_never_leaks():
    from env_factory.diversity import AdmissionLedger
    ledger=AdmissionLedger([],{'train':3,'dev':1})
    assert ledger.reserve('t0','train','key') is None
    assert ledger.reserve('d0','dev','key')=='cross_split_in_flight'
    ledger.finish('t0',accepted=False)
    for i in range(3):
        assert ledger.reserve(i,'train','key') is None
        ledger.finish(i,accepted=True)
    assert ledger.reserve('t4','train','key')=='structure_cap'
    assert ledger.reserve('d1','dev','key')=='cross_split_structure'
