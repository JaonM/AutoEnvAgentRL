from env_factory.task_contract import approval_problems


def request(goals,facts):
    return {'goal_schema':{'lookup':{},'appointment':{'approval':'approve_visit_<patient_id>'}},
            'user_facts':{'goals':[{'process':g} for g in goals],**facts}}


def test_lookup_cannot_inherit_transaction_approval():
    assert approval_problems(request(['lookup'],{'approve_visit_pat1':True}))
    assert not approval_problems(request(['lookup'],{}))


def test_declared_write_can_be_approved_or_declined():
    for decision in (True,False):
        assert not approval_problems(request(['lookup','appointment'],{'approve_visit_pat1':decision}))


def test_templates_are_escaped_and_legacy_tasks_unchanged():
    assert approval_problems(request(['appointment'],{'approve_cancel_pat1':False}))
    assert not approval_problems({'user_facts':{'approve_visit_pat1':True}})
    assert not approval_problems({'goal_schema':{'legacy':{}},
        'user_facts':{'goals':[{'process':'legacy'}],'approve_old':True}})
