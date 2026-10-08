import copy
import pytest
from env_factory.generation.review_scope import validate_review_scope
from env_factory.generation.semantic_review import SemanticReviewUnavailable


def bundle():
    return {'compiled_public_input':{'initial_user_message':'Ask attendance and priority.'},
            'source':{'interaction_contract':{'variants':[{'stages':[
                {'id':'attendance','private_fact':80},{'id':'priority','private_fact':'斗灯'}]}]},
                      'tables':[{'table_name':'layouts','columns':[{'name':'capacity','type':'integer','nullable':False}],'rows':[{'capacity':100}]}]}}


def example():
    return {'basis':'declared_script','script_assumptions':[{'variant_index':0,'stage_id':'attendance','value_json':'80'}],
            'data_mutations':[],'public_paths':[]}


def check(value,context=None):
    validate_review_scope({'findings':[{'counterexample':value}]},context or bundle())


def test_real_script_assumption_is_valid_but_invented_disclosure_is_not():
    check(example())
    value=example();value['script_assumptions'][0]['value_json']='50'
    with pytest.raises(SemanticReviewUnavailable,match='not implemented'):check(value)


def test_stock_counterfactual_remains_supported_without_changing_user_inputs():
    value=example();value['basis']='business_data'
    value['data_mutations']=[{'table':'layouts','row_index':0,'column':'capacity','value_json':'60'}]
    check(value)
    value['script_assumptions'][0]['value_json']='50'
    with pytest.raises(SemanticReviewUnavailable,match='not implemented'):check(value)


def test_business_counterexample_must_actually_change_data():
    value=example();value['basis']='business_data'
    value['data_mutations']=[{'table':'layouts','row_index':0,'column':'capacity','value_json':'100'}]
    with pytest.raises(SemanticReviewUnavailable,match='every row unchanged'):check(value)
    value['data_mutations'][0]['value_json']='99'
    check(value)


def test_public_conflict_requires_resolving_public_evidence():
    value=example();value.update(basis='public_contract',script_assumptions=[],public_paths=['compiled_public_input.initial_user_message'])
    check(value)
    value['public_paths']=['source.private_gold']
    with pytest.raises(SemanticReviewUnavailable,match='public evidence'):check(value)


def test_original_public_goal_is_valid_evidence_but_not_private_goal_fields():
    context = bundle()
    context['original_goal'] = {'public_input': {'materials': [{'text': 'Stock must be positive.'}]},
                                'private_answer': 'hidden'}
    value = example()
    value.update(basis='public_contract', script_assumptions=[],
                 public_paths=['original_goal.public_input.materials[0].text'])
    check(value, context)
    value['public_paths'] = ['original_goal.private_answer']
    with pytest.raises(SemanticReviewUnavailable, match='public evidence'):
        check(value, context)


def test_invalid_public_path_reports_exact_path_and_actual_parent_keys():
    context = bundle()
    context['compiled_public_input']['materials'] = [{'text': 'The real field.'}]
    value = example()
    value['public_paths'] = ['compiled_public_input.materials[0].content']
    with pytest.raises(SemanticReviewUnavailable) as error:
        check(value, context)
    message = str(error.value)
    assert 'compiled_public_input.materials[0].content' in message
    assert 'text' in message
    assert 'The real field.' not in message  # Diagnose shape, without copying arbitrary content.


@pytest.mark.parametrize('path', ['compiled_public_input.materials[*].content',
                                'compiled_public_input.materials[bogus]'])
def test_malformed_or_unresolved_wildcard_path_still_rejects(path):
    value = example()
    value['public_paths'] = [path]
    with pytest.raises(SemanticReviewUnavailable, match='public evidence path does not resolve'):
        check(value)


@pytest.mark.parametrize('change',['no_mutation','script_as_data','bad_column','nan','mixed_episodes'])
def test_unsupported_counterexamples_are_report_errors_not_author_defects(change):
    context=bundle();value=example();value['basis']='business_data'
    value['data_mutations']=[{'table':'layouts','row_index':0,'column':'capacity','value_json':'60'}]
    if change=='no_mutation':value['data_mutations']=[]
    if change=='script_as_data':value['data_mutations'][0]['table']='user_scripts'
    if change=='bad_column':value['data_mutations'][0]['column']='attendance'
    if change=='nan':value['data_mutations'][0]['value_json']='NaN'
    if change=='mixed_episodes':
        context['source']['interaction_contract']['variants']*=2
        value['script_assumptions'].append({'variant_index':1,'stage_id':'priority','value_json':'"斗灯"'})
    with pytest.raises(SemanticReviewUnavailable,match='SOURCE_REVIEW_INVALID:'):check(value,context)


@pytest.mark.parametrize('bad_value', ['"large"', 'null', '-1'])
def test_mutation_must_obey_runtime_types_nullability_and_checks(bad_value):
    context=bundle()
    context['source']['tables'][0]['constraints']=['capacity >= 0']
    value=example();value['basis']='business_data'
    value['data_mutations']=[{'table':'layouts','row_index':0,'column':'capacity','value_json':bad_value}]
    with pytest.raises(SemanticReviewUnavailable,match='runtime data constraints'):
        check(value,context)


def test_mutation_cannot_break_foreign_key_or_mutate_candidate():
    context=bundle();before=copy.deepcopy(context)
    context['source']['tables'].append({'table_name':'bookings','columns':[{'name':'capacity','type':'integer'}],
        'rows':[{'capacity':100}], 'foreign_keys':[{'column':'capacity','ref_table':'layouts','ref_column':'capacity'}]})
    before=copy.deepcopy(context)
    value=example();value['basis']='business_data'
    value['data_mutations']=[{'table':'layouts','row_index':0,'column':'capacity','value_json':'60'}]
    with pytest.raises(SemanticReviewUnavailable,match='foreign key'):
        check(value,context)
    assert context==before
    value['data_mutations'].append({'table':'bookings','row_index':0,'column':'capacity','value_json':'60'})
    check(value,context)
    assert context==before
