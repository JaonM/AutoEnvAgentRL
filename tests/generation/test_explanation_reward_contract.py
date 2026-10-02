import pytest
from env_factory.generation.semantic_reward import validate_explanation_rewards


def contract(description='结合菜单事实说明推荐依据', key='reason', **schema):
    return {'answer_contract': {'schema': {'properties': {key: {'type':'string', 'description':description, **schema}}}},
            'metric_implementations': [{'source':'final_agent_response', 'operator':'value_targets',
                                       'expected': {'targets':[{'key':key, 'expression':{'from_tool':{'field':'menu_note'}}}]}}]}


def test_open_reason_cannot_be_exactly_matched_to_menu_note():
    with pytest.raises(ValueError, match='OPEN_EXPLANATION_EXACT_MATCH'):
        validate_explanation_rewards(contract())


@pytest.mark.parametrize('source', [contract('原样引用菜单中的推荐理由'),
                                   contract(enum=['price', 'quality']),
                                   contract('菜品名称', key='dish_name'),
                                   contract('记录的原因代码', key='reason_code', enum=['A', 'B'])])
def test_explicit_quotes_enums_and_entity_names_remain_valid(source):
    validate_explanation_rewards(source)


def test_failure_taxonomy_does_not_parse_words_inside_review_findings():
    from examples.generate_task import _generation_failure_class
    from env_factory.generation.task_generator import TaskGenerationError
    error = TaskGenerationError('SOURCE_SEMANTIC_REVIEW_FAILED: {"source_paths":[], "reason":"schema requires meaningful explanation"}')
    assert _generation_failure_class(error) == 'GEN_SEMANTIC'
