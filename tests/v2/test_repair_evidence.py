import json
import pytest
from env_factory_v2.authoring.repair_evidence import repair_context, prompt_context, PROMPT_LIMIT
from env_factory_v2.authoring.author_workbench import AuthorWorkbench
from env_factory_v2.authoring.author_mcp import AuthorMCP
from env_factory_v2.authoring.author_session import checked_source
from env_factory_v2.contracts.contract import ContractError
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.examples import direct, simple, transaction


def test_large_failed_evidence_is_bounded_in_prompt_and_losslessly_readable():
    feedback = {'records': [{'passed': False, 'case': {'id': 'x'}, 'trace': 'evidence' * 400000}]}
    context = repair_context({'goal': 'original'}, feedback, direct())
    assert len(canonical(prompt_context(context))) < PROMPT_LIMIT
    assert context['feedback'] == feedback
    bench = AuthorWorkbench(source=direct(), context=context)
    fragments, offset = [], 0
    while True:
        read = bench.call('read_context', {'path': '/feedback/records/0/trace', 'offset': offset})
        assert len(read['json_fragment']) <= 8192
        fragments.append(read['json_fragment'])
        offset = read['next_offset']
        if offset is None: break
    assert json.loads(''.join(fragments)) == feedback['records'][0]['trace']
    assert not bench.call('read_context', {'path': '/../../etc/passwd', 'offset': 0})['passed']


def test_resume_cannot_hide_prior_independent_findings():
    feedback = {'author_recovery': {'interrupted': True}, 'failed_phase': 'generation', 'error': 'timeout'}
    assert repair_context({}, feedback, direct())['repair']['mode'] == 'resume'
    feedback['requirement_review'] = {'consistent': False, 'reason': 'incorrect business result'}
    feedback['independent_review'] = feedback['requirement_review']
    context = repair_context({}, feedback, direct())
    assert context['repair']['mode'] == 'repair'
    assert len(context['repair']['issues']) == 1


def test_unchanged_repair_and_legacy_completion_are_rejected(tmp_path):
    context = repair_context({}, {'error': 'independent rejection'}, direct())
    (tmp_path/'context.json').write_text(canonical(context))
    server = AuthorMCP(tmp_path/'workbench', source=direct(), context=context)
    def call(name, args={}):
        return json.loads(server.dispatch({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}})['result']['content'][0]['text'])
    assert call('check_contract')['passed']
    assert not call('submit_contract')['passed']
    call('check_contract')
    with pytest.raises(ContractError, match='explicit submission'):
        checked_source(tmp_path)
    assert call('put_section', {'section': 'task', 'value_json': json.dumps('Reworded task')})['passed']
    assert call('check_contract')['passed']
    assert call('submit_contract')['submitted']
    assert checked_source(tmp_path)[0]['task'] == 'Reworded task'


@pytest.mark.parametrize("factory", [direct, simple, transaction])
def test_submission_does_not_claim_business_repair(factory):
    source = factory()
    context = repair_context({}, {'error': 'independent rejection'}, source)
    bench = AuthorWorkbench(source=source, context=context)
    bench.call('put_section', {'section': 'task', 'value_json': canonical(source['task'] + ' ')})
    assert bench.call('check_contract', {})['passed']
    result = bench.call('submit_contract', {})
    assert result['submitted']
    assert result['scope'] == 'authored_recipe_validation_not_business_acceptance'
    assert 'proposed_repairs' not in result
    from env_factory_v2.validation.verification import qualify, attach_independent_review
    compiled, execution = qualify(bench.snapshot())
    failed_review = {'scope': 'native_independent_contract_challenge',
                     'contract_identity': compiled['identity'], 'passed': False}
    assert not attach_independent_review(compiled, execution, failed_review)['passed']


def test_large_request_remains_available_without_inline_expansion():
    request = {'goal': 'x' * 100000}
    context = repair_context(request, None, None)
    assert prompt_context(context)['original_request']['read_context_path'] == '/request'
    assert context['request'] == request


def test_business_review_without_failed_execution_still_requires_repair():
    context = repair_context({}, {'business_consistent': False, 'reason': 'Omitted requirement', 'records': []}, direct())
    assert context['repair']['mode'] == 'repair'
    assert context['repair']['issues'][0]['reason'] == 'Omitted requirement'


def test_many_findings_use_index_pagination_without_rejection_or_loss():
    feedback = {'records': [{'consistent': False, 'reason': str(i)} for i in range(300)]}
    context = repair_context({}, feedback, direct())
    index = prompt_context(context)['repair']
    assert len(context['repair']['issues']) == index['issue_count'] == 300
    assert len(index['issues']) == 16


def test_superseded_review_and_execution_payload_are_not_active_findings():
    old = {'business_consistent': False, 'reason': 'withdrawn verdict'}
    review = {'passed': False, 'business_consistent': True, 'review_audit': {'original_review': old},
              'requirement_review': {'passed': False, 'reviews': [{'consistent': False, 'reason': 'current clause'}]},
              'records': [{'passed': True, 'case': {'id': 'fixed'}, 'first': {'consistent': False}}]}
    context = repair_context({}, {'independent_challenge': review, 'verification': {'independent_review': old}}, direct())
    assert [i['reason'] for i in context['repair']['issues']] == ['current clause']
    assert context['feedback']['independent_challenge']['review_audit']['original_review'] == old


@pytest.mark.parametrize('factory', [direct, simple, transaction])
def test_interrupted_repair_keeps_original_rejected_baseline(tmp_path, factory):
    from env_factory_v2.authoring.author_session import incomplete_draft
    source = factory()
    rejected = factory()
    rejected['task'] += ' incorrect extra requirement'
    feedback = {'error': 'incorrect extra requirement', 'failed_phase': 'independent_challenge'}
    original = repair_context({}, feedback, rejected)
    (tmp_path/'context.json').write_text(canonical(original))
    (tmp_path/'workbench').mkdir()
    (tmp_path/'workbench/draft.json').write_text(canonical(source))
    recovery = incomplete_draft(tmp_path, 'timeout').recovery
    draft = recovery.pop('previous_source')
    feedback.update(author_recovery=recovery, failed_phase='generation', error='timeout')
    resumed = repair_context({}, feedback, draft)
    assert resumed['repair'] == original['repair']
    (tmp_path/'context.json').write_text(canonical(resumed))
    server = AuthorMCP(tmp_path/'workbench', source=draft, context=resumed)
    for name in ['check_contract', 'submit_contract']:
        response = server.dispatch({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': {}}})
        assert not response['result']['isError']
    assert checked_source(tmp_path)[0] == source
    # A later rejection starts a fresh lineage, even if old recovery data remains.
    feedback['failed_phase'] = 'independent_challenge'
    assert repair_context({}, feedback, source)['repair']['initial_identity'] != original['repair']['initial_identity']


def test_paging_does_not_exhaust_edit_budget():
    bench = AuthorWorkbench(source=direct(), max_operations=2, max_reads=130)
    for _ in range(130):
        assert bench.call('read_contract', {'path': '/task', 'offset': 0})['passed']
    with pytest.raises(ContractError, match='read budget'):
        bench.call('read_contract', {'path': '/task', 'offset': 0})
    assert bench.call('check_contract', {})['passed']
    assert bench.call('submit_contract', {})['submitted']
