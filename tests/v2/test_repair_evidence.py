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
    bench = AuthorWorkbench(source=direct(), context=context, max_operations=512)
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
    feedback['review'] = {'consistent': False, 'reason': 'incorrect business result'}
    feedback['duplicate'] = feedback['review']
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
    assert not call('submit_contract')['passed']
    issue = context['repair']['issues'][0]['id']
    assert call('address_repair', {'issue_id': issue, 'paths': ['/task'], 'explanation': 'Proposed correction; independent validation still required.'})['passed']
    assert call('check_contract')['passed']
    assert call('submit_contract')['submitted']
    assert checked_source(tmp_path)[0]['task'] == 'Reworded task'


@pytest.mark.parametrize("factory", [direct, simple, transaction])
def test_edits_invalidate_fix_claims(factory):
    context = repair_context({}, {'error': 'independent rejection'}, direct())
    bench = AuthorWorkbench(source=factory(), context=context)
    issue = context['repair']['issues'][0]['id']
    claim = {'issue_id': issue, 'paths': ['/task'], 'explanation': 'Proposed correction awaiting independent review.'}
    assert not bench.call('address_repair', claim)['passed']
    bench.call('put_section', {'section': 'task', 'value_json': '"first change"'})
    assert bench.call('address_repair', claim)['passed']
    bench.call('put_section', {'section': 'task', 'value_json': '"second change"'})
    assert bench.call('check_contract', {})['passed']
    assert not bench.call('submit_contract', {})['passed']


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
