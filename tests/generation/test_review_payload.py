import json
import pytest
from env_factory.generation.review_payload import encode_shared_values, decode_shared_values, review_payload


def test_repeated_expression_roundtrip_and_size():
    value={'lookup':{'table':'stock','field':'quantity','where':{'id':'public target'}}}
    for _ in range(12):value={'if':{'condition':True,'then':value,'else':value}}
    bundle={'original_goal':'Return the observed facts.','source':value,'compiled_reward_rules':[value]}
    text, encoding=review_payload(bundle,inline_limit=10000)
    assert encoding=='lossless_json_dag_v1'
    assert len(text)<10000
    assert decode_shared_values(json.loads(text))==bundle


def test_existing_reference_like_fields_are_preserved():
    original={'__review_ref__':'business data','values':[{'$ref':'capture'},None,True,1,1.0,'中文']}
    assert decode_shared_values(encode_shared_values(original,minimum_size=1))==original


def test_small_payload_remains_plain_json():
    source={'source':{'business_requirements':[{'id':'all_rows','statement':'Return every matching row.'}]}}
    text,encoding=review_payload(source)
    assert encoding=='json'
    assert json.loads(text)==source
    assert text == json.dumps(source, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def test_unique_oversized_content_is_never_dropped():
    original={'source':{'text':'x'*10000},'negative':{'answer':'wrong'}}
    text,_=review_payload(original,inline_limit=100)
    assert len(text)>100
    assert decode_shared_values(json.loads(text))==original


@pytest.mark.parametrize('definitions', [{}, {'v1':{'r':'v1'}}])
def test_invalid_reference_is_rejected(definitions):
    with pytest.raises(ValueError,match='Missing or cyclic'):
        decode_shared_values({'encoding':'lossless_json_dag_v1','reference_key':'r','root':{'r':'v1'},'shared_values':definitions})


@pytest.mark.parametrize('mode', ['dag', 'file', 'changed'])
def test_review_dispatch_preserves_complete_candidate(tmp_path, monkeypatch, mode):
    from unittest.mock import Mock
    from env_factory.generation import semantic_review as review
    original = {'text': 'unique text' * 100_000} if mode == 'file' else {'rows': [{'rule': 'full original business obligation' * 10}] * 4000}
    captured = {}
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            captured['prompt'] = prompt
            candidate = json.loads((tmp_path/'candidate.json').read_text())
            assert candidate['source'] == original
            encoded = json.loads((tmp_path/'review_input.json').read_text())
            assert decode_shared_values(encoded) == candidate
            if mode == 'changed':
                (tmp_path/'review_input.json').write_text('{}')
            report = {'status':'pass','checks':{name:{'passed':True,'evidence':'Concrete reviewed source.'} for name in review.CHECKS},'findings':[]}
            (tmp_path/'response.json').write_text(json.dumps(report))
            kwargs['stdout'].write('{"type":"turn.completed","usage":{}}\n')
        return Mock(returncode=0,communicate=Mock(side_effect=finish))
    monkeypatch.setattr(review.subprocess, 'Popen', launch)
    args=dict(source=original, artifacts={'public_input':{},'tools':[],'metrics':[],'metric_implementations':[]},request={},root=tmp_path,model='test',timeout=60)
    if mode=='changed':
        with pytest.raises(review.SemanticReviewUnavailable,match='SOURCE_REVIEW_INPUT_CHANGED'):
            review.review_source(**args)
        return
    result=review.review_source(**args)
    assert result['provenance']['input_encoding']==('file_backed_json' if mode=='file' else 'lossless_json_dag_v1')
    assert len(captured['prompt'])<1_048_576
    if mode=='file':
        assert 'Read that file' in captured['prompt']
        assert 'no filesystem read is required' not in captured['prompt']
