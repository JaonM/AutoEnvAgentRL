"""Split inputs remain portable, immutable and semantically identical to inline worlds."""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

from env_factory.core import spec
from env_factory.certify import bundle
from env_factory.authoring import pipeline
from env_factory.__main__ import main

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import FakeRunner, library_request


def test_split_roundtrip_and_legacy_compatibility(tmp_path):
    request = library_request()
    request['behavior_contract'] = {'cases': [{'data': request['tables']}]}
    request['sop_contract'] = {'cases': [{'data': request['tables'], 'graph': {}}]}
    before = copy.deepcopy(request)
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps(request))
    folder = tmp_path / 'split'
    spec.materialize(request, folder)
    document = json.loads((folder / 'request.json').read_text())
    assert not set(spec.ASSET_NAMES).intersection(document)
    assert spec.read(folder / 'request.json') == spec.read(legacy) == before
    assert spec.load(folder / 'request.json') == spec.load(legacy)
    assert request == before
    moved = tmp_path / 'moved'
    shutil.move(str(folder), moved)
    assert spec.read(moved / 'request.json') == before
    preview = json.loads((moved / 'data_preview.json').read_text())
    assert preview['illustrative_only'] and preview['runtime_uses_full_data']


def test_named_requests_have_distinct_assets_and_no_extra_json_in_collection(tmp_path):
    one, two = library_request(), library_request()
    two['user_facts']['new_fact'] = 'different'
    spec.write(one, tmp_path / 'one.json')
    spec.write(two, tmp_path / 'two.json')
    assert sorted(p.name for p in tmp_path.glob('*.json')) == ['one.json', 'two.json']
    assert spec.read(tmp_path / 'one.json') == one
    assert spec.read(tmp_path / 'two.json') == two


@pytest.mark.parametrize('field', list(spec.ASSET_NAMES))
def test_modified_or_missing_asset_fails_closed(tmp_path, field):
    request = {**library_request(), 'sop_contract': {'cases': []}, 'behavior_contract': {'cases': []}}
    document = spec.write(request, tmp_path / 'request.json')
    if field not in request:
        request[field] = {}  # storage integrity is independent of business validation
        document = spec.write(request, tmp_path / 'request.json')
    asset = tmp_path / document['request_assets']['files'][field]['path']
    asset.write_text('{}')
    with pytest.raises(spec.RequestError, match='hash mismatch'):
        spec.read(tmp_path / 'request.json')
    asset.unlink()
    with pytest.raises(spec.RequestError, match='cannot read'):
        spec.read(tmp_path / 'request.json')


@pytest.mark.parametrize('name', ['../data.json', '/tmp/data.json', 'sub/../../data.json', './data.json', 'sub\\data.json'])
def test_external_paths_rejected(tmp_path, name):
    document = spec.write(library_request(), tmp_path / 'request.json')
    document['request_assets']['files']['tables']['path'] = name
    (tmp_path / 'request.json').write_text(json.dumps(document))
    with pytest.raises(spec.RequestError, match='unsafe'):
        spec.read(tmp_path / 'request.json')


def test_symlink_escape_and_ambiguous_sources_rejected(tmp_path):
    folder = tmp_path / 'split'
    document = spec.write(library_request(), folder / 'request.json')
    data = folder / 'data.json'
    data.rename(tmp_path / 'data.json')
    data.symlink_to(tmp_path / 'data.json')
    with pytest.raises(spec.RequestError, match='escapes'):
        spec.read(folder / 'request.json')
    document['tables'] = {}
    (folder / 'request.json').write_text(json.dumps(document))
    with pytest.raises(spec.RequestError, match='ambiguous'):
        spec.read(folder / 'request.json')


def test_split_bundle_full_build_serve_and_resign_preserves_assets(tmp_path):
    request = library_request()
    result = pipeline.build(request, tmp_path / 'build', runner=FakeRunner())
    assert result['training_ready'], result
    target = Path(result['bundle'])
    loaded = bundle.load_bundle(target)
    assert loaded['request']['tables'] == request['tables']
    assert {'data.json', 'facts.json'} <= loaded['manifest']['files'].keys()
    assert bundle.requalify(target)['passed']
    assert {'data.json', 'facts.json'} <= bundle.load_bundle(target)['manifest']['files'].keys()
    # Even rehashing an asset reference cannot remove the bundle's commitment to that asset.
    manifest = json.loads((target / 'manifest.json').read_text())
    del manifest['files']['data.json']
    (target / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='omits required input'):
        bundle.load_bundle(target)


def test_migration_cli_is_lossless_and_refuses_existing_destination(tmp_path):
    source = tmp_path / 'legacy.json'
    source.write_text(json.dumps(library_request()))
    original = source.read_bytes()
    args = ['split-request', '--request', str(source), '--output', str(tmp_path / 'new')]
    assert main(args) == 0
    assert spec.read(tmp_path / 'new/request.json') == spec.read(source)
    assert source.read_bytes() == original
    with pytest.raises(SystemExit):
        main(args)
