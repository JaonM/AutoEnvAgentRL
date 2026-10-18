import copy
import random

import pytest

from env_factory.production.world_data import allocate, author_context, integrity, remap


def library():
    ownership = {'owner_table': 'readers', 'owner_key': 'reader', 'tables': {'readers': 'reader', 'loans': 'reader'}}
    refs = [{'table': 'loans', 'path': 'books.*.book', 'target_table': 'books', 'target_path': 'book'}]
    return {'tables': {'readers': [{'id': f'u{i}', 'reader': f'u{i}'} for i in range(10)],
                       'loans': [{'id': f'l{i}', 'reader': f'u{i}', 'books': [{'book': f'b{i}'}]} for i in range(10)],
                       'books': [{'id': f'b{i}', 'book': f'b{i}', 'copies': {'site': i}} for i in range(20)]},
            'source': {'authoring_scope': {'ownership': ownership, 'references': refs,
                                           'owners_per_task': 2, 'catalog_rows': 1, 'max_bytes': 5000}}}


def test_budgets_are_exact_reproducible_and_nonuniform():
    a = allocate(1000, 500, maximum=9, rng=random.Random(9))
    assert sum(a) == 1000 and len(a) == 500 and len(set(a)) > 3
    assert a == allocate(1000, 500, maximum=9, rng=random.Random(9))
    with pytest.raises(ValueError, match='infeasible'):
        allocate(51, 5, maximum=10, rng=random.Random(1))


def test_remapping_preserves_nested_links_and_rejects_key_collision():
    source = {'id': 'u', 'wallets': {'p': {'id': 'p'}}, 'payments': [{'wallet': 'p'}], 'description': 'p is text'}
    target = remap(source, {'u': 'u2', 'p': 'p2'})
    assert target['wallets']['p2']['id'] == target['payments'][0]['wallet'] == 'p2'
    assert source['id'] == 'u' and target['description'] == 'p is text'
    with pytest.raises(ValueError, match='collapsed'):
        remap({'a': 1, 'b': 2}, {'a': 'b'})


def test_author_shard_keeps_full_referenced_rows_and_never_changes_runtime_world():
    request = library();before = copy.deepcopy(request)
    view = author_context(request, 3)
    assert request == before and len(view['tables']['readers']) == 2
    refs = request['source']['authoring_scope']['references']
    assert not integrity(view['tables'], {'references': refs})
    book_ids = {r['book'] for r in view['tables']['books']}
    for loan in view['tables']['loans']:
        assert loan['books'][0]['book'] in book_ids
    assert all('copies' in b for b in view['tables']['books'])
    assert len(request['tables']['readers']) == 10
    assert view == author_context(request, 3)
    assert view['author_view']['owners'] != author_context(request, 4)['author_view']['owners']


def test_integrity_rejects_cross_links_and_duplicate_nested_ids():
    request = library();tables = request['tables'];rules = {'references': request['source']['authoring_scope']['references']}
    tables['loans'][0]['books'][0]['book'] = 'missing'
    assert integrity(tables, rules)
    tables['books'][1]['book'] = tables['books'][0]['book']
    assert integrity(tables, {'unique': {'books': ['book']}})


def test_budget_cannot_silently_truncate_a_required_owner_graph():
    request = library();request['source']['authoring_scope']['max_bytes'] = 5
    with pytest.raises(ValueError, match='exceeds'):
        author_context(request, 0)


def test_legacy_worlds_keep_full_author_context():
    request = library();request.pop('source')
    assert author_context(request, 1) == request
