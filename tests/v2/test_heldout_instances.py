import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from prepare_v2_natural_pilot import cases
from v2_heldout_cases import heldout_cases
from audit_v2_natural import canonical_truth
from v2_natural_oracle import expand

def test_heldout_instances_preserve_mix_and_change_actual_inputs():
    original, heldout = cases(), heldout_cases()
    assert Counter(c['stratum'] for c in heldout) == {'direct':4,'simple':4,'ordinary':4,'complex':8}
    for old, new in zip(original, heldout, strict=True):
        assert old['acceptance']['kind'] == new['acceptance']['kind']
        if new['stratum'] == 'direct':
            assert old['acceptance']['params'] != new['acceptance']['params']
        else:
            assert expand(old['business']) != expand(new['business'])
        truth = canonical_truth(new['acceptance'], new['business'], new['user'])
        assert set(truth['final_state']) == set(truth['initial'])
        if new['acceptance']['mutation']:
            canonical_truth(new['acceptance'], new['business'], new['user'], mutation=new['acceptance']['mutation'])
    assert original == cases()


def test_new_dossier_relations_are_explicit_complete_and_do_not_edit_old_cases():
    import copy
    from v2_business_relations import with_business_relations
    source = heldout_cases()
    before = copy.deepcopy(source)
    enriched = with_business_relations(source)
    assert source == before
    assert len(enriched) == 20
    assert sum(len(c['business']['relations']) for c in enriched) == 19
    for case in enriched:
        for relation in case['business']['relations']:
            assert relation['policy_text'] in case['business']['policy']
    assert len(enriched[12]['business']['relations']) == 3
    assert len(enriched[19]['business']['relations']) == 3


def test_next_variant_changes_all_instances_and_preserves_scale_and_truth():
    from v2_business_relations import with_business_relations
    first, second = heldout_cases(), with_business_relations(heldout_cases(2))
    for old, new in zip(first, second, strict=True):
        assert old['acceptance']['kind'] == new['acceptance']['kind']
        assert old['stratum'] == new['stratum']
        if new['stratum'] == 'direct':
            assert old['acceptance']['params'] != new['acceptance']['params']
        else:
            assert expand(old['business']) != expand(new['business'])
        assert [len(rows) for rows in expand(old['business']).values()] == [len(rows) for rows in expand(new['business']).values()]
        assert canonical_truth(new['acceptance'], new['business'], new['user']) is not None
