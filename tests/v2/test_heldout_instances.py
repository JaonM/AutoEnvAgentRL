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
