import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from supplement_world_diversity import prepared_task, sha


def test_fixed_task_checks_evidence_and_never_changes_assignment(tmp_path):
    folder=tmp_path/'certified';folder.mkdir();(folder/'manifest.json').write_text('{}')
    pilot=tmp_path/'pilot.json';assignment={'index':123,'slot':100}
    pilot.write_text(json.dumps({'passed':True,'assignment':assignment,'bundle':str(folder)}))
    job={'domain':{'name':'generic_train'},'assignment':assignment,'prepared_task':{'pilot':str(pilot),
        'pilot_sha256':sha(pilot),'manifest_sha256':sha(folder/'manifest.json')}}
    assert prepared_task(job,assignment,tmp_path/'output')['passed']
    with pytest.raises(ValueError,match='assignment'):prepared_task(job,{'index':124,'slot':100},tmp_path/'output')
    pilot.write_text(pilot.read_text()+' ')
    with pytest.raises(ValueError,match='pilot changed'):prepared_task(job,assignment,tmp_path/'output')
