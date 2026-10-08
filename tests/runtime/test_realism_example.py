import os
from pathlib import Path
import subprocess
import sys


def test_generated_procurement_application_and_compensation(tmp_path):
    from examples.realism.procurement import materialize
    root=materialize(tmp_path/'sandbox')
    exercise=Path(__file__).resolve().parents[2]/'examples/realism/exercise.py'
    completed=subprocess.run([sys.executable,str(exercise),'--report',str(root/'report.json')],
        env={**os.environ,'PYTHONPATH':str(root),'SANDBOX_TRAINER_API_KEY':'realism-local-test'},
        capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stdout+completed.stderr
    import json
    report=json.loads((root/'report.json').read_text())
    assert report['reward']['reward']==1
    assert report['business_transition_count']==6
    assert [step['status'] for step in report['trace'] if step.get('status',200)!=200]==[409,504,503,403]
