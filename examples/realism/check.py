"""Build and exercise the procurement example locally or in Docker.

Default: local generated app. --docker binds a random loopback port and removes
only the temporary container created by this invocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid
from examples.realism.procurement import materialize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/procurement-realism'))
    parser.add_argument('--docker',action='store_true')
    parser.add_argument('--base-image',default='python:3.12-slim')
    args=parser.parse_args();root=materialize(args.output.resolve())
    project=Path(__file__).resolve().parents[2]
    runner=[sys.executable,str(project/'examples/realism/exercise.py')]
    report=root/('docker_report.json' if args.docker else 'realism_report.json')
    env={**os.environ,'PYTHONPATH':str(root),'SANDBOX_TRAINER_API_KEY':'realism-local-test'}
    container=None
    image_id=None
    try:
        if args.docker:
            tag='autoenv-realism-procurement:local'
            with (root/'docker_build.log').open('w') as log:
                subprocess.run(['docker','build','--build-arg','BASE_IMAGE='+args.base_image,'-t',tag,str(root)],
                    check=True,stdout=log,stderr=subprocess.STDOUT)
            image_id=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
            container='autoenv-realism-'+uuid.uuid4().hex[:12]
            subprocess.run(['docker','run','-d','--rm','--name',container,'-p','127.0.0.1::8080',
                '-e','SANDBOX_TRAINER_API_KEY',tag],env=env,check=True,stdout=subprocess.DEVNULL)
            address=subprocess.check_output(['docker','port',container,'8080/tcp'],text=True).strip()
            url='http://'+address
            for attempt in range(100):
                try:
                    with urllib.request.urlopen(url+'/health',timeout=1) as response:
                        if response.status==200:break
                except OSError:
                    time.sleep(.1)
            else:raise RuntimeError('container did not become healthy')
            runner+=['--url',url]
        with (root/'exercise.log').open('w') as log:
            subprocess.run(runner+['--report',str(report)],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(report.read_text())
        result['verification']={'scope':'generated runtime example, not model-generated training readiness',
            'transport':'docker_http' if args.docker else 'in_process','image_id':image_id,
            'source_hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('sandbox_runtime.py','runtime_llm.py','app.py','BUILD_CONTRACT.json')}}
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'passed':True,'report':str(report),'reward':result['reward']['reward'],'image_id':image_id}))
    finally:
        if container:
            subprocess.run(['docker','stop',container],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':main()
