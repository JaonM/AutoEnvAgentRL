"""Explicit exact replay of preparation reviews, including failed reviews.

Never resample the same input just to obtain a passing preparation decision.
"""
import copy
import hashlib
import json
from pathlib import Path

from env_factory_v2.native.code_agent import MODEL, request_body
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.contracts.contract import require, validate_value


def request_identity(role, payload, schema):
    return hashlib.sha256(canonical(request_body(role,payload,schema)).encode()).hexdigest()


class PreparationEvidence:
    def __init__(self, agent, root):
        self.agent=agent
        self.entries={}
        for path in sorted(Path(root).glob('*.json')):
            record=json.loads(path.read_text())
            provenance=record.get('provenance',{})
            if provenance.get('model')!=MODEL or provenance.get('backend')!='codex_cli':
                continue
            key=provenance.get('input_identity')
            require(isinstance(key,str) and len(key)==64 and provenance.get('response_id'),
                    'preparation evidence lacks native identity')
            if key in self.entries:
                require(self.entries[key][0]['response']==record['response'],
                        'conflicting cached preparation evidence')
            self.entries[key]=(record,path)

    def ask(self, role, payload, schema):
        key=request_identity(role,payload,schema)
        found=self.entries.get(key)
        if found is None:
            return self.agent.ask(role,payload,schema)
        record,path=found
        require(record['input']==payload,'preparation evidence input changed')
        validate_value(record['response'],schema,'cached preparation response')
        result=copy.deepcopy(record['response'])
        provenance={**copy.deepcopy(record['provenance']),'reused':True,
            'reuse_scope':'preparation_exact_request','usage':{},
            'reused_from':str(path),'reused_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        self.agent.provenance.append(provenance)
        self.agent.evidence_root.mkdir(parents=True,exist_ok=True)
        (self.agent.evidence_root/f'{len(self.agent.provenance):04d}.json').write_text(
            json.dumps({'provenance':provenance,'input':payload,'response':result},ensure_ascii=False,indent=2)+'\n')
        return result,provenance
