"""Stdio MCP endpoint exposing only a session-local contract workbench."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .author_workbench import AuthorWorkbench, SECTIONS
from .code_agent import object_schema
from .contract import ContractError
from .expressions import canonical


TOOLS = [
    {'name': 'put_section', 'description': 'Replace one draft contract section with JSON. Invalid JSON leaves the draft unchanged.',
     'inputSchema': object_schema({'section': {'type': 'string', 'enum': sorted(SECTIONS)},
                                  'value_json': {'type': 'string'}})},
    {'name': 'edit_contract', 'description': 'Apply atomic JSON Pointer edits to the current draft; values are JSON strings.',
     'inputSchema': object_schema({'edits': {'type': 'array', 'items': object_schema({
         'op': {'type': 'string', 'enum': ['add', 'replace', 'remove']},
         'path': {'type': 'string'}, 'value_json': {'type': 'string'}})}})},
    {'name': 'check_contract', 'description': 'Compile and execute authored reference scenarios and probes. Does not provide independent business judgments or oracle answers.',
     'inputSchema': object_schema({})},
]


class AuthorMCP:
    def __init__(self, output, *, source=None, request=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.workbench = AuthorWorkbench(source=source, request=request)

    def dispatch(self, request):
        identity = request.get('id')
        if identity is None:
            return None
        method = request.get('method')
        if method == 'initialize':
            result = {'protocolVersion': request.get('params', {}).get('protocolVersion', '2024-11-05'),
                      'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'envfactory_author', 'version': '1'}}
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = {'tools': TOOLS}
        elif method == 'tools/call':
            params = request.get('params', {})
            try:
                value = self.workbench.call(params.get('name'), params.get('arguments', {}))
            except ContractError as exc:
                value = {'passed': False, 'error': str(exc)}
            source = self.workbench.snapshot()
            digest = hashlib.sha256(canonical(source).encode()).hexdigest()
            value['source_identity'] = digest
            self._save('draft.json', source)
            if params.get('name') == 'check_contract' and self.workbench.last_check is not None:
                self._save(f'check-{self.workbench.operations:04d}.json',
                           {'source_identity': digest, 'result': self.workbench.last_check})
            # This receipt always describes the current draft, never an older
            # successful check after the model has changed a section.
            self._save('latest_operation.json', {'tool': params.get('name'), 'result': value,
                                                'operations': self.workbench.operations})
            with (self.output / 'operations.jsonl').open('a') as stream:
                stream.write(canonical({'tool': params.get('name'), 'result': value}) + '\n')
            result = {'content': [{'type': 'text', 'text': canonical(value)}],
                      'isError': not value['passed']}
        else:
            return {'jsonrpc': '2.0', 'id': identity,
                    'error': {'code': -32601, 'message': 'Unknown author MCP method'}}
        return {'jsonrpc': '2.0', 'id': identity, 'result': result}

    def _save(self, name, value):
        # All filenames are operator-defined constants; tools accept no paths.
        temporary = self.output / (name + '.tmp')
        temporary.write_text(canonical(value) + '\n')
        temporary.replace(self.output / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    server = AuthorMCP(args.output, source=json.loads(args.source.read_text()) if args.source else None,
                       request=json.loads(args.request.read_text()) if args.request else None)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                continue
            response = server.dispatch(request)
        except (ValueError, TypeError) as exc:
            response = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': str(exc)}}
        if response is not None:
            print(canonical(response), flush=True)


if __name__ == '__main__':
    main()
