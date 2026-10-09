"""One validation DAG, bounded repairs, mandatory independent challenge."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from .challenge import challenge
from .contract import ContractError, compile_contract, require
from .export import export_bundle, verify_container
from .expressions import canonical
from .verification import qualify, preflight_recipes


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def implementation_identity():
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.iterdir())
             if p.suffix in {'.py', '.md'}}
    return hashlib.sha256(canonical(files).encode()).hexdigest()


def build(request, *, agent, output, source=None, repairs=1, docker=True, base_image='python:3.14-slim'):
    require(type(repairs) is int and 0 <= repairs <= 2, 'repair budget must be 0..2')
    output = Path(output)
    require(not output.exists(), 'pipeline output directory must be fresh')
    output.mkdir(parents=True)
    request = copy.deepcopy(request)
    save(output / 'request.json', request)
    frozen_platform = implementation_identity()
    agent.start_scope()
    attempts, feedback, rejected = [], None, set()
    previous_source = None
    started = time.monotonic()
    for index in range(repairs + 1):
        require(implementation_identity() == frozen_platform, 'platform changed during pipeline')
        folder = output / f'attempt-{index + 1}'
        folder.mkdir()
        attempt = {'index': index + 1, 'phases': {}, 'status': 'running'}
        attempts.append(attempt)
        save(output / 'progress.json', {'attempts': attempts, 'status': 'running'})
        def phase(name):
            attempt['active_phase'] = name
            save(output / 'progress.json', {'attempts': attempts, 'status': 'running'})
        try:
            phase('generation')
            phase_start = time.monotonic()
            if source is None or index:
                repair_feedback = ({**feedback, 'previous_source': previous_source}
                                   if feedback is not None and previous_source is not None else feedback)
                candidate, provenance = agent.generate(request, feedback=repair_feedback)
                attempt['generation_provenance'] = provenance
            else:
                candidate = copy.deepcopy(source)
            previous_source = copy.deepcopy(candidate)
            attempt['phases']['generation'] = time.monotonic() - phase_start
            save(folder / 'source.json', candidate)
            phase('compile')
            phase_start = time.monotonic()
            compiled = compile_contract(candidate)
            require(request.get('category', compiled['source']['category']) == compiled['source']['category'], 'original category changed')
            require(compiled['identity'] not in rejected, 'unchanged rejected contract; no repeated model calls')
            attempt['contract_identity'] = compiled['identity']
            attempt['phases']['compile'] = time.monotonic() - phase_start
            phase('recipe_preflight')
            phase_start = time.monotonic()
            recipes = preflight_recipes(compiled)
            attempt['phases']['recipe_preflight'] = time.monotonic() - phase_start
            save(folder / 'recipe_preflight.json', recipes)
            if not recipes['passed']:
                feedback = {'original_request': request, 'recipe_preflight': recipes}
                rejected.add(compiled['identity'])
                raise ContractError('reference trajectory/answer recipe invalid before model review')
            phase('independent_challenge')
            phase_start = time.monotonic()
            review = challenge(compiled, request, agent)
            if implementation_identity() != frozen_platform:
                raise RuntimeError('platform changed during independent challenge; discard this run')
            attempt['phases']['independent_challenge'] = time.monotonic() - phase_start
            save(folder / 'challenge.json', review)
            if not review['passed']:
                rejected.add(compiled['identity'])
                feedback = {'original_request': request, 'independent_challenge': review,
                            'instruction': review.get('instruction',
                                'Repair the actual business contradictions without weakening the original goal.')}
                raise ContractError('independent business challenge did not pass')
            phase('execution')
            phase_start = time.monotonic()
            compiled, report = qualify(compiled['source'], judge=agent.judge,
                                       independent_judge=agent.independent_judge, independent_review=review)
            if implementation_identity() != frozen_platform:
                raise RuntimeError('platform changed during execution; discard this run')
            attempt['phases']['execution'] = time.monotonic() - phase_start
            save(folder / 'verification.json', report)
            if not report['passed']:
                rejected.add(compiled['identity'])
                feedback = {'original_request': request, 'verification': report}
                raise ContractError('joint execution gates did not pass')
            phase('export')
            phase_start = time.monotonic()
            bundle = output / 'sandbox'
            manifest = export_bundle(compiled, report, bundle, base_image=base_image)
            attempt['phases']['export'] = time.monotonic() - phase_start
            container = None
            if docker:
                phase('container')
                phase_start = time.monotonic()
                container = verify_container(bundle, image='envfactory-v2:' + compiled['identity'][:16],
                                             agent=agent if compiled['source'].get('explanations') else None)
                attempt['phases']['container'] = time.monotonic() - phase_start
            attempt['status'] = 'passed' if docker else 'execution_verified_without_container'
            attempt.pop('active_phase', None)
            if implementation_identity() != frozen_platform:
                raise RuntimeError('platform changed during delivery; discard this run')
            delivery = {'version': '2.0', 'qualified': docker, 'contract_identity': compiled['identity'],
                        'bundle_identity': manifest['bundle_identity'], 'platform_identity': frozen_platform,
                        'gates': {**report['gates'], 'container_execution': docker},
                        'container': container, 'attempts': attempts, 'elapsed_seconds': time.monotonic() - started,
                        'scope': 'executable_sandbox_not_learning_effectiveness'}
            save(bundle / 'delivery.json', delivery)
            save(output / 'progress.json', delivery)
            return delivery
        except ContractError as exc:
            attempt.update(status='failed', error=str(exc), failed_phase=attempt.pop('active_phase', None))
            feedback = {**(feedback or {}), 'error': str(exc),
                        'failed_phase': attempt['failed_phase'], 'original_request': request}
            save(folder / 'feedback.json', feedback)
            save(output / 'progress.json', {'qualified': False, 'attempts': attempts, 'status': 'repair_pending' if index < repairs else 'failed'})
        except Exception as exc:
            # Transport/container infrastructure is never repaired by weakening source.
            attempt.update(status='infrastructure_failed', error=f'{type(exc).__name__}: {exc}', failed_phase=attempt.pop('active_phase', None))
            result = {'qualified': False, 'status': 'infrastructure_failed', 'attempts': attempts,
                      'elapsed_seconds': time.monotonic() - started}
            save(output / 'progress.json', result)
            return result
    return {'qualified': False, 'attempts': attempts, 'elapsed_seconds': time.monotonic() - started}
