"""Execute delivered declarative reference prefixes with production mechanics."""
import copy
import json
from pathlib import Path
import tempfile

from ..sandbox_runtime import (
    AcceptanceScenarioRunner, BusinessGoalEvaluator, BusinessLifecycle,
    ContractRewardGate, ContractToolRegistry, ContractUserSimulator,
    DeclarativeToolCompiler, EpisodeStore, ManifestDataStore, RewardStateSnapshots,
    SandboxError, ExpressionBusinessState,
)


def preview_lifecycle_task(task, root, *, return_context=False, baseline_overrides=None, scenario=None):
    root = Path(root)
    success = scenario if scenario is not None else next((s for s in task['acceptance_contract']['executable_scenarios']
                    if s.get('kind') == 'goal_success'), None)
    if success is None:
        raise ValueError('lifecycle preview requires a success reference')
    with tempfile.TemporaryDirectory(prefix='envfactory-lifecycle-preview-') as directory:
        store = EpisodeStore(Path(directory) / 'episodes.sqlite3')
        manifest = task['artifacts']['data_manifest']
        data = ManifestDataStore(manifest, root / manifest['root'], store)
        implementations = task['tool_implementations']
        lifecycle = BusinessLifecycle(task.get('business_lifecycle'), data, store,
                                      [t['function']['name'] for t in task['tools']])
        registry = ContractToolRegistry(task['tools'], DeclarativeToolCompiler(data).compile_all(implementations),
            event_recorder=store.event, lifecycle=lifecycle,
            tool_contracts=task['task_spec']['tool_contracts'],
            reward_snapshots=RewardStateSnapshots(data, implementations,
                [task['metric_implementations'], task['task_spec']]))
        gate = ContractRewardGate(task['task_spec'], task['metrics'])
        user_manifest = task['artifacts'].get('user_simulation_manifest')
        user = None
        if user_manifest:
            user_root = root / user_manifest['root']
            user = ContractUserSimulator(store,
                profiles=json.loads((user_root / user_manifest['profiles_file']).read_text()),
                scripts=json.loads((user_root / user_manifest['scripts_file']).read_text()),
                execution_routes=gate.routes, route_context=lambda: {
                    'business_state': data._all(), 'initial_business_state': data.baseline,
                    'reward_state_snapshots': store.get_state(RewardStateSnapshots.KEY, {})})
        observed = []

        def call(method, path, body, headers):
            try:
                if path == '/v1/reset':
                    store.reset(seed=(body or {}).get('seed', 17))
                    data.reset()
                    for name, rows in (baseline_overrides or {}).items():
                        data.replace_table(name, copy.deepcopy(rows))
                        data.baseline[name] = copy.deepcopy(rows)
                    if user is not None:
                        user.reset()
                    result = {}
                elif path.startswith('/v1/tools/'):
                    name = path.removeprefix('/v1/tools/')
                    if user is not None:
                        user.guard_tool(name)
                    result = registry.execute(name, body, idem_key=headers.get('Idempotency-Key'))
                    observed.append({'tool_name': name, 'arguments': body, 'result': result})
                elif path == '/v1/user_simulator':
                    if user is None:
                        raise ValueError('dialogue reference requires user simulation manifest')
                    result = user.turn(body['messages'])
                elif path == '/v1/agent_response':
                    store.set_state('final_agent_response', body.get('content', ''))
                    result = {}
                else:
                    raise ValueError(f'unsupported lifecycle preview endpoint: {path}')
                return 200, result, {}
            except SandboxError as exc:
                return exc.status, exc.body('lifecycle-preview'), {}

        steps = []
        for step in success['steps']:
            if step.get('operation') == 'reward':
                break
            steps.append(step)
        call('POST', '/v1/reset', next((step.get('body', {}) for step in steps
                                       if step.get('operation') == 'reset'), {}), {})
        AcceptanceScenarioRunner(call).run({'steps': steps, 'assertions': []})
        predicates = task['task_spec'].get('goal_contract', {}).get('row_predicates')
        final_state = ExpressionBusinessState(data._all(), data.baseline,
                                             store.get_state(RewardStateSnapshots.KEY, {}))
        if predicates and not BusinessGoalEvaluator.evaluate(predicates, final_state, data.baseline):
            raise ValueError('lifecycle success preview does not establish business postconditions')
        if return_context:
            return {'business_state': final_state, 'initial_business_state': copy.deepcopy(data.baseline),
                    'reward_state_snapshots': store.get_state(RewardStateSnapshots.KEY, {}),
                    'trajectory': store.replay(),
                    'final_agent_response': store.get_state('final_agent_response', '')}
        return observed
