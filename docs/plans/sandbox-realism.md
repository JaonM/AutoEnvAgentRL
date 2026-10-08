# Sandbox realism implementation

Scope: P0/P1/P2 share the production runtime and preserve the existing
training-readiness gates. Unconfigured tasks keep their prior behavior.

## Implemented

- P0: trusted roles, episode-local grants/revocations, state transition preconditions
  and postconditions, rollback on failure, idempotent committed results.
- P1: persisted virtual time, seeded event offsets, competing external business
  changes, state/result/proposal/time-driven user branches, rejection and replanning.
  External updates are distinguished from agent collateral writes in reward scope.
- P2: bounded pre-write faults and committed-write response loss, status-query
  recovery, multi-operation partial success, cancellation/refund/release compensation.
- Generation: lifecycle is retained through compact source, task artifact and
  generated scaffold. Structural preview validates references; code-agent guidance
  explains mechanisms. Reference steps can declare idempotency keys and failure statuses.
- Training: native tool-call IDs become idempotency keys. Only declared injected
  faults are policy observations; actual infrastructure failures still propagate.
- Example: generated procurement/cancellation app uses the standard scaffold,
  including a rejected split proposal, stock competition, order response loss,
  failed payment, revoked purchase permission and complete compensation.

## Verified

- 660 tests and 131 subtests across runtime, generation and targeted native-tool/
  fault-classification suites.
- 22 targeted remote/native-tool tests passed with loopback networking permitted.
- Docker HTTP exercise of latest runtime: full compensation reward 1, incomplete
  compensation reward 0, six business transitions despite idempotent retries.
  Evidence: `output/procurement-realism/docker_report.json` (image/source hashes).
- Clock/grants/fault counts persist across database reopen and isolate episodes;
  event jitter is repeatable by seed. Failed postconditions roll back writes.

## Still under validation

Real model-authored validation completed at `output/realism_stateful_20261002`
(run `1790969541558877000-55242`): generated 0/1 accepted, built 0, training-ready 0.
The candidate adopted roles, grants, idempotency, state transitions and an
after-commit response-loss fault. Initial reference execution passed, but semantic
review correctly found an unsupported insufficient-stock outcome. Both bounded
repairs then failed execution: branch conditions still queried post-reservation
stock instead of a trusted earlier snapshot, and a state selector matched multiple
rows. The original gate rejected the task; no retry budget or threshold was changed.

Remaining yield work: make multi-outcome reward authoring reliably distinguish
initial/captured/current business facts and prove both success and no-write shortage
paths. Runtime/example verification is complete; model-generated end-to-end yield
improvement remains unproven.

### Decision-time reward checkpoint repair

- `at_tool: {name, expression}` now pins private state at the first successful
  declared read-only tool call, after scheduled events. Repeated calls cannot
  overwrite it; missing checkpoints remain unresolved. Author previews, reference
  execution and generated services share these semantics. Independent reviewers
  receive the temporal semantics and advisory mutable-query diagnostics.
- Actual failed source replay at `output/realism_checkpoint_replay_20261002`:
  original reference still fails; manually adding explicit checkpoints and a
  unique inventory selector passes reference execution and policy ablations.
  This is a targeted regression, **not** independent acceptance or generated yield.
- Docker HTTP validation at `output/realism_checkpoint_20261002/docker_report.json`
  now uses a decision checkpoint for restored stock instead of a fixed expected
  quantity. The compensation scenario passes with reward 1.
- Remaining structural issue: conditional no-write outcomes still need consistent
  route-specific process, dependency and interaction requirements. Do not treat
  the successful reservation reference as proof of the shortage branch.

### Conditional execution routes

- Explicit `execution_routes` now select required tools, causal dependencies and
  user stages from one stable initial/checkpoint decision. Missing, overlapping
  or uncovered decisions fail closed. Inactive process weights are excluded and
  remaining positive weights normalize; outcomes and penalties are not waived.
- Forbidden-write routes require both unchanged business state (allowing recorded
  external events) and absence of successful mutating calls. Irrelevant approval
  stages can be skipped without granting permissions or inventing completed turns.
- Every alternative needs a primary reference or `route_probes` on schema-valid
  changed fixtures. The parent executes actual references and ablations before
  building. These probes do not change the production training data distribution.
- The manually repaired original failure now passes 40 primary-path checks plus
  28 shortage-path checks over three user scripts. Evidence:
  `output/realism_route_replay_20261002/verification.json`.
- Generated service parity passed via actual Docker HTTP: both routes across all
  three user scripts (6 cases), reward 1. Evidence:
  `output/realism_route_replay_20261002/docker_route_report.json`.
- Generated yield still needs a fresh model-authored run. Manual repair and
  Docker parity establish runtime behavior, not end-to-end yield improvement.
- Fresh five-sample run planned at `output/realism_routes_live_20261003`, but not
  started: automatic approval review rejected external model payload transfer.
  A specific authorization request naming OpenAI/Codex, DeepSeek and the payload
  scope is pending. Local graph endpoint is `localhost:7687`. This is not a failed
  generation round and contributes no samples to yield metrics.

### Idempotent reference validation

- Reference necessity, corruption and ordering probes now operate on logical
  requests: retries share a group only with an explicit matching idempotency key,
  tool, arguments, capture provenance and episode. Removing or corrupting a request
  affects all attempts, preventing a retained retry from repairing a negative probe.
- Trace coverage permits one committed event per logical request, including a
  delayed successful retry after a pre-commit failure. Extra attempts do not inflate
  independent evidence. Certification recomputes groups from the authored reference.
- Reward `from_tool` bindings accept retries of the same read request; distinct
  reads of the same tool remain ambiguous. A later retry may introduce captures used
  downstream. This retains current-state query semantics, not snapshot semantics.
- These changes fix false rejections in local generated-service and compiler tests;
  they do not establish fresh model-authored yield.

## Fresh-run findings and follow-up implementation

The authorized run `1790996966432410000-73645` finished at
`output/realism_routes_live_20261003`: generation 2/5, delivery 0/2, end-to-end 0/5.
Its first three authored sources contain no configured external
events, faults or alternate routes. This is not proof of improved complex-sandbox
yield even if their basic multi-step deliveries pass.

- Current prompt says events/faults are appropriate-case choices. The request
  contract requires dependency chains and interaction, but no explicit realism
  dimensions. Consequently removing those mechanics can still satisfy the request.
- Next: carry runner-owned complexity requirements into authoring and qualification.
  Require the selected dimensions without imposing every mechanism on every task.
  Preserve them through repairs; report basic multi-step and requested-complexity
  yield separately. Empty configuration arrays are not coverage.
- Structural preview should validate missing required idempotency keys in positive
  references and route probes, reporting exact source paths together with other
  structural defects. Authored negative missing-key cases must remain permissible.
- Execution evidence should identify the configured external event/fault IDs that
  actually fired, completed recovery and terminal business outcome. A declared
  unreachable fault or reward after an unhandled failure is insufficient. Use the
  existing `external_business_change` and `injected_fault` replay events, including
  committed status, and preserve final Docker/live certification requirements.
- Alternate execution routes still require executable fixtures for each outcome;
  do not count a passing happy path as evidence for recovery/shortage behavior.

## Boundaries

- Branches cannot silently invent new private facts/reward bindings. Changed
  constraints must be represented in the declared stages and business/reward contract.
- External events update existing rows; they do not change primary keys.
- Failures are bounded authored scenarios; arbitrary network loss remains infra failure.
- Authoring supplies domain invariants and compensation rules; the platform does
  not infer financial correctness from a successful HTTP response.

## Verified post-run repairs

- Outer conformance accepts the runtime's `dialogue_turn` operation and identifies
  invalid scenario/step/operation in diagnostics.
- Independent mutation probes execute reference interaction/capture/idempotency
  prerequisites before observing a platform mutation marker. Generic failures do
  not count as killed mutants.
- Reset/isolation checks recognize a single validated `user_protocol` initialization
  event and compare its contents across same-seed resets, without allowing residual
  business activity. Certification consumes the same bounded evidence format.
- Original task-1 and task-3 both passed full Docker/live replay, including
  `training_ready=true`, at `output/realism_build_replay_20261003`. This is 2/2
  existing-sample recovery, not new generation yield.
- Business-condition mutations prioritize rows addressed by condition-bearing
  reward queries. The unchanged task-5 passes 37 executable checks after the fix;
  independent review remains pending for that sample.
- Structural request preflight checks missing/misplaced idempotency keys, including
  alternate route references, while preserving intentional client-error cases.
- `--realism-profile recovery` is wired through daily/generation entry points and
  requires actual changed events and post-fault recovery in successful references.
  Repairs cannot erase required mechanics. Independent review checks business
  relevance; mechanical coverage alone is not semantic proof.

Remaining: fresh recovery-profile model-authored yield, independent review reliability
and consistent meaningful interaction diversity. No claim of full goal completion.

## Recovery run diagnosis and platform repairs (2026-10-03)

The five-request run `output/realism_recovery_live_20261003` was intentionally
stopped after one generation failure and one interrupted task, with three tasks
not started. The simplified prebuild replay returned HTTP 200 for declared 504
faults. This incomplete run has no final five-sample yield.

Implemented repairs:

- Reject reward expressions in literal state selectors before execution; report
  exact paths and explain the supported `value_expressions` form.
- Serialize Decimal diagnostic evidence without rounding or changing reward operands.
- Use the shared lifecycle/tool/user runtime in both prebuild score and buildability
  checks when the task declares a business lifecycle.
- Replace tool-name vocabulary requirements with explicit behavior contradiction
  checks; retain parameter-consumption and actual state-effect validation.
- Allow negative step-removal probes to continue when a previously expected fault
  no longer fires; still evaluate their reward and reject unexpected server errors.
- Report failed ablations with omitted steps, scenario/route, actual reward and threshold.
- Evaluate final readiness with private decision checkpoints and external-event-adjusted
  preservation baselines. Ignore random request IDs only in failed HTTP error envelopes
  during determinism comparison, preserving business values and error semantics.
- Live transport supplies idempotency headers, reuses pending keys after uncertain
  failures, and distinguishes declared tool faults from infrastructure failures.
- Trainer-authenticated state includes decision checkpoints for live postcondition
  checks; public observation does not expose them.
- Interaction matching normalizes whitespace, full-width digits and single-digit
  Chinese quantities, retaining different quantities and actions as mismatches.
- Exclude `.reward_calibration_runtime` from Docker context and portable identities
  so calibration logs cannot break subsequent builds. Sensitive-file checks remain.

The repaired existing task at `output/recovery_platform_replay_20261003` has two
business routes, three materially different interaction paths, a business event,
and an after-commit response-loss recovery. Full local compilation/delivery and
independent source review pass. Repairs to the authored source are explicitly
recorded; this sample must never be counted as fresh model generation success.

An earlier Docker/live replay qualified the environment but had 0/3 policy successes.
Actual failures exposed missing idempotency headers and brittle approval phrases;
the updated replay now passes all three policy episodes. `live_rollout_verified` certifies environment
execution integrity, while `agent_success_rate` and `agent_policy_qualified` describe
policy performance. Do not conflate these or present training readiness as a policy
success measurement.

The completed replay run `1791006062324202000-89686` took 104.679 seconds
through sandbox delivery, with `training_ready=true` and live rewards 1.0 for
seeds 0, 1 and 2. Each live episode executed a write returning HTTP 504 and
subsequently queried the reservation successfully. The trace and hashed summary
are in `output/recovery_platform_replay_20261003/verified_recovery_result.json`.
This is one repaired environment with three episodes, not three generated samples.

Fresh five-task recovery validation started in
`output/recovery_fresh_validation_20261003`, run `1791006417320624000-90652`,
using GPT-6-luna authoring with an 1800-second author budget per task. Its final
generation and end-to-end yields are pending; do not infer them from the replay.

Still required: comprehensive final executed complexity evidence and the fresh
recovery-profile generation outcome to measure end-to-end yield. Independent
review consistency and generation-time script diversity remain open improvements.
