# EnvFactory V2

This package is an independent contract-first implementation. It does not import
V1's compiler, business runtime, user policy, reward evaluator or qualification
gates. Native model roles use the existing authenticated host Code Agent bridge
as transport only, fixed to GPT-6.1-Sol with tools disabled for JSON-only turns.

## Flow

Original request → native business author → structural compiler → cheap real
trajectory/answer-recipe preflight (no mock semantic scores) → independent
native contract challenge → executable counterexamples → joint user/environment
verification → immutable bundle → Docker execution → qualified delivery.

The original request remains the semantic authority. A shared executable contract
prevents internal drift but cannot establish that the original goal was understood.
The independent challenger receives the original request separately and supplies
schema/FK-valid counterexamples with independently expected outcomes. Invalid
review cases are infrastructure/review failures, not authorization to weaken an
author's source. Genuine failed cases provide bounded repair feedback.
The verifier owns invalid-action expectations from the referenced scenario;
the challenger cannot invent an error string to reject a correctly blocked
action. A model rejection with no failed executable counterexample is marked
inconclusive, never used to trigger an author repair.

Tool predicates reference named public rules. Query implementations, public tools,
parameter validation, argument provenance, answer schemas and reward components
are deterministic compilations. Writes have atomic isolated candidate state,
real-state goals, invariants and exact, single-use, revocable user approval.
Eligibility filters in reward/goal expressions must invoke those same named
rules. Handwritten duplicate predicates are rejected before any model call.
Large fixtures use compact seeded sequence/cycle/FK generators. The compiler
materializes and validates every row, while the reviewer receives exact plans
and previews; the model does not spend output tokens writing bulk records.
Public tool prose and wire schemas are generated from those rules and bindings.
Captured arguments advertise actual reference objects in the wire schema;
business value schemas remain separately available. There is no string-schema
versus reference-object contradiction. Phase transitions persist in progress.json.

The User Simulator owns private facts, actual-menu decisions and authorization
state. A hierarchical statechart gives each declared stage an independent region:
waiting (blocked/ready), resolved (disclosed/selected/refused), and consent
(awaiting_confirmation/authorized/declined/invalidated/revoked/consumed).
Dependency guards synchronize regions. Fact corrections invalidate dependent
choices and live approvals; changed proposals invalidate old consent. Revoked or
invalidated consent requires a fresh user confirmation, whereas consumed consent
is terminal. Failed atomic execution retains exact approval for retry. No custom
FSM library or authored transition code is required. Structured stage requests
work deterministically. Optional native language
interpretation sees public stages only, and cannot set user values. Optional
natural responses choose a native style/locale and deterministically render facts,
refusals and exact proposals; model prose cannot contradict the structured event.
Planned corrections overwrite a disclosed binding; legitimate fallback choices
override initial preferences in tool arguments and reward targets alike.

The clock returns configured virtual datetime, ticks and elapsed seconds. Tool
calls and user requests advance by the declared increment. wait advances without
wall-clock sleeping; reads do not advance time. Scheduled external events follow
this clock. Reset recreates data/user/time/cursors for a fresh independent episode.

All mandatory gates must pass; no weighted qualification threshold. Answer-field
weights provide useful partial reward, distinct from binary sandbox qualification.
Actual-state goals, safety and necessary process cannot be bypassed by guessed
answers. Independent explanations use actual evaluated facts, not fixture prose.

Native semantic results reuse exact contract/fact/prompt/evaluator dependencies
within one build scope. Independent replay always obtains a fresh native result;
criterion conflicts poison the dependency. No mock/external fallback is used.

## Run

Run from the repository root using its installed Python environment:

```sh
.venv/bin/python -m env_factory_v2 demo --example retrieval --output output/v2-retrieval
.venv/bin/python -m env_factory_v2 verify output/v2-retrieval/contract.json
.venv/bin/python -m env_factory_v2 generate \
  --request request.json --connection output/code_agent_runtime_service/connection.json \
  --output output/v2-generated --repairs 1
.venv/bin/python -m env_factory_v2 build \
  --source contract.json --request request.json \
  --connection output/code_agent_runtime_service/connection.json --output output/v2-built
```

Requests include `category` and `goal`; additional business requirements are
preserved in the independent review input. `--base-image` selects an operator's
Python 3.14 base, including a locally cached mirror. `--without-docker` is explicitly
an execution-only preview, never a qualified sandbox. Every output directory must
be fresh. Inspect progress.json, attempts, native evidence and sandbox/delivery.json.

Examples: fixed public direct answer; one-tool interactive lookup; 512-record
complete retrieval; 12-result dependency chain; authorized atomic transfer;
8-write atomic transaction. Examples are test fixtures, not generation templates.

## Public service

```sh
.venv/bin/python -m env_factory_v2 serve contract.json --port 8080
```

POST /reset with {} creates an episode and returns public instructions. Use its
opaque identifier in X-Episode for POST /tool, /user, /answer, /wait, /revoke, /delete
and /collect, and GET /time or /user-state. The latter returns only stage states
and transition history, without private facts or proposals, and consumes no tick.
GET /health is global. /collect assembles complete
pages already observed by that episode; it never performs extra business reads.
Container verification replays a successful trajectory through this actual HTTP
API and checks that its reward and virtual time agree with in-process execution.
Private facts, source tables, reference answer
recipes and initial mutation controls are not API endpoints. Agent access is
through this tool API; the trusted server's filesystem is not an agent tool.
Explicit-stage user requests return authoritative structured events directly;
requests without a stage use native interpretation and natural expression.
Model-assisted endpoints use V2_BRIDGE_URL and V2_BRIDGE_TOKEN_FILE. Containers use
a temporary read-only bridge token mount; host Codex credentials stay on the host.

## Current scope

Supports typed relational fixtures, select/projection/order/pagination, expression
aggregates, bounded dependency chains, existing-row atomic multi-table updates,
user disclosure/choices/approval/refusal/revocation, virtual time and scheduled
events. Unsupported insertion/deletion, physical simulation and real third-party
side effects fail explicitly; extending them requires a reviewed platform capability.
No graph sampling or RL trainer coupling is assumed in this first standalone V2.
Executable qualification proves these tested obligations, not learning improvement
or exhaustive semantic correctness. Offline examples and unit mocks never count
toward native production yield.

## RL delivery

V2 now has a local HTTP AgentRuntime adapter, exact token/logprob trajectory
recording through the existing PPO/GRPO runner, and a separate readiness certificate.
First requalify the sandbox on the current V2 platform. Historical delivery
reports are rejected after runtime changes; never upgrade a certificate by editing
its passed flags. Then run:

```sh
.venv/bin/python -m env_factory_v2 certify-training output/new-build/sandbox/contract.json
.venv/bin/python -m rl.train --sandbox output/new-build/sandbox \
  --sandbox-backend local --environment-factory rl.environments.v2:V2Episode \
  --model models/Qwen3-0.6B-4bit --tuning lora --thinking-mode no-thinking \
  --algorithm ppo --rollout-workers 1 --rollout-concurrency 1 --rollout-group 2 \
  --epochs 1 --updates 1 --output output/v2-rl-smoke
```

The trainable policy is the explicit local `--model`; generation/review/semantic
judge remain native GPT-6.1-Sol. Semantic environments require
ENVFACTORY_V2_CONNECTION pointing to the host bridge connection file.
training.json binds delivery, adapter, platform and trajectory protocol identities.
It certifies reference adapter conformance and policy-visible isolation, not
learning effectiveness. Final answers must use v2_submit or the declared JSON
fields; malformed answers and finite episodes without an answer get zero reward.
Infrastructure/judge errors escape and never become invented policy rewards.
Native business calls use the compiled public parameters directly. Queries add
only the reserved optional _v2_cursor transport field; there is no extra business
argument envelope. Native tool call IDs are assigned by the runtime when absent.
Snapshots contain private world context in trainer-only continuations, never
policy messages or loss targets. Rollout seeds identify policy groups; the world
uses the contract's fixed fixtures and configured virtual epoch.

Trainer continuations also retain `eligibility_initial`: the post-reset world,
clock and user lifecycle basis used by initial transaction precondition queries.
Restoring current data and time must not regenerate that basis from default
fixtures. Snapshots missing this basis are rejected before resetting the current
episode; start a new episode instead of silently changing reward semantics.

The current supported training backend is local HTTP in isolated environment
processes. The V1 Docker service manager is explicitly rejected for V2. Container
qualification remains mandatory before readiness. A task manifest supports V2
train/eval splits; mixing V1 and V2 in one run is currently rejected. Related task
families must be kept in one split. A smoke update proves integration, not reward
convergence or general task success.

### Behavioral and communication scope

The pre-generation original-source review produces grounded
`communication_requirements` and `behavior_requirements` in the same call as
answer scope. Both lists travel in `generation_output_scope`, bound to the
original request and policy identity. Citations must be exact original text;
these are model interpretations, not new authority or executable truth. Neither
list waives rules absent from its summary. Historical scope artifacts without
both lists remain readable; fresh response schemas request both.

Communication requirements describe what the user must hear, including relevant
branch conditions and ordering. Behavioral requirements describe what execution
must enforce. An internal calculation rule alone does not require a tutorial,
while actual restrictions, choices, results and exact confirmation terms still
require explanations when grounded in the original request or policy.

Independent reverse review records `communication_basis` for each obligation
group. A compound communication criterion with supported and unsupported speech
is `mixed_overconstraint`, not consistent merely because one clause is supported.
The host rejects a consistent verdict paired with `behavior_only` or
`mixed_overconstraint`; it does not infer semantic correctness from the label.
Original-source review and executable independent cases remain required. This
adds fields to existing reviews, not another scoring call or a keyword gate.

### Clock sensitivity evidence

Author checks and the existing independent review receive bounded no-model
`clock_sensitivity`/`temporal_diagnostics` evidence. At most 16 wait-only probes
use numeric candidate boundaries from facts, events, rules and goals (a bounded
4096-node scan). The report identifies boolean predicate goals changing from
false to true while business data, version and user lifecycle remain unchanged.
It is finite diagnostic coverage, not a complete search or full reward test.
World-changing events are excluded from this particular comparison.

These findings never alter rewards or the local check verdict. Waiting may be a
valid task obligation. Independent reviewers must use the original requirements
and executable cases to distinguish valid waiting from erasing an unfulfilled
obligation when its eligibility expires. Neither replacing all time bases with
`initial` nor banning waits is a valid general repair.

### Production generation entry

`NativeCodeAgent.generate`, `build`, the CLI and the measurement runner all use
the host author workbench by default and require it. `author_tools=False` is
rejected; `--author-tools` remains a no-op compatibility flag for existing
measurement commands. Workbench failure never falls back to a different generator.

String/object/tree generation and their atomic response-repair route have moved
out of the production module to `scripts/experiments/v2_legacy_generation.py`.
That explicit historical diagnostic helper has no production entry or fallback.
Existing artifact decoding and compilation remain supported. This migration
unifies generation transport; it does not imply a measured quality or speed improvement.

Fresh workbench drafts accept only `environment` and `task` object documents.
The advertised MCP section schema and atomic edit validation enforce this same
boundary. Either document may be written first, and a resumed task-only draft
retains the definition format. Flat contracts remain runtime/compiler artifacts
and explicit historical imports: an imported flat draft can be repaired in its
own format but cannot mix environment-definition sections into it. Definition
compilation still owns flattening and preserves separate environment/task
identities; this boundary does not by itself deduplicate all authored semantics.

### Combined contract and trajectory feedback

The reverse-review obligation catalog and implementation paths are shared through
`obligation_index`. Each compiled candidate records a derived `obligations.json`
with original textual clauses, current enforced obligations and reference
dependency identities. It does not introduce another authored business schema.
The index is not proof of semantic correctness and cannot authorize score reuse.

Author checks report changes since the preceding successful compilation, and
pipeline repair attempts save/forward `reference_changes.json`. Changes to the
environment, user, time, public context or reward contract conservatively affect
all references. A recipe-only change affects its own materialized trajectory and
inheriting probes. No-change results mean unchanged dependencies, not semantic
validation; judge/provider and runtime identity checks remain separately required.
The report does not change local pass/fail or permit deleting affected coverage.

After deterministic recipe checks and tool augmentation, the pipeline executes
all authored trajectories and performs independent business review for the same
frozen candidate. A failed trajectory does not suppress independent review:
its rubric may itself be wrong. The review is bound to the unchanged execution
report by evidence hash and contract identity, without rerunning trajectories.
Both kinds of findings are returned in one repair package, so contract fixes and
affected references can be repaired together. Both gates and container validation
remain mandatory. This avoids hiding later-stage findings, but does not establish
a measured speed gain; collecting both can cost more for a failing attempt.

## Code organization

The root contains the public API (`Sandbox`, `compile_contract`, `ContractError`),
CLI, pipeline orchestration, examples, grammar documents and package inventory.
Import implementation modules from their owning package; old flat internal import
paths are not retained as forwarding modules. The CLI remains `python -m env_factory_v2`.

| Package | Responsibility |
| --- | --- |
| `requirements` | Demand provenance, graph grounding, obligation index, tool interference |
| `authoring` | Workbench, MCP author session, constraints and repair feedback |
| `contracts` | Definition lowering, DSL validation, expressions, fixtures and target resolution |
| `execution` | Sandbox state, transactions, rewards, user simulation and scenario values |
| `validation` | Authored execution/replay, independent challenge, diagnostics and HTTP conformance |
| `delivery` | Export/container packaging, HTTP service and RL readiness certification |
| `native` | Native Code Agent transport and response codecs |

`pipeline.py` owns stage ordering. Validators provide evidence; they do not restart
production authoring. Historical generation experiments remain in
`scripts/experiments/v2_legacy_generation.py`; they are not production entry points.
This organization does not claim to remove every existing cross-package dependency.

`package.py` owns the recursive source inventory for both export and fingerprints.
Paths are relative to the package root, so nested files and repeated basenames
cannot disappear from the bundle or overwrite fingerprint entries. Code changes
invalidate existing implementation-bound evidence; moving files is not permission
to reuse an old delivery certificate.

## Acceptance ownership

Compilation rejects invalid contracts before any qualification report exists; it
is no longer represented by an always-true `structural_contract` gate.
`joint_execution_and_counterexamples` checks expectations of both executions;
`deterministic_reset_and_replay` checks their full-state equivalence. The record's
`passed` combines both, while the gates diagnose distinct failures. Independent
business review and container execution remain mandatory for delivery. Training
certification still verifies adapter conformance, privacy and evidence identities.
`validation/gates.py` centralizes the delivery gate vocabulary; historical reports
may still contain the former structural flag, but false/missing required gates or
unknown gates are rejected. Clock/communication diagnostics remain diagnostic hints,
not new acceptance gates. No independent business or authorization check was removed.

The pipeline now exports `semantic_evidence.json` from its scoped native judge.
It is covered by the bundle manifest. Container qualification and the training
adapter restore it only after bundle verification. Entries bind the model,
recursive Python implementation, complete semantic payload and response schema.
Only exact matches reuse scores; all world executions and HTTP/adapter checks still
run. Explicit independent reassessment stays fresh, and conflicting assessments
poison the dependency. The ledger is private delivery material, never policy input.
This avoids repeating the same semantic question across delivery stages without
making the independent original-requirement reviewer trust the platform's score.
