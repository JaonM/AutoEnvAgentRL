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
state. Structured stage requests work deterministically. Optional native language
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
and /collect, and GET /time. GET /health is global. /collect assembles complete
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
