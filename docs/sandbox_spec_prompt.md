# RL Sandbox Design and Implementation Contract

The outer workflow pre-generates an AutoEnvAgentRL-owned `app.py` composition and
`task_impl.py` extension module. Preserve the composition architecture. Put
task-specific handlers, observation projection and uncompiled custom metric scores
in `task_impl.py`; do not replace the shared runtime or the user simulator
with a task-specific framework.

You are the Code Agent developing one RL sandbox task. Read `BUILD_CONTRACT.json` first. It is the read-only task contract formed by removing only the top-level `actions` field from `task.json`. The outer workflow will first create a module development DAG, then invoke you for each production module and finally for structured defect repairs.

AutoEnvAgentRL deterministically generates the read-only module-level `development_plan.json`. This is an implementation DAG, not a task-action plan: it contains production modules, dependencies, inputs, outputs, and executable validation commands. The outer workflow invokes the Code Agent once per node in dependency order. Do not regenerate the plan, create `spec.md` or `action_plan.json`, or turn task actions into development nodes. Shared platform files, compiled tool handlers, compiled metric scores and external_llm_judge execution are platform-owned. Never override them or replace semantic evaluation with fixed answers or keywords. Only implement the current node's declared extension scope.

The implementation must be task-specific and executable. Do not merely restate the input or build a generic demo. Preserve the task's success and failure semantics. If a required task field is absent, report the omission instead of inventing a default.

The outer workflow may provide a read-only `BUILD_CONTRACT.json`. It is an
authoritative projection with exactly the same JSON structure and values as
`task.json` after removing only the top-level `actions` field, not a Code Agent
proposal and not a second source of requirements. It must contain no
platform obligation, capability, endpoint, evaluation rule, action, or other
business information that is absent from `task.json`. Use only the fields and
values present in `task.json`; if the task omits a required implementation
contract such as observation schema or reward definition, report the omission
instead of inventing a default.

## Global action and tool rules

1. Analyze every `environment.action` as a semantic milestone before designing tools. `classification` has exactly two allowed values:

   - `atomic`: from the Agent's perspective, the action has one observable decision/input boundary and one corresponding environment operation. The sandbox may still perform multiple hidden implementation steps—validation, normalization, database transactions, media preprocessing, OCR, model calls, persistence, and hidden-truth evaluation—but those steps remain inside the one Trainer action and do not make the classification `composite`.
   - `composite`: from the Agent's perspective, the action requires two or more independently observable operations or decisions, whose outputs are consumed by later operations. It must be decomposed into multiple `public_llm_tool` and/or terminal `llm_generate` steps, with dependencies, serial/parallel groups, inputs, outputs, state effects, persistence effects, time cost, and rationale. Independent lookups should be separate steps; fan-out and join points must be explicit.

   Use this decision test: if the Agent supplies one request/evidence boundary and the sandbox returns the result of that operation, use `atomic`; if the Agent must choose or invoke multiple separate operations, or one operation's public output is required as another operation's input, use `composite`. Do not classify an action based only on its internal implementation complexity, external-model usage, synchronous/asynchronous behavior, or whether it eventually produces text. Never invent values such as `composite_internal_execution`, `direct_terminal_generation`, or `internal`; internal processing is described in the action's implementation fields, not in `classification`.

   Generic examples: `request_user_evidence` is usually `atomic` because it is one interaction request; `recognize_from_submitted_evidence` may be `atomic` when one evidence-to-result tool call is the Agent boundary even though the sandbox performs OCR and model inference internally; `prepare_report` is `composite` when it requires separate data lookups followed by a final response, for example `lookup_facts` and `lookup_constraints` in parallel, then `llm_generate` as the terminal step. These are decision examples, not fixed tool names for every task.
2. The `tools` section in `BUILD_CONTRACT.json` is authoritative. Implement each declared tool's complete Agent-visible input, output, validation, persistence, and business behavior. There is no separate task-action or Trainer-action registry, spec, or `action_plan.json` artifact. Each public tool has one canonical `tool_name`; user interaction is handled by the separate `user_simulator` endpoint and must not be emitted as an LLM Tool. `llm_generate` is an internal runtime step with no tool name and is never emitted as an LLM tool.
3. Every atomic operation the Agent must choose or perform is a public LLM Tool. The Trainer submits the tool call to the sandbox, and the sandbox executes that tool; the contract does not declare a separate Trainer-action registry. Validation, normalization, transactions, persistence, and hidden evaluation are implementation details inside the tool execution, not separate plan steps.
   The contract may also include `noise_tools` metadata. Every corresponding noise tool is exposed in `tools.json` and has a normal executable endpoint, but is not bound to a task action, must not mutate task-critical business data, and must not produce task-progress reward. `unrelated` tools are unrelated to the task; `related_irrelevant` tools are topic-related but irrelevant to completing the task.
4. Any operation that retrieves knowledge or business data, extracts information, interprets evidence, makes a decision, or supplies context needed by the LLM must be a public LLM Tool. Independent lookups must be separate tools and may share a parallel group; do not hide them in a high-level wrapper.
5. User interaction is not an LLM Tool. `POST /v1/user_simulator` is an internal Trainer-only endpoint and is never exposed in the tools available to the Agent. The Trainer must pass the complete ordered `messages` conversation array; the endpoint must not accept a single-message shortcut or Agent-supplied profile/script identifiers. Each successful endpoint call triggers exactly one User Simulator turn: the sandbox validates and persists the request, reads the current episode's profile/script/user state, delivers the complete context to the independent LLM user simulator, persists the generated `user_query` and `should_end` flag, and exposes them through the next observation. Reset, timers, observation reads, and business-tool calls must never trigger the User Simulator automatically.
6. `llm_generate` means the LLM replies directly from the current context. It is not an LLM Tool or endpoint.
7. LLM tools use the standard Function Tool shape:

   `{"type":"function","function":{"name":"...","description":"...","parameters":{"type":"object","properties":{},"required":[]}}}`

   `function.description` and every declared parameter's `description` must be non-empty natural-language strings, including nested object properties and object items inside arrays. Expose only values the LLM can know or provide. Never expose hidden truth, credentials, hidden-state fields, predicted answers as required inputs, or internal `role=` annotations.

   ID provenance is strict: an ID-valued LLM-tool parameter may only be
   copied from a prior public tool result or the current redacted observation
   that explicitly exposes that public reference. It must never be inferred
   from or copied out of a User Simulator message. User Simulator output is
   natural-language user content plus registered attachments, not an ID
   namespace.

   For confirmation actions, a user's confirmation is an environment event
   produced by the `user_simulator` endpoint and persisted by the sandbox. Do not expose
   internal acceptance record IDs, message IDs, evidence IDs, database keys,
   or fields such as `acceptance_evidence_id` / `acceptance_message_id` as
   required LLM-tool inputs. The confirmation action must resolve and validate
   the latest valid confirmation for the pending proposal internally. The
   Agent may provide a public proposal or candidate reference when needed to
   select the pending operation, but those references must come from tool
   results/observations. It must not submit an ID proving that the user
   accepted.

8. If a task contains media, generate it programmatically as task data. Do not call OCR or a multimodal model, and do not create a separate media-reference or media-content evaluation. The Agent is evaluated only on producing a valid tool name and parameters, correct process transitions, and task-goal completion. Never replace a required tool parameter with hidden truth or hard-coded task answers.

9. Evidence-recognition tools receive only observable evidence references/content and selectors such as `entity_id` or `item_id`. They must produce the interpretation, extraction, classification, verification result, confidence, or uncertainty. Never require `claim`, `answer`, `predicted_*`, `*_prediction`, `material`, or other answer fields as required inputs to an action whose purpose is to recognize or infer that answer. A claim may appear only as a sandbox result or as an explicitly separate user-provided claim being verified.

10. If a task requires independent knowledge or business-data lookups before a final response, model each Agent-selectable lookup as its own `public_llm_tool`, with separate outputs and dependencies. `llm_generate` is the final direct response step; it has no tool or Trainer action and must be the last step of its action chain. Do not add a high-level `submit_summary`, `summarize_*`, or equivalent public tool after `llm_generate` merely to submit the generated text. A submission tool is allowed only when the original task explicitly requires a separate externally observable submission operation.

## 1. Task understanding and scope

Use the fields in `BUILD_CONTRACT.json` to implement the task, type, complexity, uncertainty, expected interaction style, success/failure conditions, out-of-scope behavior, and the declared tool boundaries. The top-level `actions` field in `task.json` is context only and is not a sandbox registry.
If `BUILD_CONTRACT.json` contains explicit contract fields, implement those
fields exactly. Do not invent an obligation section or add requirements that
are absent from the task input.

## 2. Requirement decomposition and implementation logic

For every declared tool, specify preconditions, public inputs, business logic, reads, writes, time cost, user interaction, success result, invalid/rejected result, failure behavior, and acceptance tests. Do not describe task-environment state variables in the environment-generation stage; derive runtime state only from the complete business records and tool execution. For media-based tools, specify only the generated-code execution/persistence boundary, tool-parameter handling, process transitions, and task-goal evaluation; do not add media recognition or separate media-reference evaluation. Include the complete tool execution relationships when tools depend on one another.

`tool_name`, `inputs`, `outputs`, `depends_on`, `parallel_group`, and
`state_effects`.

Do not add `obligation_ids`, Trainer-action mappings, or another planning
layer unless those fields are present in `BUILD_CONTRACT.json`. Implement
every declared tool directly; do not silently omit a tool because it is
inconvenient.

Show fan-out, joins, and mixed serial/parallel execution. A task action may map directly to one tool only when the analysis proves it is atomic; copying all task action names without decomposition is invalid.

## 3. Data model and persistence

Use `sandbox_runtime.ManifestDataStore` to load and validate the declared data
manifest, calculate the baseline hash, and create an isolated business-data
copy on reset. Task code may query or replace declared tables through this
store; do not regenerate JSONL import, baseline copying, or episode isolation.

Define entities, relationships, types, visibility, defaults, and invariants. Prefer SQLite unless the task requires another store. Specify tables, keys, foreign keys, indexes, uniqueness, JSON fields, migrations, transaction boundaries, per-task isolation, crash recovery, concurrency, reset, replay, idempotency, timestamps, state, hidden truth, evidence, actions, tool calls, user messages, rewards, and errors. Before the Agent can act, the sandbox must generate or load an authoritative ground-truth record from the task data model and deterministic seed, assign it a version/hash, and keep it immutable for that episode. Persist the user model, generated behavior script, `user_known_facts`, `user_beliefs`, conversation memory, and user-state transitions with provenance and visibility flags; keep ground truth separate from user knowledge, Agent claims, observations, and model outputs, and never expose it through observation. Never persist API keys.

## 4. Data simulation

Read `artifacts.data_manifest` and the files it references: `data_document.md`, one schema JSON per table, and one rows JSONL file per table. The manifest is the persistence handoff for the Code Agent and keeps `task.json` compact. The document describes each business entity/table, atomic fields, types, visibility, primary/foreign keys, constraints, initialization order, relationships, and persistence requirements. Initialize the database from every non-empty rows JSONL file; do not replace these files with a state/data-model proposal. Tables may be generated and loaded independently, but cross-table foreign-key and business-consistency checks must run before the sandbox is considered ready. If the task requires media, read `artifacts.media_generation`, install only its declared dependencies, run its declared Python entrypoint during sandbox build/startup as specified, and persist only the task-relevant generated media files. Do not add binding tables, truth references, media-evidence mappings, OCR, or multimodal recognition solely because media exists. The Agent is evaluated through tool schema, process transitions, and task-goal completion. If an external LLM is used for user simulation or data proposal, define its adapter, prompt/output schema, timeout, retry, fallback, and configuration. Runtime simulation uses only externally supplied model credentials, never Code Agent credentials.

Business handlers must not inspect replay/tool-call history to enforce capability-DAG ordering or raise
`PRECONDITION_FAILED` based on an earlier call position. They validate declared arguments and current business state.
The shared reward gate and compiled process metrics own causal order and must recognize a later correct retry after an
early mistake; otherwise the sandbox teaches an unrecoverable implementation artifact instead of the business task.

## 5. LLM user simulator

Use `sandbox_runtime.ContractUserSimulator` for seeded profile/FSM selection,
episode-isolated turn state, memory persistence, `should_end`, attachments and
bounded conservative fallback. Its external-LLM renderer is platform-owned;
do not replace the renderer or regenerate the simulator state machine. The
following simulator requirements describe platform invariants, not permission
to implement a separate simulator in task_impl.py.

Design an independent, stateful user simulator, separate from the Agent and from business tools. Read the profile and FSM script files referenced by `artifacts.user_simulation_manifest`; they are not embedded in `task.json`. Initialize the runtime user state from these validated inputs. The external LLM classifies and renders each live turn, but must not regenerate or replace the FSM.

At runtime, only a successful `POST /v1/user_simulator` call starts one simulator turn. The Trainer passes the complete ordered `messages` array; an external LLM classifies the dialogue outcome and renders one natural-language `user_query` from the current profile, FSM state, legal transitions, variables, and complete conversation; a validator checks the outcome/transition pairing and output schema; then the Trainer persists the response, `should_end` flag, attachments, and user state. The simulator emits exactly one user message per endpoint call. Reset, observation reads, timers, and business-tool calls must not invoke it. Infrastructure failure cannot establish acceptance, rejection, correction, or goal completion: emit the FSM recovery policy's non-advancing `unrecognized` outcome, preserve normal state and variables, increment the bounded recovery counter, and terminate as `unresolved_dialogue` only when that bound is reached. The simulator must not read hidden truth, credentials or evaluation data, and must not act as the Code Agent.

The simulated user sees the dialogue and public evidence, not internal tool traces. It must not reject an otherwise
verifiable answer merely because raw tool-call records are absent. Validate both JSON shape and FSM cross-field semantics
inside the bounded RuntimeLLMClient feedback/retry loop; only exhausted retries may use the conservative fallback.

## 6. Real-time simulation

Use real wall-clock timestamps and durations, not an abstract integer clock. Define start time, timezone, `current_time`, action start/end, waiting/deadlines, concurrent ordering, replay, and accelerated versus wall-clock behavior. State the time equation, for example `simulated_now = start_time + elapsed_simulated_seconds`. Every time-consuming action has a duration and observable effect; tests must avoid unnecessary sleeping.

## 7. Tools and tool execution chain

Use `sandbox_runtime.SandboxApplication` as the HTTP/WSGI boundary. Wire task
callbacks into it; do not regenerate routing, Trainer authentication, reset,
tool discovery, tool dispatch, error envelopes, replay, or WSGI parsing in
`app.py`. The generated `app.py` should be a thin composition and launcher.

Compile every entry in top-level `tool_implementations` with
`sandbox_runtime.DeclarativeToolCompiler`. Do not write Python handlers for
those tools. Implement handlers only for declared business tools that have no
declarative implementation specification.

Use the supplied `sandbox_runtime.ContractToolRegistry` and
`validate_json_schema` for Function Tool validation, dispatch, mutation hooks,
and trace recording. Implement only task-specific handler functions in the
generated tool module. Do not copy or reimplement the generic schema walker or
tool execution envelope.

Pass `BUILD_CONTRACT.noise_tools` to `ContractToolRegistry`. Do not implement
task-specific handlers for declared noise tools: the shared registry executes
them with a safe, observable result, records `noise=true`, and never changes
task-critical business state. Noise calls remain visible in the trajectory but
must not satisfy key steps or receive positive process/outcome reward.

The implementation must expose the complete Agent-visible tool design. For
each public tool, specify the exact OpenAI Function schema, preconditions,
effects, time cost, observation changes, persistence writes, return structure,
and errors. The Code Agent must implement the root `tools.json` as the
standard Function Tool array declared by the contract. Do not create
`action_plan.json`, or a separate Trainer-action registry. The outer workflow
owns `development_plan.json`; the Code Agent must not replace it after planning.
A tool request returns only the business result of that tool. It must
not calculate or return observation or reward. The sandbox records the tool
call and its effects in the current session trace. The Trainer obtains public
observation only by calling the observation endpoint, and obtains reward only
by calling the declared reward endpoint; the reward endpoint then evaluates
the accumulated trace and current business data for that request.

## 8. External LLM configuration and Docker injection

The user script must define a boolean `should_end` on every decision branch.
The User Simulator consumes the selected branch and returns both `message` and
`should_end`; `true` means the user stops asking questions and `false` means
the conversation continues. The Agent Simulator never emits this flag. The
controller persists the flag and ends the dialogue only when it is `true` and
the configured minimum dialogue length has been reached; the maximum length
is only a safety fallback.

Identify runtime features requiring an external model, including user-script generation, user-message simulation, data simulation, and optional trajectory-quality evaluation. Media handling must remain deterministic and programmatic; no media-recognition model or media API key is part of the sandbox runtime. The sandbox action executor, not the RL Trainer, owns any runtime model calls. Use the common OpenAI-compatible Chat Completions contract (`POST {base_url}/chat/completions`, Bearer API key, JSON `messages`) and define the script-generation and response-generation prompts, output schemas, provider/base URL/model, timeout, retries, environment variables (`SANDBOX_LLM_API_KEY`, `SANDBOX_LLM_BASE_URL`, `SANDBOX_LLM_MODEL`, `SANDBOX_LLM_TIMEOUT_SECONDS`, `SANDBOX_LLM_MAX_RETRIES`), Docker `-e`/`--env-file` usage, missing-key behavior, deterministic fallback or disabled mode. Never bake keys into images, observations, logs, or databases; Code Agent credentials must never be runtime credentials.

## 9. Reward and evaluation

Execute `acceptance_contract.executable_scenarios` through the supplied
`sandbox_runtime.AcceptanceScenarioRunner`. Do not translate structured
scenarios back into task-specific shell or Python assertions.

Use the supplied `sandbox_runtime.ContractRewardAggregator` for metric range
validation, weighting, and clipping. The generated reward module is
responsible only for producing each declared metric score from runtime state;
it must not reimplement the aggregation formula.

Evaluate every top-level `metric_implementations` entry with
`sandbox_runtime.DeclarativeMetricEvaluator`. Do not write custom evaluators
for those metrics. Custom code or the runtime LLM is allowed only for metrics
without a declarative implementation.

Translate every metric into executable step, terminal, and trajectory logic. Keep only process metrics strongly related to task completion, plus outcome metrics for goal completion or measurable business-data changes. If a task has no key tool actions or can be completed by direct generation, process metrics may be empty. AutoEnvAgentRL compiles accepted success-path process calls into `metric_implementations` with `operator=contains_tool_call`; these are deterministic runtime rules and must be evaluated by `DeclarativeMetricEvaluator`. Only metrics that remain explicitly model-based use the external evaluator boundary. Tool selection/parameter mismatch is a process result, not a penalty. Define formulas, weights, rewards, penalties, clipping/normalization, once-only rules, invalid actions, hidden-state evaluation, terminal success/failure, deterministic replay, response shape, and hidden-truth leakage prevention.

The reward evaluator must be contract-driven, not a task-specific shortcut. Before generating metrics, the task contract contains `reward_key_steps`, the minimal goal-critical Agent actions identified from the task, business model, tools, and dialogue evidence. Ordinary lookups, optional exploration, noise tools, and every tool merely because it exists are not process-reward candidates. Process metrics may target only `reward_key_steps[*].action_name`; if there are no key steps, process metrics must be empty. Do not replace an evaluator with keyword presence, fixed tool-list positions, successful-call coverage, or a hard-coded final-document flag. For every metric, load its `evaluator` object and corresponding top-level `metric_implementations` entry from `BUILD_CONTRACT.json`: compiled process and rule-based outcome/penalty metrics execute deterministic runtime rules; model-based metrics call the configured external LLM; hybrid outcome metrics execute both declared branches. Acceptance must exercise passing and failing cases including wrong arguments, wrong numeric results, missing business changes, repeated invalid calls, and user-intent deviation.

The generated runtime must remain contract-generic after generation. Do not copy generated metric IDs, metric weights, fixed expected-call dictionaries, or fixed turn-count termination rules into runtime code. Load the complete metric/evaluator list, reward formula, tool registry, user profiles, and user scripts at runtime. Implement separate ToolRegistry, RewardEvaluator, and UserSimulator responsibilities.

## 10. Trainer API and state transitions

Design health, reset, observation, tool discovery, tool execution, the `user_simulator` request delivery endpoint, the Trainer-only `POST /v1/agent_response` endpoint, reward/status, replay/export, and optional shutdown interfaces. `agent_response` accepts a non-empty `content`, persists it as the current episode's `final_agent_response`, and records it in replay; it is not exposed as an LLM Tool. Tool execution must return only its business result; it must not calculate or return observation or reward. Only the observation endpoint returns public observation, and only the reward endpoint calculates and returns reward when explicitly called.

Define and expand these equations for the task:

`S_(t+1) = F(S_t, A_t, P_t)`
`T_(t+1) = G(T_t, A_t)`
`O_(t+1) = H(S_(t+1))`
`R_t = Q(S_t, A_t, S_(t+1))`

Explain observable state, hidden state, persistent records, tool effects, time effects, and how each tool changes variables and storage.

## 11. Acceptance and verification

Define executable tests for the standard tool schema, action chains, state transitions, invalid actions, hidden-state leakage, user scripts, external-LLM mocks, media-generation code dependencies/entrypoint execution, generated media persistence, timestamps/durations, concurrency/isolation, rewards, API contracts, reset/replay determinism, Docker build/startup, health checks, and an end-to-end trajectory. Assert that no OCR/multimodal call or binding-generation path is required and that valid tool parameters, process interfaces, state transitions, and task-goal completion are evaluated. Do not add tests whose only purpose is media-reference or media-recognition correctness. Specify evidence and the condition for the outer workflow to write a successful `status.json`.

The outer workflow additionally runs executable mutation testing after the
normal acceptance succeeds. The runtime must honor the declared
`SANDBOX_MUTATION_MODE` modes with `disabled` as the production default.
`acceptance.sh` must inherit that variable and fail for every non-disabled
mutant; a surviving mutant is a build failure. The outer workflow also runs
independent HTTP conformance for each mutant and sends the surviving mode,
stdout/stderr, and runtime log back to the same Code Agent for repair, with a
maximum of three implementation attempts.

## Design decisions and unresolved assumptions

Record task-specific decisions, rejected alternatives, assumptions, risks, and unresolved questions in `IMPLEMENTATION_REPORT.md`. Resolve implementation details without changing task success semantics.

## Runtime HTTP contract

`BUILD_CONTRACT.json.requirements.runtime_interface` is the authoritative HTTP
boundary for the sandbox and RL Trainer. Implement every declared system
endpoint, every `kind=llm_tool` endpoint, and the `kind=reward_function`
endpoint exactly as listed. The tool endpoint is `POST /v1/tools/{tool_name}`;
the endpoint's `request_schema` must be enforced against the corresponding
OpenAI Function Tool parameters. Do not invent an action endpoint or a second
tool registry. The reward endpoint must return the runtime reward computed from
the declared metrics and `reward_formula`, including component scores and
component scores when declared by the contract. There is no task-level
`termination` field or termination evaluator.

The generated contract normally includes `GET /health`, `POST /v1/reset`,
`GET /v1/observation`, `GET /v1/tools`, `POST /v1/user_simulator`,
`POST /v1/agent_response`, one endpoint entry for every LLM tool, and
`GET /v1/reward`. `user_simulator`, `agent_response`, and `reward` are marked
`access=rl_trainer_only`; none may be exposed as an Agent tool. These are a contract, not a requirement to start the
service during image construction; acceptance may start it temporarily.

## Implementation requirements

Read `BUILD_CONTRACT.json`, `task.json`, and all referenced artifacts. First
generate a valid module-level `development_plan.json`; implementation is then
performed node by node in dependency order. Generate all production code and
delivery artifacts directly from the contract, including
business-data loading, persistence, user simulator, tool execution,
observation/reward interfaces, root `tools.json` in standard OpenAI Function
Tool format, tests, `acceptance.sh`, `IMPLEMENTATION_REPORT.md`, `Dockerfile`,
`docker_build.sh`, and `docker_run.sh`. Do not create `spec.md`,
`action_plan.json`, topology scripts, child specs, or task-action plans. Do not
stop at a demo or a design report.

The outer workflow may retry this same Code Agent with the captured
stderr, test output, acceptance output, or missing-file list. On retry, fix
the reported defect in the existing sandbox and finish the complete build;
do not replace `BUILD_CONTRACT.json` or weaken the task contract.

Provide only Docker artifacts: `Dockerfile`, `.dockerignore`, `docker_build.sh`, and `docker_run.sh`; never create `Containerfile` or Apple Container files. `.dockerignore` is platform-owned and must preserve the canonical exclusion of credentials, runtime databases, logs, reviewer transcripts, and certification evidence. Default the base image to `docker.m.daocloud.io/library/python:3.14-slim`, validate reachability before building, and use a compatible mirror fallback when needed. The outer production builder resolves that reference to the current platform's immutable manifest digest and writes the resolved identity back to the final Dockerfile. Building the image must not start or leave a service container running; the outer builder may run a short-lived, network-disabled pytest smoke container and must remove it. `docker_run.sh` is an explicit operator command only. When HTTP is declared, implement exactly the endpoints in `requirements.runtime_interface`, including the separate reward endpoint.

Create a real executable `acceptance.sh` and non-empty `IMPLEMENTATION_REPORT.md`. Acceptance must use the runtime in-process boundary by default (`SANDBOX_ACCEPTANCE_MODE=in_process`) so a host that forbids local TCP binding cannot be mistaken for a business failure. It may start the service temporarily for HTTP checks only when `SANDBOX_ACCEPTANCE_MODE=http` is explicitly selected; it must always terminate that process before exiting and must never turn it into a background service after construction. The in-process checks must exercise the same public application boundary and request schemas as the HTTP handlers, not private business shortcuts. It must capture server stderr, use bounded health retries, distinguish startup errors from an environment permission failure on local TCP binding, and retain HTTP checks as an explicit diagnostic mode. Every business assertion must include diagnostic context: endpoint, HTTP method, request payload, mutation mode, response status, response body, relevant episode ID, and the preceding tool results. On failure, write these details to `acceptance_failure.log` and print them to stderr; do not use bare assertions whose failure only reports `AssertionError`. When `BUILD_CONTRACT.json` declares a non-empty capability `required_trace`, acceptance must write JSONL `runtime_trace.jsonl` at the sandbox root; each event must include `event` and may include `capability_id` and `action_id`. The outer workflow validates the required event order independently. Do not create an `OK` marker. The outer workflow writes `status.json` with `status=pending` when a task starts, updates it to `success` after successful phases, file checks, acceptance, trace validation, and optional Docker build, and records `failed` plus the exit code when construction fails.
The sandbox must include `requirements-dev.txt` with an exact `pytest==...` pin and exact pins for every directly installed test dependency, install it in the Docker image, and run `python3 -m pytest -q`. Version ranges or unpinned dependencies are not production reproducibility evidence. A missing pytest installation is a failed build, not a passed or silently skipped test. The shared Trainer-only `GET /v1/state` endpoint exposes initial/final business state for certification evidence; it must require Trainer Bearer authentication and must never appear in policy observations. After acceptance, write `acceptance_result.json` with `business_acceptance: "passed"` and `http_conformance: "passed"` or `"skipped"`; if HTTP is skipped, include a non-empty `http_skip_reason`.

The outer workflow separately generates an AutoEnvAgentRL-owned conformance plan from `task.json`, `BUILD_CONTRACT.json`, and `tools.json` after implementation. It checks the contract projection, OpenAI tool schemas, declared HTTP endpoints, reward key-step coverage, evaluator declarations, score ranges, and normalized reward formula. It also records invalid-input cases for every tool and pass/fail evaluator cases for every metric. This plan is regenerated in a temporary directory at final acceptance, so files or checks written by the Code Agent cannot weaken it. The sandbox `acceptance.sh` is supplementary and must not be treated as the authority for contract compliance.

Production gates are mandatory: use the provided `runtime_llm.py` and `sandbox_runtime.py` primitives; implement the runtime interface's Trainer Bearer authentication with `SANDBOX_TRAINER_API_KEY`; only LLM tool endpoints are Agent-facing, while reset, observation, user simulator, reward, and replay are Trainer-only. Use an isolated database and trace for every episode, accept a seed on reset, make reset/replay deterministic, support the declared `Idempotency-Key`, and expose replay metadata including seed, schema version, trace hash, and business-data hash. User simulation and evaluator calls must use one shared OpenAI-compatible adapter with the declared `SANDBOX_LLM_*` variables, bounded timeout/retry, deterministic fallback or mock mode via `SANDBOX_EVALUATOR_MOCK`, structured call traces, and secret redaction. All HTTP failures must use the declared JSON error schema and request IDs; logs must be structured and redact credentials. Acceptance must independently exercise unauthorized access, cross-episode leakage, reset determinism, idempotent retry, replay integrity, LLM timeout/fallback, evaluator mock/real schema parity, and reward changes after business-data changes. The Docker image must run as a non-root user, contain no credentials, and be started with read-only root filesystem, dropped capabilities, `no-new-privileges`, and bounded CPU, memory, and process limits.

For HTTP acceptance helpers, parse both successful responses and expected error
responses. A request that intentionally exercises validation, a missing
precondition, a conflict, or a not-found case must catch `HTTPError`, decode
its JSON error envelope, and assert the documented error code; it must never
let `urlopen()` abort the acceptance script before the response is inspected.
Unexpected status codes or non-JSON bodies must still fail with the URL,
request payload, status code, and response body so the implementation agent
can repair the actual contract mismatch.

Before finishing, run the acceptance plan, including tool-schema
validation, public-tool/Trainer-action correspondence, persistence checks,
reward checks, and an end-to-end trajectory. Fix implementation failures
rather than merely reporting them. Do not ask for approval or interactive
input during construction.
