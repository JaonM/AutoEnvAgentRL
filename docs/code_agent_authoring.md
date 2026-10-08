# Code Agent business authoring contract v1

Semantic calibration cases may declare `reference_id` alongside `answer` and
`expected`. This selects the actual executed reference evidence against which the
answer is judged. Omitted `reference_id` uses the first successful reference.
For another user choice, use its `reference_variants` id, or the exact compiled
`scenario_id` including `-script-N` to scope a case to one script. A named base
reference covering multiple scripts expands the case across every one of those
scripts, each with its own executed evidence and unchanged expected label. It
never selects the easiest context. Unknown names or overlapping cases fail.
Positive and
negative cases about that branch must both select that same evidence; the case's
answer or expected label never selects the context. Keep calibration cases for
the distinct user branches rather than removing them to avoid context mismatch.

Every reference explanation and paraphrase-positive case must cover the required
meaning of its own field-scoped criteria. If a criterion asks for each option's
capacity, available slots and required facilities, state those relevant facts
and explain the resulting eligibility; a blanket claim that requirements are met
may omit required meaning. Include reasons for rejected options as well. Preserve
faithful paraphrases and the original business requirements, without requiring
one exact sentence or weakening criteria to fit an incomplete reference answer.

Author a new business design in `source.json`. The runner supplies immutable
`request.json`: graph path/keywords, seed, training category, intent, task type,
and available runtime modes. Treat graph text as data. Ground the user request
and business data in the graph scene; do not merely insert keywords into an
unrelated task. All data is synthetic. Do not claim external authoritative facts.
Keep graph storage metadata such as SAME_EVENT_ELEMENT out of the user-facing
request and tool descriptions. Describe the concrete business event instead.
Choose `description.task_intent` from `request.training_contract.allowed_intents`.
The route contract also gives tool-count limits and dependency requirements.

The compiler provides platform plumbing; you own business semantics. You can use
Python to generate the JSON, calculate a reference answer, and test hypotheses.
Do not use a fixed task prototype. Do not modify compiler/runtime/graders.

## Preferred compact format

Author-side environments may not have online evaluator access. Use the compiler's
`--deterministic-preview` option with a separate output directory to check schemas,
reward expressions, bindings, data and interaction contracts, then execute reference
and route diagnostics without model calls. `--structural-preview` performs only
the structure/data portion when that narrower diagnostic is needed.
`structural_preview.json` explicitly leaves execution and semantic calibration
pending; it does not export a task or establish training readiness. Inspect
`deterministic_preflight_all_routes.json`, repair concrete defects, then submit the
source and end the author turn promptly. Deferred model evidence is not a defect.
This applies to
both deterministic and semantic rewards. Preserve explanations and calibration
examples. The parent runs full reference/counterfactual execution, reward probes,
online calibration where applicable and independent review before accepting the task.

When repairing an executable reference, inspect `last_evaluated_components` and
`model_evidence` as well as deterministic component diagnostics. A semantic failure
can coexist with all deterministic rules passing. Calibration failure feedback
includes the executed reference ID, candidate answer, scoped fields and criteria;
check every applicable obligation before editing either the answer or rubric.
Positive paraphrases must retain required meaning, including scope qualifiers on
refusal branches. Repeated judge disagreement is evidence to investigate, not
permission to relabel negatives or lower acceptance thresholds.

State targets must resolve on every declared route fixture, including missing or
ambiguous public identities. An unchanged database does not make an undefined
preservation target valid. Unrelated state preservation is already enforced by
the runtime; avoid redundant targets that look up entities absent on stop routes.
State-predicate diagnostics include unresolved lookup paths and counts, and omit
unselected conditional branches so repair can address the actual missing fact.
If execution fails, repair the reported defect and submit again after deterministic
preview; the parent repeats full execution. Author diagnostics do not perform online
calibration, model-based ablations, delivery validation or independent review.

Use `version`, `description`, `environment_plan`, `tables`,
`business_tools`, `reference`, and `answer_contract` for JSON answers.
For stateful tasks, prefer `business_contract` below; legacy sources use top-level
`outcomes` and `semantic_goal`. Direct-response tasks use `outcomes`.
Add `interaction_contract` when
`request.require_interaction_contract` is true (see the user interaction section).
The detailed types for descriptions, tables, rules and goals appear below.

### Shared business contract

Define each final-state predicate once in `business_contract.state_targets`.
The compiler generates both goal predicates and state reward predicates from it.
Do not also supply top-level `semantic_goal` or `outcomes`.

```json
{
  "business_contract": {
    "version": "1.0",
    "state_targets": {
      "reservation": {
        "table": "reservations", "where": {"id": "R1"},
        "values": {"status": "confirmed"}, "count": 1
      }
    },
    "outcomes": [
      {"id": "reservation_complete", "rubric": "Reservation is confirmed",
       "targets": ["reservation"]}
    ],
    "report_outcomes": [],
    "operations": [
      {"tool": "confirm_reservation", "target": "reservation",
       "selector": {"reservation_id": "id"},
       "bindings": {"status": "status"}}
    ]
  }
}
```

This fragment assumes a declared reservation table and a `confirm_reservation`
business tool with required `reservation_id` and `status` parameters. Every target
must be referenced by at least one outcome. Target syntax is the same as
`semantic_goal.row_predicates`, including dynamic `value_expressions`. Use public
identifiers and query-derived expressions appropriate to the task; the example
identifier is illustrative. Preserve semantic explanations in `report_outcomes`
using the existing `{id, rubric, semantic}` format. Deterministic answer metrics use
`{id, rubric, rule}` with `final_agent_response` and `value_targets` or `numeric_targets`.

Optional `operations` currently generates **single-table updates**. `selector`
maps required tool arguments to declared table columns that select exactly one row.
It can use a primary key returned by a query even when the goal identifies the row
by another business field. The generated identity guard checks the selected row
against the target's complete `where` conditions before writing; selecting an existing
but unrelated row still returns 409. `bindings` maps each
client-supplied target field to a distinct required argument. Static target values
constrain those argument enums. Optional `derive: ["field", ...]` selects target fields
to compute on the server from the same values/value_expressions used by goal and reward.
Exclude derived fields from `bindings` and tool arguments. For example, deriving
`status` in the fragment above uses `"derive": ["status"], "bindings": {}` and removes
the `status` parameter. Anchor query-derived write values with `at_tool` when they
depend on a completed decision read. A missing checkpoint or unresolved lookup yields
`409 DERIVED_CHANGE_UNRESOLVED` before any update; committed idempotent retries return
the cached result without recomputing fields. Generated low-level implementations use
`derived_changes` for these scalar expressions; client and server assignments cannot
overlap. Integer columns accept integral arithmetic results without rounding fractions.
Omit the tool's `implementation` and its
separate `business_lifecycle.tools` entry; put the lifecycle policy under the operation's
`policy`. The compiler adds an identity guard and defaults to required idempotency.
Existing role checks, guards and transition rules still apply. Multi-table writes
continue to require explicit supported implementations; declaring a target alone
does not establish transactional execution or fault recovery.

Cached idempotent tool retries appear in the episode trace as `tool_call_replay` events
with `payload.idempotent_replay: true`. They return the previously committed result
without re-running the business handler, transition, virtual clock or injected fault.
This records the actual recovery sequence for dependency scoring; it does not mean
another business write occurred. If a recovery read is required, its result must
inform a subsequent decision or action. Capturing a value and never using it does
not establish that dependency. A retry with changed arguments remains an
`IDEMPOTENCY_CONFLICT`, including after response loss.
Each validated tool request also records a separate `tool_attempt` with its actual
arguments, response status and SHA-256 digest of the supplied idempotency key (or
null when absent). These attempts do not count as successful business actions.
Semantic metrics that request tool evidence receive this observed request history,
so they can distinguish same-request retries from changed parameters or keys.
Raw transport keys and untriggered fault schedules are not included.

- `business_tools`: each item is `{name, description, parameters, implementation}`.
  `parameters` is the complete object JSON Schema. `implementation` is the
  declarative tool specification, omitting `tool_name` (derived from `name`).
  Optional `preconditions`, `effects`, `atomicity_rationale` describe the operation.
  Use `[]` for direct responses.
- `outcomes`: each item is `{id, rubric, rule}`. `rule` has `source`, `path`,
  `operator`, `expected`; the compiler supplies metric IDs and score mappings.
  Rules must implement actual business semantics, including every condition.
- `reference`: `{calls: [{tool_name, arguments, capture?}], answer: "..."}`.
  Arguments may reference earlier captures via `{"$ref": "name"}`. For direct
  responses use empty calls. Compute the answer independently from business data.
  A user-selected argument may also consume an earlier capture or a captured
  expression. Keep that provenance instead of substituting a fixture identifier.
  For these arguments, the compiler binds the process rule to the active script's
  disclosed value as well as the capture chain. Both must match the same call.
  Structural preview validates capture availability; full reference execution
  must still prove its actual value agrees with the user choice for every script.
- Optional `negative_scenarios`: use `from_reference: true` to derive a business
  negative from the successful trace. Supply `scenario_id`, `kind: "goal_failure"`,
  `argument_mutations: [{tool_name, arguments: {argument: wrong_value}}]`, and
  ordinary reward `assertions`. The compiler retains successful prerequisites,
  captures, user dialogue and HTTP assertions, and expands every user script.
  Each mutation must select exactly one tool call and change an existing argument.
  A derived argument-mutation negative retains its successful reference's dialogue.
  If all mutated calls have executed and the real user protocol then rejects that
  dialogue as inapplicable, the runner records the rejection, stops downstream
  business actions and reads the actual reward. The original reward assertions
  must still pass. This does not waive positive-path interactions, missing captures,
  HTTP failures or server errors. Hand-written negatives retain strict execution.
  A partial negative may preserve credit for unrelated correct outcomes. Declare an
  explicit final-reward assertion that the violated outcome component is at most
  0.2; the compiler requires that observed active component penalty and total
  reward below full credit. Earlier rewards, process-only penalties, and missing
  component evidence do not qualify. Without this evidence or a verified
  requirement binding, the default aggregate reward ceiling remains 0.2.
  Reward assertions use `$.reward` or `$.components.<metric_id>`; `$.metrics` is
  not a reward response field. Invalid paths fail structural preview.
  If the changed request is expected to be rejected, declare its exact HTTP status
  on that mutation, for example `{tool_name: "reserve", arguments: {id: "wrong"},
  expected_status: 403}`. Omission preserves the original status. In particular,
  an authorization rejection before execution does not produce a successful
  reference's after-commit timeout. Other steps and reward assertions stay intact.
  Use full explicit `steps` only for tests that intentionally omit/reorder actions
  or exercise invalid dialogue; do not combine those steps with `from_reference`.
  A negative must reach its intended defect: an unrelated HTTP rejection does not
  establish that an incorrect business result receives the right reward.
  Scenario-level `user_variables` and `initial_changes` are not supported and fail
  structural preview. Select actual user behavior through declared interaction
  variants and `user_script_indices`; use `route_probes.user_variables` for isolated
  route fixtures. Acceptance scenarios cannot use the internal diagnostic operation
  `mutate_business_state`: production HTTP execution has no generic mutation
  endpoint. Put fixture mutations in `route_probes.initial_changes` or
  business-condition probes, retaining their reward and route checks. Read-only
  `business_snapshot` remains available to the trainer scenario runner.
- `answer_contract`: required whenever an outcome uses `value_targets`.
  `{format: "json_object", schema: {type: "object", properties: {field_name:
  {type: "string", description: "meaning of this answer field"}}, required:
  ["field_name"], additionalProperties: false}}`. Use the correct JSON types:
  string, boolean, integer, number, array (`items` required), or nested object.
  Every public field must have exactly one reward target. Objects require all
  declared keys and reject extras. Arrays optionally use minItems/maxItems up to
  64. Numeric fields may use finite `minimum`/`maximum` bounds (minimum must not
  exceed maximum); nullable numeric fields retain these bounds for non-null
  values. Bounds are published and enforced, not silently discarded.
  Public categorical enums may give 2..64 possible choices, never just the
  reference answer. The compiler validates reference answers and appends this schema to the
  actual initial user message. Do not duplicate format instructions in the prose
  or place a second answer_contract in public_input. The schema owns output
  format; the prose owns the business request. Do not change a requested array
  into a comma-delimited string. Do not put private answers in schema descriptions.
  For finite text decisions expressed with `if` and literal branches, the compiler
  publishes all branch labels as a public enum automatically. An explicit enum
  must include every reachable branch label. It never publishes a sole gold value.

The compiler supplies action bindings, capture-derived dependencies, exact-call
process rewards (total 0.2), outcome weights (total 0.8; direct responses 1.0),
reset/reward steps, stable IDs, success assertions and an empty-episode failure.
Do not manually duplicate these fields when using compact format. This saves
authoring time without choosing the business scenario, tables, tools or answer.

## Source fields

- `version`: `"1.0"`.
- `description`: `task`, `task_intent`, `goal`, `expected_result`, `complexity`
  (`simple`, `standard`, `complex`), `requirements` (input/output modalities,
  currently `["text"]`), `public_input` (`initial_user_message`, `materials: []`).
  Put all facts needed for direct responses in public input. Keep private data,
  reference solutions, internal IDs and reward definitions out of public input.
- `environment_plan`: `mode` (`stateless`, `reference_data`, `stateful`),
  `requires_business_data`, `requires_persistence`, and concrete `reason`.
- `tables`: array of `{table_name, description, primary_key: [column], columns,
  foreign_keys: [], indexes: [], constraints: [], rows: [...]}`. Each column has
  `name`, `type` (`text`, `integer`, `real`, `boolean`), `nullable`. Foreign keys:
  `{column, ref_table, ref_column}`. Include multiple plausible distractor rows.
  Empty tables array is valid only for stateless tasks.
  Stateless/direct-response tasks must use empty private tables: put all input
  evidence in public_input.initial_user_message or public_input.materials.
- `tools`: OpenAI function tools `{type: "function", function: {name, description,
  parameters: {type: "object", properties, required, additionalProperties: false}}}`.
  Every parameter needs a precise description, type and business constraints.
- `tool_implementations`: declarative contracts, one per tool. Required:
  `tool_name`, `operation`, `table`. The compiler defaults `result_field` to
  `records` for select, `count` for aggregate_count, `record` for insert,
  `updated_count` for update, and `deleted_count` for delete. Supported operations:
  `select`, `aggregate_count`, `insert`, `update`, `delete`. Select filters are
  `{argument, column, operator}` with `eq/in/contains/gte/lte`; `projection` names
  the returned columns. Update uses `selector` and `changes` mappings of
  argument name to column name; insert uses `values` with the same direction.
  These are real implementations compiled against data, not canned responses.
  Filter comparisons read `row[column] operator argument`: capacity covering a
  requested party size uses `gte`; price within a budget uses `lte`. Check the
  returned rows against the reference before writing its reward. Every captured
  argument of a select/count tool must be consumed by an explicit query filter.
  A declared parameter alone does not make a real dependency between tools.
- `actions`: `{name, description, atomicity_rationale, inputs: [], outputs: [],
  preconditions: [...], effects: [...]}` for meaningful atomic actions.
- `tool_bindings`: `{tool_name, action_name}`. Do not create tools for reasoning.
- `capability_plan`: `{action_name, kind, requires_tool, dependencies: [action_name],
  reason}`; kind is `environment_operation` or `agent_reasoning`.
- `reward_key_steps`: `{step_id, action_name, required_for_goal: true,
  dependencies: [step_id], rationale}` for required tool steps.
- `metrics`: `{id, category, type: "rule-based", scope, weight, score_range,
  rubric, evaluator}`. Categories: process/outcome/penalty. Positive weights sum
  to 1; outcome exceeds process (typically .8/.2). Penalties, if present, sum
  to 1 independently. Score range `[0,1]` for positives, `[-1,0]` for penalties.
  Scope: step/state/terminal/trajectory. Evaluator: `{kind: "document_rule" or
  "business_state_rule" or "trajectory_rule", source: "runtime_rule",
  assertion, score_mapping: {pass: 1, fail: 0}}`. Process metrics also need
  `target_action`; the compiler derives exact calls and captures from references.
- `metric_implementations`: outcome/penalty executable rules `{metric_id, source,
  path, operator, expected, score_mapping: {pass: 1, fail: 0}}`. Sources:
  business_state/trajectory/final_agent_response/observation. Operators:
  eq/ne/gt/gte/lt/lte/contains/exists/count_gte/count_eq/none_tool_calls/numeric_targets/value_targets.
  Strict `gt`/`lt` require finite numbers and retain strict boundaries, including
  reward assertions; equality never passes a strict comparison.
  JSONPath supports fields, indices and literal equality filters. For final
  response use path `$`. Numeric results should use `numeric_targets`, with
  `expected: {answer_format: "single_labeled_number", targets: [{label, unit,
  tolerance: 0, expression}]}`. Expressions can reference data dynamically, e.g.
  `{aggregate: {table, field, where: {column: value}, op: "sum"}}`; a value can be
  `{lookup: {table, field, where: {column: value}}}`. Lookup selectors support
  nested lookups returning either string or numeric IDs. Arithmetic uses
  `{op: "add|sub|mul|div|max|min", args: [expression, expression]}`. Numeric
  constants are `{literal: 12}`. Aggregates support `sum`, `count`, and
  `sum_product` (use `fields: [column, column]` instead of `field`).
  Do not hardcode private-data-dependent answers in reward.
  Prefer `{from_tool: {name: "your_select_tool", field: "returned_field"}}`
  as an expression for a tool-derived fact. The compiler follows that tool's
  reference arguments and all earlier scalar captures, producing a fresh
  business-data query. It does not trust client answers or cached tool output.
  Multiple successful reference trajectories may share this binding when each
  contains the same unambiguous query, including its upstream capture chain.
  Different capture variable names or downstream user choices are allowed.
  A missing query or conflicting query scope in any successful reference cannot
  supply a shared `from_tool` binding; use an explicitly scoped expression for
  genuinely different queries.
  Structural preview reports `reward_temporal_diagnostics` for current-state
  queries over tables modified by tools or external events, including inherited
  filters and upstream queries. These are advisory review obligations, not failed
  gates. Check each decision condition and selector: a stock eligibility filter
  can stop matching after a successful reservation. Use `initial` only when the
  business rule actually refers to episode-start facts; it does not represent a
  later read after external changes. Keep final-state checks on current state.
  For decision-time facts, use the explicit checkpoint expression below.
  For a tool returning multiple records, add `where` business constraints, e.g.
  `{from_tool: {name: "your_select_tool", field: "name", where: {portions:
  {gte: 24}, vegetarian: true}}}`. The selected record must be unique; every
  selected/output field must be available in the tool's result. Repeated calls
  to the same tool are not an unambiguous from_tool reference; use explicit
  goal-linked lookup expressions in that case.
  Plain lookup/aggregate selectors also support `{gte: value}`, `{lte: value}`,
  `{in: [values]}`, `{contains: value}`, or `{eq: expression}`. Multiple bounds
  on one field are conjunctive. Empty aggregate selections count/sum to zero;
  an absent table or unresolved selector remains invalid evidence.
  Private primary/foreign-key constants are rejected in read-only reward
  selectors. A fixture ID such as 241 must be obtained through the public named
  target and the actual query chain; use from_tool to avoid duplicating joins.
  The compiler publishes the numeric label/unit format to the user. Use one
  `single_labeled_number` target for a one-number answer; use JSON value_targets
  for multiple answer fields. All final-answer outcomes, including direct
  responses, must use numeric_targets, value_targets, or calibrated semantic outcomes described below. Keyword checks such as
  `contains: "M"` cannot verify a size recommendation or its requested reasoning.
  For text, boolean, comparison or mixed structured answers use `value_targets`:
  `expected: {answer_format: "json_object", targets: [{key: "answer_field",
  expression: {lookup: {table, field, where: {column: value}}}}]}`. The final
  response must be a JSON object with exactly these keys; expose that output
  format through the required answer_contract. Values are compared to current business data, not frozen
  strings. Conditions are `{op: "eq|ne|gt|gte|lt|lte|and|or", args: [expr, expr]}`;
  literals may be text, numbers, booleans, arrays or objects (bounded finite JSON,
  no null). Dynamic ordered arrays use `{array: [expression, expression]}`.
  Arrays default to exact sequence comparison, including ties; the compiler publishes this
  policy for every array field. A target may instead declare `array_order: ["code"]` to
  require ascending code order while allowing permutations within equal-code groups, or
  `array_order: [{field: "percent", direction: "desc"}, "name"]` for mixed directions.
  `array_order: []` allows any permutation. Every mode preserves complete typed values and
  duplicate counts; sorting fields must be scalar properties in the answer's record schema.
  Declare only the ordering required by the public business goal. Query `order_by` uses the
  same field/direction syntax, with nulls last in either direction. Sort actual business
  columns directly; independently mutable negative/rank columns are not a valid replacement
  for descending order. A numeric target may add
  `tolerance: 0.000001` for a justified absolute error in the field’s declared units
  (finite, within 0..1). Choose precision from the business requirement, not to mask a
  wrong calculation. The compiler publishes each tolerance in the public input.
  Without tolerance, equality is exact; repeating ratios cannot match an internal
  Decimal exactly using a finite JSON decimal. Integer fields require exact equality;
  text, booleans and collections cannot use this numeric tolerance. Conditional results use
  `{if: {condition: expression, then: expression, else: expression}}`.
  Compact syntax also accepts raw scalars in expression positions, e.g.
  `{op: "eq", args: [lookup_expression, "cleared"]}` and `then: "CLEAR"`.
  The compiler wraps these as literals without changing their meaning.
  Multiple `value_targets` outcome metrics may check separate fields of the same
  JSON answer. Always use `path: "$"`; each target's `key` selects its field.
  Use returned counts or complete result arrays for existence and coverage decisions.
  A blank identifier is not an empty query result unless an explicit public constraint
  makes it disqualifying. Do not discard complete returned evidence by testing the first
  record's identifier against an empty string. An uncalled tool is different from an
  observed empty result; select the uncalled branch from actual route prerequisites.
  The answer must contain exactly the union of deterministic target keys and semantic outcome fields.
  The compiler may publish finite categorical decision labels, but never infers
  a public enum containing a private scripted user value. A selected identifier
  and an empty-result branch do not form a public category vocabulary. Such
  fields retain their authored shape and exact reward rules. Explicitly authored
  enums still undergo the same independent privacy review; do not put a private
  selected answer into an enum merely to constrain final-answer syntax.
  Each semantic judge receives only its owned answer fields, with an explicit metric
  field scope. Public descriptions may mention other separately scored fields; their
  absence from this projected judge input is not an omission in the submitted answer.
  Keep each metric's criteria within its declared field ownership. Cross-field business
  facts still need consistent actual tool/user evidence, and every metric retains its
  independent reference and calibration checks.
  Author each route's narrative against its actual successful prefix. A recovered
  lookup that returns no matches does not imply downstream queries were run;
  empty-result and refusal answers must not inherit a complete-route recovery claim.
  Status, analysis and recovery must agree about completed and unfinished work.
  References, route-probe answers and positive paraphrases must stay within actual
  observations. Private initial changes and external-event schedules do not prove
  the agent observed a rename, withdrawal or historical row. If current-only reads
  omit an old row, do not name it or claim its prior status without another actual
  read. A changed identifier does not by itself establish why the identity changed.
  Include each requested conclusion/fact as its own target. Read-only business
  outcomes must use dynamic `value_targets` or `numeric_targets`; static
  `contains`/`eq` phrases cannot establish business evidence fidelity. Do not
  convert a text/comparison/decision task to an unrelated counting task to pass.
  Every required lookup table must influence a dynamic outcome expression,
  including nested ID lookups and eligibility/status branches. Reading a table
  only to satisfy a process reward does not implement its business condition.
- `semantic_goal`: required for writes. `{row_predicates: [{table, where, values,
  count: 1}], expected_delta: [{table, where, field, before, after}]}`. Writes must
  change the target state, preserve unrelated rows/fields, and satisfy constraints.
  Omit this field for read-only/direct tasks.
- `scenarios`: at least one `goal_success` and one `goal_failure`, each with
  `scenario_id`, `kind`, `steps`, `assertions`. Steps use `operation`:
  `reset` (`body: {episode_id, seed}`), `tool_call` (`tool_name`, `arguments`,
  optional `capture: {variable: "$.records[0].field"}`), `agent_response`
  (`content`), `reward` (`step_id: "reward"`); use `expected_status: 200`.
  Later arguments use `{"$ref": "variable"}` to consume earlier actual results.
  Assertions: `{source: "step:reward", path: "$.reward", operator: "gte" or "lte",
  expected: 1 or 0}`. Never embed private IDs in downstream reference arguments
  where the policy must discover them through tools. Include meaningful failures.

## Route requirements

Direct response: stateless, zero business tools, public input sufficient, evaluate
the actual answer. Simple Agent: exactly one necessary business tool. Multi-step:
at least two necessary tools with a real captured result consumed downstream;
the initial user cannot know the hidden intermediate value. Business complexity
must justify the chain. Artificial splitting and redundant calls are invalid.

Compile using the command in the agent prompt. Inspect concrete compiler errors
and fix their cause. Passing preflight is necessary, not independent qualification:
the parent recompiles your source and repeats the original deterministic Agentic
gate on a fresh scaffold. Its detailed report is `prebuild_agentic_value.json`.
Sandbox acceptance and semantic review remain
separate gates. Do not claim live success from reference execution.

Write the first source draft before browsing implementation details. The field
guide above is sufficient for ordinary designs. If a specific operator fails,
read only that operator's implementation; avoid dumping entire runtime files or
searching the whole repository. The authoring deadline includes validation and
fixes. Explicit user-facing conditions must each have a matching reward rule and
a positive/negative fixture that exercises the branch.

### Discoverable query arguments and parent feedback

For exact-equality string filters, every literal in a successful reference query
must be available in the public request, the relevant tool/parameter contract,
or a preceding tool result captured with `$ref`. A private category label must
not become a guessing game. Publish a meaningful domain enum, provide discovery,
or query broadly and select using returned facts. Do not publish private answer
identifiers simply to satisfy this check. For example, a required exact `mood`
filter cannot silently require `calm and reflective` when the request says
`calm, reflective` and the tool offers no vocabulary.

Write nested source JSON using a Python helper and `json.dump`. The parent
independently compiles and preflights the source. A completed agent turn with a
candidate defect receives at most one focused repair via `parent_validation.json`,
within the shared safety budget. Keep the same graph, training category and
business goal. CLI/transport failures and changes to protected inputs do not
trigger this candidate repair. Both attempts and the original defect remain in
the runner's evidence.

### Reward fact visibility

Every private fact needed to answer must be obtainable through the actual
reference tool interfaces. A raw reward `lookup` or `aggregate` into a private
table is not evidence available to the policy. For example, a booking tool that
returns only headcount and station name cannot support an answer requiring the
station's private capacity. Expose the required fact and prefer `from_tool`.
The compiler checks field visibility for raw lookups/aggregates as well; this is
a necessary check, not a proof that all queried rows or semantic rules are sound.
Compact select tools default `result_field` to `records` when omitted.

Do not model requested facts or explanations as self-attested correctness flags
(e.g. `gold: true` meaning “my answer explains gold formation correctly”). Return
the actual factual content. Boolean business decisions such as `fits` are valid
when their meaning and evidence-derived predicate are explicit.

### Independent source review

After deterministic compilation/preflight, the parent starts a separate read-only
GPT-6-luna review before freezing the task for construction. It checks agreement
between public goal, answer format and reward; observability of every necessary
fact; dynamic business decision conditions; and actual multi-step dependencies.
The parent permits at most one structural repair and one semantic repair, with
at most three author attempts overall. Repeating a failure in an already-used
phase stops the candidate. The repaired source is compiled and reviewed again.
Review and authoring share the same safety time budget. Reviewer infrastructure failure does not spend an author repair.

The public request and published contracts are authoritative. Private rubric prose
cannot impose extra output requirements. Repairs preserve the original business
goal; clarifying an ordering rule is allowed, replacing a comparison with self-
attested correctness flags is not. Review provenance explicitly labels model
judgment; it is not executable proof or a replacement for final sandbox validation.

For stateful goals, identify target rows with stable public business keys where
possible (such as the requested order number), rather than undisclosed fixture
primary keys. Tools may discover internal keys through captures. Preconditions,
write effects and reward must implement the same requested business transition;
required reads must supply information needed for the transition, not merely
collect process points. Model decision-critical business traits as explicit data
fields so eligibility can follow changed evidence rather than a fixed trait label.

### Dynamic state targets and computed write arguments

When the requested write depends on a queried business value, use a capture in
the reference call instead of copying the fixture's number. Arithmetic write
arguments support a bounded expression, for example:

```json
{"value": {"$expr": {"op": "add", "args": [{"$ref": "configured_temperature"}, 5]}}}
```

Use `semantic_goal.row_predicates[].value_expressions` for the matching dynamic
state target. Its expressions support `from_tool` and the ordinary reward
expression operators. Keep literal targets in `values`; a column cannot appear
in both. Do not retain a fixed `expected_delta` after-value for a dynamic target.
Row-predicate `where` compares literal column values; it does not evaluate
`from_tool`, `lookup`, or `at_tool`. Use stable business attributes in `where`
and put query-derived targets (including a selected row ID) in
`value_expressions`. Do not replace a dynamic ID with a private fixture constant.
The compiler rejects numeric write arguments and literal goal values copied from
private fixture rows without a matching public numeric value (`STATE_VALUE_UNGROUNDED`).
This check runs before the answer schema is published, so schema metadata cannot
launder a hidden answer into public evidence. It is a necessary origin check;
it does not prove that a publicly mentioned number is the right business target.
Reference `$expr` arguments permit captures, literals, operators and conditions;
they cannot privately query business tables. Use one outer `$expr`; nested
arithmetic uses `op`/`args` directly, without another `$expr` wrapper. Computed
select-filter arguments are lowered through their upstream captured data by
`from_tool`, so query-derived reward facts remain sensitive to changed data.
Computed write arguments are covered by reference execution and the dependency gate.

Before submitting or repairing a task, check every public business requirement
against an observable fact and an executable outcome or state-goal condition.
For select-tool eligibility, declare top-level `business_conditions` (supported in
both compact and expanded sources). Each entry has exactly these fields:

```json
{"id":"language_support","public_requirement":"must support Persian",
 "tool":"find_venues","argument":"supports_persian","column":"supports_persian","operator":"eq"}
```

`public_requirement` is an exact excerpt from `description.public_input`, in the
task's language. Declare the argument in the tool schema and its `required` list;
provide its public/disclosed value in the reference call. The compiler adds the
select filter, then `from_tool` inherits it into the outcome lookup. Do not repeat
that filter manually; conflicting handwritten filters fail before execution.
Operators are `eq`, `gte`, `lte`, `in`, `contains`, with row-column on the left.
Missing reference bindings and conditions absent from all executable reward
lookups are rejected. This field currently covers select eligibility, not write
goals or free-text explanation quality; retain their existing contracts.
`generation_pipeline.business_condition_probes` records automatic isolated query
counterfactuals: baseline rows, changed field values, and whether other predicates
still hold. A probe changes the same condition field in every initially eligible
alternative (`row_mutations`), so a valid second option cannot invalidate the test.
At actual query execution, after scheduled events, the verifier checks that no
eligible alternative remains. An event that restores eligibility makes the probe
`needs_authored_case`; it does not establish a reward defect or a passing negative. `needs_authored_case` means the compiler could not isolate a condition
(for example, coupled predicates); it is not a passing test. These probes use the
runtime comparator but do not certify the full episode's aggregated reward.
Full compilation additionally replays every successful reference against each
applicable isolated data mutation, preserving its old tool choices and answer.
When user scripts provide different query operands, each successful reference
must bind and score the condition. The compiler retains separate query bindings
and records `reference_ids` on their probes; each probe runs only against the
corresponding reference and its expanded scripts. An ambiguous shared `from_tool`
lookup is still rejected: use observed `tool_result` facts or explicit scoped
outcome expressions. A completed
trajectory must lose full reward, or the business API must reject it with a 4xx.
Harness exceptions are failures, not evidence of rejection. Mutations that violate
the table schema or foreign keys are marked `needs_authored_case`, not passed.
`business_condition_execution.json` retains the changed field, execution outcome,
reward components and replay. These negative controls do not prove that a policy
can solve every changed-data task; positive counterfactuals still require evidence.
This proves syntactic propagation, not semantic completeness or correct predicate
direction. Independent review must still compare every condition to the public
goal; authored negative scenarios must exercise individual violating conditions.

Privacy lexical matches in structural previews are recorded as
`generation_pipeline.privacy_review_obligations`. A candidate list may mention a
private value without revealing its selection; explicit or paraphrased disclosure
is still invalid. The parent can compile these candidates with readiness pending,
but only a complete passing independent semantic review, including
`interaction_information_privacy`, clears the pending state before delivery
preflight. Direct full compilation remains strict unless the caller explicitly
owns that review workflow. A structural preview is never a training certificate.

Calling a requirements tool does not prove eligibility. Stock sufficiency,
capacity, deadlines, handling compatibility and approvals must affect success
when the public task requires them. Test a violating data value as well as the
reference value. Preserve these requirements, explanation fields, user decisions
and distractor records when repairing syntax or execution errors.
Fixed public policy must remain fixed under legal business-data changes. Implement
fixed eligibility statuses or thresholds with required single-value-enum tool
arguments and corresponding filters, rather than resolving them from mutable rule
rows. Use data-driven policy only when the public task explicitly defines it. Plan
these argument constraints in the blueprint before its tool interface is frozen.
If a query returns multiple rows, encode the public selection rule in the query
or `from_tool.where`; selecting `records[0]` alone does not establish a unique
business answer. Never remove alternatives merely to make a lookup unique.
State counts and uniqueness relative to their actual business selector. Distinguish
the full list returned for a user choice from the record uniquely matching that
choice; do not describe a multirow catalog as globally unique. Reference answers,
semantic criteria and calibration cases must make the same scope explicit.
When a structured user stage selects a key consumed through a direct `$ref`,
the compiler turns that script's process-reward captures into keyed selections
such as `$.records[?(@.offer_id=="chosen")].price`. Sibling fields from the same
producer row use the same selector. The same call must satisfy both captured
provenance and the user's disclosed choice; missing or duplicate selected keys
cannot justify a scalar argument. Qualification inspects all executable scoped
options, including wildcard alternatives. Unbound positional captures remain
invalid on a multirow result; unrelated captures and computed selections are not
automatically rewritten.

On a failed executable reference, inspect `preview/preflight_failure.json` for
ungated/gated reward components, expected answer values, actual matching rows and
state goal expectations. Repair the business mismatch without weakening the goal.

For an explicitly stateful experiment, use loop flag
`--generation-environment-mode stateful` (generator: `--environment-mode stateful`).
The `modify` intent label alone does not constrain environment mode. Five minutes
is an observation target; the default sample-wide hard deadline is disabled.


### Initial state and typed state rewards

A target such as “increase the original quantity by 2” must read the episode's
initial snapshot, rather than the already modified row. Wrap its expression:

```json
{"op":"add","args":[{"initial":{"from_tool":{"name":"read_spec","field":"plies"}}},2]}
```

The same syntax works in goal `value_expressions` and numeric/structured answer
rewards. A missing initial snapshot fails closed. The initial snapshot comes
from the trusted episode reset, never from the Agent's report.

For a decision after scheduled external changes, `initial` may be the wrong time.
Use a read checkpoint instead:

```json
{"at_tool":{"name":"read_stock","expression":{"from_tool":{"name":"read_stock","field":"available"}}}}
```

`at_tool` evaluates the enclosed expression against private business state at the
first successful execution of the named declared read-only tool. Scheduled events
for that call have already occurred. Later reads, writes, and idempotent retries
do not replace the checkpoint. Failed calls before execution establish none;
a committed call whose response is lost retains its checkpoint. Checkpoints persist
with their episode and cannot be supplied in tool arguments or final answers.
They are state snapshots, not cached query responses: `from_tool` still lowers its
declared reference selection, now against that snapshot. Only `select` and
`aggregate_count` tools can establish them; use a different named read for a
different decision time. Missing checkpoints yield unresolved evidence, never an
implicit fallback to current or initial data.

`tool_result` and `tool_argument` observations can also refer to declared write
tools. They capture the first successful matching call's validated result and
arguments in the same transaction as the write. Rolled-back calls leave no
observation; committed calls with a lost response retain it, and idempotent
replays do not replace it. A write observation cannot establish an `at_tool`
state checkpoint or substitute for the final persisted business-state goal.

For example, with reset stock 20, an external reservation leaving 16, and the
Agent reserving 12, the three values are `initial = 20`, `at_tool = 16`, and current
stock `4`. The success/shortage branch condition must use the decision checkpoint;
the persisted stock target is checkpoint stock minus the requested quantity.
Keep the actual row being scored in current state. Do not automatically wrap every
reward in a checkpoint: a final-state question must use the final state.

For a reward on persisted state, use `source: "business_state"`, `path: "$"`,
`operator: "state_predicates"`, and `expected` as a list of row predicates with
`table`, `where`, `values`, `value_expressions`, and `count`, using the same shape
as the semantic goal. This selects business rows explicitly and supports dynamic
targets. Ordinary `eq` compares literal values; it does not evaluate an object
placed in `expected`. Keep separate answer rewards when the public task also
requires reporting the result.


### Conditional writes


Prefer `decision_table` for new conditional tasks. Declare one to three observed
boolean `decisions`, then one to eight `branches`. The compiler enumerates every
boolean assignment and rejects missing or overlapping branches before replay.
This proves coverage of declared decisions; independent review still checks
whether the original task has additional business conditions.

```json
{
  "decision_table": {
    "version": "1.0",
    "decisions": {"needs_change": {"fact": "needs_change"}},
    "branches": [
      {"id": "adjust", "when": {"needs_change": true},
       "required_tools": ["lookup_zone", "read_temperature", "update_temperature"],
       "required_stages": [], "writes": "required"},
      {"id": "already_set", "when": {"needs_change": false},
       "required_tools": ["lookup_zone", "read_temperature"],
       "required_stages": [], "writes": "forbidden",
       "probe": {"initial_changes": [], "reference": {"calls": [], "answer": ""}}}
    ]
  }
}
```

Replace the example probe with real data changes, calls and answer for that branch.
Define `needs_change` in `business_contract.facts` as a boolean expression anchored
to a read checkpoint; inline expressions also work. Do not declare `execution_routes`
or `route_probes` alongside a decision table: the compiler generates them.
Primary references must name their route; every other branch needs an executable
probe. An unreachable boolean assignment must be represented by a combined
business decision rather than excused by a fake successful probe.

Optional branch `goal_values` maps shared target IDs to field expressions, for
example `{"stock": {"available": {"literal": 8}}}`. All branches must supply the
same nonempty set of existing target fields. These compile into conditional
expressions shared by semantic goals and outcome rewards. Correct refusal must
preserve state and produce the correct answer; merely skipping a write is insufficient.

For different valid execution paths in `reference_data` or `stateful` tasks,
add `execution_routes` (2..64 entries; preserve every distinct completion and truthful
early-stop branch, including long-plan prefixes). Read-only tasks use only `writes: "forbidden"`;
do not invent a state change or switch modes merely to represent a branch. Each
entry has exactly `id`, `when`, `required_tools`, `required_stages`, and `writes`
(`"required"` or `"forbidden"`). IDs use letters, digits, `_` or `-` (max 64).
`when` is a boolean reward expression; every lookup must be inside `initial` or
`at_tool`. For business decisions, exactly one condition must resolve true. Missing decision evidence,
overlapping alternatives and an uncovered decision fail closed. Every decision
checkpoint must be a required read of the route that uses it. For a later query,
guard its branch with the earlier prerequisite, for example `unique_identity AND
later_count > 0`. A known false prerequisite excludes that branch before the later
query runs; an unresolved prerequisite does not. Both expression operands must
still be valid. Prefer observed `tool_result`/`at_tool` facts for branch decisions.

The selected route determines mandatory calls, dependency edges and user stages.
Every tool called by a success reference (including probe variants) must be in
that route's `required_tools`. Do not stop the declaration at the selection menu
when completion still requires reading the selected record's details. If some
choices require a final query and others must stop, declare separate routes with
conditions based on the actual selected business record. The compiler rejects
reference/route mismatches before model evaluation; omission probes still verify
that each logical request is necessary.
Normal completion routes still need a genuine tool dependency. An early-stop
route may contain one read when it is the first step of another declared dependent
route, its data-dependent condition explains why downstream work is inapplicable,
and writes are forbidden. A downstream-only call or a constant-success branch
cannot use this exception. The task must retain a real multi-step completion route.
Early stopping still requires the exact read, a correct branch-specific answer,
required user interactions, preserved state, and executable branch/ablation probes.
Use conditional outcome expressions so unavailable downstream values are not
required after a legitimate early stop. A forbidden-write route may still require
a choice or confirmation stage protecting an omitted write; that declared dialogue
runs before the final answer. Do not call the write merely to anchor a refusal.
Mandatory calls keep
their compiled argument/capture checks. Stages protecting a required tool cannot
be omitted. Optional process metrics score zero and have zero weight; the remaining
positive weights normalize to one. Outcome metrics stay active and penalty weights
are unchanged. A forbidden-write route requires no successful mutating calls and
preserves all business state except recorded external events; writing then undoing
does not qualify. A required-write route must produce a real state change and
cannot execute a mutating tool excluded from that route.

Set `reference.route_id` on compact primary references (`execution_route` on an
expanded success scenario). Add `route_probes` for every alternative not covered
by a primary reference:

```json
{
  "route_id": "shortage",
  "initial_changes": [{"table":"inventory","where":{"sku":"A"},"values":{"available":8}}],
  "reference": {
    "calls": [
      {"tool_name":"read_order","arguments":{"ticket":"T-1"},"capture":{"sku":"$.records[0].sku"}},
      {"tool_name":"read_stock","arguments":{"sku":{"$ref":"sku"}}}
    ],
    "answer":"{\"status\":\"insufficient\",\"available\":8}"
  }
}
```

Use the task's actual tool calls, captures and answer format. Each fixture selector
must match one row, preserve primary keys and pass the original data schema/FK
constraints. These are preflight fixture probes, not extra training episodes.
All probes run the actual reference, required dialogue, skipped-step, no-tool and
wrong-answer checks. Their reward must select the declared route; merely submitting
a plausible answer is insufficient. Keep necessary process rules for every tool.
Different routes can omit inapplicable tools, but their required calls must respect
the declared input origins and real capture dependencies. Probe process bindings
are compiled from each scoped probe reference and conjoined with its actual user
disclosures; the production reference bindings are unchanged. This also permits
a normally refusing script to exercise a business branch under an explicitly
changed user-preference fixture.

If a probe reaches different user choices, give its `reference` explicit
`user_script_indices` and add sibling `reference_variants`. Each variant contains
a unique `id`, `calls`, `answer`, and `user_script_indices`, just like the primary
probe reference. The scopes must cover every user script exactly once; all share
the same `initial_changes` and declared route. For example, a primary scoped to
`[0, 2]` and an alternate scoped to `[1]` can query and report different selected
options. Every variant still runs the full route validation. Without scopes or
variants, the single reference is replayed for every script as before.

For a user refusal before necessary inputs or tool calls, declare a terminal route:

```json
{"id":"refused_date", "when":{"user_rejection":{"stage":"provide_date","branch":"decline"}},
 "required_tools":[], "required_stages":["provide_date"], "writes":"forbidden"}
```

Each interaction variant declares a private scalar preference, for example
`variables:{"refuse_date":false}`. Its `provide_date` stage has a branch with
`id:"decline"`, `outcome:"user_rejection"`, `advance:false`, a truthful refusal
`user_reply`, and `when:[{source:"user_state",path:"$.refuse_date",operator:"eq",expected:true}]`.
Other branches must be mutually exclusive; add the opposite preference condition
where needed. Terminal refusal cannot depend on `assistant_text` or grant permission.
An agent asking whether to stop does not change the user's preference.

The route is selected only after the matched refusal actually occurs. A private
preference alone, a fabricated answer or an omitted user turn is not evidence.
Observed refusal supersedes business decisions whose downstream observations are
no longer applicable. Required stages include the full dialogue prefix through
the refusal; required tools may contain only its read-only prerequisites. The
ordinary completion route still needs its genuine multi-tool dependency. Any
continued tool attempt after refusal is blocked and prevents full terminal credit.
No state changes are permitted beyond recorded external events.

Use `user_rejection` in conditional answer expressions as well: report an incomplete
status, empty undisclosed fields and no fabricated records. Outcome metrics remain
active; only inapplicable process weights are removed. Stateful goals and outcomes
must explicitly describe the appropriate unchanged state on refusal. There is no
automatic outcome credit or bypass of typed business goals.

A shared `from_tool`
query may omit an inapplicable tool on a declared refusal route only when all
references that execute it agree on the query and its full capture chain. Missing
required calls and unresolved or conflicting bindings still fail compilation.

A refusal `route_probe` may have `reference.calls:[]` and
`user_variables:{"refuse_date":true}`. This changes only declared, same-typed user
preferences in its isolated fixture. Probes still cover every script and run
real dialogue, wrong-answer and skipped-dialogue checks. Refusal may also be a
real primary/alternate user variant with the matching private preference and a
scoped zero-call reference. Earlier disclosures and read prefixes remain required
when refusal happens later in the workflow.

Semantic judges receive actual assistant/user utterances from recorded user turns,
including observed refusal or correction. Private preferences, future replies and
internal branch identifiers are not supplied as dialogue evidence. A submitted
claim that the user refused cannot substitute for an actual recorded reply.

An inventory query that filters out stock below the requested quantity cannot also
report the amount available in the shortage branch. Expose that fact through a
query without the minimum filter, and put the availability threshold in the route
condition and corresponding business outcome expression.

When the public transition depends on a condition, encode that same condition in
reference write arguments using `$expr` with `if`, and in the state target using
`value_expressions`. Capture the condition inputs from prior tool results. The
reference supplies the process reward's expected arguments, so copying just the
initial fixture's branch (for example, a literal ready status) will incorrectly
reject a correct policy when changed initial data requires the other branch.
The reference, persisted-state goal and reported answer must describe the same
transition in both branches. Keep public thresholds as literals and derive
private quantities and selected business records from captures.


### Captured object fields

References may select fields of a previously captured object, such as
`{"$ref":"job.instrument_code"}`, or an array element such as
`{"$ref":"rows[0].id"}`. The compiler lowers these into explicit captures on
the original producing call. The same binding is then used by reference replay,
query rewards, dependency gates and process rewards. Exact declared capture
names take precedence. Future captures and unsupported path syntax remain invalid.

### Direct-answer text and array contracts

Exact literal text targets for direct responses must be identifiable in public
evidence or in a meaningful public categorical enum. Do not invent a private
canonical paraphrase for a summary. The compiler checks literal text origins
before publishing the answer schema (`DIRECT_EXACT_TEXT_UNDISCLOSED`).

Define the meaning of each array element and its ordering in the public contract.
An activities array does not imply two names followed by two time strings. Use
separate named fields for distinct facts, or explicitly specify the requested
array structure. A request permitting a natural summary must not silently require
one exact wording. Preserve the requested business information when repairing
the answer format.

### Unique reward lookups and capture paths

Capture paths may use `$.records[0].id` or `records[0].id`; both have the
same runtime and reward binding meaning. Capturing the first returned row does
not prove that the public business target is unique. A `from_tool` reward lookup
requires one matching business row. If a query matches multiple records, use
publicly grounded disambiguating filters or explicit business selection rules;
do not hard-code the fixture identifier or remove the upstream dependency.

When a tool lists alternatives and the user selects a business key, capture the
selected record with an equality filter, for example
`$.records[?(@.plan_code=="chosen-code")].quoted_fee`, using the actual disclosed
key for that reference's user script. Capture the code and dependent fields from
the same record; keep `$ref` arguments and user-choice enforcement on the consumer.
Do not replace these dependencies with known fixture prices or select the first
unfiltered row. The filter returns a scalar field for exactly one match and a list
for zero or multiple matches. A scalar consumer therefore still requires a unique
match; missing or ambiguous selections need the declared recovery/abstention path.

Failed reference diagnostics include `unresolved_lookups` for each answer target:
`lookup_not_unique` with the matching row count (including zero),
`upstream_lookup_unresolved`, or `field_value_invalid`. Repair the deepest
failed lookup first while preserving the original public goal.

### Semantic explanation outcomes

When the public task permits a natural explanation or recommendation, use a
semantic outcome for those free-text fields. Keep exact classifications, numbers,
booleans and business-state changes in deterministic rules. Do not turn an open
explanation into an enum or a self-attested correctness flag.

A compact semantic outcome replaces `rule` with:

```json
{"id":"explanation","rubric":"Explain why the shared label is insufficient",
 "semantic":{"fields":["why"],
 "criteria":["Explain that the shared label covers distinct culinary roles; accept faithful paraphrases."],
 "cases":[
   {"answer":{"why":"The heading groups ingredients with different roles."},"expected":"pass"},
   {"answer":{"why":"Several distinct culinary functions share this label."},"expected":"pass"},
   {"answer":{"why":"All ingredients have the same role."},"expected":"fail"},
   {"answer":{"why":"seasonings"},"expected":"fail"}
 ]}}
```

Each expanded case must contain a complete JSON answer matching the public answer
schema, including fields owned by deterministic metrics. For compact
`business_tools/outcomes/reference` sources, prefer a named reference plus
`answer_overrides` when only the scored explanation changes:

```json
{"reference_id":"reference_success", "expected":"pass",
 "answer_overrides":{"why":"Several distinct culinary functions share this label."}}
```

The compiler copies that reference's JSON answer and replaces only the listed
top-level fields. Override keys must belong to this semantic metric. An empty
override uses the reference answer as one calibration case; it does not replace
the second distinct positive. Alternate references use their declared IDs;
`-script-N` remains available to select a specific execution context. Use either
`answer` or `answer_overrides`, never both. Expanded-source `semantic_calibration`
continues to require complete answers. The expanded cases undergo the same schema,
uniqueness, positive/negative coverage and real-judge checks.

Each field has exactly one
reward owner. Semantic fields are strings without an enum. Provide at least two
meaningfully distinct correct paraphrases and two incorrect or incomplete answers;
cases must differ in the owned fields. The compiler publishes rubric and criteria
and uses the platform runtime judge. Calibration cases are not judge instructions
or reference answers available to the policy. The actual judge must classify all
cases correctly before construction. Mock matching or network fallback is not
semantic evidence; infrastructure failure does not trigger a task-author repair.

Within one compiler preflight, identical semantic-judge inputs may reuse a prior
successful API judgment. The full metric, response schema, evidence and provider
configuration must match; failed API calls are never reused. The
`semantic_judgment_reuse.json` ledger records original judgment IDs and context
hashes, and replay cache hits cite their origin. Reuse is not an independent
judgment. Every semantic calibration case still obtains a fresh judgment, and
live runtime episodes retain their default episode-local cache.

### Dynamically selected stateful records

When the public request identifies a record through an earlier lookup, preserve
that identity in the persisted goal. A fixed private ID is not a dynamic selector.
For a target resolved through a tool, the existing row predicate can place the
primary-key column in `value_expressions`, using an `initial` lookup or
`initial` / `from_tool` expression for its expected identity, alongside the
derived mutable fields. This constrains both the matching record and allowed
writes when the initial link changes. An empty `where` alone does not express
which linked record the user requested. Do not remove identity constraints to
make a reference fixture pass. Verify the linked-record counterfactual and reject
changes to other records.


### Gate policy: environment delivery and policy performance

The construction entrypoint uses `executable_v1` qualification: executable Docker gates and data governance, with a full 10-point executable score. Live rollout, live reward calibration and trajectory privacy are not run and are not claimed. Real rollout remains part of RL training. Standalone live certification tools retain their stronger criteria.

Standalone live rollout eligibility measures verified execution, clean environment behavior,
and absence of fallback. Agent success rate and `agent_policy_qualified` are
reported separately. A zero-success clean rollout is not a solvability witness;
the executable reference and reward calibration remain required. Production
policy-performance certification retains its separately declared thresholds.
Review scores are diagnostic within the 10-point rubric: a passing, complete,
source-bound review is not rejected solely for a score below 0.8. High-severity
findings still require adjudication; unresolved evidence is not a pass.

Answer counterfactuals apply to stateful tasks whenever an outcome explicitly
reads `final_agent_response`. Correct state/process credit may survive a wrong
answer. Checks require loss relative to the declared answer weight, and a fully
correct reference must earn reward 1.0.

For an idempotent stateful goal, explicitly set `semantic_goal.allow_noop: true`
only when the user permits leaving an already-correct state unchanged. Typed
row predicates still define the goal. If initially satisfied, success must
observe the relevant business evidence and preserve the complete state; required
process and dependency metrics still apply. If initially unsatisfied, the goal
must be reached without collateral changes. Include executable cases for both
branches and a negative case that claims success without observation. Do not
invent a mandatory write in the process contract for the valid no-write branch.
The legacy model pipeline's prose-only no-op generation remains unsupported;
Code Agent must supply the explicit contract above.

A computation/validation tool may consume captured upstream values without
reading a private table again. The complete trajectory must still access private
business data, every business tool must have a runtime event, and argument,
skipped-step, dependency and meaningful-output probes remain mandatory.

## Task-specific user interactions (required for new multi-step authoring)

When `request.require_interaction_contract` is true, add `interaction_contract`
(version `"1.0"`) with exactly `request.user_script_count` variants. Legacy sources
remain readable, but new multi-step designs cannot silently fall back to generic scripts.
Each variant contains `stages`, 2..5 necessary user turns in execution order.
The compiler builds the FSM, stages acceptance conversations before the named tools,
and verifies every variant plus skipped-stage failures. Do not author redundant
confirmation turns merely to satisfy the count. Choose a business task that needs
private user input, a real decision, a constraint revision or approval.

The compiled declaration `task_spec.interaction_complexity.runtime =
deterministic_task_protocol` means the platform validates structured requests and
executes script decisions without an external user-model call. Replies and private
choices come from the active script; profile identity is persisted, but replies are
not automatically personalized by profile. This mode tests declared interaction
branches, not unrestricted conversational realism. Legacy model-driven scripts still
use the external user model and conservative failure recovery. Semantic reward judges
remain external when their metric contract declares an external evaluator.

Each stage has:

- `id`: unique within the variant.
- `kind`: `information_required`, `option_selection`, `constraint_update`,
  `execution_confirmation`, or `correction`.
- `purpose`: the concrete business reason this exchange is necessary.
- `assistant_contains_all`: natural business labels that must appear in the Agent's
  question/proposal. These are substring checks, not an LLM intent classifier.
  Keep them small and naturally discoverable; never use passwords or exact full sentences.
- `reference_response`: a valid Agent question/proposal containing those labels.
- `user_reply`: the actual user disclosure/choice/approval.
- `retry_reply`: help the Agent ask the right question without disclosing the private fact.
- `before_tool`: actual tool name whose execution must wait for this stage. An
  execution confirmation must protect an insert/update/delete tool.
- `requires_tools`: optional preceding tool names whose results are needed to make
  this decision; never include the blocked tool itself.
- `requires_evidence`: optional nonempty list of observed tool prerequisites. Use
  it whenever disclosure, correction or consent requires more than a recorded
  tool call. Each item declares `tool`, optional complete `arguments`, and
  `conditions:[{path,operator,expected}]`. The tool must be in `requires_tools`;
  operators are `eq`, `ne`, `gte`, `lte`, and `contains`. For example,
  `{"tool":"lookup_event","arguments":{"name":"Autumn event"},"conditions":[{"path":"$.count","operator":"eq","expected":1}]}`
  requires the actual public query and exactly one returned match. Every item
  must pass against that tool's latest successful observation. Empty, ambiguous,
  unrelated or missing evidence leaves the stage pending and discloses nothing.
  A corrected lookup can satisfy the prerequisite. Declare only public constants
  and observed result conditions; prose `purpose` does not enforce prerequisites.
- For disclosure/selection/update/correction: `private_fact` is a string, number or
  boolean explicitly supplied by the user, absent from initial public inputs;
  `bind_to:{tool,argument}` ties it to the protected reference call's actual argument.
  Optional `reuse_bindings:[{tool,argument}]` constrains later required inputs to
  the same disclosed value without another user turn. Every reuse must occur
  after the primary protected tool, have an executable reference, and carry a
  process metric. Each script's actual value is conjoined with the later call's
  arguments and captured provenance. Reuse cannot replace a real business-data
  dependency or be used on an execution-confirmation stage. Do not invent an
  input echo in a select response: carry already disclosed facts forward.
  Do not disclose internal IDs that should instead come from tools. Two stages in
  one variant may not repeatedly ask for the same tool argument.
- Selection additionally has `options` (at least two plausible alternatives),
  containing the chosen private_fact; the reference proposal presents all options.
- Update/correction additionally has `previous_fact`, a different constraint actually
  present in the initial message or an earlier user reply.

Example stage (within a complete variant):

```json
{
  "id": "choose_region",
  "kind": "constraint_update",
  "purpose": "用户修改配送区域，必须按新区域重新筛选",
  "assistant_contains_all": ["配送区域"],
  "reference_response": "请确认最终配送区域，是否仍按此前的西区？",
  "user_reply": "配送区域改为东区，请按新区域筛选。",
  "retry_reply": "请先向我确认最终配送区域。",
  "previous_fact": "西区",
  "private_fact": "东区",
  "bind_to": {"tool": "search_delivery_options", "argument": "region"},
  "before_tool": "search_delivery_options",
  "requires_tools": []
}
```

Keep the business reference answer and reward rules consistent with the final
constraints. Variants currently share the same business goal/reference; vary the
information path, alternatives, revision history and approval requirements, not
an unimplemented per-variant reward target. At least two interaction kinds must
be represented. ID-only copies, missing disclosures, inconsistent bindings,
unreachable stages and confirmation of read-only tools are rejected.

User turns in this protocol use task-authored replies and executable guards;
they do not spend an external LLM call to decide a fixed contract transition.
Outcome reward is withheld until every required interaction occurs. The normal
business evaluator still decides whether the final result is correct. Profiles
and the legacy LLM-driven FSM remain available for existing fixed-goal tasks.
Review semantic necessity independently: passing a dialogue gate alone is not
proof that a question is useful. Plan rollout max_steps for user turns + business
tool calls + final response, with space for recovery.


### Validation evidence for the interaction upgrade

Full local regression: 1,023 tests and 150 subtests passed, including local Metal
RL integration. New interaction tests cover all five stage kinds, numeric constraint
updates, private-information leakage, duplicate scripts, argument bindings, read-only
confirmation rejection, skipped stages, generated-app response schemas, and the final
agentic-training-value gate. The final gate was exercised with deterministic local
fixtures (`offline_mock`); this is not evidence of a new model-authored production
batch or an improvement in live rollout success rate.

### Parent repair feedback

`parent_validation.json` retains the original validator error and adds a
`feedback` packet. Each issue identifies its evidence level, cited source values,
focused repair guidance and closure conditions. A model review claim is not an
execution observation: check it against the compiled contract before changing
business requirements. For privacy defects, inspect related public rubrics,
criteria, answer schemas and reference questions together; lexical matches alone
do not establish a leak.

Blocking semantic findings must provide a structured counterexample grounded in
the public contract, an actual declared script, or explicit business-row changes.
Script assumptions must match the declared private facts. Business mutations must
satisfy the same types, nullability, uniqueness, CHECK and foreign-key constraints
as the sandbox. Invalid reviewer reports receive at most one report repair within
the existing review time budget; this does not add author repair attempts.

The reviewer also receives `reference_observations` from successful executable
references: actual goal rows, evaluated state targets, terminal reward and executed
external field changes. These observations are retained before negative probes run.
They prove the baseline only; a proposed changed-data case still needs its own replay.
Scheduled `changes[].values` are absolute assignments, not implicit decrements.
For example, assigning stock 14 makes it 14 even when the initial fixture changes
from 16 to 15. A reward based on initial stock minus two is wrong in that variant.

The reviewer receives executed reward lookup observations on the initial business
snapshot. Repair feedback can also show value-target changes under a proposed
business-data counterexample. These observations cover deterministic value targets;
they do not replay a whole episode, evaluate semantic metrics, or prove post-write
outcomes. An unresolved target is not a successful match. No diagnostic observation
overrides a rejection or grants training readiness: full parent validation and
independent review must still pass.

Structural previews also include `reward_observations` and
`unresolved_readonly_reward_targets`. For read-only tools, an unresolved value
produces `READONLY_REWARD_TARGET_UNRESOLVED` with the affected field, executed
selector and matching-row count. Inspect these warnings before requesting full
validation. Correct the business selector or answer contract; do not delete
legitimate distractor rows merely to force uniqueness. These are diagnostics,
not additional acceptance gates. For mutating tasks, initial-state observations
must not be mistaken for the required post-action result.

## Stateful realism (P0/P1/P2)

Stateful designs can include `business_lifecycle` (version `1.0`). This contract is
preserved in task artifacts, checked in structural previews, and enforced by the
same runtime in generation verification and generated Docker applications.

Tool policies may declare `guards` for conditions that must hold immediately before
the write. Each guard has a unique `id`, a `table`, a `selector` mapping columns to
literals or `{"argument": "name"}`, and `conditions` mapping columns to comparisons.
For example, inventory sufficiency can use
`"conditions": {"available": {"gte": {"argument": "quantity"}}}`.
Supported comparisons are `eq`, `gte`, `lte`, `in` and `contains`; comparison operands
may be literals or references to required tool parameters. Unknown or optional
parameter references are rejected during structural preview. Checks run after
scheduled external changes and before mutation. A failed check returns 409 with the
guard ID, matched row count and failed fields' actual/expected values. A committed
idempotent retry returns its cached response without applying obsolete preconditions
again. Declare appropriate parameter bounds, such as positive reservation quantities.

```json
{
  "version": "1.0",
  "roles": ["buyer"],
  "tools": {
    "place_order": {
      "roles": ["buyer"],
      "requires_grants": ["purchase"],
      "require_idempotency": true,
      "transition": {
        "table": "orders", "selector": {"id": {"argument": "order_id"}},
        "column": "status", "from": ["draft"], "to": "placed"
      }
    }
  },
  "dynamics": {
    "tick_seconds": 30,
    "events": [{"id": "competitor", "at_tick": 2, "jitter_ticks": 0,
      "changes": [{"table": "inventory", "selector": {"id": 1}, "values": {"available": 40}}]}],
    "faults": [{"id": "lost-response", "tool": "place_order", "call_index": 1,
      "phase": "after_commit", "status": 504}]
  }
}
```

Transitions select exactly one row. The handler performs the write; the runtime
checks the old state, permission and new state. Failed postconditions roll back
writes and idempotency records. Selectors referencing arguments must bind required
tool inputs. A principal's roles come from the trusted task contract, not tool
arguments. Execution-confirmation stages can set `grant_permissions` and
`revoke_permissions` arrays; reset clears these episode-local grants.

Time advances one tick per validated, non-cached tool attempt. User-interaction
blocks before tool dispatch do not advance it. Events run before the business
operation. `jitter_ticks` adds a seed-determined bounded offset, persisted with the
schedule for resume. External updates cannot change primary keys; data constraints
still apply. Recorded external changes are accounted for only when checking
unrelated writes, never substituted for the original snapshot in reward formulas.

Faults are finite, tied to tool attempt indices. `before` returns 429/503/504 without
performing the operation. `after_commit` preserves the write and cached response,
then returns a transport error. Retrying with the same `Idempotency-Key` returns the
original result without another tick or write. Different arguments with the same
key are a conflict. Author status-query tools so an Agent can resolve an uncertain
outcome. Reference tool steps use `idempotency_key` for this header.

`request.realism_requirements` contains runner-assigned mechanisms:
`{"version":"1.0","mechanisms":["external_change"]}`. Preserve every assigned
requirement across repairs; do not change the task intent to satisfy it. Report
incompatible requirements rather than dropping them. Optional source
`realism_requirements` can add mechanisms but cannot remove assigned requirements.
All declared events/faults require execution coverage even without an assignment.
Events must change actual business data; faults must fire and have subsequent
successful recovery in a correct terminal trajectory. External changes require
business data (`reference_data` or `stateful`); they do not force an Agent write.
Compensation requires `stateful`. Fault recovery does not also require an unrelated event.

Supported mechanisms: `clarification` requires an executed grounded user stage;
`authorization` requires executed bound consent; `external_change` and
`fault_recovery` require lifecycle events/faults; `idempotency` requires a declared
idempotency policy plus an executed same-key replay; `compensation` requires a
nonempty `compensation_tools` list naming actual write tools, which must execute
successfully. The shared goal checks the final restored/correct business state.
Independent review checks that the mechanics create meaningful decisions and that
compensation actually reverses the relevant effects; tool names alone prove neither.

Validation treats identical keyed retries as one logical request within an episode.
Necessity and argument-corruption probes affect all its attempts; retries do not
count as extra required business steps. Each scenario step still needs a unique
`step_id`. A `from_tool` binding may refer to a repeated read only when the attempts
share the same key, arguments and capture provenance. Separate reads remain
ambiguous, and `from_tool` still queries current state; use `at_tool` for a trusted
decision-time snapshot.

User stages may define up to eight `branches`, each containing `id`, nonempty
`when`, `user_reply`, `outcome`, and boolean `advance`. Each condition has `source`
(`business_state`, `last_tool_result`, `assistant_text`, or `virtual_clock`), JSON
`path`, `operator` (`eq`, `ne`, `gte`, `lte`, `contains`, `exists`), and `expected`.
A branch may override permission grants/revocations on confirmation stages. No
matching branch uses the base stage; multiple matches are an authoring error.
`advance: false` leaves the stage incomplete and consumes a bounded recovery turn.
Branches cannot silently redefine private facts or reward bindings: represent
changed business constraints explicitly in declared stages and reward contracts.
Reference dialogue steps can assert `expected_branch` and `expected_stage`.

For cross-system operations, model independently committed steps (reserve, order,
payment) and explicit compensation (cancel, refund, release). Each step must enforce
its own business invariants. A successful prefix is partial success, not completed
work. Terminal state rewards must reject outstanding obligations, duplicate writes,
unauthorized operations and unrelated modifications. Never reward a cancellation
claim without checking the actual order, payment and reservation state.

See `examples/realism/procurement.py` and `examples/realism/exercise.py` for a
reproducible runtime example. It does not claim model-generated training readiness.

### Shared dynamic business identity and decision facts

Use `business_contract.facts` for expressions shared by state targets, writes and
policy guards. Reference a fact as `{"fact":"selected_sku"}`. References are
expanded by the compiler; unknown/cyclic references fail before execution.

A state target can declare `identity: {"sku": {"fact":"selected_sku"}}`, with
`where: {}`. The identity must come from `at_tool` evidence, typically
`{"at_tool":{"name":"read_order","expression":{"from_tool":{"name":"read_order","field":"sku"}}}}`.
The compiler uses this identity for the write guard, reward/goal row and permitted
row changes. Identity fields are not write fields: omit them from `bindings` and
`derive`. Do not separately hardcode the selected SKU in readiness guards.

Guard selectors and comparator operands accept `{"expression": EXPRESSION}` as
well as literals and `{"argument":"name"}`. Use trusted observed requirements
for quantity checks, and reuse that fact for the exact quantity written. Missing
observed evidence rejects the write with 409. A shared operation can declare
`observed: {"available": CHECKPOINT_EXPRESSION, "reserved": CHECKPOINT_EXPRESSION}`
to require current fields to match the decision snapshot before committing. This
prevents stale reads from overwriting external updates. Idempotent committed
retries return the saved result without recomputing quantities.

After an after-commit timeout, a successful status query may establish completion.
The compiler does not derive a new mandatory query-to-resubmission dependency
solely from a same-key recovery replay; the runtime still requires identical
payloads for that key. Never add an unnecessary replay to satisfy a process gate.

A repeated read-only reference call may declare `optional_observation: true`.
Its omission must retain full success in parent replay and independent delivery
certification. Writes and first reads cannot be marked optional. Optional reads
must not supply essential new captures or replace required decision evidence.

Do not expose a private user choice as a singleton public parameter enum/const.
Publish its legitimate domain and use the interaction binding to enforce the
actual disclosed choice. Repeated validation issue IDs stop repair escalation;
read executed counterexample evidence before changing business definitions.

Tool descriptions must match the user's recovery goal and runtime behavior:
when a committed write loses its response, describe status confirmation and
same-key replay as alternatives unless the original user goal truly requires
both. Do not invent a mandatory retry in tool prose and then add gates solely
to enforce that invented obligation.

When a search can return multiple eligible records, bind the actual selected
record with `{"tool_result":{"name":"search_tool","path":"$.records[0].id"}}`
(or another scalar field/path). This uses the first successful observed response,
including the tool's real ordering, rather than re-running a scalar lookup over
all eligible rows. It is also allowed in shared identity facts and guard
expressions. Only declared read-only tools may supply these observations.
Retain query eligibility conditions and their counterexamples; a response count
alone is not proof that a selected business record meets the conditions.


### Approval terms and business decisions

`requires_grants` checks that authorization was granted. It does not itself bind
that grant to a price, entity, dates or quantity. Enforce the approved terms using
business lifecycle guards and include those terms in the shared business contract.
Use a decision-table no-write branch when the observed offer differs from the
finite user script's approved terms. A correct observed price is not evidence of
consent to pay that price. Independent review must check the actual user reply,
proposed write and reward target together.


### Structured approval terms

New Code Agent requests require explicit consent scope for every confirmation.
A confirmation must grant or revoke a named permission. Its protected tool's
lifecycle rule must require that grant. Every grant declares `approval_terms`:

```json
{
  "kind": "execution_confirmation",
  "before_tool": "reserve_display_bay",
  "grant_permissions": ["reserve"],
  "approval_terms": {
    "reserve": {
      "scope": "terms",
      "arguments": {"bay_id": "BAY-B"},
      "postconditions": [{"table": "reservations", "where": {"id": "R-1"},
        "values": {"daily_fee": 900}, "count": 1}]
    }
  }
}
```

Supply the usual stage trigger, replies and purpose as well. Argument terms are
checked before execution. Postconditions check actual persisted rows inside the
write transaction, before commit; mismatches return AUTHORIZATION_TERMS_MISMATCH
and roll back every write. Therefore a server-derived 950 cannot silently replace
an approved 900. Fields and values must be the actual approved terms, not a fresh
lookup of the current offer. A mismatch needs a truthful refusal/nonbooking route
or a subsequent explicit approval of revised terms, not a weakened reward.

When the user explicitly approves the displayed calculation or selected offer,
use proposal-bound terms instead of hardcoding the reference data's values:

```json
{
  "kind": "execution_confirmation",
  "before_tool": "reserve_display_bay",
  "grant_permissions": ["reserve"],
  "reference_proposal": {"bay_id": {"$ref": "selected_bay"}, "daily_fee": {"$ref": "quoted_fee"}},
  "approval_terms": {
    "reserve": {
      "scope": "terms",
      "proposal_arguments": ["bay_id", "daily_fee"],
      "postconditions": [{"table": "reservations", "where": {"bay_id": {"proposal_argument": "bay_id"}},
        "values": {"daily_fee": {"proposal_argument": "daily_fee"}}, "count": 1}]
    }
  }
}
```

This form requires structured interactions. Supply the usual stage fields and an
actual user reply approving the displayed parameters. The public material
`approval_proposal_tools` lists tools and parameter names requiring `request.proposal`.
The agent sends all write parameters as finite scalars in that object, alongside
the normal confirmation request; its question must describe the same proposal.
The simulator echoes the approved parameters and freezes them with the grant.
The write must use exactly that argument set. Postcondition references bind the
same frozen values inside the write transaction, including row identity.

`reference_proposal` is a construction hint for reference scenarios only. It may
use captured `$ref` or `$expr` values, resolved before the confirmation is sent;
real requests must contain scalar values. It never determines actual consent.
Later business changes do not update an existing approval. Obtain new approval
when terms change. Rejection branches grant nothing, and may receive the same
public proposal format. Grant branches for one tool must use one format.

Keep fixed user restrictions in literal `arguments` or literal postconditions.
Literal arguments and `proposal_arguments` must be disjoint and together cover
every write parameter; the displayed proposal must also include those fixed
arguments unchanged. Only declared proposal arguments may be referenced in
postconditions. Include a positive probe with changed initial business values
and negative probes for altered writes or persisted values after approval.
Do not use this form to turn fixed-price consent into approval of any price.

For truly unrestricted approval use `{"scope":"operation"}` explicitly. This is
not allowed when the actual user reply limits price, entity, dates or quantity.
Independent review checks this meaning. Branches changing grants must supply
matching approval_terms, including `{}` when a rejection grants nothing.
Revocation clears stored terms; another grant replaces its prior terms. Runtime
reset clears consent with other episode state. Existing artifacts without typed
terms remain readable, but new Code Agent authoring may not omit them.


## Business requirement coverage (new multi-step tasks)

When `request.require_business_requirements` is true, provide `business_requirements`.
Each row covers a concrete obligation in the public goal, rather than an implementation detail:

```json
{
  "id": "budget",
  "statement": "The selected option must fit the user budget",
  "input_refs": ["/description/public_input/initial_user_message"],
  "tools": ["query_options"],
  "outcome_ids": ["budget_satisfied"],
  "positive_scenarios": ["reference_success"],
  "negative_scenarios": ["over_budget--reference_success"]
}
```

`input_refs` are resolving JSON pointers in the expanded business source. Use stable
`description.public_input` or `interaction_contract` paths for user-visible/disclosed facts.
`tools` may be empty for an answer-only obligation. `outcome_ids` must name terminal
outcome metrics; process credit does not prove a business requirement. Every outcome
must appear in the table. Scenario IDs refer to expanded source scenarios before compiler
FSM variant suffixes; a derived negative is typically `<negative-id>--<reference-id>`.
The compiler resolves each explicit script variant without guessing arbitrary ID prefixes.

For every bound positive, all linked outcome components must reach their maximum.
For every bound negative, at least one linked component must be at most 20% of its
normalized range. Preserve legitimate partial process or unrelated outcome credit.
A negative must actually violate the named obligation, and must not only omit an
unrelated prerequisite. Use enough outcome granularity to test the obligation itself.
The independent semantic reviewer checks this meaning and coverage against the public goal.

`requirement_coverage` is exported in the task. Structural validity, paired execution,
and semantic completeness are separate fields; structural validity alone certifies neither
of the latter. `requirement_reward_validation.json` reports per-requirement/scenario
false-positive and false-negative counts, measured denominators, and inconclusive probes.
These rates describe authored probes only, not arbitrary rollout accuracy. Missing or
non-finite component scores cannot count as a correct rejection. Existing legacy tasks
without a coverage table are explicitly `not_declared`; no coverage is invented for them.

### Positive equivalence and recovery probes

New multi-step authoring runs positive variants separately from canonical step-necessity
checks. JSON object/array answers get a value-preserving formatting probe automatically.
Static reference-data environments without lifecycle behavior also get a repeated read
probe. Inapplicable families are not claimed as verified. These simple probes are not
coverage of all paraphrases or all possible recovery paths.

For meaningful alternate wording or exploratory recovery, add up to eight
`reward_pair_probes` in source:

```json
[
  {"id":"alternate_answer","reference_id":"reference_success",
   "kind":"equivalent_answer","answer":"A different, fully correct terminal answer"},
  {"id":"explore_then_recover","reference_id":"reference_success",
   "kind":"readonly_recovery","before_tool":"find_asset",
   "calls":[{"tool_name":"find_asset","arguments":{"section":"another valid section"}}]}
]
```

Use actual tool/argument names and business-valid examples. The compiler expands the
reference ID across declared user-script variants. Recovery adds at most three read-only
calls with no capture overrides; it cannot alter the original reference, business rules,
reward or assertions. Both fresh executions must finish with identical business state and
full reward. Added calls that alter lifecycle state or fail execution are inconclusive and
reject the claimed equivalence; they are not silently counted as successful recovery.
`reward_pair_validation.json` retains measured false negatives and inconclusive counts.
Do not label a wrong answer as equivalent to force the reward implementation to accept it;
the source reviewer checks the claimed equivalence against the original business goal.
Every `equivalent_answer` must satisfy all criteria for each of its scored fields,
including per-record details, current revisions, boundary comparisons and explanations.
Apply this check to both semantic calibration positives and reward-pair probes. A fact
present elsewhere in the JSON answer does not satisfy a criterion restricted to an
explanation field. Preserve the criteria when correcting an incomplete positive.

A separate `value_targets` field may report a directly observed `tool_argument`
used to filter a private `select` or `aggregate_count` query. This scope field still
requires another positive outcome reward that checks private evidence through a
lookup, aggregate or observed private read. Reporting the selected query scope alone,
copying an unused argument, or adding a zero-weight evidence metric does not qualify.

Structural preview applies the same target qualification as delivery and names the
failing metric before model evaluation. A fixture literal for the user's reply,
including a literal inside a refusal conditional, does not establish a data dependency.
Use actual observations and retain the required refusal behavior; do not remove a
requested output or weaken its reward to clear the check. For multi-row selections,
capture the row matching the disclosed unique key with a predicate path. A positional
`records[0]` capture cannot by itself prove that the selected business record is correct.

### Business blueprint before implementation

New multi-step authoring first delivers `blueprint.json`. The parent validates its
business obligations, tool input provenance, acyclic tool-result dependencies and user
interaction bindings before requesting `source.json`. Blueprint and implementation
share the author time budget. The blueprint permits one repair; accepted blueprint
content is immutable and its hash is retained in generation evidence.

Each blueprint tool declares `operation` (`select`, `aggregate_count`, `insert`,
`update`, or `delete`). Reference-data plans are read-only; stateful plans need a write.
An execution-confirmation stage must protect a write tool. Implementation must preserve
these operations, business obligations, description, tool names and input bindings.
Successful reference scenarios must actually consume the declared producer fields;
substituting known fixture constants does not satisfy a dependency.
Derived inputs can cite a user disclosure bound to another argument of the same tool
(for example, the selected format determines an offer ID and price). A derived input
may select a descendant of its declared producer result, such as an ID from a returned
records array. Direct inputs still require the exact user binding or producer field;
unrelated stages and sibling result fields do not establish provenance.

A direct input may reuse an earlier user stage with the same argument name when
the original protected tool is an ancestor through a real tool-result dependency.
Keep `kind:user` and the original stage ID in the blueprint, and add the later
`reuse_bindings` to that source interaction stage. Query APIs return their declared
records/count envelope, not arbitrary root-level request arguments. Ambiguous
producer fields named like request arguments must use an explicit returned-column
path (for example `$.records[0].id`) or cite the original disclosure. An optional
blueprint `result_field` declares a custom read envelope and must match the source.

Blueprint producer paths and observed answer fields are checked against the
runtime primitive's response envelope before freezing. Select defaults to
`records` plus `count`; aggregate_count defaults to `count`. Insert returns
`record` plus `count`; update returns only `updated_count`, and delete returns
only `deleted_count`. These writes do not echo request IDs or return updated
rows. Obtain required identities from a real read, an inserted record, or the
original disclosed input. Impossible paths must be replanned before freezing;
downstream provenance and transaction requirements remain enforced.

Pagination cursors are transport dependencies within one query. Compiled business
edges compare the actual business arguments and retain the producer captures they
need; they do not require the reference's fixed number of pages. Complete-retrieval
contracts still require an actual matching query, every intervening cursor and
exhaustion at the observed total, including an empty result's single page. When a
later business action consumes a record from a later page, its witness retains the
cursor ancestors needed to obtain that record.

Stages declare a primary `kind`. Optional `allowed_kinds` permits variations such as
option selection versus correction while preserving the same tool argument and business
obligation. It must include the primary kind and cannot mix execution confirmation with
information-disclosure kinds. The blueprint and compiled scripts require interaction
diversity. Structural blueprint approval does not certify semantic completeness or reward
correctness; execution and independent semantic review remain required.

### Structured interaction requests

New multi-step tasks publish their interaction format in the agent's public materials.
The assistant asks a natural-language question with a structured request, for example:

```json
{"question":"Which region should I query?","request":{"intent":"request_input","tool":"lookup","arguments":["region"]}}
```

Option selection also supplies `request.options`; execution confirmation uses
`intent: confirm_execution`, the protected write tool, and an empty arguments list.
Internal script and stage IDs are not exposed. Authors provide natural-language
`reference_response`; the compiler wraps it in this protocol automatically.
`assistant_contains_all` is unnecessary in this mode. Legacy scripts retain phrase
matching. Tool prerequisites, parameter bindings, privacy and authorization checks
still apply. Text-based branch conditions inspect only `question`, so offered options
in protocol metadata cannot accidentally select a branch.

`task_spec.interaction_complexity.behavioral_diversity` distinguishes declared
decision values, interaction behavior (including corrections and branch conditions),
and compiled reference tool-call sequences. Wording changes and renamed stage IDs do
not count as behavioral diversity. `wording_only_variants` flags scripts that merely
paraphrase the same interaction; `same_terminal_decisions` also permits genuine
corrections that converge to the same final choice. These are diagnostics of the
declared contract and compiled references, not proof of executed rollout coverage or
learning value. Missing per-script reference evidence is listed explicitly.

Expanded source scenarios may declare `user_script_indices: [0, 2]` to select
zero-based interaction variants. Omission preserves expansion to every script.
Each variant must have a successful reference with its own disclosed argument values;
a reference scoped to a different variant cannot validate that binding. Scoped
negative scenarios must also set `derive_interactions: true`. All selectors must be
nonempty lists of distinct in-range integers. The compiler removes the selector after
expansion and keeps the usual `-script-N` scenario suffix for evidence mapping.
This selects executable references only; it does not automatically change a reward
rule. Rewards must still represent the actual user's choice, and all generated
references and negative probes must pass execution before the task is accepted.
For a query whose selected record varies by script, a reward can use the existing
read-time expression `{"tool_result":{"name":"read_temperature","path":"$.records[0].temperature"}}`
when the business goal is to report that observed value. This refers to the platform's
first successful result snapshot for that read tool, not an arbitrary final-state
constant. It is unsuitable for goals requiring a later/current value after state changes.
For complete structured read results, `tool_result` accepts bounded JSON arrays and
objects, for example `{"tool_result":{"name":"read_evidence","path":"$.records"}}`.
An observed empty array is valid; an absent result, null, nonfinite number, nesting
deeper than 12 or any collection larger than 64 is unresolved. Use a public answer
schema describing the same record structure and order. Do not expand a tree for each
possible result length. This only checks reported evidence; retain the metrics for
business decisions, eligibility and explanations.
To report a user's accepted query selection even when zero rows match, use
`{"tool_argument":{"name":"read_evidence","path":"$.theme"}}`. It reads the actual
argument from the first successful call matching the active script's process binding.
Never infer a user's choice from `records[0]` when empty results are legal. Argument
agreement alone cannot establish correct business analysis or meaningful tool use.
`from_tool` currently requires one unambiguous success-reference query. With distinct
`reference_variants`, use the script-bound checkpoint expressions above where their
read-time semantics match the business goal; keep later-state goals as state checks.

The chosen tool arguments and captured dependencies must match the current script;
other scripts' mutually exclusive capture plans are not simultaneous obligations.
Captured producer results must also satisfy their own active-script process bindings,
recursively through the dependency chain. Unrelated exploratory reads cannot anchor
a downstream reward snapshot, and returning to the correct chain remains valid.
Include scoped negatives that swap the answers between the choices. A valid expression
alone does not prove the tool implements the business rule correctly; independent
semantic review and the existing execution checks still apply.

In compact source, scope the original `reference` with `user_script_indices`, and add
up to eight `reference_variants` records with `id`, `user_script_indices`, `calls`,
`answer`, and optional `route_id`. Their IDs become success scenario IDs; the original
remains `reference_success`. Capture shorthand is expanded for each reference.
For a derived negative, `reference_id` selects its successful base (default
`reference_success`); the negative inherits that reference's script scope. Process
rewards are compiled per script, so another script's correct arguments cannot satisfy
the active script's process obligation.

New multi-step authoring with more than one user script requests
`require_behavioral_diversity`. It rejects wording-only variants and requires at least
two distinct executed behaviors among successful references. Different final choices
are useful but not mandatory: a genuine correction or authorization response may
converge to the same result. Merely declaring an unexecuted branch does not satisfy the
execution check. The verification report includes
`distinct_executed_interaction_behaviors`; this is finite reference coverage, not
coverage of arbitrary policy trajectories. Historical requests without this flag retain
their prior policy when explicitly continued from a checkpoint.

### Fixed-design continuation after interruption

Once a blueprint is accepted, generation saves `authoring_checkpoint/manifest.json`
with content-addressed request, blueprint and any parseable candidate source. A timeout
does not make the candidate accepted. The checkpoint records consumed structural and
semantic repairs, implementation attempts, prior issue IDs and cumulative author time.

Within the configured route-attempt limit, the main generator may continue a checkpoint
after an author timeout, CLI failure or missing completed turn. It retains the original
request, seed and business design, and writes each continuation under
`generation_retries/attempt-N/`. It does not retry a rejected business design indefinitely:
the three implementation-attempt limit and one repair per validation phase carry across
continuations. Each continuation gets the configured per-invocation author time budget;
the checkpoint separately retains cumulative time. Original failures remain failures.

Continuation requires a fresh author completion, full compilation/execution and independent
review. A successful continuation republishes relative data/tool files beside `task.json`,
preserving the previous files under the continuation's `previous_delivery/` directory.
The Python `generate(..., resume_from=checkpoint_path)` API requires a fresh output directory
and an unchanged normalized request; it never silently overwrites the failed run.

### Reproducible bulk data and bounded pagination

A table may add `row_recipe: {count, columns}` to generate synthetic rows before
its explicit `rows`. The columns mapping must cover every declared column. The
runner's request seed, table name, column name and zero-based row index determine
each seeded value using `sha256-index-v1`; no generated code is executed. Supported
scalar generators are:

- `{literal: value}` for a finite JSON scalar, including nullable values.
- `{sequence: {start: 1000, step: 1}}` for integer identities.
- `{string_sequence: {prefix: "record-", start: 0, step: 1, suffix: ""}}`.
- `{cycle: [values...]}` for repeating known categories or valid foreign keys.
- `{seeded_choice: [values...]}` and `{integer: {min: 0, max: 999}}` for seeded data.

Bounds are 100,000 total expanded rows, 32 columns per recipe, 2,000,000 generated
cells and 64 MiB of generated row JSON. Choice lists contain 1..100 scalars.
All expanded rows still undergo type, primary key, foreign key, unique and CHECK
validation. Supported CHECK predicates accept either the expression alone or
its SQL `CHECK (...)` wrapper, with identical planning and runtime enforcement.
An explicit `rows: []` is valid for an empty initial table, including an insertion
destination. Omitted rows are invalid; primary/foreign keys, CHECK predicates and
the actual final-state goal still apply to all inserted or existing rows.
Explicit business rows may supplement generated distractors; they
must have distinct keys. The compiled manifest records the recipe, seed, algorithm,
expanded count and content digest. Runtime loading rejects a changed row file
that no longer matches that digest. Recipe metadata is private construction
evidence, not an answer disclosed to the policy.

For a select tool, declare `pagination: {cursor_argument: "cursor", page_size: 100}`
and an optional open string `cursor` parameter. Page size is fixed at 1..100.
`order_by` must include every primary-key column, giving unique stable ordering;
the cursor argument cannot also be a business filter. The compiler publishes
these semantics in the tool description and output schema:

- Omit the cursor for the first page, then copy `next_cursor` while retaining the
  same query arguments. `count` counts this page, `total_count` counts all matches.
- Empty `next_cursor` and `has_more=false` identify the last page, including an
  empty result. Retrying an unchanged page returns the same page.
- Cursors must have been issued in this episode for this tool/query. Unknown,
  tampered or cross-query cursors fail with `PAGINATION_CURSOR_INVALID` (400).
- Cursors bind the content of the queried table and active resolver tables.
  Changed content gives `PAGINATION_CURSOR_STALE` (409); restart and discard old
  pages. A restored identical snapshot is compatible. Reset invalidates issued
  cursors; a new query receives new opaque tokens. Logical record order is seeded
  and reproducible, while cursor tokens are episode-local transport values.

Pagination and row volume alone do not prove a hard task. References must actually
use page captures, and final rewards must check the business result's completeness
and exclusions. Count alone is observable on the first page and does not establish
that later pages are needed. Profile coverage, action/context budgets and bounded
semantic evidence must be validated separately before claiming large-retrieval
coverage; never send a ten-thousand-row database to a semantic judge as a substitute
for scoped evidence.

### Shared answer field meanings

New multi-step requests set `require_answer_fields=true`. In `blueprint.json`,
declare `answer_fields` as `{field_name: {kind, meaning}}`; this public business
meaning and role are immutable during implementation/repair. In compact source,
define each field once under `answer_fields`. Its `schema.description` must equal
the blueprint meaning. Do not separately author `answer_contract` or answer target
expressions in outcomes. Deterministic outcomes select field IDs:

```json
{
  "answer_fields": {
    "current": {
      "kind": "observed",
      "schema": {"type": "boolean", "description": "Current flag of the first returned record"},
      "observation": {"tool": "read_dossiers", "path": "$.records[0].is_current"},
      "missing": false
    },
    "count": {
      "kind": "observed",
      "schema": {"type": "integer", "description": "Number of returned records"},
      "observation": {"tool": "read_dossiers", "path": "$.count"},
      "missing": 0
    },
    "unique": {
      "kind": "derived",
      "schema": {"type": "boolean", "description": "Whether exactly one record was returned"},
      "expression": {"op": "eq", "args": [{"answer_field": "count"}, 1]}
    }
  },
  "outcomes": [{"id": "facts", "rubric": "Report observations and uniqueness accurately",
                "fields": ["current", "count", "unique"]}]
}
```

`observed` permits only `kind`, `schema`, `observation`, `missing` and optional
`tolerance`/`array_order` comparison options. The compiler emits `observed_value`
with a `tool_result` and a literal missing value. Actual false, zero, empty strings
and empty collections or explicit JSON null remain observations; missing applies only when the path is
unresolved. Multiple records never by themselves overwrite an observed flag.
Observation paths use the runtime JSON-path parser, including `$` for the entire
tool response. Declare the complete bounded response schema when using `$`;
`$.` is not a valid substitute. A nullable deterministic answer field uses
`type: ["integer", "null"]` (or another concrete type plus `null`), retaining all
object/array constraints on non-null values. Public descriptions must distinguish
an unqueried null default from an observed empty result or observed null.
Other unions of the supported JSON types are permitted, for example heterogeneous
array items with `type: ["string", "integer", "boolean", "null"]`. Every object
or array branch still requires its complete bounded schema. Scalar ordering and
numeric tolerance options must remain compatible with all possible non-null types.
Internally, absent tools, missing paths and missing checkpoints remain unresolved;
submitting null cannot satisfy such an unresolved target. Only an explicit literal
or declared `observed_value.missing` rule can supply a null target without that read.
Such answer normalization never satisfies a required process or retrieval gate.
The path refers to the existing first successful process-matching read checkpoint;
use a separate tool/checkpoint when the business goal requires a distinct later read.
All origin and missing rules are public. Never put a private fixture answer in missing.
At the blueprint stage, every observed field must already declare the same
`observation: {tool, path}` binding to one declared business tool. The planner and
compiler use the same observation grammar; missing bindings, pseudo-tools, malformed
paths, multi-tool `observation.tools` and invented user/trajectory observation forms
fail before the blueprint is frozen. Implementation must preserve the chosen binding
as well as the field meaning and kind. Use separate fields for distinct observed
responses, supported derived expressions for decisions, and semantic explanations
for obligations grounded in actual user replies/tool statuses. This checks that a
binding can be represented; execution and independent business-meaning review still
establish whether it is the right evidence. Old unfinished blueprints without these
bindings require replanning; completed task artifacts do not depend on blueprint validation.

`derived` uses one `expression` and may reference other deterministic fields with
`{"answer_field":"name"}`. References are expanded from the same definitions;
unknown names, cycles and excessive expansion fail. Its schema description must
state the actual public decision rule. A role declaration does not prove that rule
matches the business goal; independent review and executed counterexamples remain
required. `explanation` has an open string schema and remains owned by an ordinary
semantic outcome, with explicit criteria and positive/negative calibration cases.
Stateful `business_contract.report_outcomes` also accepts `fields` references.

An explanation field may declare `context_fields: ["count", "unique"]`. Freeze
the same list in its blueprint entry. These must uniquely name observed or derived
fields, never another explanation. The compiler derives the semantic outcome's
context from its owned explanation fields; do not repeat `context_fields` in a
shared-field outcome. Legacy semantic outcomes may declare the list directly.
Each supporting field must have exactly one deterministic value target.

The runtime independently checks each supporting value against that target using
the actual executed state/read checkpoints and the same tolerance/array rules.
The judge receives only successful checks in `verified_answer_context`, alongside
the projected explanation and actual evidence. Failed checks are listed by name,
without their submitted values, and retain their deterministic penalty. A failed
unrelated fact does not automatically fail the explanation. Neither aggregate
metric scores nor submitted claims of verification establish a supporting fact.
Context is part of judgment cache identity and is rebuilt in calibration cases.

For large data, an explanation may additionally declare `evidence_mode: fields`,
also frozen in its blueprint. This requires nonempty `context_fields`; every
explanation owned by one semantic metric must use the same mode. The default
`runtime` preserves full declared runtime evidence. Do not repeat the mode in a
shared-field outcome. In `fields` mode, the judge receives `executable_field_facts`
computed from the deterministic targets and actual state/read checkpoints even
when submitted structured values are wrong. It also receives the owned prose,
public task, actual user turns and compact tool attempt statuses. Raw business
tables, tool arguments and result rows are excluded. This changes only the
semantic evidence transport: deterministic scoring still uses all actual data.
The evidence context has a 65,536-byte UTF-8 limit; overflow is an explicit error,
never silent truncation. Use sufficient bounded aggregates for large collections.

Unresolved targets are listed without fabricated values. `observation_availability`
records every referenced observed path, including inactive branches; a missing
path's default is normalization, not evidence that the business observation occurred.
Explicit null facts stay resolved; their availability flag distinguishes an observed
null from the same value supplied by a declared missing default. Reviewer replay
and parent failure diagnostics retain resolution separately from the JSON value.
`decimal_values` preserves exact decimal results alongside JSON numeric values.
The runner ignores submitted claims of executable facts. Independent review must
check that the declared premises cover every required business inference; omitting
a required premise to fit the bound is invalid. Mode changes invalidate judgment
reuse. This mode does not prove that the shared business rule itself is correct.

Criteria should state the inference still required, such as why multiple current
records prevent a unique selection. Correct structured facts can be referenced
without copying them into prose; they do not substitute for that inference. Include
two faithful paraphrases and negatives for contradiction and missing/incorrect
reasoning. Independent review must still check the original goal and reject a
wrong business rule shared consistently across generated artifacts.

The compiler derives the final answer schema, publishes `answer_field_semantics`
and binds each field to exactly one outcome. Do not repeat a structured fact as a
separate prose obligation unless the original goal requires it; preserve actual
reasoning obligations. Existing checkpoint requests retain their original contract
requirements. Shared fields currently require compact source authoring.

### Runner-owned difficulty profiles and complete retrieval

`request.difficulty_profile` is immutable through repairs and checkpoint resume.
The runner allocates `--difficulty-mix` before sampling candidates and records each
profile in `sample_manifest.json`. Hard profiles require `multi_step_agentic`;
incompatible category or environment-mode allocations fail rather than downgrade.
`task_spec.difficulty_requirements` is compiler-owned; never author or override it.

- `standard` preserves ordinary tasks.
- `long_horizon_planning` requires at least eight distinct business tools in an
  executed result-to-argument dependency chain, with an actually triggered fault
  and successful recovery in a passing reference. Repeated calls, unconsumed
  capability edges and `next_cursor` hops do not establish business depth.
- `large_scale_retrieval` requires at least 10,000 actual rows, reads across at
  least two business tables, bounded pagination, and a passing reference completing
  at least three distinct pages. Every select over a table of 10,000 or more rows
  must use pagination. Semantic explanations must use `evidence_mode: fields`
  with sufficient declared premises. Use reproducible `row_recipe` data, relevant
  distractors, cross-table predicates and complete aggregates; never truncate a
  requested answer to fit the existing 64-entry answer-expression bound.
- `complex_write_transactions` requires a passing stateful reference with committed
  changes to at least two tables and three distinct primary-key objects. Restored
  objects count as genuinely touched; external changes and no-op writes do not.
  Every write needs `require_idempotency`, `requires_grants`, and tool-bound
  `approval_terms` with `scope: terms`. Declare `realism_requirements.mechanisms`
  including `compensation` and name its actual writes in `compensation_tools`.
  A declared compensation tool must restore an outstanding changed field after
  an actual fault, and an executed idempotent replay must leave state unchanged.
  A successful compensation response alone is insufficient. Failed uncommitted
  calls must leave no write behind; response loss after commit is audited as a
  committed write and must not be confused with rollback. Final goals and unrelated
  state preservation still apply. The independent delivered-sandbox gate recomputes
  this witness from its own execution, rather than trusting the authoring report.
  This is a compensating workflow across atomic single-tool transactions, not a
  single globally atomic multi-tool transaction. Describe its business semantics
  accurately; recovery steps must be necessary for the original user goal.

For ordinary tasks that explicitly require full retrieval, declare
`complete_retrieval_tools: ["search_items"]` in source. Entries must uniquely name
paginated tools. Large-retrieval profiles automatically require this for every
paginated business tool. The compiler publishes the traversal obligation and
derives process bindings into `task_spec.retrieval_contract`; do not author this
compiled object. Runtime reward requires a complete chain starting from a query
that satisfies its actual process/capture/user-script binding. Later pages must
follow that same query's returned cursor, preserve total count, and together cover
the entire result. An unrelated completed query or repeated page cannot substitute.
Empty results complete in one page. Inactive route tools are not required.

Full-result retrieval and numeric correctness are distinct obligations: a known
correct sum with an omitted page does not receive full reward. Ablations freeze
business decisions but retain cursor capture references, because opaque cursors
belong to the current episode. Missing producers still fail; the runner never
manufactures a cursor or silently skips a page. Independent review must check
whether full traversal is part of the original goal, whether every page contributes
necessary evidence and whether filters exclude the right distractors.

Profile coverage is execution evidence, not a proof of business-rule semantics.
After full execution verification, hard profiles receive compiler-owned
`task_spec.execution_budget`. Actual successful reference responses and resolved
arguments are retained as trainer-only `policy_budget_references` in generation
verification, including independent routes. Do not author this evidence. It is
excluded from CLI feedback and policy sampling. The RL adapter uses its current
tokenizer to report serial-reference action/context/output sizing and rejects a
current prompt plus configured completion budget that cannot fit. Missing hard-profile
evidence requires rebuilding the artifact. One native action may contain multiple
ordered calls, so dependency depth and reference length do not prove a minimum
number of model actions. Reference capacity is advisory, includes transport/template
margin, and does not guarantee sufficient thinking tokens. Serial, concurrent,
batched and resumed rollout paths preserve this distinction without raising budgets
automatically. Business-meaning review still runs in the full production pipeline.

Each pipeline invocation writes `difficulty_summary.json` alongside
`realism_summary.json`. Difficulty groups are disjoint and use the runner's frozen
sample assignments, including failed generation requests. The overall denominator
includes untracked planned requests. Historical samples without an assignment stay
`unassigned`; free-text complexity labels cannot retroactively assign difficulty.
The report includes readiness, terminal/pending counts, reservation-to-terminal
latency and first recorded actionable validation feedback, each with observed
sample count, median and maximum. Failed generation now records its terminal
timestamp too. Missing historical timestamps remain missing rather than becoming
zero latency. Per-profile throughput is its ready-task contribution per shared
cohort wall hour, not a claim about isolated profile execution speed.
First feedback includes execution events, parent `actionable_feedback_at` markers,
and failed local blueprint-validation completion timestamps, including checkpoint
continuations. The per-sample report links the earliest source file and timestamp.
Generic failed model/review phases and author timeouts do not by themselves prove
an actionable business defect. Events outside reservation-to-terminal bounds are
excluded; missing feedback remains missing.

### Generation timing evidence

Business-condition counterfactuals preserve scheduled task events. If an event
restores eligible rows after the initial negative fixture mutation, the parent
replans the single-predicate intervention against actual query-time state. It
validates data constraints and unchanged remaining predicates, and verifies zero
eligible rows without changing the query operand. Harness changes are explicitly
marked `counterfactual` in replay; they do not count as successful reference realism
coverage. Only their changed fields are excluded from unrelated-write attribution.
Coupled or invalid mutations remain inconclusive and fail validation. Repair feedback
includes their condition, scenario, query state and event evidence; preserve the
business goal and dynamics rather than removing an event to satisfy a probe.

`code_agent_generation.json` records each attempt's `author_seconds` separately from
`phase_seconds`: `blueprint_validation`, `compile_and_execution`,
`delivery_verification`, and `independent_review`. Only phases actually attempted are
present; failed calls retain their elapsed time. Compilation may include semantic reward
calibration, so its duration must not be interpreted as CPU-only execution time. The
attempt's total `seconds` also includes orchestration and artifact I/O. These measurements
do not change the author or review timeout budgets.

During parent validation, `attempts[].active_phase` is published before the work
starts and removed when it returns or raises. The phase duration is then published
even on failure. This distinguishes a completed author turn from a still-running
compiler or reviewer; it is not a heartbeat or proof that a subprocess is alive.

`execution_progress.jsonl` additionally records start/end events for reference
scenarios, reward evaluator invocations, and individual semantic calibration calls.
Parent validation runs at most two calibration cases concurrently, with a separate
SQLite episode and evaluator cache for each case. Each case still obtains a fresh
judgment; identical cases are not coalesced. Contexts and expected labels stay fixed,
and reports/replay evidence are merged in contract order. A failed label or unavailable
judge emits actionable feedback immediately; acceptance waits for all cases. The shared
runtime bridge also bounds model concurrency across pipelines.
Each pair has an `operation_id`, a validation `run_id`, and elapsed `seconds` on
completion or failure. Resuming appends a new run instead of erasing earlier evidence.
These records omit answers and judge payloads. `completed` means the invocation
returned, not that its verdict passed; consult the validation reports for acceptance
and evaluator availability. Nested durations overlap and must not be summed as
end-to-end time. The timings include cache lookup or local evaluation when applicable;
they are not network latency, token usage, or monetary cost measurements.

Each attempt's `transport` records explicit CLI reconnect and HTTPS fallback notices,
including attempts terminated by timeout. `network_seconds` remains null because those
notices do not contain enough timing evidence to assign network duration. Zero notices
does not establish that the network was healthy. CLI command output or agent text quoting
a reconnect message is not counted as a transport event.

### Read checkpoint qualification

The first-successful-read rules above use compiled `contains_tool_call` process
bindings when present. A read must match the active script's arguments and capture
constraints before establishing its `tool_result` / `at_tool` checkpoint. An unrelated
query before the correct query therefore cannot poison the reward snapshot. Without a
compiled process binding, the original first-successful-named-read behavior applies.
After capture, later reads cannot replace the checkpoint. These rules preserve read-time
semantics; they do not convert a snapshot into a final-state query.

Requirement-bound negative scenarios may retain credit for other correct outcomes.
Their fresh replay must reject the bound outcome (normalized score at most 0.2), satisfy
all negative requirement bindings, and produce total reward below 1. Unbound partial
negatives must assert a final active outcome component at most 0.2 (with the exact
authored comparator enforced) and lose total reward. Without either form of evidence,
the total-reward ceiling is 0.2. Compiler and delivery use the same component rule;
a failing requirement binding cannot fall back to a different passing component.
An explanation-only defect needs its semantic outcome assertion or requirement
binding; asserting only total reward below 1 is insufficient. Authored assertions and independent
semantic review remain required; declaring a requirement does not certify its meaning.

### Options supplied by a tool

When a structured `option_selection`, `correction`, or `constraint_update` stage
chooses from values returned by a prerequisite tool, declare their origin rather
than copying the fixture choices alone:

```json
"options_from": {
  "tool": "find_catalog",
  "path": "$.records[0].supported_types",
  "separator": "、"
}
```

The producer must appear in the stage's `requires_tools` and must differ from its
consumer tool. Omit `separator` when the JSON path already returns a string array.
The latest successful producer response supplies the allowed choices; an unsuccessful
call does not replace a prior successful response. Choices are nonempty strings,
deduplicated for the menu without dropping business records. A projected menu with
one remaining option retains its collection type. It must contain the script's
private selection before a choice can advance.
An actual non-advancing refusal does not disclose that selection and can still execute
when the preference is unavailable. Missing or invalid option evidence keeps the
interaction pending. The agent's structured request must present the
observed choices, not an obsolete fixture list. Keep the static `options` declaration
for baseline fixture validation. This binding does not automatically replan a script
whose desired choice is no longer available; author a reachable alternative when the
business task requires recovery from that change.
If one variant binds an input to observed options, correction/update variants for
the same input must also declare `options_from`, baseline `options`, and the
producer dependency. Keep the correction kind and prior disclosed fact; changing
dialogue behavior must not turn a catalog-constrained choice into unrestricted text.

For menus that change in route probes, use an explicit reference question template:

```json
"reference_response": "实际查询结果：{{observed_result}}；可选编号：{{observed_options}}。请选择。"
```

This requires structured interactions and `options_from`. `{{observed_options}}`
is required; `{{observed_result}}` optionally displays the producer's entire actual
successful response, including names and other catalog details. The reference
runner renders the question and structured request options immediately before the
dialogue using only completed tool calls. It cannot inspect a private preference,
the underlying database, future calls or a probe's initial changes. Missing/invalid
observations produce an empty menu that the ordinary simulator rejects. The runtime
never rewrites a live agent's question, and a literal reference question stays literal.

### Disclosed user values before a downstream call

`{"user_value":{"stage":"selection"}}` evaluates to the scalar actually disclosed
by a successful advancing turn at that stage. It is unresolved before disclosure,
after a non-advancing refusal, and after reset. It never reads future private user
preferences. Use it in a checkpoint-based predicate that tests the **selected** key
for ambiguity before calling a downstream tool; unselected duplicates alone must
not trigger a stop. Every route using the value must require that disclosure stage.
The stage and its tool/argument binding must exist consistently across user variants.
An ambiguity-stop reference can include the disclosure while omitting its downstream
tool; interleaving places that required question before the final answer.

A direct `user_value` answer field can report a query's scope only alongside a
separate positive private-data outcome and when its bound parameter filters that
query. User preference alone does not establish business eligibility. Normal
runtime, preview, route probes and semantic review share these meanings.

### Presenting observed context before a choice

Reading a list internally is insufficient when the task asks to show it before
asking a question. Declare `context_from` on the stage, even if the options are
fixed topics rather than record identifiers:

```json
{
  "requires_tools": ["list_eligible"],
  "context_from": {"tool": "list_eligible", "path": "$.records"},
  "reference_response": "合格记录：{{observed_context}}。请选择研究主题。"
}
```

Structured questions must present `request.context` equal to the complete observed
JSON value from the latest successful producer response. Missing, partial, stale,
or differently typed values cannot advance the user protocol. Empty lists are valid
observations. Public `user_interaction_context` material describes the required
source for each question. Reference rendering supplies actual observed context and
may be combined with observed option templates; live agent messages are never repaired.

### Repair budget after observed transport retries

Fresh Codex blueprint invocations with at least 240 seconds of remaining author
budget receive a 120-second startup activity check. Thread/turn start events and
transport error notices alone do not count as substantive activity. A CLI item,
completed turn or changed candidate/protected workspace file keeps the same process
running against its original deadline. If there is still no such activity, the
runner stops that process group and retries the same request at most once, using
only the remaining author budget. No blueprint rejection or business repair is
charged for that silent invocation; its elapsed time, log digest, activity check
and invocation still remain in evidence. A second silent startup fails explicitly
as `GEN_STARTUP_STALL`, without launching another task or inferring a network cause.
Short-budget calls, implementation work and resumed fixed-blueprint work retain
their normal deadlines. This bounds unproductive startup waiting; it is not proof
of a provider outage or a guarantee that restarting improves a particular task.

For author budgets of at least 120 seconds, if an earlier invocation emitted a reconnect
notice and fewer than 120 positive seconds remain, the runner defers the next author
invocation. `deferred_author_invocation` records the remaining time and
`attempt_consumed: false`. The 120-second reserve is a scheduling policy, not a measured
network delay. The checkpoint retains the failed validation evidence and consumed repair
budget. The main generator can resume it within its existing route-attempt limit; a
standalone `generate` caller receives `CODE_AGENT_TIMEOUT` and must explicitly resume.
No acceptance gate or retry ceiling is waived.

Native CLI `error`/`turn.failed` events that explicitly report selected-model
capacity are classified as `CODE_AGENT_CAPACITY` infrastructure failures. Authoring
and scoring each permit at most one same-model retry after a five-second wait,
inside the original request budget; authoring charges that wait but no business
repair for an invocation without a completed candidate. Completed turns and text
inside assistant/tool items cannot trigger this classification. Exhaustion stays
a failure in the original denominator; no model fallback or verdict is invented.
Scoring writes provenance for failed attempts too, links retry request IDs and
returns only a sanitized failure code/request ID through the local bridge. Those
IDs allow correlation with native evidence without exposing prompts or credentials.
Availability feedback is reported separately from actionable business defects.

### Delivery checks with semantic rewards

The fresh-scaffold delivery preflight uses the real configured runtime evaluator
when any metric declares `external_llm_judge` or `hybrid_outcome`. Matching an answer
against offline fixtures cannot certify a semantic negative, especially when the
same wording is correct under another user's choices. Pure rule tasks retain the
90-second offline preflight; semantic preflight has a separate 900-second ceiling.
This replays fixed acceptance trajectories, not an agent-policy rollout.

Missing reports, evaluator timeouts and observed fallback judgments are infrastructure
failures (`SEMANTIC_CALIBRATION_UNAVAILABLE`), not author-repair instructions. Actual
judged failures still reject delivery. A successful `select` returning a consistent
zero count and empty record list is meaningful evidence; the private-data dependency,
causality and reward-sensitivity checks still apply. Reordering references is not a
substitute for checking these properties.

Before launching the scaffold preflight, generation also runs the same task
eligibility scorer used at construction entry. Observed numeric rewards are checked
with production read-time snapshots and bounded private-data mutations, including
query membership changes for zero counts. Typed semantic fields use the platform
factuality wrapper; they need not copy a particular legacy criterion sentence.
Missing evidence inputs, contradictory factuality instructions and undeclared fields
remain invalid. This qualification is not a sandbox build or policy-training result.

Offline semantic fixtures select successful reference answers for the active user
protocol. The compiled script inventory and each reference's reset seed identify its
script, so another user's valid choice cannot give an incorrect answer full credit.
Historical contracts without a script inventory use only the exact reset seed's
fixture when a protocol is active. Non-interactive legacy fixtures retain their
original matching behavior. These checks still report `semantic_verification: false`;
passing them does not replace real evaluator calibration or independent review.

## Pipeline concurrency

`run_pipeline.sh --count 10 --generation-workers 3 --build-workers 2` runs up to three independent authors and two sandbox builds. Both worker counts default to one. Generation still reserves the full batch and deterministic per-task seeds before scheduling; each task keeps its route, repair, and validation budgets. Only completed manifests with matching task checksums enter the bounded build pool. Per-task directories and container tags remain distinct. A build exception records an infrastructure failure for that task; it does not discard other samples or remove a failure from the denominator. Pipeline interruption cancels pending launches and terminates owned process groups. Measure throughput on a complete batch; concurrency alone does not improve qualification criteria or imply linear speedup.

## Defined interactive inputs at qualification

Phrases such as “先问我” do not by themselves prove that a state-changing task has undefined inputs. File qualification can establish concrete inputs by validating delivered interaction scripts and replaying every successful reference prefix using production guards. Declared user facts must bind to real tool parameters and match the resolved arguments, with coverage across script variants. Missing files, invalid bindings, incomplete dialogue, unavailable captures, and uncovered variants retain the unresolved-input rejection. In-memory scoring without a task root remains conservative. Other qualification, reward, and independent semantic-review gates still apply.

## Meaningful repairs

The parent records the canonical JSON digest of a rejected source. Returning the same source after whitespace or key-order changes stops with CODE_AGENT_REPAIR_NO_CHANGE before another expensive execution or judge pass. The original defect IDs and repair budgets remain in evidence and checkpoints. A changed candidate still undergoes every original gate; changing bytes is not proof that the defect is fixed. Independent-review claims must be checked against actual source evidence. Do not resubmit an unchanged candidate hoping for a different judge verdict.

For query tasks that promise obstacle reporting, zero or ambiguous matches require a valid conditional route. Downstream choices, captures and reads apply only when their prerequisites exist. Verify those routes with data-changing probes. Semantic criteria must track actual retrieved evidence; do not require the initial fixture’s conclusion after the supporting evidence changes.

Semantic judges that declare tool_results also receive observed_tool_failures from already-recorded injected-fault events. These contain the affected tool, status, occurrence, phase and commit outcome. The future fault schedule and unused private identifiers are excluded. Recovery explanations should reflect these actual outcomes; a successful retry does not erase the earlier failure.

### Local validation judgment reuse

A single in-process live delivery validation may reuse a completed model judgment
only for identical metric, rubric, criteria, schema, runtime evidence and evaluator
configuration. It still executes each business trajectory and mutation. Changed
answers or evidence require a fresh judgment; fallbacks are never reused. The report
records `semantic_judgment_reuse` with the original judgment identifiers. Reused calls
are not independent model samples. Generated sandbox servers and normal training
episodes keep episode-isolated judgments by default; separate validation invocations
do not share this cache.

Private-data sensitivity probes compare tool HTTP statuses and complete business
payloads across repeated runs. A platform error envelope's per-request correlation
ID is excluded from this comparison; its error code, message and details remain
significant. Business fields with the same name are not removed. A transient fault
followed by a valid retry can therefore establish stable business evidence without
hiding real result changes or accepting a stale answer.

## Code Agent runtime evaluator

Task authoring, sandbox implementation and independent source review already use
Code Agent sessions. Set `SANDBOX_LLM_BACKEND=codex` to route semantic scoring,
independent calibration and model-driven user simulation through an authenticated
host service as well. Each request starts a fresh, read-only Codex CLI session with
shell, search and delegation disabled. Native completion and the original response
schema are required. A failed invocation is an evaluator availability error, never
a passing judgment; this backend does not fall back to an external model API.

Start the host service from the repository root in a persistent terminal:

```sh
.venv/bin/python -m env_factory.code_agent_runtime \
  --host 127.0.0.1 --port 55794 --model gpt-6.1-sol \
  --workers 6 --timeout 180 --output output/code_agent_runtime_service
```

The service writes `connection.json` and a mode-600 `token` file. Configure local
`.env` with backend `codex`, model and allowed response model `gpt-6.1-sol`,
`SANDBOX_LLM_BASE_URL=http://127.0.0.1:55794/v1`,
`SANDBOX_LLM_CONTAINER_BASE_URL=http://host.docker.internal:55794/v1`,
`SANDBOX_LLM_TIMEOUT_SECONDS=190`, and the token as `SANDBOX_LLM_API_KEY`.
Do not put the token in source control, command arguments or generated artifacts.
Restarting with the same output directory reuses the token; SIGTERM stops the
service and its active evaluation processes. An authenticated `GET /health`
reports the backend and configured model. Keep this service running for the whole
pipeline batch. Size its bounded worker pool for concurrent pipeline consumers.

Docker Desktop can reach this loopback service using `host.docker.internal`;
container launchers substitute the container URL. Other Docker installations
need an explicitly reachable host endpoint; a Linux host-gateway alias alone does
not expose a service bound only to host loopback. Codex login credentials remain
on the host. The container token permits access only to this fixed-model JSON
service. Per-request evidence records the requested CLI model, completion event,
usage and hashes. CLI completion does not independently report a provider model
revision; the recorded model identifies the explicit CLI argument.

Deterministic compilers, reward rules and structured interaction transitions stay
programmatic. Equal models do not establish semantic consistency: authors and
judges retain separate sessions, observed evidence, and positive/negative checks.
The live RL policy is outside this production-pipeline migration; existing live
certification requirements for rollout/evaluator separation remain in force.

## Feedback and validation scheduling

### Virtual time

Every sandbox exposes read-only `GET /v1/time`: `virtual_datetime` and
`origin_datetime` in ISO 8601 UTC, `tick`, `elapsed_seconds`, `tick_seconds`, and
`advances_on_tool_attempt`. The default origin is `2026-01-01T00:00:00+00:00`.
Set `business_lifecycle.clock: {"origin_datetime": "2026-10-08T09:00:00+08:00"}`
with lifecycle `version: "1.0"` to use another fixed origin. The origin must have
a timezone and whole-second precision; it never follows host time or the seed.
Clock configuration alone leaves time static. A nonempty `dynamics` config enables
the existing tick policy: one tick per validated, noncached tool attempt, including
business failures and injected faults, at `dynamics.tick_seconds` per tick (default 1).
An idempotent replay of a committed request does not advance time. Reading time
does not fire events, append replay events, change business data, or run a judge.
Reset restores the baseline; episode selection and snapshot restore preserve isolation.
RL exposes this endpoint through a native time tool and injects Shanghai time at reset.
Do not author a competing wall-clock business tool or a hidden wait operation.

### Early contract diagnostics

Select tools with no `projection` and no `projection_aliases` return every declared
table column. Their output schema derives those columns and types from the table;
an alias-only select returns only its declared aliases. Capture paths and downstream
types undergo the same qualification during preview as during final delivery.

After a user value is actually disclosed, all calls using its `bind_to` or
`reuse_bindings` argument must preserve that value. The runtime returns
`409 USER_INPUT_MISMATCH` before the business handler for a different typed value.
A correct retry is allowed. Do not classify an unrelated successful read with a
different bound user argument as a positive recovery probe; no such records are
exposed. This rule does not constrain unrelated tools or unbound arguments.

For an open string answer mixing observed values with literal stop/empty-result
markers, publish the exact marker and its condition in the field description or
public instructions. For example, distinguish an unqueried field from an observed
empty result. Unpublished literal markers fail with `ANSWER_FALLBACK_UNDISCLOSED`.
The compiler never publishes a private reward literal automatically. Preserve
observed facts on normal routes and keep qualification facts separate from unique
identity or overall eligibility decisions.

Compilation first checks executable parameter consumption, then runs a model-free
pass over reference and isolated stop-route scenarios. Disclosed parameters count
as used only when the compiled process binding enforces the actual user value;
lifecycle guard operands also count. The cheap execution pass uses conservative
reward upper bounds. Unknown semantic evidence is deferred, and its report always
has `training_ready: false`. Structural preview remains a structure/data check.

Fresh multi-step requests require dynamic choice coverage. For each distinct stage
with `options_from`, execute a reference or route probe with a nonempty actual menu
that excludes the script's private preference. A declared non-advancing branch must
actually execute and complete a valid stop or recovery route with full reward.
Empty menus, a refusal while the preference remains available, a stale menu, or an
unexecuted branch declaration cannot establish coverage. Use legitimate fixture
changes and user behavior; never invent a choice or override menu membership.
The deterministic pass detects missing cases before model grading; the full pass
counts only passing scenarios. Model-dependent unresolved trajectories defer this
check to full execution. This is finite coverage, not proof over all possible menus.

Full reference and route validation uses at most two independent workers per task,
with separate episode databases and probe directories. The runtime bridge bounds
model concurrency across tasks. Independent branch failures are collected into
`validation_findings.json` and one repair feedback package. Dependent ablations are
not attempted on broken reference trajectories. Final delivery and independent
source review remain required after these checks.

Semantic judgments return criterion IDs, satisfaction flags and brief observable
evidence. Full credit conflicting with a failed criterion is rejected. These are
assessment evidence, not hidden reasoning. Calibration includes this evidence in
its failed-case feedback.

One runner-owned in-memory cache can span author repair attempts. Reuse requires
identical metric ownership, rubric, criteria, answer schema, normalized observed
context, runtime code and provider configuration. Author-writable files are never
loaded as cache authority. Calibration still requests fresh judgments; independent
final delivery receives no repair cache. Opposite labels for the same dependency
key invalidate reuse, retain both judgments and prevent acceptance for that key.
A resumed author process starts with an empty cache.
Concurrent validation requests with the identical dependency key share one in-flight
judgment. Different keys remain concurrent; transient fallback results are not cached,
and fresh calibration judgments are never coalesced.

`execution_progress.jsonl` timestamps actionable findings as they appear;
`code_agent_generation.json` records author/repair and validation phase windows.
Compare full-cohort yield together with first-feedback latency, total elapsed time
and training-ready tasks per hour. Historical diagnostic controls do not enter a
fresh cohort's yield denominator.
