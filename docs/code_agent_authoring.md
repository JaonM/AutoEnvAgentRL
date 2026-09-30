# Code Agent business authoring contract v1

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

Write only `version`, `description`, `environment_plan`, `tables`,
`business_tools`, `outcomes`, `reference`, `answer_contract` for JSON answers,
and `semantic_goal` for stateful tasks.
The detailed types for descriptions, tables, rules and goals appear below.

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
- Optional `negative_scenarios`: additional full scenarios as described below.
- `answer_contract`: required whenever an outcome uses `value_targets`.
  `{format: "json_object", schema: {type: "object", properties: {field_name:
  {type: "string", description: "meaning of this answer field"}}, required:
  ["field_name"], additionalProperties: false}}`. Use the correct JSON types:
  string, boolean, integer, number, array (`items` required), or nested object.
  Every public field must have exactly one reward target. Objects require all
  declared keys and reject extras. Arrays optionally use minItems/maxItems up to
  64. Public categorical enums may give 2..64 possible choices, never just the
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
  eq/ne/gte/lte/contains/exists/count_gte/count_eq/none_tool_calls/numeric_targets/value_targets.
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
  Arrays are compared element by element, with strict JSON types. Conditional results use
  `{if: {condition: expression, then: expression, else: expression}}`.
  Compact syntax also accepts raw scalars in expression positions, e.g.
  `{op: "eq", args: [lookup_expression, "cleared"]}` and `then: "CLEAR"`.
  The compiler wraps these as literals without changing their meaning.
  Multiple `value_targets` outcome metrics may check separate fields of the same
  JSON answer. Always use `path: "$"`; each target's `key` selects its field.
  The answer must contain exactly the union of deterministic target keys and semantic outcome fields.
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
The compiler rejects numeric write arguments and literal goal values copied from
private fixture rows without a matching public numeric value (`STATE_VALUE_UNGROUNDED`).
This check runs before the answer schema is published, so schema metadata cannot
launder a hidden answer into public evidence. It is a necessary origin check;
it does not prove that a publicly mentioned number is the right business target.
Reference `$expr` arguments permit captures, literals, operators and conditions;
they cannot privately query business tables. Computed write arguments are covered
by reference execution and the dependency gate. Computed select-filter bindings
are not yet supported by `from_tool` reward lowering.

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

For a reward on persisted state, use `source: "business_state"`, `path: "$"`,
`operator: "state_predicates"`, and `expected` as a list of row predicates with
`table`, `where`, `values`, `value_expressions`, and `count`, using the same shape
as the semantic goal. This selects business rows explicitly and supports dynamic
targets. Ordinary `eq` compares literal values; it does not evaluate an object
placed in `expected`. Keep separate answer rewards when the public task also
requires reporting the result.


### Conditional writes

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

Every case supplies a complete JSON answer matching the public answer schema,
including any fields owned by deterministic metrics. Each field has exactly one
reward owner. Semantic fields are strings without an enum. Provide at least two
meaningfully distinct correct paraphrases and two incorrect or incomplete answers;
cases must differ in the owned fields. The compiler publishes rubric and criteria
and uses the platform runtime judge. Calibration cases are not judge instructions
or reference answers available to the policy. The actual judge must classify all
cases correctly before construction. Mock matching or network fallback is not
semantic evidence; infrastructure failure does not trigger a task-author repair.

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

Live rollout eligibility measures verified execution, clean environment behavior,
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
