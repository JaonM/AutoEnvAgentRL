# Reusable environment and task definitions

`compile-definition --environment environment.json --task task.json --output DIR`
compiles two version `1.0` documents to the existing executable `2.0` contract.
The output directory must be new. Compilation does not qualify the sandbox or
replace independent acceptance against the original request.

The environment owns `id`, `tables`, `rules`, `tools`, `time` and optional
`invariants`, `events`, and `communications`. The task owns `id`, `category`, `request`, `user`,
`scenarios`, `probes`, `limits` and optional `answers`, `reports`, `messages`, `explanations`,
`success`, `goals`, and `demand`. Both documents require `version`.
Task documents cannot override environment fields. Environment, task and compiled
contract identities are recorded separately in `compilation.json`.

## Derived collection reports

A task may declare a report instead of repeating a query's eligibility expression
in an answer specification:

```json
{
  "id": "matching_records",
  "meaning": "All matching records",
  "query": "lookup",
  "arguments": {"size": {"ref": "user.size"}},
  "columns": ["id"],
  "weight": 1
}
```

The compiler derives an unordered array of typed objects, taking the field schemas
from the environment table. It selects the complete business collection using the
query's named rules and projects only published columns. It does not use observed
tool output as the expected answer, and it does not truncate truth to one page.
The task's reference trajectory must still retrieve the data; ordinary execution,
authorization, observation and negative-case checks remain in force.

Reports currently describe unordered complete collections, not ranked subsets or
transaction completion. `answers` remains available for existing contracts;
Transaction goals may use the declarative plans described below. New generation requests use an envelope with exactly `environment`
and `task`; both JSON transports and the author workbench support this envelope.
The build pipeline stores `definition.json` and `compilation.json` before tool
augmentation. `source.json` contains the executable, potentially augmented contract.
Repair feedback preserves the original envelope rather than converting derived
reports back to handwritten answer expressions. Existing flat contracts remain
accepted for historical reproduction and repair.
These are implementation limits, not grounds to reduce task difficulty or bypass
the independent business verifier.

## Transaction outcome plans

A task goal may be `{id, rule_ids, transaction_plan: [{tool, arguments, when?}]}`
instead of a handwritten `predicate`. `tool` names an environment transaction;
`arguments` are expressions over `initial`, `user`, and `user_state`. Optional
`when` is a boolean over the same roots and defaults to true. The plan states the
intended business actions, not the actions the agent happened to execute.

The evaluator starts from the actual initial world, applies the referenced
operations in declaration order with the intended parameters, and compares all
tables and columns against the final world, ignoring storage order. Operations
may update existing records or create new records. A disabled intention leaves
the projected state unchanged. Actual committed-result messages cannot substitute
for final state. Process checks still enforce evidence and authorization.

Plans deliberately do not infer commit times. Environments with scheduled events,
or operations depending on time or consent lifecycle state, must use explicit
state predicates. This restriction applies to this projection form only: keep
these capabilities and task difficulty in the sandbox. Do not delete events or
time dependencies merely to use a plan. Independent acceptance must still verify
that the intended operations and their definitions implement the original request;
matching an incorrectly authored operation is not independent business proof.

## Shared communication obligations

Environment `communications` contains `{id, rule_id, obligation, when?}`. Each entry
states one communication obligation grounded in a public rule. Optional `when` is
a boolean expression over authoritative `data`, `initial`, `user`, `user_state`,
and `time`; it cannot depend on submitted answers or collected evidence. Omitted
`when` means always applicable. Use conditions only for branches required by the
original business specification, never to hide an unmet obligation.

Task `messages` contains `{id, meaning, obligations: [communication_id],
context_fields: [answer_id], weight}`. The compiler derives semantic explanations
from these references. A task cannot rewrite the referenced obligation. Message
fields are graded semantically, accepting faithful paraphrases; they do not carry
an exact expected sentence. Keep identifiers, required verbatim literals, numbers,
sets and other structured facts in deterministic `answers`/`reports`.

Only applicable criteria are sent to the semantic judge. When none apply, the
component is satisfied without a model call; evidence records the inactive IDs.
All structured answer, process, safety, and world-outcome checks still apply.
The independent verifier must check that conditions faithfully implement the
original requirement and cover both applicable and inapplicable branches.

## Independent execution challenges

Independent review cases name a base scenario but may supply `steps_json` as a
JSON array to construct a different trajectory using existing tools and user
stages. JSON `null` (or omission in historical evidence) inherits the trajectory;
`[]` deliberately performs no actions. New trajectories are validated against
the executable grammar and run against the unchanged contract, with separate
execution and replay. They cannot inherit the base scenario's expected runtime
error: an accidental exception is not evidence of correct reward discrimination.
Original-request interpretation remains independent of authored rules and answers.
This capability permits missing-branch probes; it does not itself prove complete
coverage or correct business interpretation.
