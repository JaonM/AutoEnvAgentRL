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
report columns must match requested output scope. A tool can expose extra fields
without requiring the agent to repeat them in its answer or earn reward for them.
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

For conditional consent in event-free, time-independent transactions, an intention
may declare `completion: "consent_or_unavailable"` instead of `when`. Intended
arguments may read initial data and private user facts, never actual consent state.
Target existence and the transaction's own boolean preconditions determine whether
the operation is available in the projected world. An unavailable operation leaves
it unchanged and requires no pointless approval. An available operation requires
either the exact intended successful commit or explicit refusal/revocation bound
to the latest matching proposal. Approval without execution cannot complete it.
All successful commits must match the plan; all final tables must match projection.
This derives precondition eligibility, not general feasibility: post-commit
invariants still apply and independent review must detect incomplete guards.
Time-dependent guards and event-driven worlds retain timeline or explicit goals;
do not remove time behavior to use this mode. Process requirements must likewise
follow the original request rather than unconditionally requiring inapplicable
consent stages.

For scheduled events or time-dependent operations use `transaction_timeline`
instead of `transaction_plan`, with the same intention entries. This form matches
each enabled intention to the next successful transaction in the actual trace by
tool and independently evaluated intended arguments; missing, extra or different
transactions fail. Only the actual commit clock is used, never result summaries or
actual final state as the expected state. Due events apply before each commit;
remaining events apply through episode end. Equal-time events preserve declaration
order. Monotonic tick and elapsed time are required. Transaction arguments and
conditions still cannot copy actual tool arguments or final state.

Operations depending on consent lifecycle state still use explicit state
predicates. This restriction applies to this projection form only: keep these
capabilities and task difficulty in the sandbox. Do not delete dependencies merely
to use a plan. Independent acceptance must still verify
that the intended operations and their definitions implement the original request;
matching an incorrectly authored operation is not independent business proof.

## Bounded author recovery

If a host author turn completes within its allowed tool scope but reports
incomplete work or lacks a current successful check, the pipeline retains its
data-only draft for the next existing repair attempt. `author_draft.json` and
`feedback.json` record that handoff. A check receipt is attached only when its
source identity matches the retained draft. A retained draft is not qualified;
the next attempt still requires compilation, independent review and execution.
Recovery adds no attempts. Process failures, timeouts and failed transport turns
remain infrastructure failures, not requests to weaken the business contract.

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

For contracts carrying cited demand requirements, independent review is
bidirectional. `requirement_reviews` trace every original clause to implementation
and executed cases. `obligation_reviews` trace every cataloged mandatory answer
(including nested properties), communication criterion, state goal and explicit
success constraint back to original clauses and cases. Unsupported obligations
are rejected even when forward coverage and execution pass. Necessary derived
outputs and decision evidence can be justified by business meaning; exact word
matches are not required. Table membership alone is not a justification.
This uses the existing independent review call, not an extra scoring stage.

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

Independent reports include `communication_coverage` for every conditional
criterion. Coverage records applicable/inapplicable case IDs only when execution
and replay agree on the evaluated condition. Failed executions and historical
evidence without applicability are unobserved. One observed branch is a coverage
gap to investigate against original requirements and reachability, not an automatic
rejection or permission to weaken the condition. Two observed branches also do
not prove that the condition implements the correct original business meaning.
