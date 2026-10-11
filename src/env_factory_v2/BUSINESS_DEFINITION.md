# Reusable environment and task definitions

`compile-definition --environment environment.json --task task.json --output DIR`
compiles two version `1.0` documents to the existing executable `2.0` contract.
The output directory must be new. Compilation does not qualify the sandbox or
replace independent acceptance against the original request.

The environment owns `id`, `tables`, `rules`, `tools`, `time` and optional
`invariants` and `events`. The task owns `id`, `category`, `request`, `user`,
`scenarios`, `probes`, `limits` and optional `answers`, `reports`, `explanations`,
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
transaction goals and semantic communication obligations are not yet derived by
this compiler. New generation requests use an envelope with exactly `environment`
and `task`; both JSON transports and the author workbench support this envelope.
The build pipeline stores `definition.json` and `compilation.json` before tool
augmentation. `source.json` contains the executable, potentially augmented contract.
Repair feedback preserves the original envelope rather than converting derived
reports back to handwritten answer expressions. Existing flat contracts remain
accepted for historical reproduction and repair.
These are implementation limits, not grounds to reduce task difficulty or bypass
the independent business verifier.
