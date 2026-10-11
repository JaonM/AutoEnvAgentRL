"""Prompts for the model roles. brief.md is the only semantic authority for all of them."""

SCHEMA = """You define the answer contract for a business task. Read brief.md (data.json and
facts.json hold the initial tables and the user's private facts).

Write answer_schema.json: the JSON object the agent must return to the user at the end.
Format: {{"fields": {{"<name>": {{"type": ..., "description": ..., ...}}}}}}
- types: string, integer, number, boolean, array (needs "items"), object (optional "properties");
  "nullable": true when a value can legitimately be absent; arrays may set "order": "any"
  when order carries no meaning.
- Cover exactly what the user asked to learn or be told, nothing decorative. Express every
  outcome with ids, integers, booleans or string enums ("enum": [...]) so that two correct,
  independent implementations produce identical values. Every field needs a precise
  description, including units and how absent/failed cases are reported.
- Free text (an explanation, reason or message) can never be compared exactly. Avoid it; if
  the user must get one, add at most one such field with "graded": false and put the actual
  outcome in a graded field (for example a "failure" enum next to it). The same applies to
  string properties inside objects and array items.
- Do not decide the business outcome here; only its shape.

Run `{python} -m env_factory_v3.checks schema` until it prints PASS. Do not read or write
files outside this folder. End with one line naming the fields.
"""

AUTHOR = """You build an executable business environment for agent RL training, in plain Python.

Folder: brief.md (business rules and user request; the only authority), data.json (initial
tables), facts.json (the user's private facts; an agent learns them only by asking),
answer_schema.json (the required final answer), request.json (all of the above).

Write env.py:
    from env_factory_v3.runtime import write, BusinessError

1. make_tools() -> dict of name -> function.
   - Read tool: fn(db, now, **args) -> JSON. Give each tool a docstring that tells an agent
     what it does and what its arguments mean.
   - Write tool: decorate with @write; fn(db, now, **args) mutates db in place and returns a
     JSON receipt. Raise BusinessError to reject; the runtime rolls the write back.
   - Tools enforce every rule in brief.md on whatever data they receive. They run on
     perturbed worlds, so never depend on particular ids or values from data.json.
   - Expose what an agent needs to do the task through tools; never expose facts.json.
2. reference(api) -> dict: a deterministic agent that handles the request correctly using only
   api.ask(fact_key), api.call(tool, **args) and api.confirm(approval_key, proposal).
   Before every write: token = api.confirm(approval_key, {{"tool": name, "args": exact_args}}),
   then api.call(name, consent=token, **exact_args). confirm returns None when the user
   declines; then that write must not happen. Catch BusinessError where the rules allow
   partial success. Return the answer exactly in answer_schema.json's shape.

Keep table names and row columns exactly as in data.json; new rows follow the record format
in brief.md and the columns of existing rows of that table. Standard library only (no I/O,
no randomness, no clock: virtual time is the `now` argument).

Run `{python} -m env_factory_v3.checks env` and iterate until it prints PASS. Do not read or
write files outside this folder. End with a two-line summary.
"""

ORACLE = """You are an independent auditor. Compute, from first principles, the outcome of a
correct handling of the user's request.

Folder: brief.md (business rules and user request; the only authority), data.json, facts.json,
answer_schema.json (the required answer shape), request.json.

Write oracle.py:
    from env_factory_v3.runtime import InvalidWorld   # optional

def expected(data, facts, now) -> {{"answer": ..., "final_state": ...}}
- final_state: every table after all operations a correct handling performs. An operation
  happens only if the rules allow it at that point AND the user's matching approval fact is
  true. Committed operations stay committed unless the rules say otherwise.
- answer: what the user must be told, exactly in answer_schema.json's shape.
- It runs on perturbed worlds and flipped approvals: never depend on particular ids or
  values. Raise InvalidWorld only when the data itself breaks an invariant stated in
  brief.md (e.g. a dangling reference); never to avoid a hard case.
Keep table names and row columns as in data.json. Pure function, standard library only.

Run `{python} -m env_factory_v3.checks oracle` and iterate until it prints PASS. Do not read
or write files outside this folder. End with a two-line summary.
"""

ARBITER = """Two independent implementations of the same business disagree.
- env.py: environment tools and a reference agent
- oracle.py: an independent computation of the expected outcome
- mismatches.json: the worlds where they disagree, with both answers and state differences
brief.md is the only authority.

For each disagreement decide which side violates brief.md and fix that file (both if both are
wrong). Never special-case ids or worlds; keep both interfaces; do not edit brief.md,
data.json, facts.json, request.json or answer_schema.json.

Write verdict.json:
{{"decisions": [{{"issue": "...", "wrong_side": "env" | "oracle" | "both" | "ambiguous",
                 "worlds": ["<world names from mismatches.json this decision covers>"],
                 "brief_quote": "<exact text from the user request or business rules that decides it>",
                 "reason": "..."}}]}}
Each brief_quote must be copied verbatim from the "User request" or "Business rules"
section of brief.md (never from the answer schema). If brief.md genuinely supports both
readings, use "ambiguous", quote the ambiguous text and change nothing for that issue: an
ambiguous business rule must be clarified by its owner, not decided by you. Always list the
worlds each decision covers; the platform drops perturbed worlds whose rules are ambiguous.

Run `{python} -m env_factory_v3.checks diff`, `{python} -m env_factory_v3.checks env` and
`{python} -m env_factory_v3.checks oracle` until they pass (or only ambiguous issues remain).
Do not read or write files outside this folder.
"""

AUDITOR = """You audit one kind of decision in a business task: what happens after the user refuses
an approval. brief.md is the only authority (data.json and facts.json hold the original world).

branches.json lists refusal cascades: worlds where the user refused approval `denied`, and the
implementation then never took the later steps in `skipped_steps` (confirmations and tool calls
with their arguments) that it does take when everything is approved. Each entry shows the
approvals asked, the committed writes and the final answer in both worlds.

For each entry decide whether brief.md requires those later steps to be dropped once `denied`
is refused (read-only lookups the dropped steps needed go with them):
- "supported": brief.md says so (a rule makes the later step depend on the refused one, or the
  user explicitly states what to do when that step is refused or does not happen).
- "unsupported": brief.md requires the later steps to be offered anyway (for example the steps
  are authorized or validated independently).
- "ambiguous": brief.md does not decide it. Ordering words alone ("do A, then B", "after A, B")
  describe the planned sequence, not what happens when A is refused; if they are the only
  support, the verdict is "ambiguous".
- Steps that only follow up on the refused write itself (verifying or re-reading what it would
  have changed) are naturally dropped: "supported", quoting the rule that requires the approval.

Write audit.json:
{{"branches": [{{"world": "<world from branches.json>", "verdict": "supported" | "unsupported" | "ambiguous",
                "brief_quote": "<exact text from the user request or business rules that decides it, or the ambiguous text>",
                "reason": "..."}}]}}
Cover every entry. Each brief_quote must be copied verbatim from the "User request" or "Business rules"
section of brief.md (never from the answer schema). Do not write any other
file and do not read or write files outside this folder. End with one line per world.
"""

BRANCH_REPAIR = """An independent audit found a mistake env.py and oracle.py share: after the user
refuses one approval, both drop later steps that brief.md requires.
- branches.json: each refused world, the steps the implementation skipped there, and both runs
- audit.json: the auditor's verdict per world; fix the worlds marked "unsupported", following
  the quoted rule and reason
brief.md is the only authority.

Fix env.py's reference agent (and its tools if needed) and oracle.py so that, after a refusal,
everything the rules still require is handled: later steps are offered for their own approval
and carried out when approved and valid, and the answer reports them. Change nothing else.
Never special-case ids or worlds; keep both interfaces; do not edit brief.md, data.json,
facts.json, request.json, answer_schema.json, branches.json or audit.json.

Run `{python} -m env_factory_v3.checks env`, `{python} -m env_factory_v3.checks oracle` and
`{python} -m env_factory_v3.checks diff` until all three print PASS.
Do not read or write files outside this folder. End with a two-line summary of the fix.
"""

REPAIR = """The environment in this folder failed qualification. gate_report.json lists the
failed gates with evidence. brief.md remains the only authority.

Fix env.py (and only env.py) so the failures are resolved for the right reason. Do not weaken
business rules, special-case ids or worlds, or change answer_schema.json.
Run `{python} -m env_factory_v3.checks env` until it prints PASS.
Do not read or write files outside this folder. End with a two-line summary of the fix.
"""
