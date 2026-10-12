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
- Do not decide the business outcome here; only its shape. Descriptions define meaning, units
  and how an absent value is written; they must not prescribe control flow (what happens after a
  refusal or failure, whether later steps still run, which rule wins) beyond what brief.md says.
  An enum may list possible outcomes; it must not imply which branch the business takes.

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
   - Read tool: fn(db, now, **args) -> JSON. The signature and docstring are the only tool
     description an agent sees: annotate every argument with its type (str, int, float, bool,
     dict, list[...], or X | None) and give the docstring a summary plus an `Args:` section
     with one `name: meaning` line per argument (units, id format, allowed values).
   - Write tool: decorate with @write; fn(db, now, **args) mutates db in place and returns a
     JSON receipt. Raise BusinessError to reject; the runtime rolls the write back.
   - Tools enforce every rule in brief.md on whatever data they receive. They run on
     perturbed worlds, so never depend on particular ids or values from data.json.
   - Expose what an agent needs to do the task through tools; never expose facts.json.
2. approval_key(tool, args, facts) -> str (only when facts.json has approval topics): the
   approval topic in facts.json that governs this exact write, derived from the arguments and
   the user's facts (for example: the loan is the first id the user listed -> approve_first).
   A user talking freely approves proposals, never topic keys, so this mapping must follow the
   business meaning of the write, not the order calls happen to arrive in. It runs on the
   platform side only.
3. reference(api) -> dict: a deterministic agent that handles the request correctly using only
   api.ask(fact_key), api.call(tool, **args) and api.confirm(approval_key, proposal).
   Before every write: token = api.confirm(approval_key, {{"tool": name, "args": exact_args}}),
   then api.call(name, consent=token, **exact_args). confirm returns None when the user
   declines; then that write must not happen. Catch BusinessError where the rules allow
   partial success. Return the answer exactly in answer_schema.json's shape. When data.json has
   scheduled_events, time passes only through api.wait(seconds); the events change rows as the
   clock passes them and tools never see that table.

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
- scheduled_events (if present): apply them with env_factory_v3.runtime.apply_event in `at`
  order for the waiting a correct handling does; keep that table unchanged in final_state.
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

PERTURBATION_AUDITOR = """You audit perturbed worlds of a business task. brief.md is the only authority (data.json
and facts.json hold the original world).

perturbations.json lists worlds where cells of rows the task acts on were changed
(`changed_cells`: table, id, field, original and perturbed value, and whether the task writes
that row), yet the implementation committed the same writes and gave the same answer as in the
original world (shown).

For each world decide whether brief.md makes the outcome depend on any changed cell:
- "consistent": no rule makes these changes matter here (for example a threshold is still met,
  or the field plays no role); the unchanged outcome is right.
- "should_change": a rule makes the outcome depend on a changed cell (an eligibility flag, a
  status, an amount, a date boundary), so the same outcome is wrong. Quote that rule.
- "ambiguous": brief.md does not decide whether the change matters. Quote the unclear text.
- A status or eligibility flag (for example active, cancelled, live, enabled, quarantined)
  changed on a row the task writes: if brief.md does not say how a row in that state is handled,
  the verdict is "ambiguous" (quote the rule that governs that write), never "consistent".
  Writing to a row as if its state did not matter needs a rule that says so.

Write perturbation_audit.json:
{{"worlds": [{{"world": "<world from perturbations.json>", "verdict": "consistent" | "should_change" | "ambiguous",
              "brief_quote": "<exact text from the user request or business rules; required unless consistent>",
              "reason": "..."}}]}}
Cover every entry. Each brief_quote must be copied verbatim from the "User request" or "Business
rules" section of brief.md (never from the answer schema). Do not write any other file and do not
read or write files outside this folder. End with one line per world.
"""

PERTURBATION_REPAIR = """An independent audit found rules env.py and oracle.py both ignore: in some perturbed
worlds a changed cell should change the outcome, yet both produce the original outcome.
- perturbations.json: each world, its changed cells, and the unchanged writes and answer
- perturbation_audit.json: the verdict per world; fix the worlds marked "should_change",
  following the quoted rule and reason
brief.md is the only authority.

Fix env.py (its tools and reference agent) and oracle.py so both enforce the quoted rules on
any data. Change nothing else. Never special-case ids or worlds; keep both interfaces; do not
edit brief.md, data.json, facts.json, request.json, answer_schema.json, perturbations.json or
perturbation_audit.json.

Run `{python} -m env_factory_v3.checks env`, `{python} -m env_factory_v3.checks oracle` and
`{python} -m env_factory_v3.checks diff` until all three print PASS.
Do not read or write files outside this folder. End with a two-line summary of the fix.
"""

REQUEST_AUTHOR = """You design one new business task for training customer-facing agents. Business domain:
{domain}. {detail}
Write request.json in this folder (assignment.json repeats the assignment):
{{"id": "{id}", "family": "<short domain slug>", "category": "{category}", "reward_type": "{reward_type}",
  "title": "<short Chinese title>", "user_scene": {{...}}, "business_rules": "...", "tables": {{...}},
  "user_facts": {{...}}, "time": {{"start": 10, "tick": 0}}}}
Do not write the user's opening message: another writer turns user_scene into it without ever
seeing the rules, tables or facts.

Task shape: {shape}

- business_rules (Chinese): the complete and only authority. State every rule the outcome
  depends on precisely (eligibility, limits, rounding, ordering, ties, what happens on refusal or
  failure, partial success). Amounts are integer cents in data; times are virtual integers.
  No rule may be open to two readings. For every row a write creates or changes, state the
  value of every column of that table (including columns that stay null or unchanged): the
  final state is compared exactly, so an unstated column is a guess.
- tables: synthetic rows, each with a unique "id"; columns consistent within a table; enough
  rows that wrong rules give different results (decoys, boundary values, an ineligible row).
- user_facts: what only the user knows and an agent must ask for (ids, quantities, choices),
  plus one boolean approve_* fact per write the user is asked to approve.
- user_scene (Chinese), the only input of the opening writer:
  {{"situation": "<who the user is and what happened, one sentence>",
    "goal": "<what they want done or decided, in their own terms, no rules>",
    "known": {{"<item>": <value>, ...}}, "wants_to_know": ["<result the user wants to be told>", ...]}}
  known holds only what the user can state up front (a direct task needs every number of its
  computation here, as the user would read it off a receipt or a price board; amounts in cents
  are fine, the writer says them in 元). Never put business rules, confirmation steps, transaction
  semantics or virtual clock values ("time 10", "day 30") in the scene; give times relative to now
  ("starts in 20 minutes", "expires in 2 days") in the unit the rules use. user_facts stay private:
  do not copy them into known unless the user would open with them.
- Time does not pass in this task: queries, approvals and writes all happen at time.start. Do
  not describe a clock that ticks per action; time-dependent rules compare against time.start.
- Use a domain different from library loans, procurement, subscriptions, document access,
  parcels, parking, warehouse lots and file versions.

Run `{python} -m env_factory_v3.checks request` until it prints PASS. Do not read or write
files outside this folder. End with one line: the title.
"""

AUTHOR_RUBRIC = """
This is a subjective task graded by a rubric, not by a fixed expected answer: answer_schema.json
has a single "reply" field, the final message to the user. reference(api) must gather what the
request needs through the tools (and the user), then return {{"reply": "<the message a skilled
assistant would send>"}}: built from the tool results and the user's facts (so it changes with
the data), in the user's language, addressing everything the user asked for, and stating only
what the tools and the user actually said.
"""

RUBRIC_AUTHOR = """You define how a subjective business task is graded. Read brief.md (data.json and facts.json
hold the initial tables and the user's private facts). You never see the implementation.

Write rubric.json: {{"criteria": [{{"id": "<snake_case>", "weight": 1-5, "description": "...",
  "levels": {{"0": "...", "1": "...", "2": "..."}}, "basis": "<exact text from brief.md>"}}]}}
- 3 to 6 criteria covering what makes the reply and the conversation good for THIS request:
  what the user asked to learn or decide, the business rules the reply must respect or explain,
  asking for missing information instead of assuming, explaining refusals or trade-offs.
- Each criterion is observable in the conversation (tool calls, what was asked, what was told);
  levels describe concrete behaviour, not adjectives ("states the refund amount and why", not
  "good explanation"). Weight what matters most to the user higher.
- basis: copied verbatim from the "User request" or "Business rules" section of brief.md, the
  text that makes the criterion matter.
- Do not add a criterion about factual accuracy: the platform always adds a "grounded" criterion.
  Do not grade wording style unless the request asks for it.

Run `{python} -m env_factory_v3.checks rubric` until it prints PASS. Do not read or write files
outside this folder. End with one line naming the criteria.
"""

OPENING_WRITER = """你是一位普通用户，正在和客服/业务助理在线聊天。根据下面的场景，写出你发的第一条消息。

你的处境：{situation}
你想要的：{goal}
你手头已知、开口时会提到的信息：{known}
你希望最后被告知：{wants}
你的说话风格：{persona}（{persona_hint}）
{feedback}
要求：一到三句口语，像真人；已知信息里的每个数字都要说到（金额按元说，1元=100分，如180分说“1.8元”）；
不要复述任何规则或办理流程，不要说“先告诉我……等我确认再……”，不要出现“当前时间是10”“第10天”这类编号；
不要总以“帮我”“麻烦”开头。只输出这条消息本身。"""

REPAIR = """The environment in this folder failed qualification. gate_report.json lists the
failed gates with evidence. brief.md remains the only authority.

Fix env.py (and only env.py) so the failures are resolved for the right reason. Do not weaken
business rules, special-case ids or worlds, or change answer_schema.json.
Run `{python} -m env_factory_v3.checks env` until it prints PASS.
Do not read or write files outside this folder. End with a two-line summary of the fix.
"""
