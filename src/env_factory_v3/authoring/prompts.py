"""Prompts for the model roles. brief.md is the only semantic authority for all of them."""

SCHEMA = """You define the answer contract for a business task. Read brief.md (data.json and
facts.json hold the initial tables and the user's private facts).

Write answer_schema.json: the JSON object the agent must return to the user at the end.
Format: {{"fields": {{"<name>": {{"type": ..., "description": ..., ...}}}}}}
- types: string, integer, number, boolean, array (needs "items"), object (optional "properties");
  "nullable": true when a value can legitimately be absent; every array sets "order": "any"
  when order carries no meaning or "exact" when it is a ranking or sequence. Amounts are integers
  in the smallest unit (cents, fen), never strings.
- Cover exactly what the user asked to learn or be told, nothing decorative. Express every
  outcome with ids, integers, booleans or string enums ("enum": [...]) so that two correct,
  independent implementations produce identical values. Every field needs a precise
  description, including units and how absent/failed cases are reported.
- Free text (an explanation, reason or message) can never be compared exactly. Avoid it; if
  the user must get one, add at most one such field with "graded": false and put the actual
  outcome in a graded field (for example a "failure" enum next to it). The same applies to
  string properties inside objects and array items.
- The agent delivers the answer by talking to the user, and the graded values are read back from
  what it said. Every graded value must be something a good customer-service agent would actually
  say: amounts, dates, counts, outcomes, and ids the user knows or needs (a booking reference, a
  new confirmation number). Never ask for system-internal row ids or keys the user never sees
  ("i1", "s2", "item-row-7"), internal clock ticks or day indexes (give relative or human units
  instead), or odd formats ("dollars 42.50"). Keep objects small and flat; fewer, meaningful
  fields beat exhaustive dumps of the database.
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
   - `now` is time.start: an int, or for a calendar clock an ISO string "YYYY-MM-DDTHH:MM" (compare with
     datetime.fromisoformat; dates in the tables are ISO strings too).
   - Read tool: fn(db, now, **args) -> JSON. The signature and docstring are the only tool
     description an agent sees: annotate every argument with its type (str, int, float, bool,
     dict, list[...], or X | None) and give the docstring a summary plus an `Args:` section
     with one `name: meaning` line per argument (units, id format, allowed values).
   - Write tool: decorate with @write; fn(db, now, **args) mutates db in place and returns a
     JSON receipt. Raise BusinessError to reject; the runtime rolls the write back. Its arguments are
     business parameters (ids, enums, amounts, lists of ids), never an object: when the write must
     confirm amounts the agent worked out and told the customer (a penalty, a refund), take each as its
     own integer argument (penalty_cents, refund_cents) and reject a wrong one, never a bundle of terms.
   - Tools look like a real business API. Each read answers one business question about
     specific records (look up by id, list one customer's orders, search with filters) and
     returns only the records and fields that question needs. Never write a generic table
     reader (a tool taking a table name, read_table, list_rows) or return a whole table.
   - Reads return facts (amounts, statuses, times, limits), not the decision the task asks
     for: do not compute the final eligibility verdict, total, refund, quote or ranking inside
     a read tool; applying business_rules to those facts is the agent's job. Write tools still
     enforce every rule in brief.md and raise BusinessError on a violation. Tools run on
     perturbed worlds, so never depend on particular ids or values from data.json.
   - Expose what an agent needs to do the task through tools; never expose facts.json.
   - The tools are the business's whole service API, not a kit cut for this one request: besides
     the reads this task needs, add at least three reads a colleague would use for neighbouring
     questions in the same business (for example tracking, loyalty balance, opening hours,
     another record type) that this task never needs. They read the same data and must work.
   - Ids of internal records (items, payments, tickets, segments) come only from what another
     read returns (an order's items from the order), never from something the user could know.
   - In `Args:` describe each id's format with a made-up example that is not in data.json
     (for example "order id such as 'ord-000'"); agents must take real ids from the user or
     from tool results, never from the example.
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
   declines; then that write must not happen. api.ask raises KeyError for a fact the user does
   not know (brief.md lists forgotten facts under "How the user behaves"): identify the user the
   other way the rules allow. Handle every goal the user has, in order.
   World mode: when request.json has "goal_schema", this environment serves many tasks on one
   database. The user's goals are the fact "goals" ([{{"process", "params"}}]): reference(api)
   must handle any process in goal_schema with any valid params (read them with api.ask), never
   only the sample task in facts.json. When goal_schema gives each process an "outcome", the answer
   is {{"goal_1": ..., "goal_2": ...}}: one field per goal in order, in that process's outcome shape
   (answer_schema.json shows it for the sample goals). An "inquiry" goal is answered from the rules
   and its params alone, with no tool call. Catch BusinessError where the rules allow
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
(`now` is time.start: an int, or an ISO string "YYYY-MM-DDTHH:MM" for a calendar clock.)
- final_state: every table after all operations a correct handling performs. An operation
  happens only if the rules allow it at that point AND the user's matching approval fact is
  true. Committed operations stay committed unless the rules say otherwise.
- answer: what the user must be told, exactly in answer_schema.json's shape.
- alternatives (optional, rare): when brief.md genuinely leaves a choice open that facts.json
  does not settle (e.g. "either of two equivalent options is acceptable"), return the other
  correct outcomes as "alternatives": [{{"answer": ..., "final_state": ...}}] (at most 3, each
  different from the main one). Never list a wrong or partial outcome, never "do nothing" when
  the main outcome changes the world, and never use it to hedge a rule you find unclear.
- scheduled_events (if present): apply them with env_factory_v3.runtime.apply_event in `at`
  order for the waiting a correct handling does; keep that table unchanged in final_state.
- World mode: when request.json has "goal_schema", the same oracle serves many tasks on one
  database: compute the outcome for whatever goals the fact "goals" lists, for any process in
  goal_schema, never only the sample task in facts.json. When each process has an "outcome", the
  answer is {{"goal_1": ..., "goal_2": ...}}: one field per goal in order, in that process's
  outcome shape.
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
If brief.md explicitly lets either outcome stand (a stated free choice, not an unclear rule),
the oracle is the side to fix: add the env's outcome to oracle.py's "alternatives" and record
wrong_side "oracle".

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
  "language": "{language}", "title": "<short title in {language_name}>", "user_scene": {{...}},
  "business_rules": "...", "tables": {{...}},
  "user_facts": {{...}}, "time": {{"start": "2024-05-15T15:00", "tick": 0}}}}
Do not write the user's opening message: another writer turns user_scene into it without ever
seeing the rules, tables or facts.

Task shape: {shape}

{script}

- business_rules ({language_name}): the complete and only authority. State every rule the outcome
  depends on precisely (eligibility, limits, rounding, ordering, ties, what happens on refusal or
  failure, partial success). Amounts are integer cents in data; times are virtual integers.
  No rule may be open to two readings. For every row a write creates or changes, state the
  value of every column of that table (including columns that stay null or unchanged): the
  final state is compared exactly, so an unstated column is a guess.
- Refusals and failures: for every step the user may refuse or that may fail, the rules say
  (a) whether the remaining steps still run, (b) the exact row values and result label left for
  that item, or that nothing is written, (c) whether it uses up or frees any budget, quota,
  capacity, sequence number or position that later steps depend on, and (d) how the final answer
  reports it. If a later step's outcome depends on an earlier refusal, say exactly how; one
  refused or failed step must leave exactly one possible final state.
- tables: synthetic rows, each with a unique "id"; columns consistent within a table; enough
  rows that wrong rules give different results (decoys, boundary values, an ineligible row).
- user_facts: what only the user knows and an agent must ask for (ids, quantities, choices),
  plus one boolean approve_* fact per write the user is asked to approve.
- user_scene ({language_name}), the only input of the opening writer:
  {{"situation": "<who the user is and what happened, one sentence>",
    "goal": "<what they want done or decided, in their own terms, no rules>",
    "known": {{"<item>": <value>, ...}}, "wants_to_know": ["<result the user wants to be told>", ...]}}
  known holds only what the user can state up front (a direct task needs every number of its
  computation here, as the user would read it off a receipt or a price board; amounts in cents
  are fine, the writer says them in {currency}). Keep it short enough to say in one breath: a direct task
  has at most 3 records and 8 numbers in known (design the computation to fit). wants_to_know has
  at most 3 short items in the user's own words; everything the final answer must report goes
  into business_rules ("最终必须告知……"), not into the scene. Besides the outcome of any change,
  the answer must include at least one concrete fact the agent can only learn from a lookup (how many
  options are available, the new departure time, the remaining balance), the way real customers ask
  "what are my options" or "when does it leave". Never put business rules, confirmation steps, transaction
  semantics or virtual clock values ("time 10", "day 30") in the scene; give times relative to now
  ("starts in 20 minutes", "expires in 2 days") in the unit the rules use. user_facts stay private:
  do not copy them into known unless the user would open with them.
  An interactive task (not direct) opens the way real customers do: known has at most 2 items and
  no identifiers (ids, references, emails, phone numbers, exact amounts); the goal is stated broadly.
  Everything else the agent needs lives in user_facts and comes out only when the agent asks.
- Time does not pass in this task: queries, approvals and writes all happen at time.start. Do
  not describe a clock that ticks per action; time-dependent rules compare against time.start.
- Use calendar time, as real businesses do: time = {{"start": "<YYYY-MM-DDTHH:MM, now>", "tick": 0}};
  dates and times in the tables are ISO strings ("2024-05-15" or "2024-05-15T09:30"); rules are stated
  in calendar terms ("within 24 hours of booking", "at least 3 days before departure", "before 18:00
  on the delivery day") so the agent has to reason about dates, including across days and months.
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
what the tools and the user actually said. Write it the way a good customer-service agent
talks: plain words, amounts in the user's currency (元 in Chinese, dollars in English), no internal field names, table ids, status codes, raw cents,
virtual clock values or quoted rule text, a few short paragraphs at most.
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
- Do not add a criterion about factual accuracy, answering the request at all, or plain
  language: the platform always adds "grounded", "plain_language" and "answers_request" (a gate:
  a reply that does not deliver what the user asked for scores 0).
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
想被告知的内容简单带过即可，不必逐项列出，用你自己的说法收尾（也可以不专门收尾）；{avoid}
不要复述任何规则或办理流程，不要说“先告诉我……等我确认再……”，不要出现“当前时间是10”“第10天”这类编号；
不要总以“帮我”“麻烦”开头。只输出这条消息本身。"""

REPAIR = """The environment in this folder failed qualification. gate_report.json lists the
failed gates with evidence. brief.md remains the only authority.

Fix env.py (and only env.py) so the failures are resolved for the right reason. Do not weaken
business rules, special-case ids or worlds, or change answer_schema.json.
Run `{python} -m env_factory_v3.checks env` until it prints PASS.
Do not read or write files outside this folder. End with a two-line summary of the fix.
"""

POLICY_WRITER = """你要把一份写给开发者的业务实现规格，改写成给一线客服看的业务政策手册。客服只看这份手册办事。

实现规格：
{rules}

要求：
1. 保留每一条会影响处理结果的规则：资格条件、金额怎么算和怎么取整、处理顺序、上限和名额、被拒绝或失败后怎么办、部分成功、哪些操作需要用户同意、最后要告诉用户什么。数值、阈值和边界（包括是否含等号）必须和规格完全一致。
2. 用业务语言写：不要出现表名、字段名、状态码、英文标识符、time.start、“虚拟时间”之类的实现细节。时间说成“现在”“开始前 30 分钟”这样的相对说法，时间单位照规格写（分钟、小时或天），但不要加“虚拟”二字（英文手册不要出现 virtual）。{language_rule}
3. 只写政策，不写解题步骤（不要写“首先查询……然后……”），也不要提工具名。
4. 不要加入规格里没有的规则，不要写具体客户的编号或数据。
{feedback}
只输出手册正文，分条列出。"""

POLICY_CHECKER = """下面是一份业务实现规格，和据此改写的客服政策手册。客服只能看到手册。逐条对照：规格里任何会影响处理结果的规则（资格、计算、取整、顺序、上限、拒绝或失败后的处理、需要用户同意的操作、最后要告诉用户的内容），如果在手册里缺失、被改变、数值或边界不一致、或者说得含糊到可以有两种理解，就列出来。只看业务含义，不管措辞和实现细节。
客服除了手册，还能看到当前时间的数值、查询工具返回的数据和最终答案的字段说明，所以手册不需要写出当前时间的具体数值、表名字段名或数据里的具体记录；这类内容不算缺失。手册里的时间单位（分钟、小时、天）与规格一致即可，不写“虚拟”二字不算问题。

实现规格：
{rules}

客服政策手册：
{policy}

只输出 JSON：{{"problems": ["<一句话说明一条缺失或不一致的规则>", ...]}}；没有问题时输出 {{"problems": []}}。"""

OPENING_WRITER_EN = """You are an ordinary customer chatting online with a service agent. From the scene below, write
the first message you send, in English.

Your situation: {situation}
What you want: {goal}
What you know and will mention when you start: {known}
What you want to be told in the end: {wants}
Your speaking style: {persona} ({persona_hint})
{feedback}
Requirements: one to three casual sentences, like a real person; mention every number you know (say
amounts in dollars: 180 cents is "$1.80"); mention what you want to be told only briefly, in your own
words; {avoid}
never restate rules or procedures, never dictate how the agent should confirm things with you, and never
mention internal numbers such as "time 10" or "day 10". Output only the message itself."""

WORLD_AUTHOR = """You design one complete customer-service business world for training agents: one database and
one policy manual on which many different customer tasks will later be posed. Business domain: {domain}. {detail}
Write request.json in this folder:
{{"id": "{id}", "family": "<short domain slug>", "category": "multi_step_agentic", "reward_type": "business",
  "language": "{language}", "title": "<short title in {language_name}>", "business_rules": "...",
  "goal_schema": {{...}}, "tables": {{...}}, "user_facts": {{...}}, "user_scene": {{...}},
  "time": {{"start": "<YYYY-MM-DDTHH:MM, now>", "tick": 0}}}}

- business_rules ({language_name}): the full policy manual of this business, the only authority. Cover 6 to 10
  business processes customers come for (for example look up, book, change, cancel, refund, exchange,
  compensate, upgrade). State which requests are outside this business's scope and how they are answered
  (decline, point elsewhere, or a handover this business actually provides). For each: eligibility, limits, fees and
  rounding, deadlines relative to now, what happens on refusal or failure, partial success, and which writes need
  the customer's approval. Customers must be identified (state how: e.g. name plus postcode, or email) before any
  account data is shown or changed. For every row a write creates or changes, state the value of every column.
  No rule may be open to two readings. Amounts are integer cents; dates and times are ISO strings ("2024-05-15", "2024-05-15T09:30") against a fixed
  calendar now (time.start), and rules are stated in calendar terms (hours, days, "before 18:00").
- goal_schema: {{"<process>": {{"description": "<what the customer wants>", "kind": "account" | "inquiry",
  "params": {{"<name>": "<type and meaning>"}}, "outcome": <field spec>}}}} for those 6 to 10 processes. A task's
  goals are one to three of them with concrete params.
  - kind "inquiry" (one or two processes): a policy question answered from the rules and the numbers the customer
    states (for example the fee for a change on a fare and a date they name): no identification, no table
    lookup, no write; its params are those numbers.
  - kind "advice" (at least one): the customer wants a recommendation, comparison or explanation about their own
    situation (which plan or option suits them, why a charge happened); the agent looks up their records and
    applies the rules; no write. Its outcome holds the key points a good answer must contain: the recommended
    option (an enum or record id; when the rules make several equally good, the oracle lists the others as
    alternatives) and the reasons as enums of the deciding rules, plus the amounts or dates the answer needs.
  - Every other process is "account".
  - outcome: what the customer must be told when a goal of this process is handled, as one answer field spec of
    type "object" with "properties" (the same rules as an answer schema: ids, integers, booleans and string enums,
    a description with units and how absent or refused cases are written; only values a good agent would say).
  - approval (every process whose handling writes): the approve_* fact key pattern its writes are approved
    under, with the record id when one customer may decide differently per record, e.g. "approve_cancel_<exchange_id>".
    env.approval_key must return exactly this pattern filled in for the write, and every task names its approval
    facts by it.
    It must cover refusal and partial cases too: a "status" enum, and for an account process a "reason" string
    enum naming why it was refused or could not be done (null on success), so an agent that refuses without
    checking the data cannot match a correct refusal. Each task's answer has one field
    per goal, goal_1, goal_2, ..., in the order of its goals, each in its process's outcome shape.
- tables: a realistic database: at least 40 customers and at least 200 rows in all, with several payment methods,
  statuses and edge cases (expired, ineligible, at a limit, out of stock); unique "id" per row; columns consistent.
- user_facts: one sample task: "goals": [{{"process": ..., "params": {{...}}}}], the identity facts the rules ask
  for, and one boolean approve_* fact per write the customer is asked to approve.
- user_scene ({language_name}) for that sample task: {{"situation", "goal", "known", "wants_to_know"}} as a
  customer would state them, with no rules, protocol or virtual clock values.
- Use a domain different from library loans, procurement, subscriptions, document access, parcels, parking,
  warehouse lots and file versions.

Run `{python} -m env_factory_v3.checks world` until it prints PASS. Do not read or write files outside this
folder. End with one line: the title.
"""

TASK_AUTHOR = """You write one customer task on an existing business world. Files in this folder: world.json (the
policy manual "business_rules", "goal_schema" and the database "tables"), assignment.json (what this task must
contain). Read them first.

Write task.json:
{{"title": "<short title in {language_name}>",
  "user_facts": {{"goals": [{{"process": "<from goal_schema>", "params": {{...}}}}], <identity facts the rules ask
  for>, <one boolean approve_* fact per write the customer is asked to approve>}},
  "user_scene": {{"situation", "goal", "known", "wants_to_know"}},
  "user_script": {{"traits": [...], ...}},
  "overlay": [{{"table": "<table>", "id": "<row id>", "set": {{"<column>": <value>}}}}]}}

- Processes for this task: {processes}. Pick a customer and records from the tables for which these goals make
  sense under the rules; params must name real rows.
- Name every approve_* fact by the "approval" pattern of its process in goal_schema (for example
  approve_cancel_<exchange_id> -> approve_cancel_HX1), one per write the customer is asked about.
- {shape}
- Make this task a different exercise from the world's other tasks: vary the case the rules distinguish (another
  fare or tier, another timing window, an exception), not only the amounts.
- Use the size of the world to make the conversation real: prefer a customer who has several similar records
  (so "my booking" needs a follow-up question), let goals across processes depend on each other (the outcome of
  the first changes what the second can do), and keep user_scene.known to at most two items so the agent has to
  ask for the rest.
- overlay: optional small changes to existing rows (only existing tables, ids and columns) that create the
  situation this task needs (for example make the first-choice item unavailable, set a date just past a deadline).
- user_scene ({language_name}): what the customer knows and says, in their own words: no rules, protocol or
  virtual clock values; ids only if a customer would have them.
- {script}

Run `{python} -m env_factory_v3.checks task` until it prints PASS. Do not read or write files outside this folder.
End with one line: the title.
"""
