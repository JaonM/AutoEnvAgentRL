# V2 executable business contract

All contracts are JSON data. No Python, SQL, eval, network code or arbitrary filters.
The implementation in contract.py is authoritative and rejects unknown keys.
Generator examples are demonstrations only; generation must design a new business.

Required top-level keys: version="2.0", id, category (direct_response,
simple_agentic, multi_step_agentic), task (original public business goal), rules,
tables, tools, user, answers, scenarios, probes, limits, time.
Optional: explanations, invariants, events, success, goals.
At least one positive scenario must succeed on the original seeded data without
any initial mutations, including inherited mutations. Changed-data positives
belong in probes or additional scenarios. Include a negative scenario as well;
negative probes alone do not replace it. A repair must fix the specific missing
baseline or negative trajectory, preserving the other valid tests.
Scenarios may reuse another scenario with extends=PARENT_ID. The child must
declare its own id and expect_success. Other supplied fields replace inherited
fields (answer and mutations replace completely); omitted fields are inherited.
Optional append_steps adds actual steps after the inherited/replaced steps.
Parents may appear later in the list. Cycles, unknown parents and more than 32
inheritance levels fail. The compiler expands full scenarios before all validation,
independent review, execution and replay. Use inheritance for wrong answers,
revocations or stale captures instead of repeating long trajectories. Expectations
must remain grounded in original business; inheritance never skips execution.

Expressions have exactly one operator. Literals always use {"literal":VALUE};
paths use {"ref":"data.accounts.0.balance"}. Roots depend on context. Operators:
eq/ne/lt/le/gt/ge/in/contains/add/sub/mul/at take two expression operands;
and/or take nonempty expression lists; not/count/sum/unique take one expression
operand in a list; list takes expression operands; object maps field names to
expressions; if takes {condition,then,else}; filter/map take {items,value} with
row available while evaluating value. sum expects its single expression to return
a numeric list. Type-strict comparisons; false and zero are real observations.
at takes [collection_expression,index_expression].
sort takes {items:EXPRESSION,order_by:[{field:COLUMN,direction:"asc"|"desc"}]}.
It stably sorts up to 10000 row objects by the declared keys, with nulls last in
either direction. Each key must contain one scalar type (int/float share numeric
ordering); strings use case-sensitive lexical ordering. Mixed scalar types,
missing fields and non-finite numbers fail. Use sort on authoritative filtered
rows before mapping ordered answers; storage order is not business order.
walk takes {items:EXPRESSION,start:EXPRESSION,key:COLUMN,next:COLUMN,max_nodes:INTEGER}.
It returns current rows in linked order, following next from start until null;
null start returns an empty list. Column names are literal strings, max_nodes is
1..10000, and items has at most 10000 rows. Duplicate keys, missing links, cycles
and over-limit chains fail explicitly. Keys are non-null scalars, with typed
identity (true is not 1; 1 and 1.0 are equal). Use a public rule to select the
business start where required, then map the walked rows for IDs or sum mapped
costs. Never use table positions as route order or hard-code a reference path.
This is pure authoritative-state computation, not a substitute for required
agent tool observations, pagination or user authorization.
Shared predicates use
{"rule":{"id":"public_rule_id","arguments":{"size":{"ref":"user.size"}}}}.
Supply exactly the named args referenced by that public rule. The current row,
data and time remain in scope. Public rule definitions may not invoke other rules.

rules: list of {id,public,predicate}. public explains an actual business condition;
predicate can reference row,args,data,initial,user,time. Tool filters are ONLY references
to these rules, never handwritten extra predicates. A display name cannot become
an identity constraint without an original public requirement.

tables: {id,schema,primary_key,rows,foreign_keys?}. Row object schemas declare
properties, required, additionalProperties=false. Supported types string, integer,
number, boolean, object, array; optional enum and nullable=true. Arrays require
items schema. foreign_keys contain {column,target_table,target_column}. Unique
non-null primary keys; all initial mutations retain schemas and foreign keys.
For a foreign-key column declaring nullable=true, null means no relationship
(for example a terminal next_id or an unselected purchase) and needs no target
record. Every non-null value must match a real target; dangling values are rejected.
For bulk data, replace rows with generator={count:0..10000,seed:integer,columns:{column:SPEC}}.
Do not write hundreds of literal rows. Column specs: {kind:"constant",value};
{kind:"sequence",start,step,prefix?,width?} generates start+i*step, optionally
formatted as prefix+zero-padded integer; {kind:"cycle",values,offset?} picks
values[(i+seed+offset)%len(values)]; {kind:"foreign_key",table,column,offset?}
cycles through an earlier materialized table with the same index formula.
All generated rows undergo full schema, primary-key and FK validation. Reviewers
receive exact generator plans and small concrete previews; execution uses ALL rows.

tools use id,kind,description,parameters (closed object schema),rule_ids.
Query tools also use table,columns,order_by ([{field,direction:"asc"|"desc"}]),
page_size (1..128). Results: items,total,offset,next_cursor,query_identity,
state_version. Pagination cursors are bound to episode/query/state.
Transaction tools use operations=[{table,key:EXPR,set:{column:EXPR}}]. Their
rule_ids are preconditions over the immutable state BEFORE this transaction,
including the original target row. They are never reevaluated against partially
written state. Write expressions evaluate in order on one isolated candidate
state, then validate schemas, FKs and invariants before atomic commit. Invariants
see the candidate post-commit user state, including consumption of this exact
authorization. Data and consent consumption commit together; failed validation
leaves both unchanged. Preconditions still see the live pre-commit user state. key
identifies exactly one existing row; primary-key writes, insertion and deletion
are currently unsupported.

Tool rule_ids apply to EVERY operation row, including operations on different
tables. An activity-only row predicate cannot also guard a registration row.
For heterogeneous transactions, express shared conditions over data and args
(select identities through public rules), not fields unique to one target row.
Collection-local row references inside filter/map belong to that collection and
do not refer to the operation row. The compiler rejects operation-row references
to fields absent from any target table; this is a schema error, not a valid negative.
Transaction tools MUST name authorization=user_stage_id and have actual-state
goals. Do not simulate transactions merely through final-answer accuracy.
For transfers and settlement, goals and answer targets must identify the exact
authorized recipient row using the disclosed selection and observed account
identity. A supplier, quote or customer relationship can match multiple accounts;
it does not mean every related account must receive the payment. Include a
counterexample with another valid account for the same related object, and check
that only the intended recipient changes while unrelated balances remain intact.
Public rule descriptions must state only what their predicates enforce. A rule
matching an available row to an argument does not itself prove the user's actual
selection; publish and enforce that obligation through user_bindings and the
actual choice stage. Review the combined rule, binding and authorization behavior.
Optional tool keys: user_bindings={argument:fact_name},
result_bindings=[{argument,tool,path}], complete_retrieval=true.
result_bindings require actual observed references, not matching literals;
path may use '*' for a numeric result-array index, e.g. items.*.id.
The binding tool can be a list of permitted providers. A tool may consume its own
earlier result: runtime observations must already exist, preventing temporal cycles.
Each result binding may specify recency="latest" (default "any"). latest requires
the observation ID of the most recent successful business tool call, in addition
to the provider and path constraints. User, time, wait, state and collect events
do not replace that call; failed calls do not produce results. For a chain requiring
only the previous result's next_id, use latest and include an executed negative
trajectory that reuses an older capture after a successful successor query.
Observation coverage alone does not enforce this succession constraint.
Source parameters describe resolved business values. The compiler separately
publishes wire schemas: a captured argument is a {$result,path} object, not a
string. User-bound arguments may use their business value or a {$user,path}
object. Do not copy compiled wire schemas back into source parameter schemas.
Public tool prose is compiler-generated; authored descriptions cannot impose
separate behavior. Query nulls sort last in either direction.

user: {facts:{private_name:VALUE},stages:[...]}. Each stage has id,kind,depends_on.
fact/correction stages have fact; optional as_fact declares the current public
binding updated by that disclosure. Planned corrections use a different private
fact key and overwrite the same as_fact only after their prerequisites.
choice stages have fact,source_tool,source_path (e.g.items),
value_field (e.g.id),fallback (refuse|first_available). They decide ONLY from the
actual observed menu; missing preference may legitimately refuse. Authorization
stages have tool and facts must explicitly include approve_STAGE_ID:boolean.
They receive {tool,arguments} proposals, authorize exact parameters once, and
support revocation. Models cannot mutate facts, choices or authorizations.
The platform compiles stages to parallel regions of a hierarchical statechart;
no additional authored FSM schema is required. Dependency guards enable waiting
stages. Facts resolve as disclosed; choices resolve as selected or refused.
Consent enters awaiting_confirmation, then authorized or declined. Changing a
schema-valid proposal or attempting different authorized parameters invalidates
old consent. Revoked, invalidated or declined consent can receive another explicit
proposal/user decision; consumed consent is terminal and cannot be reused.
Transaction failures before commit retain exact consent for a retry. A correction
of disclosed facts invalidates dependent choices and live consent, including
consent without a declared dependency edge. Repeated completed facts/choices are
rejected unless a dependency invalidated that stage. Corrected preferences govern
repeated choices. GET /user-state exposes only states and transition events,
never private context, and does not advance the clock.
Stages depend only on earlier stages. Do not expose private facts in public text.
Reward user bindings use actual disclosed choices/corrections, overriding initial
preferences. A permitted fallback choice must not be graded against an unavailable
original preference. Refusal/no-write success branches must also make state goals
and answer targets conditional on that actual decision.
Expressions also have root user_state, containing the same non-secret snapshot
as GET /user-state: root, regions:{stage_id:state}, and transition history. Use
user_state.regions.STAGE to distinguish consent.consumed (successful write),
consent.declined or consent.revoked (legitimate refusal/no-write), and consent.authorized
(approval exists but execution is unfinished). Initial approve_STAGE facts are
preferences, not proof of current authorization or a committed write. A refusal
branch must require an actual refusal/revocation state and unchanged business data;
merely failing to execute an approved write must not earn refusal credit. Goals
and answer targets must follow the same branch decision as success requirements.

answers: [{id,meaning,expected:EXPR,weight,comparison:"exact"|"unordered"|"numeric",schema}].
Expected roots: data,initial,user,user_state,time. Never reference submitted answer
or caller-chosen observation IDs. evidence is ONLY for scenario answer recipes;
reward targets, state goals, branch conditions and required observation targets
must remain valid when a policy uses different IDs for the same actual calls.
Use numeric for scalar decimal calculations. It accepts at most eight binary
floating-point ULPs, capped at 1e-9 absolute error; this is representation cleanup,
not a business rounding rule. Integers remain exact and booleans are not numbers.
Current record facts must follow actual records, not fixture literals. Meaning
and comparison are public. Weights including explanations sum to one.
Every eligibility filter in reward targets, invariants and actual-state goals MUST invoke
named public rules (or boolean combinations of them). Never duplicate a tool's
eligibility predicate manually in an expected expression. Use filter/map with
the rule operator to compute the authoritative complete set independently of
the agent's possibly incomplete tool observations. Scenario/challenge answer
recipes may use independent expressions to challenge the rule itself.
explanations: [{id,meaning,criteria:[{id,rule_id,obligation}],context_fields:[answer_id],weight}].
Criteria describe invariant reasoning obligations over actual runtime facts,
not immutable wording or mutable fixture content. Code Agent grades explanations
only against authoritative evaluated context_fields and actual user turns.

invariants: [{rule_id,predicate:EXPR}], roots data,initial,user,time.
goals: [{id,rule_ids,predicate:EXPR}], roots data,initial,user,user_state,time.
Goals enforce real world outcomes separately from answer accuracy. Writes require
at least one goal. Example: actual destination balance equals initial balance+20.
success: [{id,public,when:EXPR,required_tools:[id],required_stages:[id]}].
required_stages means stages still completed in the current simulator state.
Simple agentic means one business tool, not a mandatory single call. Optional
min_business_calls/max_business_calls explicitly bound attempts for a success
branch, only when justified by the original request or business dossier. Publish
the limit in branch public text. Do not invent call caps from a category label.
Actual consent.declined and consent.revoked are different user outcomes. Cover
real declines; do not use a revoke trajectory as a substitute for user rejection.
User withdrawal is a trusted user event, never a policy tool to manufacture.
Optional required_observations on a success branch is a list of
{id,tools:[QUERY_ID],columns:[COLUMN],expected:EXPRESSION}. expected returns a list
of row objects with exactly those columns, derived from current authoritative
data (reuse public filtering rules). Every expected row must occur in an actual
successful result from one of the declared query tools. Providers must also be
required_tools and publish all these columns. This checks observations, not just
correct guesses. For linked tasks require all currently reachable chain IDs
(derive them using walk), plus the actual terminal owner ID from the owner query.
If links extend or reorder the chain, obligations change with the actual data.
Do not freeze the expected list to initial IDs, a fixed length or test call IDs.
When facts must be fresh, include the relevant fact columns alongside identity.
Translate every original quantitative obligation into an explicit executable
goal or active success condition, not just seed size or public prose. If the
agent must obtain at least N eligible options, require the current eligible count
to be at least N using the same public filter, plus complete retrieval/required
observations. Apply that obligation to every applicable completion branch,
including refusal where the original requires obtaining the menu first. Include
N-1 and exactly N counterexamples. Distinguish these behavior requirements from
fixture-only dataset scale; do not invent agent obligations from fixture size.
Revocation removes completion of the revoked consent stage. A valid refusal or
revocation branch must use its actual user_state region as proof and MUST NOT
require that revoked consent stage in required_stages. Fact/choice prerequisites
still apply when the original business requires them. Preserve current data
against initial data for legitimate no-write outcomes, including already-existing
business records. Positive scenarios and probes undergo the SAME deterministic
workflow, safety, pagination, invariant and goal checks as final scoring before
native independent review; no semantic judge is simulated in this preflight.
Explicit
branches allow successful refusal/no-write outcomes. Exactly applicable business
obligations must be preserved, never require an inactive branch's tools. Default
requires all declared tools and user stages. Unauthorized protected writes always
prevent full success. Retrieval tools marked complete_retrieval require every
page of one unchanged query before full credit.
When such a tool supplies a user choice, retrieve and collect the entire menu
BEFORE that user turn. The source_id must identify a complete menu from the
current state and the same query; later reads cannot repair an earlier choice.
An incomplete or stale menu raises `choice requires the complete current observed menu before selection`.
Tools without complete_retrieval retain their declared partial-menu behavior.
Direct has zero business tools, stages and tables; simple has one tool; multistep has at least two tools and an
actual result_binding. No padding to satisfy these requirements.

Answer meanings distinguish CURRENT STATE from THIS EXECUTION. A historical
paid/registered/settled flag does not prove this episode submitted a transaction.
For an answer about whether the current authorized transaction committed, use
`user_state.regions.STAGE == "consent.consumed"`: consent is consumed only after
a successful atomic commit, never on approval alone or a rejected write. Decline,
revocation, and unfinished approval therefore mean false even if the initial
record was already paid. For answers about current record state instead, expose
that record through actual queries and grade its current value. Include a no-write
case with an already-completed initial record to distinguish these two meanings.

limits: {max_actions:1..4096}. User turns, business calls and waits consume budget.
time: {base:timezone-aware ISO datetime,seconds_per_action:0..86400}.
Reading time does not advance it. wait advances virtual time without sleeping.
events: [{at_seconds,table,key:literal_primary_key,set:{column:literal_value}}].
Events apply deterministically before actions at their scheduled virtual time.

scenarios: [{id,steps,answer:{answer_id:EXPR},expect_success:boolean,mutations?,expect_error?}].
Provide at least one successful and one failed valid-answer trajectory. An
intentional invalid-action negative must name expect_error substring explicitly;
an accidentally broken negative is not evidence of reward discrimination.
Common exact runtime error substrings for intentional negatives:
- Violating a declared result_binding (including a literal instead of the required
  capture or a capture from the wrong tool/path): `required observed-result dependency missing`.
- Violating a user_binding, including calling before disclosure with a literal:
  `tool argument must use the actual disclosed user value`.
- Resolving a nonexistent capture when no earlier binding check rejects it:
  `unknown observed result`.
- A transaction without live exact consent:
  `authorization missing, revoked, consumed or parameters changed`.
Checks execute in order. A negative meant to test insufficient balance/capacity
must first obtain valid exact consent and valid references so it reaches the
business precondition. An earlier authorization/dependency error is not proof
of insufficient-balance handling or rollback. Do not guess synonyms for errors.
steps: call/traverse {kind,tool,id,arguments}; call optionally cursor.
References inside arguments: {"$result":"previous_call_id","path":"items.0.id"}
or {"$user":"completed_stage_id","path":"value"}.
User steps: {kind:"user",stage,question,source_id?,proposal?}.
Wait steps: {kind:"wait",seconds}; revoke: {kind:"revoke",stage}.
traverse executes all real pages, each consuming one action, and creates evidence
under the traversal id with items,total,next_cursor=null,pages.
The helper consumes one collection action after all page calls. Public agents
fetch pages through /tool, then /collect with {pages:[observed_page_id],id};
collection rejects duplicate, mixed-query, out-of-order or incomplete pages.
Answer recipes may reference data,initial,evidence,user,time. These are fixture
recipes, never available as agent actions and never used as reward targets.
Stateless direct tasks can test pure public-rule boundaries without adding tables,
tools, private data or extra answer prose. In a positive probe's answer recipe,
invoke the actual named public rule with boundary arguments. Use an if expression
that returns the independently correct answer for the original task only when
the boundary rule returns its business-correct value; otherwise return a wrong
but schema-valid answer. This executes the rule at the boundary and fails if its
predicate is wrong. Merely submitting another input's answer as a negative for
the fixed original input does not verify that boundary. Empty data mutations are
appropriate when there is no mutable data. Boundary tests are evaluator fixtures,
not additional agent answer obligations unless the original request requires them.
mutations: [{table,key,column,value}] apply to initial fixtures by primary key.

probes: [{id,scenario,rule_ids,mutations,expect_success,answer?,expect_error?}].
Cover correct changed-data answers AND wrong stale answers, display-field
invariance, missing/wrong dependencies, menu absence, exact consent, rollback,
time boundaries where applicable. Correct independent partial fields may retain
partial reward, but failed goal/process/safety never receives full success.

Unsupported capabilities fail explicitly; do not approximate physical behavior,
external effects or complex transactions with fabricated narrative feedback.

Every mutable fact required in an agent answer must be accessible through actual
permitted observations or actual user disclosures. Fixture answer recipes reading
data are not agent capabilities. Preserve unrelated records through invariants,
without requiring hidden unrelated balances in the answer. Transaction goals must
select the actual authorized object IDs and compare unrelated objects against
initial state; never require every existing row to become the selected row.
When the request requires an actual authorization refusal, include a separate
authorization stage with an actual false preference and a valid declined/no-write
trajectory. Revocation alone does not test user refusal. Enforce original minimum
result counts in an unconditional goal, rather than only activating observations
above that count. Exercise the exact threshold and one below it.

For a requested executor verification after writing, add required_observations
with after_tools:[TRANSACTION_ID,...]. These must name required transaction tools.
Only actual query results after the last successful call of every named write,
at the final current state_version, satisfy that observation. Earlier reads,
correct guessed answers and evaluator-only state checks do not verify the write.
For required insufficient-resource rejection/rollback tests, obtain exact consent
and valid actual bindings, then attempt the real transaction. Read-only predicates
are not evidence that a transaction was rejected or left all records unchanged.

Answer observability applies to EVERY success branch, including refusal/revocation:
an unchanged receipt is still an exact mutable answer obligation. Provide real
query access to required existing fields and record IDs, or restrict the answer
to the original requested observable facts. A write result exposes only its
changed fields and key, not unmodified hidden fields of a record.
Empty-menu probes must still use actual $user bindings for disclosed values;
mutate authoritative rows to remove candidates rather than replacing a bound
argument with a fabricated literal. After a query is filtered to return no rows,
never dereference items.0 as if a row existed. A stale-reference test must capture
an actual earlier row and reach the intended binding/recency protection, rather
than failing while resolving a nonexistent result. Do not relabel an accidental
empty-result failure as proof of stale-reference protection.

## Tool selection interference

The build pipeline adds up to two executable read-only partial views after the original
reference preflight and before independent review. `build(tool_interference=0..4)` controls
this budget (default 2); direct-response tasks remain tool-free. Only existing query
columns are projected; filters, pagination, user bindings and result dependencies are
preserved. No new private facts, unrelated data or writes are fabricated. Each added view
is executed using a valid reference prefix, and the full reference preflight runs again.
All added tools pass the same review, qualification, export and container gates.

Tools may carry internal `selection_role: task|distractor` (default task). Distractors must
be queries, are excluded from the simple-task core-tool count and default completion
obligations, and the label is stripped from public tools. Actual projected columns are
explicitly described publicly. Existing explicit success obligations are preserved.
This first version covers partial-view selection; cross-domain distractors and independent
measurement of robustness gains require separate evaluation. A successful build does not
prove training improvement.
