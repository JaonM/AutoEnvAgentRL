# V2 executable business contract

All contracts are JSON data. No Python, SQL, eval, network code or arbitrary filters.
The implementation in contract.py is authoritative and rejects unknown keys.
Generator examples are demonstrations only; generation must design a new business.

Required top-level keys: version="2.0", id, category (direct_response,
simple_agentic, multi_step_agentic), task (original public business goal), rules,
tables, tools, user, answers, scenarios, probes, limits, time.
Optional: explanations, invariants, events, success, goals.
At least one positive scenario must succeed on the original seeded data without
any initial mutations or user_overrides, including inherited variants. Changed-data positives
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
bounded_cover:{items:EXPR,target:EXPR,max_states:1..10000,
max_transitions:1..1000000,max_plans:1..10000} exactly solves bounded integer
covering. Each item is {key:unique nonempty string,units:positive integer,
cost:nonnegative integer,minimum:positive integer,maximum:nonnegative integer}.
Choose zero or minimum..maximum copies per group; cover target (nonnegative
integer), minimizing total cost, then total units. Returns
{feasible:boolean,cost:integer|null,quantity:integer|null,plans:[[{key,count}]]}.
All equally optimal plans are retained; no hidden key tie-break chooses one.
Infeasible returns null totals and empty plans; zero target returns zero and [[]].
Use named public rules to select eligible source records before mapping these
coefficients from their current fields. The optimizer does not invent eligibility,
authorization, budgets, expiry or tool evidence. Any computation budget excess
raises an explicit error instead of claiming infeasibility or truncating optima.
range:{start:EXPR,stop:EXPR,max_items:1..10000} constructs integers from start
inclusive to stop exclusive, with integer endpoints. Empty/reversed intervals
return []. Exceeding max_items raises an error; candidates are never truncated.
flat_map:{items:EXPR,value:EXPR} evaluates value with each item as row and
concatenates the resulting lists (at most 10000 output items). Non-list bodies
and budget overflow fail explicitly. Nested flat_map/map with lexical captures
can construct bounded Cartesian products from actual dynamic stock ranges;
keep business eligibility in named public rules and use sort for public ranking.
paths use {"ref":"data.accounts.0.balance"}. Roots depend on context. Operators:
eq/ne/lt/le/gt/ge/in/contains/add/sub/mul/at take two expression operands;
floor_div/ceil_div take two integer operands with a nonzero divisor and return
the exact integer floor/ceiling of their quotient, including negative values.
Every fixed-arity operator requires a JSON ARRAY of expression objects, even
when its arity is one. Never write {"is_integer":{"ref":"args.boxes"}}.
Correct unary syntax is {"is_integer":[{"ref":"args.boxes"}]}.
is_integer takes one operand and accepts only an integer (never boolean or float).
rational_sum_floor takes one expression returning a list of at most 10000
[integer numerator, nonzero integer denominator] pairs. It sums exact rational
terms before flooring once, returning integer JSON, without floating reciprocals.
For prorated billing, map each segment to [price_cents * days, period_days], then
apply rational_sum_floor to the complete list. Never floor each segment first.
For example, a precomputed segment-pair list uses
{"rational_sum_floor":[{"ref":"vars.segment_pairs"}]}.
An inline map must also be inside the operand array:
{"rational_sum_floor":[{"map":{"items":{"ref":"vars.segments"},"value":
{"list":[{"mul":[{"ref":"row.price_cents"},{"ref":"row.days"}]},
{"ref":"row.period_days"}]}}}]}.
The outer array contains ONE expression; the inner returned list contains the
rational pairs. Do not remove the outer array or wrap each pair as an operand.
For refunds use floor_div([paid_cents * unused_days, period_days]); for carton
coverage use ceil_div([required_units, pack_units]). These examples describe
operand values; actual operands must use the expression objects below.
and/or take nonempty expression lists; not/count/sum/unique take one expression
operand in a list; list takes expression operands; object maps field names to
expressions; if takes {condition,then,else}; filter/map take {items,value} with
row available while evaluating value. sum expects its single expression to return
a numeric list. Type-strict comparisons; false and zero are real observations.
at takes [collection_expression,index_expression].
let takes {bindings:{NAME:EXPRESSION},value:EXPRESSION}. Bind 1..16 names
(ASCII letter followed by letters/digits/underscores, at most 64 characters).
Bindings evaluate in the outer scope simultaneously; the body reads vars.NAME.
Nested lets shadow locally and do not mutate their caller. Capture an outer
row key with let before a nested filter/map: inner row refers to the inner
collection, while vars.project_id retains the captured outer project key.
To capture an outer integer index, place let outside the inner filter and use
vars.index as the second operand of at. If another binding needs that index,
put it in a nested let body: sibling bindings cannot read one another.
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
data and time remain in scope. Public rules may invoke other named public rules
with exact arguments. The dependency graph must be acyclic and at most 16 rules
deep. Indirect operation-row fields are validated on every transaction table;
collection-local rows remain local. Unknown rules and cycles fail compilation.
A row-dependent rule must be called inside filter/map or an operation-row
context. Global answer, branch and probe expressions cannot call it directly;
select the authoritative record inside a collection expression first.

rules: list of {id,public,predicate}. public explains an actual business condition;
predicate can reference row,args,data,initial,user,time. Tool filters are ONLY references
to these rules, never handwritten extra predicates. A display name cannot become
an identity constraint without an original public requirement.

tables: {id,schema,primary_key,rows,foreign_keys?}. Row object schemas declare
properties, required, additionalProperties=false. Supported types string, integer,
number, boolean, object, array; optional enum and nullable=true. Arrays require
items schema. Optional minItems/maxItems constrain array lengths; both are
integers with 0 <= minItems <= maxItems <= 10000 (defaults 0 and 10000).
Only arrays accept these keys. Do not impose a fixed result length when valid
empty or different-size result sets are permitted by the public business rules.
Mixed scalar arrays such as [quote_id, boxes] can use
{"type":"array","items":{"type":["string","integer"]}}. Scalar type unions
allow string/integer/number/boolean only, require unique type names, and still
reject booleans as integers. Keep public business rules responsible for tuple
position and quantity constraints; a union alone does not enforce them.
Use nullable=true for null; unions do not accept object/array/null variants.
foreign_keys contain {column,target_table,target_column}. Unique
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
Bound private facts must satisfy the argument schema at compilation. Arrays and
objects require their actual items/properties schemas; never stringify structured
plans or declare them as strings to bypass exact proposal validation.
result_bindings require actual observed references, not matching literals;
path may use '*' for a numeric result-array index, e.g. items.*.id.
path may also be a nonempty list of up to 16 distinct paths: any one is allowed.
For example, one stock_id binding may allow items.*.source_id or items.*.target_id.
Separate bindings remain simultaneous requirements; do not express alternatives
by repeating an argument across bindings. Providers and paths are independent
allowed sets, and runtime still requires an actual observed result reference.
The binding tool can be a list of permitted providers. A tool may consume its own
earlier result: runtime observations must already exist, preventing temporal cycles.
Each result binding may specify recency="latest" (default "any"). latest requires
the observation ID of the most recent successful business tool call, in addition
to the provider and path constraints. User, time, wait, state and collect events
do not replace that call; failed calls do not produce results.
An observation produced by collect/traverse may satisfy latest when its verified
final query page is that most recent successful business call. Collecting old pages
after an intervening successful read does not make those pages current. For a chain requiring
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
actual observed menu; missing preference may legitimately refuse.
An observed menu or source_id is evidence for the simulator, not proof that the
assistant communicated choices to the user. When the public request asks to see
options before choosing, actual assistant text must present the eligible objects
with their public identifiers or names before selection. “All options are above”
without actual displayed options does not satisfy that obligation. The reward
contract must check this delivery and reject an otherwise correct transaction
when delivery is omitted. Do not require duplicate final-answer fields for
information already truthfully delivered in dialogue. Authorization
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
named public rules (or boolean combinations of them). Declared foreign-key to
primary-key equality joins may additionally use scalar lexical captures. Membership
joins also support two nonnullable foreign keys to the same declared primary key.
Existence joins may use gt(count(filter(...)), literal 0), including nested lexical
captures, when every inner filter obeys these public-rule/relationship restrictions.
Other count thresholds are business conditions and require their own public rule.
Nullable sibling keys never establish a common entity merely by both being null.
Membership
joins require a key collection selected by public-rule filters. Matching values or
column names alone never establish a relationship; literal keys, undeclared joins,
and additional unshared business conditions remain invalid. A let wrapper may capture
arguments before invoking those rules; its value still obeys the same restriction.
Never duplicate a tool's
eligibility predicate manually in an expected expression. Use filter/map with
the rule operator to compute the authoritative complete set independently of
the agent's possibly incomplete tool observations. Scenario/challenge answer
recipes may use independent expressions to challenge the rule itself.
explanations: [{id,meaning,criteria:[{id,rule_id,obligation,when?}],context_fields:[answer_id],weight}].
Optional criterion `when` is a boolean EXPR over data,initial,user,user_state,time.
It scopes a communication obligation to the actual required business branch;
it cannot depend on submitted answers or policy evidence. Only applicable criteria
are judged. A non-boolean condition is an error, not an exemption. If all criteria
are inapplicable, the component is satisfied without a semantic model call.
Criteria describe invariant reasoning obligations over actual runtime facts,
not immutable wording or mutable fixture content. Code Agent grades explanations
against authoritative evaluated context_fields, actual user turns, and successful
executed business tool observations (actual arguments, results, times and snapshot
versions). No undisclosed table or private user facts are supplied to the judge.
Observations support explanations of actual relationships, validity intervals and
verification; stale observations cannot override the evaluated current target.
Declare obligations that have such evidence; do not require reciting unused
general rules or facts absent from both context_fields and actual observations.

invariants: [{rule_id,predicate:EXPR}], roots data,initial,user,time.
goals: [{id,rule_ids,predicate:EXPR}], roots data,initial,user,user_state,time.
Alternatively use {id,rule_ids,transaction_plan:[{tool,arguments,when?}]} for an
event-free world with time-independent transaction operations. Intention arguments
and optional boolean when use initial,user,user_state only. Expected final state
is projected from initial data by the referenced transaction operations; all tables
and columns must match, independent of record order. Plans cannot copy actual
observations or final data. For scheduled events or time-dependent operations use
transaction_timeline with the same intention format: actual successful commits
must match intended tool/arguments in order. Projection uses actual commit clocks,
applies due events before each commit and remaining events through episode end,
and rejects missing or extra successful commits. Result summaries are not truth.
Static plan intentions may use completion="consent_or_unavailable" instead of when.
Arguments then depend only on initial data and private user facts. Target existence
and declared transaction guards derive availability; unavailable intentions need no
consent, while available ones require the exact commit or explicit refusal/revocation
bound to the intended proposal. Missing, extra and different commits fail. Static
commits match by tool and arguments rather than declaration order: independent
operations may be reordered only if the entire resulting world still equals the
declaration-order projection. Explicit authorization dependencies remain enforced.
Time-dependent or event-driven consent completion retains explicit goals; temporal
plans still match commits in order. Every table and column, including unrelated
records and tables with no planned operation, remains part of the state comparison.
Exactly one of predicate, transaction_plan or transaction_timeline is required.
Keep explicit predicates for other unsupported dependencies; never simplify the
original business merely to use the projection form.
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
Direct has zero business tools, stages and tables; simple requires at least one
business tool and may use multiple necessary reads for the same user goal.
The task category is preserved from the request; table normalization or splitting
a read into tools must not force recategorization or hidden fixture shortcuts.
Multistep has at least two tools and an
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

scenarios: [{id,steps,answer:{answer_id:EXPR},expect_success:boolean,mutations?,expect_error?,user_overrides?}].
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
question may be text or a closed observed template:
{"parts":["当前可选地点：",{"$result":"sites_result","path":"items"},"。请选择。"]}.
The runtime renders actual observed values as JSON text, retains the rendered
question in the user trace, and rejects unknown observations or overlong text.
Only literal text and $result/path references are allowed (at most 32 parts).
No hidden data, undisclosed user facts, arbitrary expressions or template code.
Use templates for dynamic menus in reference scenarios; final answers need not
repeat choices already delivered before selection. Free policies still send
ordinary text through v2_chat; templates do not add a new agent-visible tool.
Wait steps: {kind:"wait",seconds}; revoke: {kind:"revoke",stage}.
Clock observation: {kind:"time"} reads the same virtual clock as the public
GET /time and v2_time interfaces, without advancing time or business-call counts.
It records an explicit platform observation; timestamps automatically attached
to other events do not prove an agent clock read. A success branch may declare
required_platform_observations:[{capability:"time",freshness:"any"|"current"}]
when the original public requirement requires that observation. "current" means
the observed elapsed_seconds equals the elapsed_seconds at grading; "any" accepts
an earlier explicit read. This is conditional on the branch being active. Do not
add clock-read obligations merely because the platform supports them. Use this
deterministic evidence instead of requiring extra answer prose about reading time.
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

probes: [{id,scenario,rule_ids,mutations,expect_success,answer?,expect_error?,user_overrides?}].
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

## Business-first training reward

Final `reward = 0.95 * business_reward + 0.05 * efficiency.score` when business success
is complete; otherwise it is `0.95 * business_reward` with no efficiency credit.
`business_reward` retains the original evidence/safety-gated weighted answer score.
`success` depends entirely on business checks, never on efficiency or reward equalling 1.
Efficiency is `1 / (1 + redundant_reads)`: duplicates of the same successful query page,
projection and data version. First reads, required pagination, changed snapshots and
failed-call retries are not penalized. This intentionally does not claim to measure
all unnecessary work. No separate reward is granted for calling or avoiding a tool label.

Equivalent query providers can satisfy required reads and observation requirements when
they use the same table, rule IDs, parameter schema, user/result dependencies and return
all required columns. Transaction and after-write phase obligations remain explicit.
A partial projection cannot substitute for a required full projection. Business success
with redundant reads is still success; training and reporting must use the `success`
flag rather than a reward==1 shortcut.

## Native free-dialogue user

`user.profile` is an optional private object for communication and preferences. Its
optional `disclosure` maps existing fact names to `never`, `on_request` or `when_needed`.
Do not include computed task answers. The export includes a private `user_profile.json`;
no private profile, event catalog or grading targets are published to the task policy.

Formal V2Episode defaults to `free_dialogue`. It requires the local native connection
for agentic tasks; no rule fallback is permitted. The old
qualification/reference interface stays structured and its certificate explicitly
states that free-dialogue rollout has not been verified by reference replay.

The policy sends `v2_chat(message, proposal?, source_id?)`; it never supplies a stage ID.
A separate native user context proposes zero or more private events and a natural reply.
An independent model call checks reply/event agreement using server-validated event
results. There is at most one correction. Events, messages and state changes commit
atomically only after both checks. Model faults are infrastructure failures and mark
trajectories non-trainable. This entails two native calls per successful interaction,
up to four for a correction; semantic validation is model-based, not a formal proof.

Each validated dialogue turn advances virtual time once, including clarification; composite
events do not count as multiple turns. User state uses a ledger and business prerequisite checks, not HSM transitions. Existing
stage declarations are a private event catalog for compatibility with business contracts.
Legacy state labels are maintained for existing predicates, not exposed as a dialogue
sequence. Follow-up questions and composite disclosure do not require fixed turns.
Canonical `disclosed_facts` accompany natural replies so business arguments use exact
values without guessing string normalization. Only already-disclosed values are returned.
Actual complete menus and exact proposals are required for choices and authorization.
Approved writes return tokens bound to exact tool/arguments. In free mode transactions
require `_v2_authorization`; revoked, replaced or consumed tokens cannot be reused.
Profile-controlled withdrawal cannot be manufactured by a policy tool.

The private continuation captures dialogue, facts, ledger and tokens, and rejects mode
mismatches. Free dialogue can be enabled on the standalone server using
`--user-mode free_dialogue`. `validate_v2_free_dialogue.py` runs diagnostic native
policy/user HTTP rollouts without claiming a fresh yield or training improvement.

## Demand provenance

Optional `demand` separates immutable `user_request`, private `task_spec` and public
`agent_context`. Task wording must equal the original user request. Every task-spec
requirement cites the exact user expression or publicly available business policy;
hidden or invented requirements cannot compile. The graph evidence is private provenance,
not vocabulary injected into user openings. The public agent context includes the complete
applicable policy, so the agent is not scored against inaccessible business instructions.

New layered demands split public clauses into individually identified requirements.
Each requirement has `id`, `text`, `source`, `visibility: "public"` and a `citation`
with zero-based Unicode character offsets `start` (inclusive) and `end` (exclusive)
within the named public source. The text must equal that exact slice, IDs must be
unique, and all substantive source characters must be covered. Existing frozen
whole-document citations remain readable. Textual coverage proves origin, not that
an executable rule faithfully implements every business meaning; independent review
and counterexamples must still verify that interpretation.

For a layered demand, the independent business challenge reviews each cited public
clause exactly once. Its response links the clause to implementation locations from
a finite contract-relative catalog and to actual independently executed case IDs.
Declared table schemas, primary keys and foreign keys may support integrity
obligations; fixture rows and seed generators cannot establish implementation
coverage. They remain input data for independent execution.
Unknown, missing, duplicate or private-fact citations cannot establish coverage.
A failed clause interpretation remains inconclusive even if every proposed case
passes or a subsequent case-label audit changes its overall verdict. This is
model-reviewed semantic evidence, not a formal proof of business completeness;
retain independent execution, reward counterexamples and held-out measurements.

Business graph ingestion uses separate EFV2 labels and frozen namespaces, preserving
existing scene nodes. Objects, policies and document hashes come from explicit dossiers;
synthetic imports remain marked synthetic. Snapshot retrieval verifies document hashes
against the frozen dossiers. It does not claim production knowledge or original scene
coverage merely because the records are stored in Neo4j.

Qualification-only user_overrides is an object mapping existing private fact keys
to values with the same exact types and object shape as the base fixture. Unknown
keys and boolean/numeric substitutions are rejected. A scenario replaces inherited
overrides; probes and independent cases overlay their explicitly supplied keys.
Use {"approve_STAGE":false} to exercise direct refusal without changing base
user.facts. Baseline and training-reference scenarios cannot contain overrides.
These variants never become public reset parameters or initial agent context.
Independent review cases carry user_overrides_json="{}" when unchanged.

Business relation robustness: resolve associated records by declared keys, not
array positions. State goals and invariants must survive schema-valid reorderings
and reciprocal association changes. Include executable probes for those changes.
A linked entry may itself be terminal; its observed terminal identifier must be
a valid provider for subsequent owner lookup. Success branches must not require
a successor call when no successor exists. Retain observed-result provenance.
