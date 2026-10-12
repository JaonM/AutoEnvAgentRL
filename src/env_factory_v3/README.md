# EnvFactory V3 (M1)

V3 builds agent-RL business environments as plain Python and qualifies them by independent
differential checks. It does not import V1 or V2 code. Design rationale and evidence:
the "EnvFactory V3 设计文档" and the 2026-10-12 spike (`experiments/v3_spike/`), where this
representation got 7/8 frozen complex cross-business samples correct against V2's
handwritten oracles (V2: 0/24).

## Flow

```
request.json ─► schema session ─► answer_schema.json
                     │
          ┌──────────┴──────────┐   (parallel, isolated folders, same schema)
   author session          oracle session
   env.py: tools+reference  oracle.py: expected(data, facts, now)
          └──────────┬──────────┘
          differential check on original, approval flips, 20 perturbed worlds
                     │ disagreement → arbiter (verbatim brief quote; ambiguous → report)
          branch audit: refusal cascades env and oracle share → fresh auditor (verbatim quote)
          gates: static · differential · negatives · category · http
                     │
               bundle/ (+ manifest hashes) ─► serve / train
```

Arbitration rules: every decision quotes the user request or the business rules verbatim (checked
by the host; the model-written answer schema and the platform's summaries in brief.md never count) and names the
worlds it covers. A rule that genuinely supports two readings is reported, never decided: on the
original world or an approval flip the build fails as `spec_ambiguous`; when only perturbed worlds
are affected, those worlds are excluded from comparison and from training and listed in
`spec_findings` for the business owner.

Branch audit: env and oracle can make the same mistake, which the differential check cannot
see. The likeliest one is what happens after the user refuses an approval, because the original
world approves everything. When an approval flip makes the reference drop a step it takes in the
original world after that approval (a later confirmation, or a tool call with arguments never seen
in the refused world), a fresh auditor that sees only the brief and the two traces must justify
each cascade with a verbatim quote: `supported` passes, `ambiguous` fails as `spec_ambiguous`
(ordering words alone never decide it). On `unsupported`, a repair session (one unit of repair
budget) fixes env.py and oracle.py together from the auditor's finding. The build fails as
`branch_unsupported` when no budget is left, when the checks fail afterwards, or when the refusal
world still calls a skipped tool, or asks a skipped approval, fewer times than the original world
(the refused write itself excepted). The comparison is by count because arguments may
legitimately differ once a refusal has changed the state. Builds without a cascade run no audit; the audit itself spends no budget.

Perturbation audit: agreement on a perturbed world also proves nothing when both sides ignore
the same rule. Example: an inactive subscription that still gets refunded. The audit runs when a
perturbation changes a row the task acts on (named by the user, written, or reported) yet the
reference commits the same writes and gives the same answer as on the original world. A fresh
auditor gives a verdict per world: `consistent`, `should_change` (with a quote), or `ambiguous`.
A status flag changed on a row the task writes is `ambiguous` unless a rule covers that state.
Rules both sides ignore are repaired on both sides (one unit of budget). Unrepaired and ambiguous
worlds are excluded from training and reported in `spec_findings`.

Answer contract rule: every outcome is graded exactly (ids, numbers, booleans, string enums).
User-facing prose must be marked `"graded": false`; the schema check rejects fields that look like
free text without an enum or that mark. Two correct implementations never agree on wording.

One repair budget (default 2, so an arbitration and a branch repair can both happen) covers every
corrective session. Infrastructure refusals (OpenAI 403/429/5xx) and sessions that time out before the model
produces a single token (network down) are retried once after a backoff and never count as
business failures.
At most two codex sessions run at once.

## Artifacts

| file | written by | meaning |
| --- | --- | --- |
| `request.json` | operator / `import-v2` | user request, business rules, tables, private user facts, frozen clock |
| `answer_schema.json` | schema session | the final answer contract shared by env, oracle and reward |
| `env.py` | author session | `make_tools()` (read tools, `@write` tools) and `reference(api)` |
| `oracle.py` | oracle session, never sees env.py | `expected(data, facts, now) -> {answer, final_state}` |
| `audit/audit.json` | audit session, sees neither implementation | verdict per refusal cascade (only when one exists) |
| `perturbation_audit/perturbation_audit.json` | perturbation audit session | verdict per outcome-insensitive perturbed world |
| `bundle/policy.json` | `policy` command | policy gate episodes, pass@k and success rate |
| `gates.json` | platform | every gate result and the worlds env/oracle agree on |
| `manifest.json` | platform | file hashes and V3 platform identity |

## Reward (no author-written reward code)

`reward = 0.95 · business + 0.05 · efficiency`, derived from the oracle. Business: the final
state must equal the oracle's (half the weight when the task changes state) and each answer
field must equal the oracle's (sharing the rest). Efficiency is paid only on success, so a wrong
or empty answer never earns it. A write attempted without valid consent is a
safety violation and zeroes the reward. Episodes may be served on any world the env and oracle
agree on, so memorised answers do not transfer between worlds.

## Gates (all mandatory)

1. **static**: sandbox rules (no I/O, no clock, no reflection), a valid answer schema, and a type
   annotation plus an `Args:` description for every tool parameter (the agent sees nothing else)
2. **differential**: reference and oracle agree on the original world, every approval flip and at least five perturbed worlds; the reference earns 1.0
3. **negatives**, all derived automatically: do nothing, replay a stale answer on a changed world, skip consent, stop after the first write, forge consent (must earn 0)
4. **category**: the reference trace has the structure the category needs (direct: no tools; simple: reads only, no result dependency; multi-step: ≥2 calls with a result dependency or a write)
5. **http**: the reference replayed through the HTTP service earns the same full reward

**Policy gate** (`policy BUNDLE --k 4`, not yet part of `build`): an independent
OpenAI-compatible model plays k episodes through the HTTP service and the shared agent view
(`agent_view.py`, which is also what `rl` trains on), one per served world. The gate passes when
at least one episode succeeds. Provider failures are infrastructure errors, not task results.
An adversarial agent is M2. `policy --user-mode natural` plays through the free-dialogue user.

## Commands

```sh
.venv/bin/python -m env_factory_v3 import-v2 --frozen output/env_factory_v2_cross_business_frozen_01 --sample 4 --output r.json
.venv/bin/python -m env_factory_v3 build --request r.json --output output/v3-build-04
.venv/bin/python -m env_factory_v3 batch --frozen output/env_factory_v2_cross_business_frozen_01 --samples 4,5,9 --output output/v3-batch
.venv/bin/python -m env_factory_v3 verify output/v3-build-04/bundle [--resign]
.venv/bin/python -m env_factory_v3 docker output/v3-build-04/bundle
.venv/bin/python -m env_factory_v3 serve output/v3-build-04/bundle --port 8090
```

Model sessions run `python -m env_factory_v3.checks {schema|env|oracle|diff}` in their folder.

## Delivery and training

`docker BUNDLE` packs this package and the bundle into `envfactory-v3:<identity>`. It runs every
gate again inside that image with no network, read-only root, all capabilities dropped,
no-new-privileges and a non-root user. On success it records the image in `docker_image.json`
next to the hashed files.

Changing any V3 platform file invalidates every bundle. `verify BUNDLE --resign` re-runs every
gate on the current platform and re-signs the bundle only when they all pass.

`rl` trains on V3 bundles directly (`--tasks` with bundle paths): `--sandbox-backend docker` starts
each environment from its verified container (`V3DockerEpisode`); `local` serves it in process
(`V3Episode`).

## Free-dialogue user

`serve --user-mode natural` (or `ENVFACTORY_V3_USER_MODE=natural` for `rl`) replaces ask/confirm
with free dialogue. A plain assistant message is speech to the user, and the simulated user's
reply comes back as the next user turn. `v3_chat {message, proposal}` asks for approval of one
exact write. Only `v3_answer` ends the episode. The agent sees no topic keys: only the user's opening, the rules
and the answer schema. A simulated user (`user_sim.py`, model from `V3_USER_*`, falling back to
`LLM_*`) knows the request and the private facts and answers in plain Chinese. It never decides
consent. When a chat message carries an exact write proposal, the platform maps it to its
approval topic with `env.approval_key(tool, args, facts)` and reads the user's approval fact.
The simulator is only told the decision so it can phrase it. Reward therefore stays as
deterministic as with the structured user. `gates.json` records natural readiness: `approval_key`
must name the topic the reference confirms on every approval world. The structured user does not
need it.

## Business owner loop

`findings BUILD` turns a build's `spec_findings` into questions for the business owner.
`clarify --request R --answer TEXT --output R2` adds the answers to the request as
`clarifications`. They become part of the business rules in brief.md and can be quoted by
verdicts. Rebuild from R2.

## Generating requests

`generate --domains domains.json --count 30 --output DIR` gives each request its own model
session. Each one must pass `checks request`: a valid request, an opening in a real user's voice
(red flags: amounts in 分, transaction-semantics phrases), and boolean `approve_*` facts.
Near-duplicates are dropped and the V2 category mix is kept. Then run `batch --requests DIR/requests`.

## HTTP protocol

`POST /reset {world?|seed?}` → episode id and public instructions (request, rules, tool specs,
askable user topics, answer schema). `POST /step {episode, action}` with action types `tool`,
`ask`, `confirm`, `wait`, `answer` (natural user mode: `chat`). With `V3_SERVICE_TOKEN` set, every
POST needs `Authorization: Bearer <token>`; `rl` gives each environment its own random token. `POST /reward {episode}` after the answer; `POST /release {episode}` frees it. Business
rejections are observations (`ok: false, kind: business`).

## M1 limits

- The clock is fixed unless the data has a `scheduled_events` table. With one, time passes only
  through `wait` (`v3_wait` for the agent), and the events (`{id, at, table, row_id, set}`) change
  rows as the clock passes them. Tools never see that table.
- The free-dialogue user is LLM-simulated, and its replies are not graded. The final answer is
  still the structured `answer` object, not a natural-language reply.
- The serving container publishes its port on 127.0.0.1, so it keeps a bridge network. Only the
  container verification runs with `--network=none`; env.py itself is limited by the static
  sandbox rules (no I/O, no clock, no reflection).
- Passing gates proves the tested obligations only, not learning effectiveness.
