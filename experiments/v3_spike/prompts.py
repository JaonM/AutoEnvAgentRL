"""Prompts for the three model roles. Business content comes only from brief.md."""

AUTHOR = """You are building an executable business environment for agent RL training, in plain Python.

Files in this folder:
- brief.md: the business rules and the user's request (the only authority)
- data.json: initial tables
- facts.json: the user's private facts and decisions (an agent learns them only by asking)
- v3_runtime.py: the runtime (read it first), check_env.py: the self-check

Write env.py defining:
1. make_tools() -> dict of tool name -> function.
   - Read tool: fn(db, now, **args) -> JSON value the agent needs. db is a private copy.
   - Write tool: decorate with @write (from v3_runtime import write, BusinessError).
     fn(db, now, **args) mutates db in place and returns a JSON receipt.
     Raise BusinessError to reject; the runtime rolls the write back atomically.
   - Tools must enforce every rule in brief.md on whatever data they receive. They will be
     run on perturbed worlds, so never depend on the particular ids or values in data.json.
2. reference(api) -> dict: a deterministic reference agent that handles the user's request
   correctly using only api.call(tool, **args), api.ask(fact_key), api.confirm(fact_key, proposal).
   Every write needs consent: token = api.confirm(approval_fact_key, {{"tool": name, "args": exact_args}}),
   then api.call(name, consent=token, **exact_args). confirm returns None when the user declines;
   then that write must not happen. Catch BusinessError where the business allows partial success.
   Return the answer to the user as a JSON object.

Virtual time is the `now` argument, fixed at {now} seconds.
Keep table names and row columns exactly as in data.json. New rows (orders, refunds, receipts)
follow the record format in brief.md and the columns of existing rows of that table.
Standard library only. Do not read or write files outside this folder.

Run `{python} check_env.py` and iterate until it prints PASS. End with a two-line summary.
"""

ORACLE = """You are an independent auditor of a business process.

Files in this folder:
- brief.md: the business rules and the user's request (the only authority)
- data.json: initial tables; facts.json: the user's private facts and decisions
- check_oracle.py: the self-check (v3_runtime.py only provides NOW)

Write oracle.py defining expected(data, facts, now) -> {{"answer": ..., "final_state": ...}}:
- final_state: every table after all operations a correct handling performs. An operation
  happens only if the business rules allow it at that point AND the user's corresponding
  approval fact is true. Earlier committed operations stay committed unless the rules say otherwise.
- answer: the JSON answer the user should receive.
Compute from first principles; it will be run on perturbed worlds and flipped approvals,
so never depend on the particular ids or values in data.json.
Keep table names and row columns as in data.json; new rows follow the record format in brief.md
and the columns of existing rows of that table. Pure function, standard library only.
Virtual time is the `now` argument, fixed at {now} seconds.
Do not read or write files outside this folder.

Run `{python} check_oracle.py` and iterate until it prints PASS. End with a two-line summary.
"""

ARBITER = """Two independent implementations of the same business disagree.
- env.py: environment tools plus a reference agent (reference(api))
- oracle.py: an independent computation of the expected outcome
- mismatches.json: worlds where their final states differ, with both answers

brief.md is the only authority. For each disagreement decide which side violates brief.md,
and fix that file (fix both if both are wrong). Do not edit brief.md, data.json or facts.json,
keep both interfaces, and never special-case particular ids or worlds.

Run `{python} diffcheck.py`, `{python} check_env.py` and `{python} check_oracle.py`
until all pass. Write a short verdict to verdict.md (which side was wrong and why).
Do not read or write files outside this folder.
"""


def brief(sample, policy, data, facts, now):
    tables = '\n'.join(f'- {name}: {len(rows)} rows, columns {sorted(rows[0]) if rows else "(empty)"}'
                       for name, rows in data.items())
    approvals = [k for k, v in facts.items() if k.startswith('approve') and isinstance(v, bool)]
    return f"""# {sample['title']}

## User request (what the user says to the agent)
{sample['raw_request']}

## Business rules
{policy}

## Tables (full rows in data.json)
{tables}

## User private facts (facts.json)
The user knows these and reveals them when asked; approval facts ({', '.join(approvals) or 'none'})
are the user's answers when asked to confirm the corresponding operation.

## Virtual time
Fixed at now = {now} seconds for this exercise.
"""
