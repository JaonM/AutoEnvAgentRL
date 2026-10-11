"""Retired generation transports, for explicit historical diagnostics only.

Not imported by production env_factory_v2. No CLI or automatic fallback.
"""
import json
from pathlib import Path
from env_factory_v2.native.code_agent import object_schema
from env_factory_v2.contracts.contract import ContractError, require
from env_factory_v2.contracts.expressions import canonical


def generate_legacy(self, request, *, feedback=None, source_object=False, source_tree=False):
    require(type(source_object) is bool, 'source object selector must be boolean')
    require(type(source_tree) is bool and not (source_object and source_tree), 'invalid JSON tree selector')
    grammar = ((Path(__import__('env_factory_v2').__file__).parent / 'BUSINESS_DEFINITION.md').read_text() + '\n\n'
               + (Path(__import__('env_factory_v2').__file__).parent / 'CONTRACT.md').read_text())
    previous = feedback.get('previous_source') if feedback is not None else None
    # Capability refusals are parsed JSON too, but contain no contract to edit.
    # Retry generation within the existing attempt budget, retaining the refusal
    # as feedback. Partial contracts still use atomic repair.
    contract_candidate = isinstance(previous, dict) and bool(
        {'task', 'tables', 'tools', 'answers', 'scenarios', 'rules'} & previous.keys())
    if contract_candidate:
        from env_factory_v2.authoring.repair import apply_edits, recover_edit_values
        previous = feedback['previous_source']
        evidence = {key: value for key, value in feedback.items() if key != 'previous_source'}
        if 'original_request' in evidence and canonical(evidence['original_request']) == canonical(request):
            evidence.pop('original_request')  # Supplied once as original_request below.
        value, provenance = self.ask(
            'Repair the supplied contract using atomic JSON Pointer edits, not a complete rewrite. '
            'Preserve every original business obligation and difficulty. Repair the actual executable '
            'evidence; do not change expectations to hide unintended errors or remove required tests. '
            'For an intentional invalid action, use its actual error text as expect_error only if the '
            'action really tests that intended protection. Prefer the smallest coherent edits. '
            'Check each negative recipe against its own effective private facts and resolved arguments. '
            'A dynamic $user reference follows user_overrides and cannot create a mismatched binding. '
            'A literal equal to the base disclosed value is also valid. To test a wrong binding use an '
            'explicit scenario with truly different schema-valid arguments and facts; preserve the '
            'baseline facts and all valid positive paths. Do not relabel an invalid fixture as passing. '
            'Each edit has op add/replace/remove, path beginning /, and value_json containing JSON '
            '(use null for remove). Paths refer to the result of earlier edits; - appends to arrays. '
            'value_json must parse as JSON: string replacements include their JSON quotation marks; '
            'never place unquoted prose in value_json. '
            'For consent repairs distinguish direct refusal (consent.declined) from later withdrawal '
            '(consent.revoked). Preserve separate executable no-write success paths and correct '
            'submission answers for both; never substitute a revocation scenario for a decline. '
            'Never execute tools.',
            {'original_request': request, 'grammar': grammar, 'previous_source': previous,
             'repair_feedback': evidence}, object_schema({'edits': {'type': 'array', 'items': object_schema({
                 'op': {'type': 'string', 'enum': ['add', 'replace', 'remove']},
                 'path': {'type': 'string'}, 'value_json': {'type': 'string'}})}}))
        try:
            edits, format_evidence = recover_edit_values(value['edits'], self)
            return apply_edits(previous, edits), {**provenance,
                **({'format_repair': format_evidence} if format_evidence else {})}
        except json.JSONDecodeError as exc:
            raise ContractError(f'repair value invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}') from exc
    from env_factory_v2.native.response_format import SOURCE_OBJECT_SCHEMA
    from env_factory_v2.native.json_tree import TREE_SCHEMA
    tree_instruction = ('Return source encoded in the closed JSON tree wire grammar: '
        'an object is {"o":[{"k":KEY,"v":NODE},...]}; arrays are ordinary JSON arrays '
        'of encoded nodes, and strings/numbers/booleans/null are ordinary JSON scalar values '
        'with no tag wrapper. Encode every object recursively; no raw contract JSON string, '
        'no omitted fields, duplicate keys, extra tags or placeholders. Preserve actual string '
        'characters, not their printed escape sequences. The decoded source must '
        'satisfy the full business grammar. Wire encoding must not simplify any obligation. '
        if source_tree else '')
    value, provenance = self.ask('Generate a NEW business contract following the supplied grammar. '
        'Return source as {"environment": ENVIRONMENT, "task": TASK} with separate version 1.0 '
        'documents as described in BUSINESS_DEFINITION. The compiler assembles executable version 2.0. '
        'Keep tables, rules, operations and time in the environment; task goals cannot override them. '
        'Use derived reports for complete unordered query record collections so eligibility and '
        'field types come from the environment, not a duplicate reward implementation. Preserve '
        'the requested output scope: tools may expose additional fields without making them '
        'mandatory answers. Include only justified report columns and reward-bearing fields. '
        'natural communication as task messages referencing shared environment communications; '
        'do not encode freely worded explanations as exact expected sentences. Ground conditional '
        'communication applicability in actual business requirements, covering both branches. '
        'For time-independent transactions in event-free worlds, use transaction_plan goals '
        'to derive intended final state from environment operations and intended parameters. '
        'For conditional authorization use intention completion=consent_or_unavailable without when: '
        'derive availability from transaction preconditions, require exact execution or bound refusal/revocation '
        'when available, and no pointless consent when unavailable. Do not duplicate eligibility in handwritten '
        'consent-completion goals or unconditional process stages. Arguments must not depend on actual consent state. '
        'Use transaction_timeline for scheduled events and time-dependent operations; use explicit '
        'state predicates for other unsupported dependencies. Never remove capabilities or lower '
        'difficulty to fit a projection form. '
        'ranked, scalar, transactional and semantic obligations with the full supported grammar. '
        'This is full generation, even when repair_feedback describes an earlier failure. '
        'There is no editable previous contract in this branch: generate the complete source from '
        'original_request and grammar; never return generation_incomplete or demand an omitted source. '
        'Preserve the original request and its category. No Python code. Model roles may not invent hidden '
        'business filters or freeze mutable facts in reasoning criteria. Include grounded positive/negative '
        'scenarios and data counterexamples. If repairing, address executable evidence without weakening '
        'the original goal. Keep descriptions concise, use seed generators for bulk data and probes '
        'to reuse trajectories rather than duplicating long scenarios. '
        'Use scenario extends and append_steps to share long reference trajectories and express only '
        'the differences; each inherited case still executes in full. '
        'Preserve all required negative branches, dependencies, scale and transaction objects. '
        'Derive related records by declared primary/foreign keys, never table row positions. State goals '
        'A result binding path can be a string or a list of alternative paths. For a shared lookup '
        'parameter accepting either source_id or target_id, use one binding with those paths as a '
        'list; separate bindings are simultaneous requirements, not alternatives. '
        'Nested filter/map replaces row with the inner collection row. Capture outer keys using '
        'let bindings before entering the inner collection and reference vars.KEY; never use inner '
        'row.id as if it were the outer record ID. '
        'and invariants must remain correct after schema-valid row reordering and reciprocal association '
        'changes. A reverse foreign key is not necessarily unique: update and grade only the forward '
        'linked record selected by the transaction, and preserve other rows even if their reverse key '
        'matches. Probe nonreciprocal but schema-valid associations. Cover these changes with executable '
        'probes. For linked traversal an entry can already '
        'be terminal: allow its observed terminal ID to feed the owner query, use disjunctive result '
        'binding providers when needed, and require successor calls only when a successor exists. '
        'Unless the original public rules guarantee a matching record, include a normal empty-result '
        'completion with truthful schema-valid answers and no write. Make downstream record queries, '
        'menu selection and consent conditional on observed existence and nonempty availability. '
        'An empty upstream query cannot provide a downstream result binding; do not unconditionally '
        'require that downstream tool for the empty branch. Include an executable empty-result '
        'scenario or probe using existing private fact overrides or schema-valid data changes. '
        'Preference correction and clarification must not require preliminary consent: consent is '
        'for an exact actual write proposal, after final preferences and current choices are known. '
        'Authorization tasks must cover direct refusal (consent.declined) separately from later '
        'withdrawal (consent.revoked). Include an executable positive no-write scenario for direct '
        'refusal using scenario user_overrides={approve_STAGE:false}, unchanged business data, and a matching '
        'success branch. Keep base user.facts exactly unchanged; overrides are qualification-only. '
        'For sequential transactions preserve earlier committed writes when later consent is refused. '
        'Transaction operations support optional create:true to atomically insert a new row. '
        'Operations also support when:EXPR, a boolean applicability condition over pre-transaction '
        'data/args/user/user_state/time (no operation row). Skipped operations do not resolve targets '
        'or run row guards. All skipped is unavailable, never a successful commit. Use this for '
        'genuine partial applicability, grounded in original rules; do not skip required writes to '
        'avoid failures. Enabled operations still obey all guards and commit atomically. '
        'Use key for its primary key and set for every required non-key column; creation set expressions '
        'cannot read row because the row does not exist yet. A duplicate key fails the entire transaction '
        'without consuming consent. Keep ledger insertion and corresponding balances or stock updates '
        'in one transaction. Grade actual created rows and include omission/duplicate/refusal tests. '
        'required_observations columns must be the minimum publicly required output or decision evidence, '
        'not every column returned by a tool. Eligibility enforced by the same named query rules need not '
        'be separately displayed when the user asks only for eligible IDs. Preserve necessary decision '
        'fields, complete pagination and result dependencies. Accept committed transaction results '
        'as write evidence when sufficient; require a post-write read only when a public business '
        'obligation or missing result evidence makes it necessary. Do not require redundant reads. '
        'Use numeric comparison for scalar number answers derived from arithmetic; keep exact comparison for discrete values. '
        'Include user.profile describing communication style, disclosure boundaries and preferences, '
        'consistent with private facts and original request; never store computed answers there. Do not execute tools. '
        + tree_instruction + ('Return source as a JSON object, not a quoted JSON string. ' if source_object else ''),
        {'original_request': request, 'grammar': grammar, 'repair_feedback': feedback},
        TREE_SCHEMA if source_tree else SOURCE_OBJECT_SCHEMA if source_object else object_schema({'source_json': {'type': 'string'}}),
        **({'source_tree': True} if source_tree else {'source_object': True} if source_object else {}))
    if source_tree:
        from env_factory_v2.native.json_tree import decode_response
        return decode_response(value), {**provenance, 'generation_format': 'closed_json_tree',
                                       'generation_wire_version': 2,
                                       'strict_contract_grammar_enforced': False,
                                       'business_grammar_enforced_by': 'compile_contract'}
    if source_object:
        from env_factory_v2.native.response_format import validate_source_response
        validate_source_response(value)
        return value['source'], {**provenance, 'generation_format': 'native_source_object',
                                 'strict_contract_grammar_enforced': False}
    try:
        return json.loads(value['source_json']), provenance
    except json.JSONDecodeError as exc:
        from env_factory_v2.native.json_syntax import recover_json
        try:
            recovered, evidence = recover_json(value['source_json'], self)
            return recovered, {**provenance, 'format_repair': evidence}
        except (ContractError, ValueError) as repair_error:
            raise ContractError(f'generated source is invalid JSON at line {exc.lineno}, column {exc.colno}: '
                                f'{exc.msg}; bounded syntax recovery failed: {repair_error}') from repair_error
