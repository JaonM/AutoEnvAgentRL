from env_factory.generation.agent_authoring import unresolved_lookup_diagnostics


def test_ambiguous_upstream_lookup_is_distinguished_from_missing_downstream_row():
    upstream = {"lookup": {"table": "shipments", "field": "id", "where": {"origin": "Herat"}}}
    expression = {"lookup": {"table": "quotes", "field": "cost", "where": {"shipment": {"eq": upstream}}}}
    state = {"shipments": [{"id": "A", "origin": "Herat"}, {"id": "B", "origin": "Herat"}],
             "quotes": [{"shipment": "A", "cost": 50}]}
    issues = unresolved_lookup_diagnostics(expression, state)
    assert issues[0]["reason"] == "lookup_not_unique"
    assert issues[0]["matched_row_count"] == 2
    assert issues[1]["reason"] == "upstream_lookup_unresolved"
    state["shipments"].pop()
    assert unresolved_lookup_diagnostics(expression, state) == []
    state["quotes"] = []
    assert unresolved_lookup_diagnostics(expression, state)[0]["matched_row_count"] == 0


def test_initial_lookup_diagnostic_uses_initial_rows_after_business_write():
    from env_factory.sandbox_runtime import ExpressionBusinessState
    lookup = {"lookup": {"table": "plans", "field": "amount", "where": {"id": "A"}}}
    current = {"plans": []}
    baseline = {"plans": [{"id": "A", "amount": 12}]}
    state = ExpressionBusinessState(current, baseline)
    assert unresolved_lookup_diagnostics({"initial": lookup}, state) == []
    assert unresolved_lookup_diagnostics(lookup, state)[0]["matched_row_count"] == 0


def test_capture_shape_error_identifies_array_property_mismatch():
    import pytest
    from env_factory.generation.agent_authoring import validate_capture_paths
    tools = [{"name": "assays", "output_contract": {"schema": {
        "type": "object", "properties": {"records": {"type": "array",
        "items": {"type": "object", "properties": {"aperture": {}}}}}}}}]
    step = {"operation": "tool_call", "tool_name": "assays",
            "capture": {"value": "$.records.records[0].aperture"}}
    with pytest.raises(ValueError, match="CAPTURE_PATH_INVALID.*records.records"):
        validate_capture_paths([{"steps": [step]}], tools)
    step["capture"]["value"] = "$.records[0].aperture"
    validate_capture_paths([{"steps": [step]}], tools)


def test_capture_missing_closed_field_fails_before_execution_but_open_schema_remains_valid():
    import pytest
    from env_factory.generation.agent_authoring import validate_capture_paths
    schema = {"type": "object", "properties": {"records": {"type": "array"}},
              "additionalProperties": False}
    tools = [{"name": "get_level_guidance", "output_contract": {"schema": schema}}]
    step = {"operation": "tool_call", "tool_name": "get_level_guidance",
            "capture": {"name": "level"}}
    with pytest.raises(ValueError, match="CAPTURE_PATH_INVALID.*field level.*records"):
        validate_capture_paths([{"steps": [step]}], tools)
    schema["additionalProperties"] = True
    validate_capture_paths([{"steps": [step]}], tools)
    schema["additionalProperties"] = False
    schema["patternProperties"] = {"^level$": {"type": "string"}}
    validate_capture_paths([{"steps": [step]}], tools)
