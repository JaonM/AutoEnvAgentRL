"""CLI differences shared by task authoring and semantic review."""
import json

AGENTS = ("codex", "claude", "opencode")


def command(agent, model, prompt, response, *, readonly=False, schema=None):
    if agent not in AGENTS:
        raise ValueError(f"Unsupported code agent: {agent}")
    if agent == "codex":
        args = ["codex", "exec", "--ephemeral", "--sandbox",
                "read-only" if readonly else "workspace-write",
                "--skip-git-repo-check", "--model", model, "--json",
                "--output-last-message", str(response)]
        if schema:
            args += ["--output-schema", str(schema)]
        return args + ["-"]
    if agent == "claude":
        args = ["claude", "--print", "--output-format", "json", "--model", model]
        args += ["--tools", "Read", "--allowedTools", "Read"] if readonly else ["--permission-mode", "acceptEdits", "--allowedTools", "Read,Edit,Write,Bash"]
        return args
    return ["opencode", "run", "--format", "json", "--model", model, prompt]


def completed_events(agent, text, response):
    """Normalize successful CLI turns; never treat exit zero alone as completion."""
    if agent == "claude":
        event = json.loads(text)
        if event.get("type") != "result" or event.get("is_error") or event.get("subtype") != "success":
            return []
        response.write_text(event.get("result", ""))
        return [{"type": "turn.completed", "usage": event.get("usage", {})}]
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    if agent == "codex":
        return [e for e in events if e.get("type") == "turn.completed"]
    if any(e.get("type") == "error" for e in events):
        return []
    finished = [e for e in events if e.get("type") == "step_finish"
                and e.get("part", {}).get("reason") == "stop"]
    if not finished:
        return []
    response.write_text("".join(e.get("part", {}).get("text", "") for e in events if e.get("type") == "text"))
    return [{"type": "turn.completed", "usage": e.get("part", {}).get("tokens", {})} for e in finished]
