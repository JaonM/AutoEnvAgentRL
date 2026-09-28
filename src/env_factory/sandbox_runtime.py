"""Reusable, dependency-free runtime primitives for generated RL sandboxes.

The Code Agent receives a copy of this module as ``sandbox_runtime.py``.  It
owns transport/application wiring, while this module owns the invariants that
must not be reimplemented differently by every generated task:
authentication, episode isolation, idempotency, deterministic replay,
structured errors, runtime LLM calls, evaluator mocks, and trace hashes.
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

try:
    from .runtime_llm import (
        RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError,
        capture_runtime_llm_trace, summarize_runtime_llm_trace,
    )
except ImportError:  # copied as a flat module into a generated sandbox
    from runtime_llm import (
        RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError,
        capture_runtime_llm_trace, summarize_runtime_llm_trace,
    )


class SandboxError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details

    def body(self, request_id: str) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details, "request_id": request_id}}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def trainer_token_from_env() -> str:
    return os.getenv("SANDBOX_TRAINER_API_KEY", "")


def require_trainer(authorization: str | None) -> None:
    expected = trainer_token_from_env()
    if not expected:
        raise SandboxError("AUTH_NOT_CONFIGURED", "SANDBOX_TRAINER_API_KEY is not configured", 503)
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not __import__("hmac").compare_digest(supplied, expected):
        raise SandboxError("UNAUTHORIZED", "Trainer authentication required", 401)


class JsonLog:
    def __init__(self, name: str = "sandbox") -> None:
        self.name = name

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "logger": self.name, "timestamp": time.time(), **fields}
        # Never include values that could contain credentials.
        for key in list(payload):
            if any(word in key.lower() for word in ("key", "secret", "token", "password")):
                payload[key] = "[REDACTED]"
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


class ExternalCapabilityClient:
    """Configured boundary for external-capability task tools.

    Training can inject deterministic JSON fixtures; production can point the
    same contract at an HTTP provider. Missing configuration is represented as
    an explicit data gap, never fabricated market/search data.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("SANDBOX_EXTERNAL_CAPABILITY_URL", "").rstrip("/")
        self.fixture_path = os.getenv("SANDBOX_EXTERNAL_FIXTURES", "")

    def query(self, capability: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.fixture_path:
            fixtures = json.loads(Path(self.fixture_path).read_text(encoding="utf-8"))
            value = fixtures.get(capability) if isinstance(fixtures, Mapping) else None
            if isinstance(value, Mapping):
                cases = value.get("cases")
                if isinstance(cases, list):
                    for case in cases:
                        if isinstance(case, Mapping) and canonical_json(case.get("arguments")) == canonical_json(arguments):
                            return dict(case["result"])
                    return {"status": "data_unavailable", "items": [], "reason": "fixture has no matching arguments"}
                if arguments:
                    raise SandboxError("EXTERNAL_FIXTURE_INVALID", "parameterized capability requires argument-indexed fixture cases", 500)
                return dict(value)
        if self.base_url:
            payload = canonical_json({"capability": capability, "arguments": dict(arguments)}).encode("utf-8")
            with urlopen(Request(
                self.base_url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            ), timeout=float(os.getenv("SANDBOX_EXTERNAL_TIMEOUT_SECONDS", "15"))) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, Mapping):
                raise SandboxError("EXTERNAL_RESPONSE_INVALID", "external provider returned a non-object", 502)
            return dict(value)
        return {
            "status": "data_unavailable", "items": [], "capability": capability,
            "source": None, "reason": "external capability provider is not configured",
        }


@dataclass(frozen=True)
class Episode:
    episode_id: str
    seed: int
    data_hash: str
    schema_version: str


class EpisodeStore:
    """SQLite-backed isolated episodes, events, idempotency and replay."""

    def __init__(self, db_path: str | Path, *, schema_version: str = "1.0") -> None:
        self.db_path = str(db_path)
        self.schema_version = schema_version
        self._lock = threading.RLock()
        self._local = threading.local()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            yield existing
            return
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        self._local.connection = connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            self._local.connection = None
            connection.close()

    @contextmanager
    def transaction(self):
        with self._lock, self._connect():
            yield

    @contextmanager
    def episode_context(self, episode_id: str | None):
        previous = getattr(self._local, "episode_id", None)
        self._local.episode_id = episode_id
        try:
            yield
        finally:
            self._local.episode_id = previous

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    seed INTEGER NOT NULL,
                    data_hash TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS events (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    idem_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, idem_key)
                );
                CREATE TABLE IF NOT EXISTS episode_state (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    state_key TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, state_key)
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(idempotency)")}
            if "request_hash" not in columns:
                db.execute("ALTER TABLE idempotency ADD COLUMN request_hash TEXT")

    def reset(self, *, episode_id: str | None = None, seed: int | None = None, data_hash: str = "") -> Episode:
        with self._lock, self._connect() as db:
            episode_id = episode_id or f"ep-{uuid.uuid4().hex}"
            if seed is None:
                seed = random.SystemRandom().randrange(0, 2**63)
            db.execute("DELETE FROM events WHERE episode_id = ?", (episode_id,))
            db.execute("DELETE FROM idempotency WHERE episode_id = ?", (episode_id,))
            db.execute("DELETE FROM episode_state WHERE episode_id = ?", (episode_id,))
            # Reset is also the Trainer's active-episode selection operation.
            # Deactivate every episode so switching back to an older id cannot
            # leave a newer episode selected by ``current()``.
            db.execute("UPDATE episodes SET active=0")
            db.execute("INSERT OR REPLACE INTO episodes(episode_id,seed,data_hash,schema_version,created_at,active) VALUES(?,?,?,?,?,1)", (episode_id, seed, data_hash, self.schema_version, time.time()))
            return Episode(episode_id, seed, data_hash, self.schema_version)

    def current(self) -> Episode:
        with self._connect() as db:
            selected = getattr(self._local, "episode_id", None)
            if selected:
                row = db.execute("SELECT * FROM episodes WHERE episode_id=?", (selected,)).fetchone()
                if row is None:
                    raise SandboxError("EPISODE_NOT_FOUND", "unknown episode", 404)
            else:
                row = db.execute("SELECT * FROM episodes WHERE active=1 ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return self.reset()
        return Episode(row["episode_id"], row["seed"], row["data_hash"], row["schema_version"])

    def event(self, event_type: str, payload: Any, result: Any = None, *, idem_key: str | None = None) -> Any:
        episode = self.current()
        with self._lock, self._connect() as db:
            if idem_key:
                existing = db.execute("SELECT response_json FROM idempotency WHERE episode_id=? AND idem_key=?", (episode.episode_id, idem_key)).fetchone()
                if existing:
                    return json.loads(existing[0])
            sequence = int(db.execute("SELECT COALESCE(MAX(sequence), 0) FROM events WHERE episode_id=?", (episode.episode_id,)).fetchone()[0]) + 1
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (episode.episode_id, sequence, event_type, canonical_json(payload), canonical_json(result), time.time()))
            if idem_key:
                db.execute("INSERT INTO idempotency(episode_id,idem_key,response_json,created_at,request_hash) VALUES(?,?,?,?,?)", (episode.episode_id, idem_key, canonical_json(result), time.time(), self._request_hash(event_type, payload)))
        return result

    @staticmethod
    def _request_hash(event_type: str, payload: Any) -> str:
        stable = {key: value for key, value in payload.items() if key not in {"timestamp", "duration_ms", "tool_call_id"}} if isinstance(payload, Mapping) else payload
        return sha256_json({"event": event_type, "payload": stable})

    def execute_idempotent(
        self,
        idem_key: str,
        event_type: str,
        payload: Any,
        operation: Callable[[], Any],
    ) -> Any:
        """Execute and record an operation at most once for the active episode."""
        with self.transaction():
            episode = self.current()
            with self._connect() as db:
                existing = db.execute(
                    "SELECT response_json,request_hash FROM idempotency WHERE episode_id=? AND idem_key=?",
                    (episode.episode_id, idem_key),
                ).fetchone()
            if existing:
                if existing[1] != self._request_hash(event_type, payload):
                    raise SandboxError("IDEMPOTENCY_CONFLICT", "idempotency key belongs to another request", 409)
                return json.loads(existing[0])
            result = operation()
            return self.event(event_type, payload, result, idem_key=idem_key)

    def set_state(self, state_key: str, value: Any) -> None:
        episode = self.current()
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO episode_state VALUES(?,?,?,?)", (episode.episode_id, state_key, canonical_json(value), time.time()))

    def get_state(self, state_key: str, default: Any = None) -> Any:
        episode = self.current()
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM episode_state WHERE episode_id=? AND state_key=?", (episode.episode_id, state_key)).fetchone()
        return default if row is None else json.loads(row[0])

    def replay(self) -> dict[str, Any]:
        episode = self.current()
        with self._connect() as db:
            rows = db.execute("SELECT sequence,event_type,payload_json,result_json,created_at FROM events WHERE episode_id=? ORDER BY sequence", (episode.episode_id,)).fetchall()
        events = [{"sequence": r[0], "event": r[1], "payload": json.loads(r[2]), "result": json.loads(r[3]) if r[3] else None, "timestamp": r[4]} for r in rows]
        # Transport identifiers and timing are diagnostics, not trajectory identity.
        stable_events = []
        for event in events:
            payload = event["payload"]
            if isinstance(payload, Mapping):
                payload = {key: value for key, value in payload.items()
                           if key not in {"tool_call_id", "request_id", "timestamp", "duration_ms"}}
            stable_events.append({"sequence": event["sequence"], "event": event["event"],
                                  "payload": payload, "result": event["result"]})
        return {"episode_id": episode.episode_id, "seed": episode.seed, "schema_version": episode.schema_version, "data_hash": episode.data_hash, "events": events, "trace_hash": sha256_json(stable_events)}


class DataManifestValidator:
    """Validate generated schema/rows before application-specific loading."""

    @staticmethod
    def validate(manifest: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
        root = Path(root)
        tables = manifest.get("tables")
        if not isinstance(tables, list):
            raise SandboxError("DATA_MANIFEST_INVALID", "data manifest has no tables", 500)
        if not tables and manifest.get("environment_mode") in {"stateless", "external_capability"}:
            return {"tables": {}, "data_hash": sha256_json({})}
        if not tables:
            raise SandboxError("DATA_MANIFEST_INVALID", "data manifest has no tables", 500)
        loaded: dict[str, list[dict[str, Any]]] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []
        for table in tables:
            name, schema_file, rows_file = table.get("table_name"), table.get("schema_file"), table.get("rows_file")
            if not all(isinstance(x, str) and x for x in (name, schema_file, rows_file)):
                raise SandboxError("DATA_MANIFEST_INVALID", "table manifest is incomplete", 500)
            schema = json.loads((root / schema_file).read_text(encoding="utf-8"))
            columns = {column["name"] for column in schema.get("columns", [])}
            rows = [json.loads(line) for line in (root / rows_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(set(row) != columns for row in rows):
                raise SandboxError("DATA_INVALID", f"table {name} rows do not match schema", 500)
            primary = schema.get("primary_key", [])
            if primary and len({tuple(row[key] for key in primary) for row in rows}) != len(rows):
                raise SandboxError("DATA_INVALID", f"table {name} primary key is not unique", 500)
            loaded[name] = rows
            for foreign_key in schema.get("foreign_keys", []):
                if isinstance(foreign_key, dict):
                    # Accept both the compact runtime vocabulary and the more
                    # descriptive names emitted by task/data generators.  The
                    # shared boundary is the right place to normalize these;
                    # generated sandboxes should not need one-off adapters.
                    ref_table = foreign_key.get("ref_table") or foreign_key.get("references_table")
                    ref_column = foreign_key.get("ref_column") or foreign_key.get("references_column")
                    foreign_keys.append((name, foreign_key.get("column"), ref_table, ref_column))
                elif isinstance(foreign_key, (list, tuple)) and len(foreign_key) == 4:
                    foreign_keys.append(tuple(foreign_key))
        for table, column, ref_table, ref_column in foreign_keys:
            if not all(isinstance(value, str) and value for value in (table, column, ref_table, ref_column)):
                raise SandboxError("DATA_INVALID", "foreign key declaration is invalid", 500)
            if ref_table not in loaded or any(
                row.get(column) not in {item.get(ref_column) for item in loaded[ref_table]}
                for row in loaded[table]
            ):
                raise SandboxError("DATA_INVALID", f"foreign key {table}.{column} is invalid", 500)
        return {"tables": loaded, "data_hash": sha256_json(loaded)}


class ManifestDataStore:
    """Episode-isolated business records loaded from a validated manifest."""

    STATE_KEY = "business_data"

    @staticmethod
    def _column_json_type(declared: Any) -> str | None:
        kind = str(declared or "").strip().lower()
        if re.match(r"^(?:serial|bigserial|tinyint|smallint|mediumint|int|integer|bigint)\b", kind):
            return "integer"
        if re.match(r"^(?:decimal|numeric|real|float|double)\b", kind):
            return "number"
        if re.match(r"^(?:char|varchar|text|date|datetime|timestamp|uuid)\b", kind):
            return "string"
        if re.match(r"^(?:bool|boolean)\b", kind):
            return "boolean"
        return None

    @staticmethod
    def _constraint_literal(raw: str) -> Any:
        import ast
        value = raw.strip()
        if value.upper() == "NULL":
            return None
        if value.upper() in {"TRUE", "FALSE"}:
            return value.upper() == "TRUE"
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                return float(value) if "." in value else int(value)
            except ValueError as exc:
                raise SandboxError(
                    "DATA_SCHEMA_INVALID", f"unsupported CHECK literal: {raw}", 500
                ) from exc

    @classmethod
    def _check_constraint(cls, row: Mapping[str, Any], expression: str) -> bool:
        clauses = re.split(r"\s+AND\s+", expression.strip(), flags=re.I)
        for clause in clauses:
            nullable_in_match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*?)\)\s+OR\s+\1\s+IS\s+NULL\s*",
                clause, flags=re.I,
            )
            if nullable_in_match:
                field, raw_values = nullable_in_match.groups()
                if field not in row:
                    raise SandboxError(
                        "DATA_SCHEMA_INVALID", f"CHECK references unknown field: {field}", 500
                    )
                if row[field] is None:
                    continue
                clause = f"{field} IN ({raw_values})"
            in_match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*)\)\s*",
                clause, flags=re.I,
            )
            if in_match:
                field, raw_values = in_match.groups()
                import ast
                try:
                    values = ast.literal_eval(f"({raw_values},)")
                except (ValueError, SyntaxError) as exc:
                    raise SandboxError(
                        "DATA_SCHEMA_INVALID", f"unsupported CHECK expression: {expression}", 500
                    ) from exc
                if field not in row or row[field] not in values:
                    return False
                continue
            match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|<>|!=|=|>|<)\s*(.*?)\s*",
                clause,
            )
            if not match:
                raise SandboxError(
                    "DATA_SCHEMA_INVALID", f"unsupported CHECK expression: {expression}", 500
                )
            field, operator, raw_expected = match.groups()
            if field not in row:
                raise SandboxError(
                    "DATA_SCHEMA_INVALID", f"CHECK references unknown field: {field}", 500
                )
            actual, expected = row[field], cls._constraint_literal(raw_expected)
            try:
                passed = {
                    "=": lambda: actual == expected,
                    "!=": lambda: actual != expected,
                    "<>": lambda: actual != expected,
                    ">": lambda: actual > expected,
                    "<": lambda: actual < expected,
                    ">=": lambda: actual >= expected,
                    "<=": lambda: actual <= expected,
                }[operator]()
            except TypeError:
                passed = False
            if not passed:
                return False
        return True

    def __init__(
        self,
        manifest: Mapping[str, Any],
        root: str | Path,
        episode_store: EpisodeStore,
    ) -> None:
        validated = DataManifestValidator.validate(manifest, root)
        self.baseline: dict[str, list[dict[str, Any]]] = validated["tables"]
        self.data_hash: str = validated["data_hash"]
        self.episode_store = episode_store
        self.schemas = {
            table["table_name"]: json.loads((Path(root) / table["schema_file"]).read_text(encoding="utf-8"))
            for table in manifest.get("tables", [])
        }
        self._validate_tables(self.baseline)

    def _validate_tables(self, tables: Mapping[str, Any]) -> None:
        for name, schema in self.schemas.items():
            rows = tables[name]
            columns = {column["name"]: column for column in schema.get("columns", [])}
            primary = schema.get("primary_key", [])
            seen: set[str] = set()
            for row in rows:
                if set(row) != set(columns):
                    raise SandboxError("DATA_INVALID", f"fields do not match schema: {name}", 400)
                for field, column in columns.items():
                    value = row[field]
                    if value is None and column.get("nullable") is True and field not in primary:
                        continue
                    if value is None and (field in primary or column.get("nullable") is False):
                        raise SandboxError("DATA_INVALID", f"null field: {name}.{field}", 400)
                    kind = self._column_json_type(column.get("type"))
                    if kind:
                        validate_json_schema({**column, "type": kind}, value, f"{name}.{field}")
                if primary:
                    key = canonical_json([row[field] for field in primary])
                    if key in seen:
                        raise SandboxError("DATA_INVALID", f"duplicate primary key: {name}", 400)
                    seen.add(key)
            for field, column in columns.items():
                if column.get("unique") is True:
                    values = [canonical_json(row[field]) for row in rows if row[field] is not None]
                    if len(values) != len(set(values)):
                        raise SandboxError("DATA_INVALID", f"duplicate unique field: {name}.{field}", 400)
            for index in schema.get("indexes", []):
                if not isinstance(index, Mapping):
                    raise SandboxError("DATA_SCHEMA_INVALID", f"invalid index: {name}", 500)
                fields = index.get("columns", [])
                if not isinstance(fields, list) or not fields or any(
                    not isinstance(field, str) or field not in columns for field in fields
                ):
                    raise SandboxError("DATA_SCHEMA_INVALID", f"invalid index columns: {name}", 500)
                if index.get("unique") is True:
                    values = [canonical_json([row[field] for field in fields]) for row in rows]
                    if len(values) != len(set(values)):
                        raise SandboxError(
                            "DATA_INVALID", f"duplicate unique index: {name}.{','.join(fields)}", 400
                        )
            for constraint in schema.get("constraints", []):
                if isinstance(constraint, str):
                    expression, constraint_name = constraint.strip(), "unnamed"
                elif isinstance(constraint, Mapping):
                    expression = constraint.get("expression")
                    constraint_name = constraint.get("name", "unnamed")
                    if str(constraint.get("type", "CHECK")).upper() != "CHECK":
                        expression = None
                else:
                    expression, constraint_name = None, "unnamed"
                if not isinstance(expression, str) or not expression.strip():
                    raise SandboxError("DATA_SCHEMA_INVALID", f"invalid constraint: {name}", 500)
                if any(not self._check_constraint(row, expression) for row in rows):
                    raise SandboxError(
                        "DATA_INVALID", f"CHECK constraint failed: {name}.{constraint_name}", 400
                    )
            for foreign in schema.get("foreign_keys", []):
                if isinstance(foreign, Mapping):
                    field = foreign.get("column")
                    parent = foreign.get("ref_table") or foreign.get("references_table")
                    target = foreign.get("ref_column") or foreign.get("references_column")
                else:
                    _, field, parent, target = foreign
                if parent not in tables or any(
                    row.get(field) is not None and not any(other.get(target) == row.get(field) for other in tables[parent])
                    for row in rows
                ):
                    raise SandboxError("DATA_INVALID", f"foreign key violation: {name}.{field}", 400)

    def reset(self, _episode: Episode | None = None) -> None:
        self.episode_store.set_state(self.STATE_KEY, copy.deepcopy(self.baseline))

    def _all(self) -> dict[str, list[dict[str, Any]]]:
        value = self.episode_store.get_state(self.STATE_KEY)
        if value is None:
            self.reset()
            value = self.episode_store.get_state(self.STATE_KEY)
        if not isinstance(value, dict):
            raise SandboxError("DATA_STATE_INVALID", "episode business data is invalid", 500)
        return value

    def table(self, name: str) -> list[dict[str, Any]]:
        tables = self._all()
        if name not in tables:
            raise SandboxError("NOT_FOUND", f"unknown business table: {name}", 404)
        return copy.deepcopy(tables[name])

    def select(self, name: str, **equals: Any) -> list[dict[str, Any]]:
        return [
            row for row in self.table(name)
            if all(row.get(field) == expected for field, expected in equals.items())
        ]

    def replace_table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        tables = self._all()
        if name not in tables:
            raise SandboxError("NOT_FOUND", f"unknown business table: {name}", 404)
        original_columns = {column["name"] for column in self.schemas[name].get("columns", [])}
        replacement = [dict(row) for row in rows]
        if any(set(row) != original_columns for row in replacement):
            raise SandboxError("DATA_INVALID", f"replacement rows for {name} do not match schema", 400)
        tables[name] = replacement
        self._validate_tables(tables)
        self.episode_store.set_state(self.STATE_KEY, tables)

    def insert(self, name: str, row: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.table(name)
        candidate = dict(row)
        expected = set(rows[0]) if rows else set(candidate)
        if set(candidate) != expected:
            raise SandboxError("DATA_INVALID", f"insert row for {name} does not match schema", 400)
        rows.append(candidate)
        self.replace_table(name, rows)
        return copy.deepcopy(candidate)

    def update(self, name: str, selector: Mapping[str, Any], changes: Mapping[str, Any]) -> int:
        rows = self.table(name)
        if rows and set(changes) - set(rows[0]):
            raise SandboxError("DATA_INVALID", f"update fields for {name} do not match schema", 400)
        changed = 0
        for row in rows:
            if all(row.get(key) == value for key, value in selector.items()):
                row.update(changes)
                changed += 1
        self.replace_table(name, rows)
        return changed

    def delete(self, name: str, selector: Mapping[str, Any]) -> int:
        rows = self.table(name)
        kept = [row for row in rows if not all(row.get(key) == value for key, value in selector.items())]
        deleted = len(rows) - len(kept)
        self.replace_table(name, kept)
        return deleted

    def snapshot_hash(self) -> str:
        return sha256_json(self._all())


class DeclarativeToolCompiler:
    """Compile common read-only business tools from constrained specifications."""

    OPERATORS = {"eq", "in", "contains", "gte", "lte"}

    def __init__(self, data: ManifestDataStore) -> None:
        self.data = data

    @staticmethod
    def _matches(actual: Any, expected: Any, operator: str) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "in":
            return actual in expected if isinstance(expected, list) else False
        if operator == "contains":
            if not isinstance(actual, (str, list)):
                return False
            if isinstance(expected, list):
                return any(item in actual for item in expected)
            return expected in actual
        if operator == "gte":
            return actual is not None and actual >= expected
        if operator == "lte":
            return actual is not None and actual <= expected
        raise SandboxError("TOOL_SPEC_INVALID", f"unsupported filter operator: {operator}", 500)

    def compile(self, spec: Mapping[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        operation = spec.get("operation")
        if operation not in {"select", "aggregate_count", "insert", "update", "delete"}:
            raise SandboxError("TOOL_SPEC_INVALID", f"unsupported declarative operation: {operation}", 500)
        table = spec.get("table")
        result_field = spec.get("result_field", "records")
        filters = spec.get("filters", [])
        projection = spec.get("projection", [])
        projection_aliases = spec.get("projection_aliases", {})
        order_by = spec.get("order_by", [])
        if not isinstance(table, str) or not table or not isinstance(result_field, str) or not result_field:
            raise SandboxError("TOOL_SPEC_INVALID", "select spec requires table and result_field", 500)
        if not isinstance(filters, list) or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("argument"), str)
            or not isinstance(item.get("column"), str)
            or item.get("operator") not in self.OPERATORS
            for item in filters
        ):
            raise SandboxError("TOOL_SPEC_INVALID", "select filters are invalid", 500)
        for item in filters:
            resolver = item.get("resolve")
            if resolver is not None and (
                not isinstance(resolver, Mapping)
                or not all(
                    isinstance(resolver.get(field), str) and resolver[field]
                    for field in ("table", "match_column", "value_column")
                )
            ):
                raise SandboxError("TOOL_SPEC_INVALID", "filter resolver is invalid", 500)
        if not isinstance(projection, list) or not all(isinstance(item, str) for item in projection):
            raise SandboxError("TOOL_SPEC_INVALID", "select projection is invalid", 500)
        if not isinstance(projection_aliases, Mapping) or any(
            not isinstance(alias, str) or not alias
            or not isinstance(column, str) or not column
            for alias, column in projection_aliases.items()
        ):
            raise SandboxError("TOOL_SPEC_INVALID", "select projection aliases are invalid", 500)
        if not isinstance(order_by, list) or not all(isinstance(item, str) for item in order_by):
            raise SandboxError("TOOL_SPEC_INVALID", "select order_by is invalid", 500)

        def selected(arguments: dict[str, Any]) -> list[dict[str, Any]]:
            rows = self.data.table(table)
            for rule in filters:
                argument = rule["argument"]
                if argument not in arguments:
                    continue
                expected_values = [arguments[argument]]
                resolver = rule.get("resolve")
                if isinstance(resolver, Mapping):
                    requested = arguments[argument]
                    requested_values = requested if isinstance(requested, list) else [requested]
                    expected_values = [
                        row.get(resolver["value_column"])
                        for row in self.data.table(resolver["table"])
                        if row.get(resolver["match_column"]) in requested_values
                    ]
                    if rule["operator"] == "in":
                        expected_values = [expected_values]
                rows = [
                    row for row in rows
                    if any(
                        self._matches(row.get(rule["column"]), expected, rule["operator"])
                        for expected in expected_values
                    )
                ]
            if order_by:
                rows.sort(key=lambda row: tuple((row.get(field) is None, row.get(field)) for field in order_by))
            if projection or projection_aliases:
                rows = [{
                    **{field: row.get(field) for field in projection},
                    **{alias: row.get(column) for alias, column in projection_aliases.items()},
                } for row in rows]
            return rows

        def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            if operation == "insert":
                mapping = spec.get("values", {})
                row = {column: arguments[argument] for argument, column in mapping.items()}
                return {"record": self.data.insert(table, row), "count": 1}
            selector_map = spec.get("selector", {})
            selector = {column: arguments[argument] for argument, column in selector_map.items()}
            if operation == "update":
                changes_map = spec.get("changes", {})
                changes = {column: arguments[argument] for argument, column in changes_map.items()}
                return {"updated_count": self.data.update(table, selector, changes)}
            if operation == "delete":
                return {"deleted_count": self.data.delete(table, selector)}
            rows = selected(arguments)
            if operation == "aggregate_count":
                return {result_field: len(rows)}
            return {result_field: rows, "count": len(rows)}

        return handler

    def compile_all(self, specs: Sequence[Mapping[str, Any]]) -> dict[str, Callable[[dict[str, Any]], Any]]:
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        for spec in specs:
            name = spec.get("tool_name")
            if not isinstance(name, str) or not name or name in handlers:
                raise SandboxError("TOOL_SPEC_INVALID", "tool implementation name is invalid", 500)
            handlers[name] = self.compile(spec)
        return handlers


def validate_json_schema(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> None:
    """Validate the JSON-Schema subset emitted for Function Tool arguments.

    Keeping this in the reviewed runtime prevents every generated sandbox from
    inventing a subtly different validator.  Unsupported schema constructs are
    rejected during generation; the runtime covers the emitted object, array,
    scalar, enum, required and additionalProperties constraints.
    """

    kind = schema.get("type")
    expected = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }.get(kind)
    if expected is not None:
        valid = isinstance(value, expected)
        if kind in {"integer", "number"} and isinstance(value, bool):
            valid = False
        if not valid:
            raise SandboxError("INVALID_ARGUMENT", f"{path} must be {kind}", 400)
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise SandboxError("INVALID_ARGUMENT", f"{path} is not an allowed value", 400)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        import math
        if not math.isfinite(value):
            raise SandboxError("INVALID_ARGUMENT", f"{path} must be finite", 400)
        for boundary, invalid in (("minimum", lambda limit: value < limit), ("maximum", lambda limit: value > limit)):
            if boundary in schema and invalid(schema[boundary]):
                raise SandboxError("INVALID_ARGUMENT", f"{path} violates {boundary}", 400)
    if isinstance(value, (list, str)):
        low, high = ("minItems", "maxItems") if isinstance(value, list) else ("minLength", "maxLength")
        if len(value) < schema.get(low, 0) or len(value) > schema.get(high, float("inf")):
            raise SandboxError("INVALID_ARGUMENT", f"{path} has invalid length", 400)
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise SandboxError("INVALID_ARGUMENT", f"{path} is missing required properties", 400, missing)
        unknown = sorted(set(value) - set(properties))
        additional = schema.get("additionalProperties", True)
        if additional is False and unknown:
            raise SandboxError("INVALID_ARGUMENT", f"{path} has unexpected properties", 400, unknown)
        if isinstance(additional, Mapping):
            for name in unknown:
                validate_json_schema(additional, value[name], f"{path}.{name}")
        for name, child in properties.items():
            if name in value:
                validate_json_schema(child, value[name], f"{path}.{name}")
    elif kind == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_json_schema(item_schema, item, f"{path}[{index}]")


class ContractToolRegistry:
    """Generic Function Tool validation, dispatch, tracing and mutation seam."""

    def __init__(
        self,
        tools: Sequence[Mapping[str, Any]],
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
        *,
        noise_tools: Sequence[Mapping[str, Any]] = (),
        event_recorder: Callable[..., Any] | None = None,
        tool_contracts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.tools = [dict(item) for item in tools]
        self.handlers = dict(handlers)
        self.event_recorder = event_recorder
        self.output_schemas = {item["name"]: item.get("output_contract", {}).get("schema") for item in tool_contracts}
        self.noise_tools = {
            item.get("name"): dict(item)
            for item in noise_tools
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        self.schemas: dict[str, Mapping[str, Any]] = {}
        for item in self.tools:
            function = item.get("function")
            if item.get("type") != "function" or not isinstance(function, Mapping):
                raise SandboxError("TOOL_CONTRACT_INVALID", "invalid Function Tool declaration", 500)
            name, parameters = function.get("name"), function.get("parameters")
            if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
                raise SandboxError("TOOL_CONTRACT_INVALID", "tool name or parameters are invalid", 500)
            self.schemas[name] = parameters
        missing = sorted(set(self.schemas) - set(self.handlers) - set(self.noise_tools))
        if missing:
            raise SandboxError("TOOL_CONTRACT_INVALID", "declared tools have no handlers", 500, missing)

    def execute(self, name: str, arguments: Mapping[str, Any], *, idem_key: str | None = None) -> Any:
        if name not in self.schemas:
            raise SandboxError("NOT_FOUND", f"unknown tool: {name}", 404)
        if not isinstance(arguments, Mapping):
            raise SandboxError("INVALID_ARGUMENT", "tool arguments must be an object", 400)
        args = dict(arguments)
        mutation = os.getenv("SANDBOX_MUTATION_MODE", "disabled")
        if mutation != "ignore_tool_arguments":
            validate_json_schema(self.schemas[name], args)
        tool_started = time.monotonic()
        event_payload = {
            "tool_name": name, "arguments": args, "noise": name in self.noise_tools,
            "tool_call_id": f"call-{uuid.uuid4().hex}", "timestamp": time.time(),
        }

        def invoke() -> Any:
            try:
                if mutation == "constant_tool_result":
                    return {"mutation": "constant_tool_result", "value": None}
                if mutation == "skip_business_write" and name not in self.noise_tools:
                    return {"status": "ok", "mutation": "skip_business_write"}
                if name in self.noise_tools:
                    metadata = self.noise_tools[name]
                    rows = copy.deepcopy(metadata.get("records", []))
                    for argument, column in metadata.get("parameter_columns", {}).items():
                        if argument in args:
                            rows = [row for row in rows if row.get(column) == args[argument]]
                    return {"records": rows, "count": len(rows)}
                result = self.handlers[name](args)
                output_schema = self.output_schemas.get(name)
                if isinstance(output_schema, Mapping):
                    try:
                        validate_json_schema(output_schema, result, "tool_result")
                    except SandboxError as exc:
                        raise SandboxError("TOOL_RESULT_INVALID", str(exc), 500) from exc
                return result
            finally:
                event_payload["duration_ms"] = round((time.monotonic() - tool_started) * 1000, 3)
        recorder_owner = getattr(self.event_recorder, "__self__", None)
        if isinstance(recorder_owner, EpisodeStore):
            with recorder_owner.transaction():
                if idem_key:
                    return recorder_owner.execute_idempotent(idem_key, "tool_call", event_payload, invoke)
                result = invoke()
                recorder_owner.event("tool_call", event_payload, result)
                return result
        result = invoke()
        if self.event_recorder is not None:
            self.event_recorder("tool_call", event_payload, result, idem_key=idem_key)
        return result


class ContractRewardAggregator:
    """Apply the generated metric ranges, weights and clipping formula."""

    def __init__(self, metrics: Sequence[Mapping[str, Any]]) -> None:
        self.metrics = [dict(metric) for metric in metrics]
        if not self.metrics:
            raise SandboxError("REWARD_CONTRACT_INVALID", "metrics must not be empty", 500)

    def aggregate(self, scores: Mapping[str, Any]) -> dict[str, Any]:
        components: dict[str, float] = {}
        raw_reward = 0.0
        declared_ids = {
            metric.get("id") for metric in self.metrics
            if isinstance(metric.get("id"), str) and metric.get("id")
        }
        missing = sorted(declared_ids - set(scores))
        if missing:
            raise SandboxError(
                "REWARD_SCORE_MISSING",
                "declared metrics have no runtime score",
                500,
                missing,
            )
        for metric in self.metrics:
            metric_id = metric.get("id")
            if not isinstance(metric_id, str) or not metric_id:
                raise SandboxError("REWARD_CONTRACT_INVALID", "metric id is invalid", 500)
            value = float(scores[metric_id])
            score_range = metric.get("score_range")
            if not isinstance(score_range, list) or len(score_range) != 2:
                raise SandboxError("REWARD_CONTRACT_INVALID", f"metric {metric_id} range is invalid", 500)
            low, high = map(float, score_range)
            if not low <= value <= high:
                raise SandboxError("REWARD_SCORE_INVALID", f"metric {metric_id} score is outside its range", 500)
            weight = float(metric.get("weight", 0.0))
            components[metric_id] = value
            raw_reward += weight * value
        return {
            "reward": max(-1.0, min(1.0, raw_reward)),
            "raw_reward": raw_reward,
            "components": components,
        }


class BusinessGoalEvaluator:
    """Finite row predicates used by generation, execution and acceptance."""

    @staticmethod
    def evaluate(predicates: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> bool:
        if not predicates:
            return False
        for predicate in predicates:
            table = predicate.get("table")
            selector = predicate.get("where", {})
            if table not in state or not isinstance(selector, Mapping):
                return False
            rows = [row for row in state[table] if all(row.get(key) == value for key, value in selector.items())]
            expected = predicate.get("values", {})
            matching = [row for row in rows if all(row.get(key) == value for key, value in expected.items())]
            if len(matching) != predicate.get("count"):
                return False
        return True

    @staticmethod
    def preserves_unrelated(goal: Mapping[str, Any], baseline: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
        if set(baseline) != set(state):
            return False
        predicates = goal.get("row_predicates", [])
        for table, old_rows in baseline.items():
            relevant = [item for item in predicates if item.get("table") == table]
            if not relevant:
                if old_rows != state[table]:
                    return False
                continue
            keys = goal.get("table_primary_keys", {}).get(table, [])
            if not keys:  # legacy contracts have no row identity metadata
                continue
            index = lambda rows: {canonical_json([row.get(key) for key in keys]): row for row in rows}
            before, after = index(old_rows), index(state[table])
            for identity in before.keys() | after.keys():
                old, new = before.get(identity), after.get(identity)
                if old == new:
                    continue
                candidates = [item for item in relevant if all((old or new).get(key) == value for key, value in item.get("where", {}).items())]
                if old is None:
                    valid = any(item["count"] > 0 and all(new.get(key) == value for key, value in item["values"].items()) for item in candidates)
                elif new is None:
                    valid = any(item["count"] == 0 for item in candidates)
                else:
                    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
                    valid = any(changed <= set(item["values"]) for item in candidates)
                if not valid:
                    return False
        return True


class ContractRewardGate:
    """Apply curriculum-level causal prerequisites before reward aggregation."""

    def __init__(self, task_spec: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> None:
        self.task_spec = dict(task_spec)
        self.metrics = [dict(item) for item in metrics]

    def apply(self, scores: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(scores)
        training = self.task_spec.get("training_contract", {})
        category = training.get("category") if isinstance(training, Mapping) else None
        dag = self.task_spec.get("capability_dag", {})
        required = set(dag.get("nodes", [])) if isinstance(dag, Mapping) else set()
        trajectory = context.get("trajectory", {})
        events = trajectory.get("events", []) if isinstance(trajectory, Mapping) else []
        all_calls = [
            str(event.get("payload", {}).get("tool_name"))
            for event in events if isinstance(event, Mapping)
            and event.get("event") == "tool_call"
            and isinstance(event.get("payload"), Mapping)
        ]
        business_calls = [
            str(event.get("payload", {}).get("tool_name"))
            for event in events if isinstance(event, Mapping)
            and event.get("event") == "tool_call"
            and isinstance(event.get("payload"), Mapping)
            and event.get("payload", {}).get("noise") is not True
        ]
        called = set(business_calls)
        edges = dag.get("edges", []) if isinstance(dag, Mapping) else []
        # A later valid retry can establish a dependency after an early mistake.
        calls = [event for event in events if isinstance(event, Mapping) and event.get("event") == "tool_call"
                 and isinstance(event.get("payload"), Mapping)]
        def established(edge: Mapping[str, Any]) -> bool:
            for index, event in enumerate(calls):
                if event["payload"].get("tool_name") != edge.get("to_tool"):
                    continue
                for previous in calls[:index]:
                    if previous["payload"].get("tool_name") != edge.get("from_tool"):
                        continue
                    if edge.get("argument_path") and edge.get("result_path"):
                        source = DeclarativeMetricEvaluator._resolve(previous.get("result"), edge["result_path"])
                        target = DeclarativeMetricEvaluator._resolve(event["payload"].get("arguments", {}), edge["argument_path"])
                        if source is not None and source == target:
                            return True
                    else:
                        return True
            return False
        paths = dag.get("success_paths") or [list(required)]
        dependency_progress = any(
            bool(path) and set(path) <= called and any(
                edge.get("from_tool") in path and edge.get("to_tool") in path for edge in edges
            ) and all(established(edge) for edge in edges
                      if edge.get("from_tool") in path and edge.get("to_tool") in path)
            for path in paths
        )
        unresolved = any(
            isinstance(event, Mapping)
            and event.get("event") == "user_turn"
            and isinstance(event.get("result"), Mapping)
            and event.get("result", {}).get("termination_reason") == "unresolved_dialogue"
            for event in events
        )
        causal_progress = (
            not unresolved
            and (
                (
                    category == "direct_response"
                    and isinstance(context.get("final_agent_response"), str)
                    and bool(context["final_agent_response"].strip())
                )
                or (category == "simple_agentic" and bool(required & called))
                or (
                    category == "multi_step_agentic"
                    and bool(edges)
                    and dependency_progress
                )
            )
        )
        goals = self.task_spec.get("goal_contract", {})
        predicates = goals.get("row_predicates", [])
        if predicates:
            baseline = context.get("initial_business_state", {})
            state = context.get("business_state", {})
            causal_progress = causal_progress and BusinessGoalEvaluator.evaluate(predicates, state)
            if goals.get("requires_state_change"):
                causal_progress = causal_progress and not BusinessGoalEvaluator.evaluate(predicates, baseline) and state != baseline
                causal_progress = causal_progress and BusinessGoalEvaluator.preserves_unrelated(goals, baseline, state)
        # Tool presence alone is not evidence that the Agent chose the right
        # arguments or completed every required step.  Compiled process
        # metrics encode those exact causal obligations.  Outcome credit is
        # available only when each declared process metric reaches its best
        # score; a later correct retry can still satisfy a trajectory rule.
        process_complete = all(
            metric.get("id") in result
            and float(result[str(metric["id"])]) >= float(metric.get("score_range", [0, 1])[-1])
            for metric in self.metrics
            if metric.get("category") == "process"
        )
        causal_progress = causal_progress and process_complete
        if not causal_progress:
            for metric in self.metrics:
                # Process shaping must not reward an isolated downstream call.
                # Until the declared chain and state goal are jointly valid,
                # both process and outcome evidence is non-causal.
                if metric.get("category") in {"process", "outcome"} and metric.get("id") in result:
                    low = metric.get("score_range", [0, 1])[0]
                    result[str(metric["id"])] = float(low)
        return result


class ContractEvaluatorRuntime:
    """Cached, trace-visible boundary for contract-declared LLM evaluators."""

    STATE_KEY = "evaluator_cache"

    def __init__(self, episode_store: EpisodeStore) -> None:
        self.episode_store = episode_store

    def json_judge(
        self,
        metric: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        fallback: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        # Evaluator calls are trace events, not task progress. Excluding them
        # keeps repeated reward reads idempotent and makes the cache effective.
        def stable(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: stable(item) for key, item in value.items()
                        if key not in {"timestamp", "duration_ms", "created_at", "trace_hash", "request_id", "tool_call_id", "sequence"}}
            if isinstance(value, list):
                return [stable(item) for item in value if not (
                    isinstance(item, Mapping)
                    and (item.get("event") or item.get("kind"))
                        in {"evaluator_call", "runtime_llm_call"}
                )]
            return value
        stable_context = stable(context)
        context_hash = sha256_json({
            "metric_id": metric.get("id"), "evaluator": metric.get("evaluator"),
            "context": stable_context,
        })
        cache = self.episode_store.get_state(self.STATE_KEY, {})
        if isinstance(cache, Mapping) and context_hash in cache:
            return dict(cache[context_hash])
        result: Mapping[str, Any]
        used_fallback = False
        try:
            evaluator = metric.get("evaluator", {})
            mapping = evaluator.get("score_mapping", {}) if isinstance(evaluator, Mapping) else {}
            criteria = metric.get("criteria", [])
            if not isinstance(criteria, list):
                criteria = [criteria]
            judge_instructions = {
                "task": "Evaluate whether the runtime evidence satisfies this one metric.",
                "rules": [
                    "Apply only the supplied rubric and criteria; do not invent requirements.",
                    "Judge semantic equivalence, not wording, unless the criteria explicitly require exact syntax.",
                    "Treat the public task and public materials as the evaluation target.",
                    "When conversation is present, evaluate all assistant responses cumulatively; a later acknowledgement or closing message does not erase a correct earlier answer.",
                    "Use the highest-scoring label only when every material criterion is satisfied; use an intermediate label for partial evidence when available.",
                    "Return exactly one label declared in label_scores.",
                ],
                "rubric": metric.get("rubric", ""),
                "criteria": [str(item) for item in criteria if str(item).strip()],
                "label_scores": mapping,
            }
            result = RuntimeLLMClient().json_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Return only one JSON object that matches the requested response schema. "
                            + canonical_json(judge_instructions)
                        ),
                    },
                    {"role": "user", "content": canonical_json({"runtime_evidence": stable_context})},
                ],
                response_schema=response_schema or {"type": "object"},
            )
            if not isinstance(result, Mapping):
                raise RuntimeLLMError("evaluator returned a non-object")
        except (RuntimeLLMError, OSError, ValueError, TypeError):
            result, used_fallback = dict(fallback), True
        updated = dict(cache) if isinstance(cache, Mapping) else {}
        updated[context_hash] = dict(result)
        self.episode_store.set_state(self.STATE_KEY, updated)
        self.episode_store.event("evaluator_call", {
            "metric_id": metric.get("id"), "context_hash": context_hash,
            "cached": False, "used_fallback": used_fallback,
        }, dict(result))
        return dict(result)


class ContractModelMetricEvaluator:
    """Platform-owned semantic outcome evaluator and explicitly limited fixture mode."""

    def __init__(self, contract: Mapping[str, Any], store: EpisodeStore) -> None:
        self.contract, self.store = contract, store
        self.runtime = ContractEvaluatorRuntime(store)

    def evaluate_all(self, context: Mapping[str, Any], existing: Mapping[str, float]) -> dict[str, float]:
        scores = {}
        mock = os.getenv("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"}
        for metric in self.contract.get("metrics", []):
            metric_id = metric["id"]
            evaluator = metric.get("evaluator", {})
            if metric_id in existing or evaluator.get("kind") not in {
                "external_llm_judge", "hybrid_outcome",
            }:
                continue
            mapping = evaluator.get("score_mapping", {})
            labels = {name: value for name, value in mapping.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)}
            if not labels:
                raise SandboxError("EVALUATOR_CONTRACT_INVALID", "semantic evaluator requires numeric label mapping", 500)
            low, high = metric.get("score_range", [0, 1])
            if any(not low <= value <= high for value in labels.values()):
                raise SandboxError("EVALUATOR_CONTRACT_INVALID", "semantic score mapping exceeds declared range", 500)
            failure = min(labels, key=labels.get)
            if mock:
                references = [step.get("content")
                              for scenario in self.contract.get("acceptance_contract", {}).get("executable_scenarios", [])
                              if scenario.get("kind") == "goal_success"
                              for step in scenario.get("steps", []) if step.get("operation") == "agent_response"]
                response = context.get("final_agent_response")
                label = max(labels, key=labels.get) if response and response in references else failure
                scores[metric_id] = float(labels[label])
                self.store.event("evaluator_call", {"metric_id": metric_id, "mode": "offline_fixture", "semantic_verification": False}, {"label": label})
            else:
                declared_inputs = metric.get("evaluation_inputs", [])
                if not isinstance(declared_inputs, list):
                    declared_inputs = []
                aliases = {
                    "recent_conversation": "conversation",
                    "terminal_observation": "public_observation",
                    "tool_results": "trajectory",
                    "business_data": "business_state",
                }
                selected_context: dict[str, Any] = {}
                observation = context.get("observation", {})
                if not isinstance(observation, Mapping):
                    observation = {}
                for name in declared_inputs:
                    if not isinstance(name, str):
                        continue
                    source = aliases.get(name, name)
                    if source in context:
                        selected_context[name] = context[source]
                    elif source in observation:
                        selected_context[name] = observation[source]
                # The judge must know what the public user actually requested.
                # This is safe to add independently of model-authored
                # ``evaluation_inputs`` because it contains no hidden truth.
                public_input = self.contract.get("public_input")
                if isinstance(public_input, Mapping):
                    selected_context["public_input"] = dict(public_input)
                public_task = self.contract.get("task")
                if isinstance(public_task, str) and public_task.strip():
                    selected_context["task"] = public_task.strip()
                # Keep the judge input bounded and contract-directed.  Older
                # contracts without evaluation_inputs still receive the two
                # canonical response fields rather than the entire replay and
                # business database.
                if not selected_context:
                    for name in ("conversation", "final_agent_response"):
                        if name in context:
                            selected_context[name] = context[name]
                result = self.runtime.json_judge(metric, selected_context, fallback={"label": failure}, response_schema={
                    "type": "object", "required": ["label"],
                    "properties": {"label": {"type": "string", "enum": list(labels)}},
                })
                scores[metric_id] = float(labels[result["label"]])
        return scores


class DeclarativeMetricEvaluator:
    """Evaluate deterministic metric predicates from a constrained DSL."""

    OPERATORS = {
        "eq", "ne", "gte", "lte", "contains", "exists", "count_gte",
        "count_eq", "changed", "unchanged", "subset", "none_tool_calls",
        "contains_tool_call",
    }

    @staticmethod
    def path_tokens(path: str) -> list[tuple[str, Any]]:
        """Supported paths: fields, indices, array wildcards and equality filters."""
        import ast
        if path in {"", "$"}:
            return []
        remaining = path[1:] if path.startswith("$") else "." + path
        tokens = []
        while remaining:
            field = re.match(r"\.([^\s.\[\]]+)", remaining)
            index = re.match(r"\[(\d+)\]", remaining)
            wildcard = re.match(r"\[\*\]", remaining)
            selector = re.match(r"""\[\?\(@\.([^\s.\[\]=]+)\s*==\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|-?\d+(?:\.\d+)?|true|false|null)\s*\)\]""", remaining)
            if field:
                tokens.append(("field", field[1]))
                remaining = remaining[field.end():]
            elif index:
                tokens.append(("index", int(index[1])))
                remaining = remaining[index.end():]
            elif wildcard:
                tokens.append(("wildcard", None))
                remaining = remaining[wildcard.end():]
            elif selector:
                literal = selector[2]
                expected = ast.literal_eval(literal) if literal.startswith("'") else json.loads(literal)
                tokens.append(("filter", (selector[1], expected)))
                remaining = remaining[selector.end():]
            else:
                raise SandboxError("METRIC_SPEC_INVALID", "unsupported JSON path syntax", 500, {"path": path})
        return tokens

    @classmethod
    def _resolve(cls, value: Any, path: str) -> Any:
        current = value
        for kind, part in cls.path_tokens(path):
            if kind == "wildcard":
                current = list(current) if isinstance(current, list) else []
                continue
            if kind == "filter":
                column, expected = part
                current = [row for row in current if isinstance(row, Mapping) and row.get(column) == expected] if isinstance(current, list) else []
                continue
            if isinstance(current, Mapping):
                current = current.get(part)
            elif isinstance(current, list) and (kind == "index" or str(part).isdigit()):
                index = int(part)
                current = current[index] if index < len(current) else None
            elif isinstance(current, list) and kind == "field":
                values = [row.get(part) for row in current if isinstance(row, Mapping)]
                current = values[0] if len(values) == 1 else values
            else:
                return None
        return current

    @classmethod
    def _compare(cls, actual: Any, operator: str, expected: Any) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gte":
            return actual is not None and actual >= expected
        if operator == "lte":
            return actual is not None and actual <= expected
        if operator == "contains":
            if isinstance(actual, list) and isinstance(expected, Mapping):
                return any(
                    isinstance(row, Mapping)
                    and all(row.get(key) == value for key, value in expected.items())
                    for row in actual
                )
            if isinstance(actual, Mapping) and isinstance(expected, Mapping):
                return all(actual.get(key) == value for key, value in expected.items())
            return expected in actual if isinstance(actual, (str, list, dict)) else False
        if operator == "exists":
            return (actual is not None) is bool(expected)
        if operator in {"count_gte", "count_eq"}:
            count = len(actual) if isinstance(actual, (str, list, dict)) else 0
            return count >= expected if operator == "count_gte" else count == expected
        if operator == "changed":
            return actual != expected
        if operator == "unchanged":
            return actual == expected
        if operator == "subset":
            return set(actual).issubset(set(expected)) if isinstance(actual, list) and isinstance(expected, list) else False
        if operator == "none_tool_calls":
            if not isinstance(actual, list) or not isinstance(expected, list):
                return False
            blocked = set(expected)
            return not any(
                isinstance(event, Mapping)
                and event.get("event") == "tool_call"
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("tool_name") in blocked
                for event in actual
            )
        if operator == "contains_tool_call":
            if not isinstance(actual, list) or not isinstance(expected, Mapping):
                return False
            tool_name = expected.get("tool_name")
            arguments = expected.get("arguments")
            captures = expected.get("captures", [])
            if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
                return False
            resolved: dict[str, Any] = {}
            if not isinstance(captures, list):
                return False
            for capture in captures:
                if not isinstance(capture, Mapping):
                    return False
                capture_name = capture.get("name")
                capture_tool = capture.get("tool_name")
                capture_path = capture.get("path")
                if not all(isinstance(value, str) and value for value in (
                    capture_name, capture_tool, capture_path
                )):
                    return False
                source = next((
                    event.get("result") for event in actual
                    if isinstance(event, Mapping)
                    and event.get("event") == "tool_call"
                    and isinstance(event.get("payload"), Mapping)
                    and event["payload"].get("tool_name") == capture_tool
                ), None)
                value = cls._resolve(source, capture_path)
                if value is None:
                    return False
                resolved[capture_name] = value

            def resolve(value: Any) -> Any:
                if isinstance(value, Mapping) and set(value) == {"$ref"}:
                    return resolved.get(value["$ref"], value)
                if isinstance(value, Mapping):
                    return {key: resolve(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [resolve(item) for item in value]
                return value

            canonical_arguments = resolve(arguments)
            return any(
                isinstance(event, Mapping)
                and event.get("event") == "tool_call"
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("tool_name") == tool_name
                and event["payload"].get("arguments") == canonical_arguments
                for event in actual
            )
        raise SandboxError("METRIC_SPEC_INVALID", f"unsupported metric operator: {operator}", 500)

    def evaluate(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        source = spec.get("source")
        path = spec.get("path", "$")
        operator = spec.get("operator")
        if source not in {"business_state", "trajectory", "final_agent_response", "observation"}:
            raise SandboxError("METRIC_SPEC_INVALID", "metric source is invalid", 500)
        if not isinstance(path, str) or operator not in self.OPERATORS:
            raise SandboxError("METRIC_SPEC_INVALID", "metric path or operator is invalid", 500)
        actual = self._resolve(context.get(source), path)
        passed = self._compare(actual, operator, spec.get("expected"))
        score_mapping = spec.get("score_mapping", {"pass": 1.0, "fail": 0.0})
        if not isinstance(score_mapping, Mapping):
            raise SandboxError("METRIC_SPEC_INVALID", "metric score_mapping is invalid", 500)
        value = score_mapping.get("pass" if passed else "fail")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SandboxError("METRIC_SPEC_INVALID", "metric mapped score is invalid", 500)
        return float(value)

    def evaluate_all(
        self, specs: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for spec in specs:
            metric_id = spec.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id or metric_id in scores:
                raise SandboxError("METRIC_SPEC_INVALID", "metric implementation id is invalid", 500)
            scores[metric_id] = self.evaluate(spec, context)
        return scores


class ContractUserSimulator:
    """Episode-isolated FSM user driven by a runtime LLM with safe fallback."""

    STATE_KEY = "user_simulator"
    NORMAL_OUTCOMES = frozenset({
        "goal_satisfied", "information_required", "user_correction",
        "user_rejection", "user_acceptance",
    })
    RECOVERY_OUTCOMES = frozenset({
        "agent_off_topic", "agent_premature_completion", "unrecognized",
    })

    def __init__(
        self,
        episode_store: EpisodeStore,
        *,
        profiles: Sequence[Mapping[str, Any]] = (),
        scripts: Sequence[Mapping[str, Any]] = (),
        renderer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.episode_store = episode_store
        self.profiles = {str(item.get("profile_id")): dict(item) for item in profiles if isinstance(item, Mapping)}
        self.scripts = {
            str(item.get("script_id")): self._normalize_script(item)
            for item in scripts if isinstance(item, Mapping)
        }
        self.renderer = renderer
        if not self.profiles or not self.scripts:
            raise SandboxError("USER_SIMULATION_INVALID", "profiles and FSM scripts must not be empty", 500)

    @classmethod
    def _normalize_script(cls, source: Mapping[str, Any]) -> dict[str, Any]:
        """Upgrade legacy FSMs to the typed dialogue-outcome protocol."""
        script = copy.deepcopy(dict(source))
        terminal_ids = {
            item.get("state_id") for item in script.get("states", [])
            if isinstance(item, Mapping) and item.get("terminal") is True
        }
        transitions = []
        for raw in script.get("transitions", []):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            if item.get("outcome_category") not in cls.NORMAL_OUTCOMES:
                condition = str(item.get("condition", "")).casefold()
                if item.get("to_state") in terminal_ids:
                    outcome = "user_acceptance"
                elif any(marker in condition for marker in ("补充", "信息", "澄清", "clarif", "information")):
                    outcome = "information_required"
                elif any(marker in condition for marker in ("纠正", "更正", "correct")):
                    outcome = "user_correction"
                elif any(marker in condition for marker in ("拒绝", "不接受", "reject")):
                    outcome = "user_rejection"
                else:
                    outcome = "goal_satisfied"
                item["outcome_category"] = outcome
            transitions.append(item)
        script["transitions"] = transitions
        recovery = dict(script.get("recovery_policy", {}))
        recovery.setdefault("max_recoveries", 2)
        recovery.setdefault("user_behavior", "指出回复没有解决当前问题，并要求 Agent 重新回答。")
        recovery["handled_outcomes"] = sorted(cls.RECOVERY_OUTCOMES)
        script["recovery_policy"] = recovery
        return script

    def reset(self, episode: Episode | None = None) -> None:
        episode = episode or self.episode_store.current()
        rng = random.Random(episode.seed)
        script_id = sorted(self.scripts)[rng.randrange(len(self.scripts))]
        profile_id = sorted(self.profiles)[rng.randrange(len(self.profiles))]
        script = self.scripts[script_id]
        self.episode_store.set_state(self.STATE_KEY, {
            "turn_index": 0, "memory": [],
            "script_id": script_id, "profile_id": profile_id,
            "state_id": script.get("initial_state"), "variables": copy.deepcopy(script.get("variables", {})),
            "recovery_count": 0, "termination_reason": None,
            "conversation_prefix": [],
        })

    @staticmethod
    def _llm_render(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        transitions = {
            item.get("transition_id"): item
            for item in payload.get("transitions", [])
            if isinstance(item, Mapping) and isinstance(item.get("transition_id"), str)
        }

        def validate_semantics(candidate: Mapping[str, Any]) -> None:
            match_status = candidate.get("match_status")
            outcome = candidate.get("outcome_category")
            transition_id = candidate.get("transition_id")
            selected = transitions.get(transition_id)
            if not isinstance(candidate.get("user_query"), str) or not candidate["user_query"].strip():
                raise RuntimeLLMError("user_query must be non-empty")
            if outcome in ContractUserSimulator.NORMAL_OUTCOMES and (
                match_status != "matched"
                or selected is None
                or selected.get("outcome_category") != outcome
            ):
                raise RuntimeLLMError(
                    "normal outcome must select a current transition with the same category"
                )
            if outcome in ContractUserSimulator.RECOVERY_OUTCOMES and (
                match_status == "matched" or transition_id not in {None, ""}
            ):
                raise RuntimeLLMError("recovery outcome must not select a transition")
            if outcome in {"agent_off_topic", "agent_premature_completion"} and match_status != "unmatched":
                raise RuntimeLLMError("recognized recovery outcome must be unmatched")
            if outcome == "unrecognized" and match_status not in {"unmatched", "ambiguous"}:
                raise RuntimeLLMError("unrecognized outcome must be unmatched or ambiguous")

        return RuntimeLLMClient().json_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Act as the simulated user described by the supplied profile and current FSM state. "
                        "Classify the dialogue into exactly one declared outcome_category. Normal outcomes are "
                        "goal_satisfied, information_required, user_correction, user_rejection, user_acceptance. "
                        "Recovery outcomes are agent_off_topic, agent_premature_completion, unrecognized. "
                        "Normal outcomes require match_status=matched and one listed transition with the same category. "
                        "Recovery outcomes require unmatched, except unrecognized may be ambiguous, and no transition. Return only JSON. "
                        "The simulated user sees the dialogue, not hidden tool traces. Never reject an otherwise verifiable answer only "
                        "because raw tool-call records are absent; judge the visible answer and public evidence. "
                        "Do not answer the user's own task, reveal hidden state, or invent business facts."
                    ),
                },
                {"role": "user", "content": canonical_json(payload)},
            ],
            response_schema={
                "type": "object",
                "required": ["user_query", "match_status", "outcome_category", "reason_code"],
                "properties": {
                    "user_query": {"type": "string"},
                    "transition_id": {"type": "string"},
                    "match_status": {"type": "string", "enum": ["matched", "unmatched", "ambiguous"]},
                    "outcome_category": {
                        "type": "string",
                        "enum": sorted(
                            ContractUserSimulator.NORMAL_OUTCOMES
                            | ContractUserSimulator.RECOVERY_OUTCOMES
                        ),
                    },
                    "reason_code": {"type": "string"},
                },
            },
            semantic_validator=validate_semantics,
        )

    @classmethod
    def _fallback_render(cls, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Infrastructure failure cannot establish user acceptance or success."""
        return {
            "user_query": "暂时无法确认当前回复，请稍后重试。",
            "transition_id": None,
            "match_status": "unmatched",
            "outcome_category": "unrecognized",
            "reason_code": "simulator_unavailable",
        }

    def turn(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise SandboxError("INVALID_ARGUMENT", "messages must be an array", 400)
        if any(not isinstance(item, Mapping) or item.get("role") not in {"user", "assistant", "system"} or not isinstance(item.get("content"), str) for item in messages):
            raise SandboxError("INVALID_ARGUMENT", "messages contain invalid entries", 400)
        state = self.episode_store.get_state(self.STATE_KEY)
        if not isinstance(state, dict):
            self.reset()
            state = self.episode_store.get_state(self.STATE_KEY)
        normalized_messages = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in messages
        ]
        expected_prefix = state.get("conversation_prefix", [])
        if not isinstance(expected_prefix, list):
            expected_prefix = []
        if (
            not normalized_messages
            or normalized_messages[:len(expected_prefix)] != expected_prefix
            or len(normalized_messages) <= len(expected_prefix)
            or normalized_messages[-1]["role"] != "assistant"
        ):
            raise SandboxError(
                "INCOMPLETE_CONVERSATION",
                "messages must contain the complete ordered conversation and a new assistant turn",
                409,
            )
        script = self.scripts.get(str(state.get("script_id")), {})
        profile = self.profiles.get(str(state.get("profile_id")), {})
        if state.get("termination_reason"):
            return {
                "user_query": "本次对话已经结束。",
                "should_end": True,
                "attachments": [],
                "termination_reason": state["termination_reason"],
            }
        index = int(state["turn_index"])
        state_before = str(state.get("state_id", ""))
        transitions = [
            item for item in script.get("transitions", [])
            if isinstance(item, Mapping) and item.get("from_state") == state.get("state_id")
        ] if isinstance(script, Mapping) else []
        current_state = next((
            dict(item) for item in script.get("states", [])
            if isinstance(item, Mapping) and item.get("state_id") == state.get("state_id")
        ), {}) if isinstance(script, Mapping) else {}
        payload = {
            "messages": list(messages), "profile": profile,
            "goal": script.get("goal", ""),
            "user_input": copy.deepcopy(script.get("user_input", {})),
            "current_state": current_state, "state_id": state.get("state_id"),
            "variables": copy.deepcopy(state.get("variables", {})),
            "transitions": transitions, "turn_index": index,
            "recovery_count": int(state.get("recovery_count", 0)),
            "recovery_policy": dict(script.get("recovery_policy", {})),
        }
        used_fallback = False
        try:
            candidate = (self.renderer or self._llm_render)(payload)
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("user_query"), str) or not candidate["user_query"].strip():
                raise RuntimeLLMError("user renderer returned an invalid user_query")
            match_status = candidate.get("match_status")
            if match_status is None:  # custom renderer compatibility
                match_status = "matched" if candidate.get("transition_id") else "unmatched"
            if match_status not in {"matched", "unmatched", "ambiguous"}:
                raise RuntimeLLMError("user renderer returned an invalid match_status")
            outcome = candidate.get("outcome_category")
            if outcome is None and match_status == "matched":  # custom renderer compatibility
                selected = next((item for item in transitions if item.get("transition_id") == candidate.get("transition_id")), {})
                outcome = selected.get("outcome_category")
            if outcome not in self.NORMAL_OUTCOMES | self.RECOVERY_OUTCOMES:
                raise RuntimeLLMError("user renderer returned an invalid outcome_category")
            transition_ids = {item.get("transition_id") for item in transitions}
            if match_status == "matched" and candidate.get("transition_id") not in transition_ids:
                raise RuntimeLLMError("user renderer selected an invalid transition")
            selected = next((item for item in transitions if item.get("transition_id") == candidate.get("transition_id")), None)
            if outcome in self.NORMAL_OUTCOMES and (
                match_status != "matched" or selected is None or selected.get("outcome_category") != outcome
            ):
                raise RuntimeLLMError("normal outcome does not match the selected transition")
            if outcome in self.RECOVERY_OUTCOMES and (
                match_status == "matched" or candidate.get("transition_id") not in {None, ""}
            ):
                raise RuntimeLLMError("recovery outcome must not select a normal transition")
            if outcome in {"agent_off_topic", "agent_premature_completion"} and match_status != "unmatched":
                raise RuntimeLLMError("recognized recovery outcome must be unmatched")
            if outcome == "unrecognized" and match_status not in {"unmatched", "ambiguous"}:
                raise RuntimeLLMError("unrecognized outcome must be unmatched or ambiguous")
            result = {
                "user_query": candidate["user_query"].strip(),
                "transition_id": candidate.get("transition_id") if match_status == "matched" else None,
                "match_status": match_status,
                "outcome_category": outcome,
                "reason_code": str(candidate.get("reason_code") or match_status),
                "attachments": [],
            }
        except Exception:
            used_fallback = True
            fallback = self._fallback_render(payload)
            result = {**fallback, "attachments": []}
        applied_transition = None
        if result["match_status"] != "matched":
            recovery_policy = script.get("recovery_policy", {})
            maximum = int(recovery_policy.get("max_recoveries", 2))
            state["recovery_count"] = int(state.get("recovery_count", 0)) + 1
            result["should_end"] = state["recovery_count"] >= maximum
            if result["should_end"]:
                state["termination_reason"] = "unresolved_dialogue"
                result["termination_reason"] = "unresolved_dialogue"
        elif transitions:
            requested = result.get("transition_id") if isinstance(result, Mapping) else None
            transition = next((item for item in transitions if item.get("transition_id") == requested), None)
            if transition is not None:
                applied_transition = transition
                state["recovery_count"] = 0
                state["state_id"] = transition.get("to_state")
                state["variables"] = {**state.get("variables", {}), **dict(transition.get("updates", {}))}
                terminal_states = {
                    item.get("state_id") for item in script.get("states", [])
                    if isinstance(item, Mapping) and item.get("terminal") is True
                }
                result["should_end"] = bool(transition.get("should_end") or state["state_id"] in terminal_states)
                if result["should_end"]:
                    state["termination_reason"] = "completed"
                    result["termination_reason"] = "completed"
            else:
                result["should_end"] = False
        else:
            result["should_end"] = bool(current_state.get("terminal"))
        result["fsm_script_id"] = str(state.get("script_id", ""))
        result["fsm_transition_id"] = (
            str(applied_transition.get("transition_id"))
            if applied_transition is not None else None
        )
        result["fsm_state_before"] = state_before
        result["fsm_state_after"] = str(state.get("state_id", ""))
        result["fsm_transition_applied"] = applied_transition is not None
        result["fsm_recovery_count"] = int(state.get("recovery_count", 0))
        result.pop("transition_id", None)
        state["turn_index"] = index + 1
        state["memory"] = [*state.get("memory", []), {"messages": list(messages), "result": result}]
        state["conversation_prefix"] = [
            *normalized_messages,
            {"role": "user", "content": result["user_query"]},
        ]
        self.episode_store.set_state(self.STATE_KEY, state)
        self.episode_store.event("user_turn", {"messages": list(messages), "used_fallback": used_fallback}, result)
        return result


class SandboxApplication:
    """Dependency-free HTTP application boundary shared by generated tasks.

    Task code supplies business callbacks.  This class owns the stable Trainer
    protocol, authentication, error envelopes, reset/replay, tool discovery,
    idempotency forwarding and WSGI transport.
    """

    def __init__(
        self,
        *,
        episode_store: EpisodeStore,
        tool_registry: ContractToolRegistry,
        observation: Callable[[], Mapping[str, Any]],
        reward: Callable[[], Mapping[str, Any]],
        user_turn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
        business_snapshot: Callable[[], Mapping[str, Any]] | None = None,
        reset_hook: Callable[[Episode], None] | None = None,
        data_hash: str = "",
    ) -> None:
        self.episode_store = episode_store
        self.tool_registry = tool_registry
        self.observation_callback = observation
        self.reward_callback = reward
        self.user_turn_callback = user_turn
        self.business_snapshot = business_snapshot
        self.reset_hook = reset_hook
        self.data_hash = data_hash

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        wanted = name.casefold()
        return next((str(value) for key, value in headers.items() if key.casefold() == wanted), None)

    def handle(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        started = time.monotonic()
        request = request_id()
        supplied = dict(headers or {})
        response_headers = {"Content-Type": "application/json", "X-Request-ID": request}
        try:
            episode_id = self._header(supplied, "X-Episode-ID")
            with self.episode_store.episode_context(episode_id):
                with self.episode_store.transaction():
                    with capture_runtime_llm_trace() as llm_trace:
                        payload = self._dispatch(method.upper(), path, body, supplied)
                    if llm_trace and path != "/health":
                        self.episode_store.event(
                            "runtime_llm_call",
                            {
                                "request_path": path,
                                "summary": summarize_runtime_llm_trace(llm_trace),
                            },
                            {"recorded": True},
                        )
            status, response = 200, dict(payload)
        except SandboxError as exc:
            status, response = exc.status, exc.body(request)
        except Exception as exc:  # keep the public protocol stable
            error = SandboxError("INTERNAL_ERROR", "sandbox request failed", 500, {"type": type(exc).__name__})
            status, response = error.status, error.body(request)
        tool_call_id = None
        if path.startswith("/v1/tools/"):
            tool_call_id = f"call-{uuid.uuid4().hex}"
            response_headers["X-Tool-Call-ID"] = tool_call_id
        try:
            with self.episode_store.episode_context(self._header(supplied, "X-Episode-ID")):
                episode_id = self.episode_store.current().episode_id if path != "/health" else None
        except Exception:
            episode_id = None
        JsonLog("sandbox.request").emit(
            "request_completed", request_id=request, episode_id=episode_id,
            tool_call_id=tool_call_id, method=method.upper(), path=path, status=status,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
        )
        return status, response, response_headers

    def _dispatch(self, method: str, path: str, body: Any, headers: Mapping[str, str]) -> Mapping[str, Any]:
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        if method == "GET" and path == "/v1/tools":
            return {"tools": self.tool_registry.tools}
        trainer_paths = {"/v1/reset", "/v1/observation", "/v1/state", "/v1/user_simulator", "/v1/agent_response", "/v1/reward", "/v1/replay"}
        mutation = os.getenv("SANDBOX_MUTATION_MODE", "disabled")
        if path in trainer_paths and mutation != "bypass_trainer_auth":
            require_trainer(self._header(headers, "Authorization"))
        if method == "POST" and path == "/v1/reset":
            if body is None:
                body = {}
            if not isinstance(body, Mapping):
                raise SandboxError("INVALID_ARGUMENT", "reset body must be an object", 400)
            unknown = sorted(set(body) - {"episode_id", "seed"})
            if unknown:
                raise SandboxError("INVALID_ARGUMENT", "reset has unexpected properties", 400, unknown)
            selected = self._header(headers, "X-Episode-ID")
            if selected and body.get("episode_id", selected) != selected:
                raise SandboxError("EPISODE_CONFLICT", "reset episode differs from X-Episode-ID", 409)
            episode = self.episode_store.reset(
                episode_id=body.get("episode_id") or selected, seed=body.get("seed"), data_hash=self.data_hash
            )
            if self.reset_hook:
                self.reset_hook(episode)
            return {
                "episode_id": episode.episode_id,
                "seed": episode.seed,
                "data_hash": episode.data_hash,
                "schema_version": episode.schema_version,
            }
        if method == "GET" and path == "/v1/observation":
            return dict(self.observation_callback())
        if method == "GET" and path == "/v1/state":
            if self.business_snapshot is None:
                raise SandboxError(
                    "STATE_UNAVAILABLE", "business snapshot callback is unavailable", 500
                )
            return {"business_state": dict(self.business_snapshot())}
        if method == "POST" and path == "/v1/user_simulator":
            if not isinstance(body, Mapping) or not isinstance(body.get("messages"), list):
                raise SandboxError("INVALID_ARGUMENT", "messages must be an array", 400)
            if set(body) != {"messages"} or any(
                not isinstance(message, Mapping) or set(message) != {"role", "content"}
                for message in body["messages"]
            ):
                raise SandboxError("INVALID_ARGUMENT", "undeclared message or request properties", 400)
            return dict(self.user_turn_callback(body["messages"]))
        if method == "POST" and path == "/v1/agent_response":
            if not isinstance(body, Mapping) or not isinstance(body.get("content"), str) or not body["content"].strip():
                raise SandboxError("INVALID_ARGUMENT", "agent response requires non-empty content", 400)
            if set(body) != {"content"}:
                raise SandboxError("INVALID_ARGUMENT", "agent response has undeclared properties", 400)
            content = body["content"].strip()
            self.episode_store.set_state("final_agent_response", content)
            self.episode_store.event("agent_response", {"content": content}, {"accepted": True})
            return {"accepted": True}
        if method == "GET" and path == "/v1/reward":
            if mutation == "constant_reward":
                return {"reward": 0.0, "raw_reward": 0.0, "components": {"__mutation__": 0.0}}
            return dict(self.reward_callback())
        if method == "GET" and path == "/v1/replay":
            return self.episode_store.replay()
        prefix = "/v1/tools/"
        if method == "POST" and path.startswith(prefix) and len(path) > len(prefix):
            if body is None:
                body = {}
            if not isinstance(body, Mapping):
                raise SandboxError("INVALID_ARGUMENT", "tool body must be an object", 400)
            result = self.tool_registry.execute(
                path[len(prefix):], body, idem_key=self._header(headers, "Idempotency-Key")
            )
            if not isinstance(result, Mapping):
                return {"result": result}
            return dict(result)
        raise SandboxError("NOT_FOUND", f"unknown endpoint: {method} {path}", 404)

    def __call__(self, environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET"))
        path = str(environ.get("PATH_INFO", "/"))
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(length) if length else b""
        request_headers = {
            key[5:].replace("_", "-"): value
            for key, value in environ.items() if key.startswith("HTTP_")
        }
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_request_id = request_id()
            error: SandboxError
            trainer_paths = {
                "/v1/reset", "/v1/observation", "/v1/state", "/v1/user_simulator",
                "/v1/agent_response", "/v1/reward", "/v1/replay",
            }
            try:
                if (
                    path in trainer_paths
                    and os.getenv("SANDBOX_MUTATION_MODE", "disabled") != "bypass_trainer_auth"
                ):
                    require_trainer(self._header(request_headers, "Authorization"))
                error = SandboxError("INVALID_JSON", "request body is not valid JSON", 400)
            except SandboxError as exc:
                error = exc
            status, payload, headers = error.status, error.body(malformed_request_id), {
                "Content-Type": "application/json",
                "X-Request-ID": malformed_request_id,
            }
        else:
            status, payload, headers = self.handle(method, path, body, request_headers)
        encoded = canonical_json(payload).encode("utf-8")
        response_headers = list(headers.items()) + [("Content-Length", str(len(encoded)))]
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 409: "Conflict", 500: "Internal Server Error", 503: "Service Unavailable"}.get(status, "Error")
        start_response(f"{status} {reason}", response_headers)
        return [encoded]


class AcceptanceScenarioRunner:
    """Execute EnvFactory-owned structured black-box scenarios."""

    def __init__(
        self,
        call: Callable[..., tuple[int, Any, Mapping[str, str]]],
        *,
        trainer_headers: Mapping[str, str] | None = None,
        business_snapshot: Callable[[], Any] | None = None,
        mutate_business_state: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.call = call
        self.trainer_headers = dict(trainer_headers or {})
        self.business_snapshot = business_snapshot
        self.mutate_business_state = mutate_business_state

    @staticmethod
    def _resolve(value: Any, variables: Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping) and set(value) == {"$ref"}:
            return variables.get(str(value["$ref"]))
        if isinstance(value, Mapping):
            return {key: AcceptanceScenarioRunner._resolve(item, variables) for key, item in value.items()}
        if isinstance(value, list):
            return [AcceptanceScenarioRunner._resolve(item, variables) for item in value]
        return value

    def run(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        variables: dict[str, Any] = {}
        step_results: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        last_body: Any = None
        for index, step in enumerate(scenario.get("steps", [])):
            if not isinstance(step, Mapping):
                raise SandboxError("SCENARIO_INVALID", f"step {index} is not an object", 500)
            operation = step.get("operation")
            if operation == "reset":
                method, path, auth, body = "POST", "/v1/reset", True, self._resolve(step.get("body", {}), variables)
            elif operation == "tool_call":
                method, path, auth = "POST", f"/v1/tools/{step.get('tool_name', '')}", False
                body = self._resolve(step.get("arguments", {}), variables)
            elif operation == "agent_response":
                method, path, auth = "POST", "/v1/agent_response", True
                body = {"content": str(self._resolve(step.get("content", ""), variables))}
            elif operation in {"observation", "reward", "replay"}:
                method, path, auth, body = "GET", f"/v1/{operation}", True, None
            elif operation == "business_snapshot":
                if self.business_snapshot is None:
                    raise SandboxError("SCENARIO_INVALID", "business snapshot callback is unavailable", 500)
                status, last_body = 200, self.business_snapshot()
                method, path, auth, body = "INTERNAL", "business_snapshot", False, None
            elif operation == "mutate_business_state":
                if self.mutate_business_state is None:
                    raise SandboxError("SCENARIO_INVALID", "business mutation callback is unavailable", 500)
                mutation = self._resolve(step.get("mutation", {}), variables)
                status, last_body = 200, self.mutate_business_state(mutation)
                method, path, auth, body = "INTERNAL", "mutate_business_state", False, mutation
            else:
                raise SandboxError("SCENARIO_INVALID", f"unsupported operation: {operation}", 500)
            if method != "INTERNAL":
                status, last_body, _ = self.call(method, path, body, self.trainer_headers if auth else {})
            expected_status = step.get("expected_status", 200)
            if status != expected_status:
                raise SandboxError("SCENARIO_ASSERTION_FAILED", f"step {index} expected HTTP {expected_status}, got {status}", 500, {"body": last_body, "actual_status": status})
            capture = step.get("capture", {})
            if isinstance(capture, Mapping):
                for name, capture_path in capture.items():
                    variables[str(name)] = DeclarativeMetricEvaluator._resolve(last_body, str(capture_path))
            step_id = step.get("step_id")
            if isinstance(step_id, str) and step_id:
                step_results[step_id] = last_body
            history.append({"operation": operation, "status": status, "body": last_body})
        for assertion in scenario.get("assertions", []):
            if not isinstance(assertion, Mapping):
                raise SandboxError("SCENARIO_INVALID", "assertion must be an object", 500)
            source_name = assertion.get("source")
            if source_name == "variables":
                source = variables
            elif isinstance(source_name, str) and source_name.startswith("step:"):
                source = step_results.get(source_name.split(":", 1)[1])
            else:
                source = last_body
            actual = DeclarativeMetricEvaluator._resolve(source, str(assertion.get("path", "$")))
            expected = self._resolve(assertion.get("expected"), variables)
            if not DeclarativeMetricEvaluator._compare(actual, str(assertion.get("operator")), expected):
                raise SandboxError("SCENARIO_ASSERTION_FAILED", "scenario assertion failed", 500, {"actual": actual, "assertion": dict(assertion)})
        return {"scenario_id": scenario.get("scenario_id"), "history": history, "variables": variables, "step_results": step_results}


def request_id() -> str:
    return f"req-{uuid.uuid4().hex}"
