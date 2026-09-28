"""Source file formats accepted by dataset-backed task generation."""

from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    ".xlsx", ".parquet", ".sqlite", ".sqlite3", ".db",
})
ARCHIVE_EXTENSIONS = frozenset({".zip", ".gz", ".tar", ".tar.gz", ".tgz"})
SUPPORTED_SOURCE_EXTENSIONS = SUPPORTED_EXTENSIONS | ARCHIVE_EXTENSIONS


def source_extension(name: str) -> str:
    lowered = Path(name).name.lower()
    return ".tar.gz" if lowered.endswith(".tar.gz") else "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
