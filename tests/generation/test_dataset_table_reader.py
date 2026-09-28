import csv
import gzip
import io
import json
import sqlite3
import tarfile
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from env_factory.generation.dataset_table_reader import sample_table, selected_member
from env_factory.generation.task_generator import TaskGenerationError


ROWS = [
    {"order_id": f"O{index}", "category": "Books" if index % 2 else "Food",
     "amount": index * 10}
    for index in range(1, 8)
]


@pytest.mark.parametrize("suffix", [
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    ".xlsx", ".parquet", ".sqlite", ".sqlite3", ".db",
])
def test_supported_table_formats_produce_same_business_rows(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"orders{suffix}"
    if suffix in {".csv", ".tsv"}:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]),
                                    delimiter="," if suffix == ".csv" else "\t")
            writer.writeheader()
            writer.writerows(ROWS)
    elif suffix == ".json":
        path.write_text(json.dumps({"data": ROWS}), encoding="utf-8")
    elif suffix in {".jsonl", ".ndjson"}:
        path.write_text("\n".join(json.dumps(row) for row in ROWS) + "\n", encoding="utf-8")
    elif suffix == ".xlsx":
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(list(ROWS[0]))
        for row in ROWS:
            sheet.append(list(row.values()))
        workbook.save(path)
    elif suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(ROWS), path)
    else:
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE orders (order_id TEXT, category TEXT, amount INTEGER)")
        connection.executemany("INSERT INTO orders VALUES (:order_id, :category, :amount)", ROWS)
        connection.commit()
        connection.close()
    headers, rows = sample_table(path)
    assert headers == list(ROWS[0])
    assert rows == [{key: str(value) for key, value in row.items()} for row in ROWS]


def test_nested_values_and_unsupported_formats_fail_closed(tmp_path: Path) -> None:
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps([{"order_id": "O1", "category": {"name": "Books"},
                                   "amount": 10}] * 5), encoding="utf-8")
    with pytest.raises(TaskGenerationError, match="nested"):
        sample_table(nested)
    unsupported = tmp_path / "orders.pdf"
    unsupported.write_bytes(b"not a table")
    with pytest.raises(TaskGenerationError, match="unsupported dataset format"):
        sample_table(unsupported)


@pytest.mark.parametrize("suffix", [".zip", ".tar", ".tar.gz", ".tgz", ".csv.gz"])
def test_archived_csv_is_sampled_without_extracting_all_members(tmp_path: Path, suffix: str) -> None:
    content = "order_id,category,amount\n" + "".join(
        f"O{index},Books,{index * 10}\n" for index in range(1, 8)
    )
    path = tmp_path / f"orders{suffix}"
    if suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("docs/readme.txt", "metadata")
            archive.writestr("data/orders.csv", content)
    elif suffix == ".csv.gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(content)
    else:
        with tarfile.open(path, "w:gz" if suffix != ".tar" else "w") as archive:
            payload = content.encode()
            info = tarfile.TarInfo("data/orders.csv")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    assert selected_member(path) == ("orders.csv" if suffix == ".csv.gz" else "data/orders.csv")
    headers, rows = sample_table(path)
    assert headers == ["order_id", "category", "amount"]
    assert rows[0] == {"order_id": "O1", "category": "Books", "amount": "10"}


def test_zip_parquet_materializes_only_selected_member(tmp_path: Path) -> None:
    parquet = tmp_path / "orders.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), parquet)
    archive_path = tmp_path / "orders.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(parquet, "tables/orders.parquet")
        archive.writestr("notes.txt", "not a table")
    assert selected_member(archive_path) == "tables/orders.parquet"
    assert sample_table(archive_path)[1][0]["amount"] == "10"


def test_large_csv_samples_first_rows_without_loading_entire_file(tmp_path: Path) -> None:
    path = tmp_path / "large.csv"
    with path.open("wb") as stream:
        stream.write(b"order_id,category,amount\n" + b"O1,Books,10\n" * 500)
        stream.truncate(110_000_000)
    assert path.stat().st_size > 100_000_000
    assert len(sample_table(path)[1]) == 500


def test_archive_rejects_unsafe_path_and_configured_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../orders.csv", "order_id,category,amount\n" * 6)
    with pytest.raises(TaskGenerationError, match="unsafe"):
        sample_table(path)
    large = tmp_path / "large.zip"
    with zipfile.ZipFile(large, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("orders.csv", "order_id,category,amount\n" * 1000)
    with pytest.raises(TaskGenerationError, match="size limit"):
        sample_table(large, max_source_bytes=1_000)
