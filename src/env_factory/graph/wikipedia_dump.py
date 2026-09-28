"""Build a local SQLite FTS index from a Wikimedia pages-articles dump."""

from collections.abc import Iterator
import bz2
import logging
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

from env_factory.graph.wikipedia import WikipediaError, WikipediaResponse, WikipediaResult

logger = logging.getLogger(__name__)
NS = "{http://www.mediawiki.org/xml/export-0.11/}"


class WikipediaDumpIndexer:
    """Index namespace-0 Wikipedia articles for local full-text search."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def build(self, dump_path: str | Path, *, replace: bool = False) -> int:
        dump_path = Path(dump_path)
        if not dump_path.is_file():
            raise FileNotFoundError(f"dump file does not exist: {dump_path}")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            if replace:
                connection.execute("DROP TABLE IF EXISTS pages_fts")
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts "
                "USING fts5(title, url UNINDEXED, text)"
            )
            count = 0
            with connection:
                for title, text in self._iter_articles(dump_path):
                    connection.execute(
                        "INSERT INTO pages_fts(title, url, text) VALUES (?, ?, ?)",
                        (title, self._page_url(title), text),
                    )
                    count += 1
                    if count % 10_000 == 0:
                        logger.info("Wikipedia 索引进度：已处理 %d 个页面", count)
            logger.info("Wikipedia 本地索引完成：%d 个页面，数据库=%s", count, self.database_path)
            return count
        finally:
            connection.close()

    def _iter_articles(self, dump_path: Path) -> Iterator[tuple[str, str]]:
        opener = bz2.open if dump_path.suffix == ".bz2" else open
        with opener(dump_path, "rb") as stream:
            for _, page in ET.iterparse(stream, events=("end",)):
                if page.tag != f"{NS}page":
                    continue
                namespace = page.findtext(f"{NS}ns")
                title = page.findtext(f"{NS}title") or ""
                text = page.findtext(f"{NS}revision/{NS}text") or ""
                if namespace == "0" and title and text:
                    yield title, text
                page.clear()

    @staticmethod
    def _page_url(title: str) -> str:
        from urllib.parse import quote

        return f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


class LocalWikipediaClient:
    """Search a local Wikipedia SQLite FTS5 index."""

    def __init__(self, database_path: str | Path, *, limit: int = 10) -> None:
        self.database_path = Path(database_path)
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Wikipedia index does not exist: {self.database_path}")
        self.limit = limit

    def search(self, query: str, *, limit: int | None = None, **_: object) -> WikipediaResponse:
        if not query.strip():
            raise ValueError("query must not be empty")
        result_limit = limit or self.limit
        if result_limit <= 0:
            raise ValueError("limit must be greater than zero")
        connection = sqlite3.connect(self.database_path)
        try:
            rows = connection.execute(
                "SELECT title, url, snippet(pages_fts, 2, '', '', ' … ', 80) "
                "FROM pages_fts WHERE pages_fts MATCH ? LIMIT ?",
                (query, result_limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise WikipediaError(f"invalid local Wikipedia query: {query}") from exc
        finally:
            connection.close()
        return WikipediaResponse(
            query=query,
            results=tuple(
                WikipediaResult(title=str(title), url=str(url), content=str(content), engine="wikipedia-local")
                for title, url, content in rows
            ),
        )
