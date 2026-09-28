"""Client for the official MediaWiki Action API."""

from dataclasses import dataclass
import html
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


class WikipediaError(RuntimeError):
    """Raised when the Wikipedia API fails or returns invalid data."""


@dataclass(frozen=True)
class WikipediaResult:
    """A normalized Wikipedia search result."""

    title: str
    url: str
    content: str = ""
    engine: str = "wikipedia"
    score: float | None = None


@dataclass(frozen=True)
class WikipediaResponse:
    """A normalized Wikipedia search response."""

    query: str
    results: tuple[WikipediaResult, ...]


class WikipediaClient:
    """Search a Wikipedia language edition through MediaWiki Action API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 10.0,
        user_agent: str = "env-factory/0.1 (knowledge graph builder)",
    ) -> None:
        self.base_url = (
            base_url or os.getenv(
                "WIKIPEDIA_API_URL", "https://zh.wikipedia.org/w/api.php"
            )
        ).rstrip("?")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not user_agent.strip():
            raise ValueError("user_agent must not be empty")
        self.timeout = timeout
        self.user_agent = user_agent

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        language: str | None = None,
        **_: Any,
    ) -> WikipediaResponse:
        """Search page titles and text in the configured Wikipedia edition."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|size|wordcount",
            "format": "json",
            "formatversion": 2,
        }
        payload = self._request(params)
        raw_results = payload.get("query", {}).get("search", [])
        if not isinstance(raw_results, list):
            raise WikipediaError("Wikipedia response field 'search' must be a list")
        results = tuple(
            WikipediaResult(
                title=str(item.get("title", "")),
                url=self._page_url(str(item.get("title", ""))),
                content=self._clean_snippet(str(item.get("snippet", ""))),
                score=float(item["wordcount"]) if item.get("wordcount") is not None else None,
            )
            for item in raw_results
            if isinstance(item, dict) and item.get("title")
        )
        return WikipediaResponse(query=query, results=results)

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise WikipediaError(f"Wikipedia returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WikipediaError("Unable to reach Wikipedia API") from exc
        except json.JSONDecodeError as exc:
            raise WikipediaError("Wikipedia returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise WikipediaError("Wikipedia response must be a JSON object")
        return payload

    @staticmethod
    def _page_url(title: str) -> str:
        return f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", snippet))).strip()
