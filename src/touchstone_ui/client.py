"""Thin client for the Touchstone graphs served by a LangGraph server.

The UI never imports the graphs directly: it talks to the same LangGraph API that
LangGraph Studio uses (``make dev``), so it works unchanged against a deployed
server - just point ``TOUCHSTONE_API_URL`` at it.

Run settings are sent as the run ``context`` (the graphs' context schema), never as
``config["configurable"]``: the server rejects a run that sets both.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
from langgraph_sdk import get_sync_client

RETRIEVAL_GRAPH = "retrieval_graph"
INDEX_GRAPH = "indexer"


class BackendError(RuntimeError):
    """The LangGraph server could not be reached or the run failed."""


def _describe(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        detail = error.response.text.strip()
        return f"HTTP {error.response.status_code}: {detail or error.response.reason_phrase}"
    if isinstance(error, httpx.ConnectError | httpx.TimeoutException):
        return "could not reach the Touchstone backend - is `make dev` running?"
    return str(error) or type(error).__name__


class TouchstoneClient:
    """Ask questions and index documents through the LangGraph API."""

    def __init__(self, api_url: str) -> None:
        """Create a client for the LangGraph server at ``api_url``."""
        self.api_url = api_url.rstrip("/")
        self._sdk = get_sync_client(url=self.api_url)

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Return True if the server answers its health check."""
        try:
            response = httpx.get(f"{self.api_url}/ok", timeout=timeout)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def create_thread(self) -> str:
        """Create a conversation thread and return its id."""
        try:
            thread = self._sdk.threads.create(metadata={"source": "touchstone-ui"})
        except httpx.HTTPError as e:
            raise BackendError(_describe(e)) from e
        return str(thread["thread_id"])

    def ask(
        self, thread_id: str, question: str, context: Mapping[str, Any]
    ) -> Iterator[tuple[str, Any]]:
        """Stream ``(event, data)`` updates while the agents answer ``question``.

        ``context`` holds the run settings (``user_id``, models, web search).
        """
        payload: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
        try:
            for part in self._sdk.runs.stream(
                thread_id,
                RETRIEVAL_GRAPH,
                input=payload,
                context=dict(context),
                stream_mode="updates",
            ):
                yield part.event, part.data
        except httpx.HTTPError as e:
            raise BackendError(_describe(e)) from e

    def index(self, docs: Sequence[str], context: Mapping[str, Any]) -> None:
        """Add ``docs`` (plain text or URLs) to the knowledge base of ``context``'s user."""
        payload: dict[str, Any] = {"docs": list(docs)}
        try:
            result = self._sdk.runs.wait(
                None,
                INDEX_GRAPH,
                input=payload,
                context=dict(context),
                raise_error=False,
            )
        except httpx.HTTPError as e:
            raise BackendError(_describe(e)) from e
        error = result.get("__error__") if isinstance(result, dict) else None
        if isinstance(error, dict):
            raise BackendError(str(error.get("message") or error.get("error") or error))
