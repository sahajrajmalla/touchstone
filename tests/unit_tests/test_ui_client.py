from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from touchstone_ui import client as client_module
from touchstone_ui.client import BackendError, TouchstoneClient
from touchstone_ui.documents import decode_uploads, parse_urls, split_text
from touchstone_ui.settings import DEFAULT_API_URL, UISettings, studio_url


class FakeSDK:
    """Records calls made through langgraph_sdk's sync client."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.wait_result: Any = {"docs": []}
        self.error: Exception | None = None
        self.threads = SimpleNamespace(create=self._create_thread)
        self.runs = SimpleNamespace(stream=self._stream, wait=self._wait)

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if self.error:
            raise self.error

    def _create_thread(self, **kwargs: Any) -> dict[str, str]:
        self._record("threads.create", **kwargs)
        return {"thread_id": "t-1"}

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        self._record("runs.stream", *args, **kwargs)
        yield SimpleNamespace(event="metadata", data={"run_id": "r"})
        yield SimpleNamespace(event="updates", data={"respond": {}})

    def _wait(self, *args: Any, **kwargs: Any) -> Any:
        self._record("runs.wait", *args, **kwargs)
        return self.wait_result


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    fake = FakeSDK()
    monkeypatch.setattr(client_module, "get_sync_client", lambda url: fake)
    return fake


def test_ask_streams_updates_for_the_retrieval_graph(sdk: FakeSDK) -> None:
    client = TouchstoneClient("http://server:2024/")
    events = list(client.ask("t-1", "What is DNA?", {"user_id": "alice"}))

    assert events == [("metadata", {"run_id": "r"}), ("updates", {"respond": {}})]
    name, args, kwargs = sdk.calls[0]
    assert (name, args) == ("runs.stream", ("t-1", "retrieval_graph"))
    assert kwargs["input"] == {
        "messages": [{"role": "user", "content": "What is DNA?"}]
    }
    # Settings travel as the run context; the server rejects context + configurable.
    assert kwargs["context"] == {"user_id": "alice"}
    assert "config" not in kwargs
    assert kwargs["stream_mode"] == "updates"
    assert client.api_url == "http://server:2024"


def test_create_thread(sdk: FakeSDK) -> None:
    assert TouchstoneClient(DEFAULT_API_URL).create_thread() == "t-1"
    assert sdk.calls[0][2] == {"metadata": {"source": "touchstone-ui"}}


def test_index_runs_the_indexer_statelessly(sdk: FakeSDK) -> None:
    TouchstoneClient(DEFAULT_API_URL).index(["a", "b"], {"user_id": "alice"})
    name, args, kwargs = sdk.calls[0]
    assert (name, args) == ("runs.wait", (None, "indexer"))
    assert kwargs["input"] == {"docs": ["a", "b"]}
    assert kwargs["context"] == {"user_id": "alice"}
    assert "config" not in kwargs
    assert kwargs["raise_error"] is False


def test_index_reports_run_errors(sdk: FakeSDK) -> None:
    sdk.wait_result = {
        "__error__": {"error": "ValueError", "message": "Could not extract"}
    }
    with pytest.raises(BackendError, match="Could not extract"):
        TouchstoneClient(DEFAULT_API_URL).index(["https://bad"], {"user_id": "a"})


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("refused"), "is `make dev` running"),
        (
            httpx.HTTPStatusError(
                "bad",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(422, text='{"detail":"invalid"}'),
            ),
            'HTTP 422: {"detail":"invalid"}',
        ),
    ],
)
def test_http_errors_become_backend_errors(
    sdk: FakeSDK, error: Exception, message: str
) -> None:
    sdk.error = error
    client = TouchstoneClient(DEFAULT_API_URL)
    with pytest.raises(BackendError, match=message):
        list(client.ask("t", "q", {}))
    with pytest.raises(BackendError, match=message):
        client.index(["x"], {})
    with pytest.raises(BackendError, match=message):
        client.create_thread()


def test_is_healthy(sdk: FakeSDK, monkeypatch: pytest.MonkeyPatch) -> None:
    client = TouchstoneClient(DEFAULT_API_URL)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200))
    assert client.is_healthy()
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(503))
    assert not client.is_healthy()

    def refuse(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)
    assert not client.is_healthy()


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert UISettings.from_env() == UISettings(DEFAULT_API_URL, "")
    monkeypatch.setenv("TOUCHSTONE_API_URL", "https://example.com/")
    monkeypatch.setenv("USER_ID", " alice ")
    assert UISettings.from_env() == UISettings("https://example.com", "alice")
    assert studio_url("http://127.0.0.1:2024") == (
        "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
    )


def test_split_text() -> None:
    text = "  First paragraph.\nStill first.\n\n \nSecond paragraph.  \n"
    assert split_text(text, split_paragraphs=True) == [
        "First paragraph.\nStill first.",
        "Second paragraph.",
    ]
    assert split_text(text, split_paragraphs=False) == [text.strip()]
    assert split_text("   ", split_paragraphs=True) == []


def test_parse_urls() -> None:
    assert parse_urls("https://a.org\n\n  http://b.org/x.pdf \nnot a url\nftp://c") == (
        ["https://a.org", "http://b.org/x.pdf"],
        ["not a url", "ftp://c"],
    )


def test_decode_uploads() -> None:
    assert decode_uploads(
        [
            ("a.md", "﻿# Notes\n".encode()),
            ("empty.txt", b"  "),
            ("bin.txt", b"\xff\xfe\x00"),
        ]
    ) == (["# Notes"], ["bin.txt"])
