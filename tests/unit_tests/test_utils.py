import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from retrieval_graph import utils


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing ``text``."""
    stream = f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


######################
# Messages & tokens
######################


def test_get_message_text_handles_strings_and_blocks() -> None:
    assert utils.get_message_text(HumanMessage(content="Hello")) == "Hello"
    blocks = AIMessage(
        content=[
            {"type": "text", "text": "Hello"},
            " ",
            {"type": "tool_use", "id": "1", "name": "x", "input": {}},
            {"type": "text", "text": "World"},
        ]
    )
    assert utils.get_message_text(blocks) == "Hello World"


def test_estimate_tokens() -> None:
    assert utils.estimate_tokens("") == 0
    assert utils.estimate_tokens("abcd" * 10) == 10


def test_truncate_to_token_limit() -> None:
    assert utils.truncate_to_token_limit("short", max_tokens=10) == "short"
    assert utils.truncate_to_token_limit("", max_tokens=10) == ""
    long_text = "x" * 1000
    truncated = utils.truncate_to_token_limit(long_text, max_tokens=10)
    assert truncated.startswith("x" * 36)
    assert "CONTENT TRUNCATED" in truncated
    assert len(truncated) < len(long_text)


######################
# Document formatting
######################


def test_format_docs() -> None:
    docs = [Document(page_content="Hello"), Document(page_content="World")]
    assert utils.format_docs(docs) == (
        "<documents>\n<document>\nHello\n</document>\n"
        "<document>\nWorld\n</document>\n</documents>"
    )
    assert utils.format_docs(None) == "<documents></documents>"


def test_format_docs_limits_count_and_length() -> None:
    docs = [Document(page_content="y" * 50) for _ in range(5)]
    formatted = utils.format_docs(docs, max_content_length=10, max_total_docs=2)
    assert formatted.count("<document>") == 2
    assert "Showing 2 of 5 documents" in formatted
    assert "y" * 10 + "... [TRUNCATED]" in formatted


def test_format_docs_safe_renders_metadata() -> None:
    formatted = utils.format_docs_safe(
        [Document(page_content="Body", metadata={"source": "a.txt"})]
    )
    assert "source='a.txt'" in formatted
    assert "Body" in formatted
    assert utils.format_docs_safe([]) == "<documents></documents>"


def test_format_docs_safe_respects_token_budget() -> None:
    docs = [Document(page_content="z" * 4000) for _ in range(10)]  # ~1000 tokens each
    formatted = utils.format_docs_safe(docs, max_total_tokens=2500)
    assert formatted.count("<document>") < 10
    assert "documents (others truncated" in formatted


def test_format_docs_safe_truncates_a_single_huge_document() -> None:
    formatted = utils.format_docs_safe(
        [Document(page_content="h" * 400_000)], max_total_tokens=5000
    )
    assert "TRUNCATED" in formatted
    assert utils.estimate_tokens(formatted) <= 5000


def test_format_docs_with_citations() -> None:
    formatted = utils.format_docs_with_citations(
        [
            Document(
                page_content="A", metadata={"title": "T", "source": "S", "type": "pdf"}
            ),
            Document(page_content="B"),
        ]
    )
    assert '<document title="T" source="S" type="pdf">' in formatted
    assert 'title="Document 2" source="Document_2" type="unknown"' in formatted
    assert utils.format_docs_with_citations(None) == "<documents></documents>"


######################
# Models
######################


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("anthropic/claude-haiku-5-5", ("claude-haiku-5-5", "anthropic")),
        ("openai/gpt-4o", ("gpt-4o", "openai")),
        ("claude-haiku-5-5", ("claude-haiku-5-5", None)),
    ],
)
def test_load_chat_model_splits_provider(
    monkeypatch: pytest.MonkeyPatch, name: str, expected: tuple[str, str | None]
) -> None:
    seen: list[tuple[str, str | None]] = []

    def fake_init_chat_model(model: str, *, model_provider: str | None) -> str:
        seen.append((model, model_provider))
        return "model"

    monkeypatch.setattr(utils, "init_chat_model", fake_init_chat_model)
    assert utils.load_chat_model(name) == "model"
    assert seen == [expected]


######################
# URL ingestion (local HTTP server, no internet)
######################

LONG_TEXT = "Photosynthesis turns light into chemical energy. " * 10


@pytest.fixture
async def site() -> AsyncIterator[str]:
    async def html(_: web.Request) -> web.Response:
        body = (
            "<html><head><style>p{color:red}</style><script>var x=1;</script></head>"
            f"<body><h1>Plants</h1><p>{LONG_TEXT}</p></body></html>"
        )
        return web.Response(text=body, content_type="text/html")

    async def pdf(_: web.Request) -> web.Response:
        return web.Response(
            body=make_pdf("Photosynthesis in plants"), content_type="application/pdf"
        )

    async def slow(_: web.Request) -> web.Response:
        await asyncio.sleep(2)
        return web.Response(text="late")

    async def broken_pdf(_: web.Request) -> web.Response:
        return web.Response(body=b"not a pdf", content_type="application/pdf")

    app = web.Application()
    app.router.add_get("/page.html", html)
    app.router.add_get("/notes.pdf", pdf)
    app.router.add_get("/broken.pdf", broken_pdf)
    app.router.add_get("/slow", slow)
    runner = web.AppRunner(app)
    await runner.setup()
    tcp = web.TCPSite(runner, "127.0.0.1", 0)
    await tcp.start()
    port = tcp._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def test_extract_text_from_web_url_strips_markup(site: str) -> None:
    text = await utils.extract_text_from_web_url(f"{site}/page.html")
    assert text.startswith("Plants")
    assert LONG_TEXT.strip() in text
    assert "var x" not in text and "color:red" not in text
    assert "<" not in text


async def test_extract_text_from_web_url_reports_http_errors(site: str) -> None:
    assert await utils.extract_text_from_web_url(f"{site}/missing") == (
        "Failed to access URL: HTTP 404"
    )


async def test_extract_text_from_pdf_url(site: str) -> None:
    assert await utils.extract_text_from_pdf_url(f"{site}/notes.pdf") == (
        "Photosynthesis in plants"
    )
    assert await utils.extract_text_from_pdf_url(f"{site}/missing.pdf") == (
        "Failed to download PDF: HTTP 404"
    )
    assert (await utils.extract_text_from_pdf_url(f"{site}/broken.pdf")).startswith(
        "Error processing PDF:"
    )


async def test_url_fetch_times_out(site: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "URL_FETCH_TIMEOUT_SECONDS", 0.2)
    assert await utils.extract_text_from_web_url(f"{site}/slow") == (
        "Error processing web page: timed out after 0.2s"
    )


async def test_create_documents_from_urls(site: str) -> None:
    urls = [f"{site}/page.html", f"{site}/notes.pdf", f"{site}/missing"]
    page, pdf, missing = await utils.create_documents_from_urls(urls)

    assert page.metadata == {
        "source": urls[0],
        "type": "webpage",
        "title": "page.html",
        "extraction_success": True,
        "content_length": len(page.page_content.strip()),
    }
    # Fetched fine, but too little text to be useful.
    assert pdf.metadata["type"] == "pdf"
    assert pdf.metadata["extraction_success"] is False
    assert missing.metadata["extraction_success"] is False
    assert missing.page_content == "Failed to access URL: HTTP 404"


async def test_create_documents_from_urls_keeps_one_document_per_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def explode(url: str) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(utils, "extract_text_from_web_url", explode)
    (doc,) = await utils.create_documents_from_urls(["https://example.com/a"])
    assert doc.metadata["type"] == "error"
    assert doc.metadata["extraction_success"] is False
    assert "boom" in doc.page_content
