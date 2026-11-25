import importlib

import pytest
from langchain_core.documents import Document

from retrieval_graph import index_graph
from retrieval_graph.configuration import Configuration, IndexConfiguration

from .conftest import FakeKnowledgeBase, context

index_module = importlib.import_module("retrieval_graph.index_graph")

PAGE_TEXT = "Photosynthesis is how plants turn light into chemical energy. " * 5


def fake_url_documents(failures: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    """Replacement for create_documents_from_urls; URLs in ``failures`` fail."""
    failures = failures or {}

    async def create_documents_from_urls(urls: list[str]) -> list[Document]:
        docs = []
        for url in urls:
            if url in failures:
                docs.append(
                    Document(
                        page_content=failures[url],
                        metadata={"source": url, "extraction_success": False},
                    )
                )
            else:
                docs.append(
                    Document(
                        page_content=PAGE_TEXT,
                        metadata={
                            "source": url,
                            "type": "webpage",
                            "extraction_success": True,
                        },
                    )
                )
        return docs

    return create_documents_from_urls


async def test_indexes_plain_text_documents(knowledge_base: FakeKnowledgeBase) -> None:
    result = await index_graph.ainvoke(
        {"docs": ["Photosynthesis converts light.", "The Civil War lasted 1861-1865."]},
        context=context("alice"),
    )
    assert result == {"docs": []}
    assert [d.page_content for d in knowledge_base.added] == [
        "Photosynthesis converts light.",
        "The Civil War lasted 1861-1865.",
    ]
    assert {d.metadata["user_id"] for d in knowledge_base.added} == {"alice"}
    assert knowledge_base.opened == knowledge_base.closed == 1


async def test_indexes_a_single_string(knowledge_base: FakeKnowledgeBase) -> None:
    await index_graph.ainvoke({"docs": "Cats swim in bowls."}, context=context("alice"))
    (doc,) = knowledge_base.added
    assert doc.page_content == "Cats swim in bowls."
    assert doc.metadata["id"] and doc.metadata["user_id"] == "alice"


async def test_indexes_urls_and_text_together(
    knowledge_base: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index_module, "create_documents_from_urls", fake_url_documents()
    )
    await index_graph.ainvoke(
        {"docs": ["https://example.com/plants", "Some plain notes."]},
        context=context("alice"),
    )
    contents = [d.page_content for d in knowledge_base.added]
    assert contents == [PAGE_TEXT, "Some plain notes."]
    # The URL itself is never indexed as text.
    assert "https://example.com/plants" not in contents
    assert knowledge_base.added[0].metadata["source"] == "https://example.com/plants"


async def test_partial_url_failure_indexes_the_rest(
    knowledge_base: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index_module,
        "create_documents_from_urls",
        fake_url_documents({"https://bad.example": "Failed to access URL: HTTP 404"}),
    )
    await index_graph.ainvoke(
        {"docs": ["https://good.example", "https://bad.example"]},
        context=context("alice"),
    )
    assert [d.metadata["source"] for d in knowledge_base.added] == [
        "https://good.example"
    ]


async def test_all_urls_failing_raises_a_helpful_error(
    knowledge_base: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index_module,
        "create_documents_from_urls",
        fake_url_documents({"https://bad.example": "Failed to access URL: HTTP 404"}),
    )
    with pytest.raises(
        ValueError, match=r"https://bad.example: Failed to access URL: HTTP 404"
    ):
        await index_graph.ainvoke(
            {"docs": ["https://bad.example"]}, context=context("alice")
        )
    assert knowledge_base.added == []


async def test_short_extractions_are_rejected(
    knowledge_base: FakeKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def tiny(urls: list[str]) -> list[Document]:
        return [
            Document(page_content="tiny", metadata={"extraction_success": True})
            for _ in urls
        ]

    monkeypatch.setattr(index_module, "create_documents_from_urls", tiny)
    with pytest.raises(ValueError, match="Insufficient content extracted"):
        await index_graph.ainvoke(
            {"docs": ["https://example.com"]}, context=context("alice")
        )


async def test_empty_input_does_not_touch_the_store(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    assert await index_graph.ainvoke({"docs": []}, context=context("alice")) == {
        "docs": []
    }
    assert knowledge_base.opened == 0


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("https://example.com/a.pdf", True),
        ("  http://example.com  ", True),
        ("https://example.com is a nice site", False),
        ("Photosynthesis", False),
        ("ftp://example.com", False),
    ],
)
def test_is_url_document(content: str, expected: bool) -> None:
    assert index_module.is_url_document(Document(page_content=content)) is expected


def test_user_id_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_ID", "bob")
    (doc,) = index_module.ensure_docs_have_user_id(
        [Document(page_content="x", metadata={"source": "s"})], IndexConfiguration()
    )
    assert doc.metadata == {"source": "s", "user_id": "bob"}


async def test_settings_in_configurable_still_work_but_are_deprecated(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    with pytest.warns(DeprecationWarning, match=r"config\['configurable'\]"):
        await index_graph.ainvoke(
            {"docs": ["Some notes."]}, {"configurable": {"user_id": "bob"}}
        )
    assert knowledge_base.user_ids == ["bob"]
    assert knowledge_base.added[0].metadata["user_id"] == "bob"


async def test_a_retrieval_configuration_is_accepted_as_context(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    # A Configuration is an IndexConfiguration, so one object serves both graphs.
    settings = Configuration(user_id="carol", response_model="anthropic/x")
    await index_graph.ainvoke({"docs": ["Some notes."]}, context=settings)
    assert knowledge_base.added[0].metadata["user_id"] == "carol"


async def test_the_context_rejects_unknown_settings(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    # A dict context must only hold this graph's settings (LangGraph builds the
    # IndexConfiguration from it). The LangGraph server drops unknown keys instead.
    with pytest.raises(TypeError, match="response_model"):
        await index_graph.ainvoke(
            {"docs": ["Some notes."]},
            context={"user_id": "carol", "response_model": "anthropic/x"},
        )
    assert knowledge_base.added == []
