"""Indexing graph: add documents (plain text and/or URLs) to a user's knowledge base.

``docs`` may mix plain text and ``http(s)://`` URLs::

    index_urls -> index_docs

* ``index_urls`` downloads each URL (web pages and PDFs), extracts the text and
  indexes it, then hands the remaining plain-text documents to ``index_docs``.
* ``index_docs`` indexes those plain-text documents.

Every document is stamped with the configured ``user_id`` so retrieval is scoped to
that user. The graph ends by clearing ``docs`` (the output is ``{"docs": []}``).

Run settings (:class:`~retrieval_graph.configuration.IndexConfiguration`) are the
graph's context schema: nodes read them from ``runtime.context``.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from retrieval_graph import retrieval
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState
from retrieval_graph.utils import create_documents_from_urls

logger = logging.getLogger(__name__)

MIN_EXTRACTED_CHARS = 200
"""Pages yielding less text than this are treated as failed extractions."""
_ERROR_PREFIXES = ("Failed", "Error", "Document processing dependencies")
_MAX_REPORTED_FAILURES = 3


def is_url_document(doc: Document) -> bool:
    """Return True if the document is just an ``http(s)://`` URL to fetch."""
    content = doc.page_content.strip()
    return content.startswith(("http://", "https://")) and not any(
        ch.isspace() for ch in content
    )


def ensure_docs_have_user_id(
    docs: Sequence[Document], configuration: IndexConfiguration
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

    Args:
        docs: A sequence of Document objects to process.
        configuration: The run settings. Their ``user_id`` defaults to the
            ``USER_ID`` environment variable.

    Returns:
        A new list of Document objects with updated metadata.
    """
    user_id = configuration.user_id
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in docs
    ]


async def _add_documents(
    docs: Sequence[Document], configuration: IndexConfiguration
) -> None:
    """Stamp ``docs`` with the user id and add them to the vector store."""
    async with retrieval.make_retriever(configuration) as retriever:
        await retriever.aadd_documents(ensure_docs_have_user_id(docs, configuration))


async def index_urls(
    state: IndexState,
    config: RunnableConfig,
    runtime: Runtime[IndexConfiguration] | None = None,
) -> dict[str, Any]:
    """Index the documents behind any URLs in ``docs``; pass the rest on.

    Each URL is fetched and its text extracted. Pages that yield no meaningful
    content are reported; if none of the URLs can be extracted, the run fails with
    a ValueError explaining why.
    """
    urls = [doc.page_content.strip() for doc in state.docs if is_url_document(doc)]
    remaining = [doc for doc in state.docs if not is_url_document(doc)]

    if urls:
        processed_docs = await create_documents_from_urls(urls)

        valid_docs: list[Document] = []
        failures: list[tuple[str, str]] = []
        for url, doc in zip(urls, processed_docs, strict=True):
            content = doc.page_content.strip()
            if (
                not doc.metadata.get("extraction_success", False)
                or content == url
                or content.startswith(_ERROR_PREFIXES)
                or len(content) < MIN_EXTRACTED_CHARS
            ):
                reason = (
                    content
                    if content.startswith(("Failed", "Error"))
                    else "Insufficient content extracted"
                )
                failures.append((url, reason))
            else:
                valid_docs.append(doc)

        if valid_docs:
            configuration = IndexConfiguration.from_runtime(runtime, config)
            await _add_documents(valid_docs, configuration)
            logger.info("Indexed %d document(s) from URLs.", len(valid_docs))

        if failures:
            details = "\n".join(
                f"  - {url}: {reason}"
                for url, reason in failures[:_MAX_REPORTED_FAILURES]
            )
            error_msg = (
                f"Failed to extract content from {len(failures)} URL(s):\n{details}"
                + ("..." if len(failures) > _MAX_REPORTED_FAILURES else "")
                + "\n\nPlease verify URLs are accessible and contain extractable "
                "content, or upload document text manually."
            )
            logger.warning("URL processing error: %s", error_msg)
            if not valid_docs:
                raise ValueError(
                    "Could not extract content from any of the provided URLs. "
                    f"{error_msg}"
                )

    # Hand the plain-text documents to `index_docs`. Returning a list *replaces*
    # `docs` (see `reduce_docs`); an empty list leaves nothing to index.
    return {"docs": remaining}


async def index_docs(
    state: IndexState,
    config: RunnableConfig,
    runtime: Runtime[IndexConfiguration] | None = None,
) -> dict[str, str]:
    """Index the plain-text documents in ``docs``, then clear them from the state."""
    if state.docs:
        configuration = IndexConfiguration.from_runtime(runtime, config)
        await _add_documents(state.docs, configuration)
        logger.info("Indexed %d text document(s).", len(state.docs))
    return {"docs": "delete"}


# Run settings are the context schema, as in graph.py.
builder = StateGraph(IndexState, context_schema=IndexConfiguration)
builder.add_node("index_docs", index_docs)
builder.add_node("index_urls", index_urls)
builder.add_edge(START, "index_urls")  # Process URLs first
builder.add_edge("index_urls", "index_docs")  # Then the remaining plain-text docs
builder.add_edge("index_docs", END)

graph = builder.compile()
graph.name = "IndexGraph"
