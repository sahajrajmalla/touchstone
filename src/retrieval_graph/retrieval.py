"""Create retrievers for the configured vector store.

Only a local Elasticsearch deployment (``retriever_provider="elastic-local"``) is
currently supported. Every search is filtered by ``user_id`` so that each user only
sees the documents they indexed.

Usage::

    async with make_retriever(config) as retriever:
        docs = await retriever.ainvoke("photosynthesis")
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from retrieval_graph.configuration import IndexConfiguration

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

SUPPORTED_RETRIEVER_PROVIDERS = ("elastic-local",)
ELASTICSEARCH_INDEX_NAME = "langchain_index"
"""Elasticsearch index that stores every user's documents (filtered by user_id)."""
USER_ID_FIELD = "metadata.user_id.keyword"
"""Exact-match field for the per-user filter.

Metadata is mapped dynamically, so ``metadata.user_id`` is an *analyzed* text field
(lowercased and split on punctuation). Filtering on it made "Alice" or "student-42"
match nothing, and let "alice" also match the documents of "alice-smith". The
``.keyword`` sub-field holds the exact value (up to 256 characters).
"""


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder.

    Args:
        model: Embedding model as ``provider/model``. A bare model name is treated
            as a Google model.

    Raises:
        ValueError: If ``model`` is empty or the provider is not supported.
    """
    if not model:
        raise ValueError("Embedding model cannot be empty or None")

    provider, model_name = (
        model.split("/", maxsplit=1) if "/" in model else ("google", model)
    )
    match provider:
        case "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=model_name)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


def _elasticsearch_settings() -> tuple[str, str, str]:
    """Read the Elasticsearch connection settings from the environment.

    The URL must be non-empty. User and password must be set but may be empty
    (e.g. for a local cluster with security disabled).

    Raises:
        ValueError: With the names of the missing variables, if any.
    """
    settings = {
        "ELASTICSEARCH_URL": os.environ.get("ELASTICSEARCH_URL") or None,
        "ELASTICSEARCH_USER": os.environ.get("ELASTICSEARCH_USER"),
        "ELASTICSEARCH_PASSWORD": os.environ.get("ELASTICSEARCH_PASSWORD"),
    }
    missing = [name for name, value in settings.items() if value is None]
    if missing:
        raise ValueError(
            "Elasticsearch is not configured: set "
            f"{', '.join(missing)} in your .env file (see .env.example)."
        )
    url, user, password = settings.values()
    return str(url), str(user), str(password)


def _make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> tuple[VectorStoreRetriever, "AsyncElasticsearch"]:
    """Build a user-scoped Elasticsearch retriever and return it with its client.

    No network I/O happens here; connections are opened lazily on first use.
    """
    from elasticsearch import AsyncElasticsearch
    from langchain_elasticsearch import AsyncElasticsearchStore

    url, user, password = _elasticsearch_settings()
    client = AsyncElasticsearch(hosts=[url], basic_auth=(user, password))
    store = AsyncElasticsearchStore(
        index_name=ELASTICSEARCH_INDEX_NAME,
        embedding=embedding_model,
        client=client,
    )

    search_kwargs = dict(configuration.search_kwargs)
    # Restrict results to this user's documents (data isolation). Build a new list
    # so the caller's configuration is never mutated.
    search_kwargs["filter"] = [
        *search_kwargs.get("filter", []),
        {"term": {USER_ID_FIELD: configuration.user_id}},
    ]
    search_kwargs.setdefault("k", 1)
    return store.as_retriever(search_kwargs=search_kwargs), client


@asynccontextmanager
async def make_retriever(config: RunnableConfig) -> AsyncIterator[VectorStoreRetriever]:
    """Create a retriever for the agent, based on the current configuration.

    The underlying Elasticsearch client is closed when the ``async with`` block
    exits, even if it raises.

    Raises:
        ValueError: If the configuration or environment is invalid.
    """
    configuration = IndexConfiguration.from_runnable_config(config)
    if configuration.retriever_provider not in SUPPORTED_RETRIEVER_PROVIDERS:
        raise ValueError(
            "Unrecognized retriever_provider in configuration. "
            f"Expected one of: {', '.join(SUPPORTED_RETRIEVER_PROVIDERS)}\n"
            f"Got: {configuration.retriever_provider}"
        )

    embedding_model = make_text_encoder(configuration.embedding_model)
    retriever, client = _make_elastic_retriever(configuration, embedding_model)
    try:
        yield retriever
    finally:
        try:
            await client.close()
        except Exception:  # Never let cleanup hide the original error.
            logger.warning("Failed to close the Elasticsearch client", exc_info=True)
