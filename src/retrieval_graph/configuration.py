"""Configurable parameters for the indexing and retrieval graphs.

Values are read from the ``configurable`` section of a run's config - for example the
configuration panel in LangGraph Studio, the Touchstone UI's sidebar, or
``graph.ainvoke(inputs, {"configurable": {"user_id": "alice"}})``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Self

from langchain_core.runnables import RunnableConfig, ensure_config

from retrieval_graph import prompts

# Defaults. The previous defaults (claude-3-5-sonnet-20240620,
# claude-3-haiku-20240307 and Google's text-embedding-004) have been retired by
# their providers and now fail with "model not found".
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-001"
DEFAULT_RESPONSE_MODEL = "anthropic/claude-sonnet-5-5"
DEFAULT_QUERY_MODEL = "anthropic/claude-haiku-5-5"


def normalize_embedding_model(name: str) -> str:
    """Return ``name`` in ``provider/model`` form, adding Google's ``models/`` prefix.

    Names without a provider are assumed to be Google models.

    Examples:
        >>> normalize_embedding_model("google/gemini-embedding-001")
        'google/models/gemini-embedding-001'
        >>> normalize_embedding_model("gemini-embedding-001")
        'google/models/gemini-embedding-001'
    """
    if "/" not in name:
        name = f"google/{name}"
    provider, model = name.split("/", maxsplit=1)
    if provider == "google" and not model.startswith("models/"):
        return f"google/models/{model}"
    return name


@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(
        default_factory=lambda: os.getenv("USER_ID", ""),
        metadata={
            "description": "Unique identifier for the user. Documents are indexed "
            "and retrieved per user. Defaults to the USER_ID environment variable."
        },
    )

    elasticsearch_host: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_HOST", "localhost"),
        metadata={
            "description": "Informational only: the connection uses the "
            "ELASTICSEARCH_URL environment variable."
        },
    )

    elasticsearch_port: int = field(
        default_factory=lambda: int(os.getenv("ELASTICSEARCH_PORT", "9200")),
        metadata={
            "description": "Informational only: the connection uses the "
            "ELASTICSEARCH_URL environment variable."
        },
    )

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={
            "description": "Name of the embedding model to use, as provider/model. "
            "Only Google ('google/...') embedding models are supported."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic-local"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic-local",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic-local'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"k": 1, "score_threshold": 0.7},
        metadata={
            "description": "Keyword arguments for the retriever's similarity search. "
            "'k' is the number of documents returned per search (default 1). "
            "Note: 'score_threshold' is not applied by the default similarity search."
        },
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.user_id or self.user_id == "default_user":
            raise ValueError("Please provide a valid user_id in the configuration.")
        self.embedding_model = normalize_embedding_model(
            self.embedding_model or DEFAULT_EMBEDDING_MODEL
        )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> Self:
        """Create an instance from the ``configurable`` section of a RunnableConfig.

        Unknown keys (for example ``thread_id``, which the LangGraph server adds) are
        ignored.

        Args:
            config: The configuration object to use.

        Returns:
            An instance of the class with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={
            "description": "Reserved: not currently used. Specialist prompts live "
            "in retrieval_graph/prompts.py."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=DEFAULT_RESPONSE_MODEL,
        metadata={
            "description": "The language model used by the specialist and critique "
            "agents. Should be in the form: provider/model-name."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "Reserved: not currently used. The classifier prompt "
            "lives in retrieval_graph/prompts.py."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=DEFAULT_QUERY_MODEL,
        metadata={
            "description": "The language model used to classify queries into a "
            "subject. Should be in the form: provider/model-name."
        },
    )

    enable_web_search: bool = field(
        default=True,
        metadata={
            "description": "Whether the specialist agents may use the web_search "
            "tool (Tavily)."
        },
    )

    web_search_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", ""),
        metadata={
            "description": "API key for web search service (Tavily). Defaults to "
            "the TAVILY_API_KEY environment variable."
        },
    )

    min_retrieval_threshold: int = field(
        default=2,
        metadata={"description": "Reserved: not currently used."},
    )
