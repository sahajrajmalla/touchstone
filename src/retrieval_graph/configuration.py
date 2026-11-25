"""Run settings for the indexing and retrieval graphs.

The settings are the graphs' context schema. They arrive as the run's context - for
example from the configuration panel in LangGraph Studio, the Touchstone UI's sidebar,
``client.runs.stream(..., context={...})`` with the LangGraph SDK, or
``graph.ainvoke(inputs, context={"user_id": "alice"})``.

Runs without a context fall back to the ``configurable`` section of the run's config,
the way settings were passed before. That fallback is deprecated (see
:meth:`IndexConfiguration.from_runtime`).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Self

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.runtime import Runtime

from retrieval_graph import prompts

# Defaults. The previous defaults (claude-3-5-sonnet-20240620,
# claude-3-haiku-20240307 and Google's text-embedding-004) have been retired by
# their providers and now fail with "model not found".
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-001"
DEFAULT_RESPONSE_MODEL = "anthropic/claude-sonnet-5-5"
DEFAULT_QUERY_MODEL = "anthropic/claude-haiku-5-5"

CONFIGURABLE_DEPRECATION_MESSAGE = (
    "Passing Touchstone settings in config['configurable'] is deprecated; pass them as "
    "the run context instead, e.g. graph.ainvoke(inputs, context={'user_id': 'alice'}), "
    "or context={...} with the LangGraph SDK. The 'configurable' fallback will be "
    "removed in a future release (see docs/configuration.md)."
)


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
    def field_names(cls) -> frozenset[str]:
        """Return the names of the settings this class accepts."""
        return frozenset(f.name for f in fields(cls) if f.init)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Create an instance from ``values``, ignoring keys that are not settings."""
        names = cls.field_names()
        return cls(**{k: v for k, v in values.items() if k in names})

    @classmethod
    def from_runtime(
        cls, runtime: Runtime[Any] | None, config: RunnableConfig | None = None
    ) -> Self:
        """Return the settings of the current run.

        The run's context wins: LangGraph has already turned a ``context={...}`` dict
        into an instance of the graph's context schema, with defaults filled in.

        Without a context (``runtime`` is None, or a library call without
        ``context=``), the settings are read from ``config["configurable"]`` instead,
        with a ``DeprecationWarning`` if it contains any setting. ``thread_id`` and
        other non-settings keys stay in ``configurable`` and never warn.

        Args:
            runtime: The ``runtime`` LangGraph injects into a node, or None.
            config: The run's config; only read when there is no context.

        Raises:
            TypeError: If the context is neither settings nor a mapping.
            ValueError: If no valid ``user_id`` is configured.
        """
        context = runtime.context if runtime is not None else None
        if isinstance(context, cls):
            return context
        if isinstance(context, IndexConfiguration):
            # The indexer's settings given to the retrieval graph: keep the values.
            return cls.from_mapping(vars(context))
        if isinstance(context, Mapping):
            return cls.from_mapping(context)
        if context is not None:
            raise TypeError(
                f"Expected the run context to be a {cls.__name__} or a dict, "
                f"got {type(context).__name__}."
            )
        configurable = ensure_config(config).get("configurable") or {}
        if not cls.field_names().isdisjoint(configurable):
            warnings.warn(
                CONFIGURABLE_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2
            )
        return cls.from_mapping(configurable)

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> Self:
        """Create an instance from the ``configurable`` section of a RunnableConfig.

        Unknown keys (for example ``thread_id``, which the LangGraph server adds) are
        ignored. Graph nodes use :meth:`from_runtime`, which prefers the run context.

        Args:
            config: The configuration object to use.

        Returns:
            An instance of the class with the specified configuration.
        """
        return cls.from_mapping(ensure_config(config).get("configurable") or {})


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
