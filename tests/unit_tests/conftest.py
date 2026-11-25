"""Shared fixtures for the offline unit tests.

Nothing here talks to an LLM, Elasticsearch or the internet:

* ``fake_llm`` replaces ``load_chat_model`` with a scripted model.
* ``knowledge_base`` replaces ``retrieval.make_retriever`` with an in-memory store.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langgraph.runtime import Runtime

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration

graph_module = importlib.import_module("retrieval_graph.graph")

# Environment variables the code reads; cleared so a developer's .env or shell
# never changes test results.
_ISOLATED_ENV_VARS = (
    "USER_ID",
    "TAVILY_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELASTICSEARCH_URL",
    "ELASTICSEARCH_USER",
    "ELASTICSEARCH_PASSWORD",
    "ELASTICSEARCH_HOST",
    "ELASTICSEARCH_PORT",
    "TOUCHSTONE_API_URL",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ISOLATED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # The UI loads .env on startup; never let a developer's .env leak into tests.
    import touchstone_ui.settings

    monkeypatch.setattr(touchstone_ui.settings, "load_dotenv", lambda **_: False)


def context(user_id: str = "alice", **settings: Any) -> dict[str, Any]:
    """Build a run context: the settings callers pass as ``context=``."""
    return {"user_id": user_id, **settings}


def runtime(user_id: str = "alice", **settings: Any) -> Runtime[Configuration]:
    """Build the ``runtime`` LangGraph injects into nodes, for direct node calls."""
    return Runtime(context=Configuration(**context(user_id, **settings)))


######################
# Fake chat model
######################

SpecialistReply = AIMessage | Callable[[str, list[str]], AIMessage]


@dataclass
class LLMCall:
    """One call made to the fake model."""

    role: str  # "classify" | "specialist" | "critique"
    prompt: str
    tools: list[str]
    model_name: str


@dataclass
class FakeLLM:
    """Scripted stand-in for every chat model used by the retrieval graph.

    * classification always returns ``subject``;
    * specialists pop ``specialist_replies`` (an AIMessage, or a callable receiving
      the rendered prompt and the bound tool names), then answer "final answer";
    * the critic pops ``critique_decisions``, then answers "respond".
    """

    subject: str = "science"
    specialist_replies: list[SpecialistReply] = field(default_factory=list)
    critique_decisions: list[str] = field(default_factory=list)
    calls: list[LLMCall] = field(default_factory=list)

    def __call__(self, model_name: str) -> _FakeModel:
        return _FakeModel(self, model_name)

    def prompts(self, role: str) -> list[str]:
        return [call.prompt for call in self.calls if call.role == role]

    def specialist_calls(self) -> list[LLMCall]:
        return [call for call in self.calls if call.role == "specialist"]


class _FakeModel:
    def __init__(
        self,
        llm: FakeLLM,
        model_name: str,
        schema: type | None = None,
        tools: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.schema = schema
        self.tools = tools or []

    def with_structured_output(self, schema: type) -> _FakeModel:
        return _FakeModel(self.llm, self.model_name, schema=schema)

    def bind_tools(self, tools: list[Any]) -> _FakeModel:
        return _FakeModel(self.llm, self.model_name, tools=[t.name for t in tools])

    async def ainvoke(self, value: PromptValue, config: Any = None) -> Any:
        prompt = value.to_string()
        llm = self.llm
        if self.schema is graph_module.QueryClassification:
            llm.calls.append(LLMCall("classify", prompt, [], self.model_name))
            return graph_module.QueryClassification(subject=llm.subject)
        if self.schema is graph_module.CritiqueDecision:
            llm.calls.append(LLMCall("critique", prompt, [], self.model_name))
            decision = (
                llm.critique_decisions.pop(0) if llm.critique_decisions else "respond"
            )
            return graph_module.CritiqueDecision(
                decision=decision, reasoning=f"critique #{len(llm.prompts('critique'))}"
            )
        llm.calls.append(LLMCall("specialist", prompt, self.tools, self.model_name))
        reply = llm.specialist_replies.pop(0) if llm.specialist_replies else None
        if reply is None:
            return AIMessage(content="final answer")
        return reply(prompt, self.tools) if callable(reply) else reply


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(graph_module, "load_chat_model", llm)
    return llm


def tool_call(
    name: str = "retrieve_documents", query: str = "q", id: str = "call-1"
) -> AIMessage:
    """An AIMessage requesting one tool call."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {"query": query}, "id": id}]
    )


######################
# Fake knowledge base
######################


@dataclass
class FakeKnowledgeBase:
    """In-memory replacement for the Elasticsearch-backed retriever."""

    search_results: list[Document] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    added: list[Document] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    """The ``user_id`` of the settings each retriever was opened with."""
    opened: int = 0
    closed: int = 0
    error: Exception | None = None


class _FakeRetriever:
    def __init__(self, kb: FakeKnowledgeBase) -> None:
        self.kb = kb

    async def ainvoke(self, query: str, config: Any = None) -> list[Document]:
        if self.kb.error:
            raise self.kb.error
        self.kb.queries.append(query)
        return list(self.kb.search_results)

    async def aadd_documents(self, docs: list[Document]) -> None:
        self.kb.added.extend(docs)


@pytest.fixture
def knowledge_base(monkeypatch: pytest.MonkeyPatch) -> FakeKnowledgeBase:
    kb = FakeKnowledgeBase()

    @asynccontextmanager
    async def fake_make_retriever(configuration: Any) -> AsyncIterator[_FakeRetriever]:
        kb.opened += 1
        kb.user_ids.append(configuration.user_id)
        try:
            yield _FakeRetriever(kb)
        finally:
            kb.closed += 1

    monkeypatch.setattr(retrieval, "make_retriever", fake_make_retriever)
    return kb
