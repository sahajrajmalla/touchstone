"""End-to-end runs of the retrieval graph with scripted models (no network)."""

import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_tavily.tavily_search import TavilySearchAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver

from retrieval_graph import graph
from retrieval_graph.state import State

from .conftest import FakeKnowledgeBase, FakeLLM, config, tool_call

graph_module = importlib.import_module("retrieval_graph.graph")


async def ask(question: str | list[Any], **configurable: Any) -> dict[str, Any]:
    messages = [HumanMessage(question)] if isinstance(question, str) else question
    return await graph.ainvoke({"messages": messages}, config(**configurable))


######################
# Routing
######################


@pytest.mark.parametrize(
    ("subject", "prompt_marker"),
    [
        ("science", "expert science educator"),
        ("history", "expert historian"),
        ("literature", "expert literature scholar"),
        ("general", "knowledgeable educator specializing in general knowledge"),
    ],
)
async def test_routes_to_the_matching_specialist(
    fake_llm: FakeLLM,
    knowledge_base: FakeKnowledgeBase,
    subject: str,
    prompt_marker: str,
) -> None:
    fake_llm.subject = subject
    result = await ask("A question")
    assert result["classification"] == subject
    assert prompt_marker in fake_llm.prompts("specialist")[0]


def test_unknown_classification_falls_back_to_general() -> None:
    state = State(messages=[], classification="astrology")
    assert graph_module.route_to_specialist(state) == "general_agent"
    assert graph_module.after_tools_routing(state) == "general_agent"


async def test_uses_configured_models(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    await ask("Hi", query_model="anthropic/q-model", response_model="anthropic/r-model")
    models = {(call.role, call.model_name) for call in fake_llm.calls}
    assert models == {
        ("classify", "anthropic/q-model"),
        ("specialist", "anthropic/r-model"),
        ("critique", "anthropic/r-model"),
    }


######################
# Happy paths
######################


async def test_direct_answer_without_tools(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    fake_llm.specialist_replies = [AIMessage("Hello! How can I help?")]
    result = await ask("Hello")

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == "Hello! How can I help?"
    assert result["critique_decision"] == "respond"
    assert knowledge_base.queries == []


async def test_retrieves_documents_then_answers(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    knowledge_base.search_results = [
        Document(
            page_content="Cats swim in their water bowls.",
            metadata={"source": "cats.txt"},
        )
    ]
    fake_llm.specialist_replies = [
        tool_call("retrieve_documents", "cat swimming"),
        AIMessage("In their water bowls [Source: cats.txt]."),
    ]
    result = await ask("Where do cats swim?")

    assert knowledge_base.queries == ["cat swimming"]
    assert knowledge_base.opened == knowledge_base.closed == 1
    first_prompt, second_prompt = fake_llm.prompts("specialist")
    assert "You MUST use retrieve_documents" in first_prompt
    assert "Cats swim in their water bowls." in second_prompt
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content.startswith("Retrieved 1 relevant document(s)")
    assert [d.page_content for d in result["retrieved_docs"]] == [
        "Cats swim in their water bowls."
    ]
    assert result["messages"][-1].content == "In their water bowls [Source: cats.txt]."


async def test_stored_documents_are_truncated(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    knowledge_base.search_results = [Document(page_content="x" * 10_000)]
    fake_llm.specialist_replies = [tool_call()]
    result = await ask("Q")
    (doc,) = result["retrieved_docs"]
    assert doc.page_content.startswith("x" * graph_module.MAX_STORED_CHARS_PER_DOC)
    assert doc.page_content.endswith("[TRUNCATED FOR TOKEN EFFICIENCY]")


async def test_answer_text_is_extracted_from_content_blocks(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    fake_llm.specialist_replies = [
        AIMessage(content=[{"type": "text", "text": "Block answer"}])
    ]
    result = await ask("Q")
    assert result["agent_response"] == "Block answer"
    assert result["messages"][-1].content == "Block answer"


async def test_empty_answer_gets_a_fallback(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    fake_llm.specialist_replies = [AIMessage("")]
    result = await ask("Q")
    assert "couldn't generate a proper response" in result["messages"][-1].content


######################
# Conversations (regression: the first question was answered forever)
######################


async def test_follow_up_questions_answer_the_latest_question(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    await ask(
        [
            HumanMessage("What is photosynthesis?"),
            AIMessage("It is how plants make food."),
            HumanMessage("Who wrote Hamlet?"),
        ]
    )
    assert "Who wrote Hamlet?" in fake_llm.prompts("classify")[0]
    assert "User Question: Who wrote Hamlet?" in fake_llm.prompts("specialist")[0]
    assert "User Question: Who wrote Hamlet?" in fake_llm.prompts("critique")[0]


async def test_per_turn_state_is_reset_in_a_persistent_thread(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    thread_graph = graph_module.builder.compile(checkpointer=InMemorySaver())
    run_config = {
        **config(),
        "configurable": {**config()["configurable"], "thread_id": "t1"},
    }

    # Turn 1: retrieve a document, get critiqued once, then answer.
    knowledge_base.search_results = [Document(page_content="Turn one document")]
    fake_llm.specialist_replies = [
        tool_call(),
        AIMessage("draft"),
        AIMessage("answer 1"),
    ]
    fake_llm.critique_decisions = ["retry", "respond"]
    await thread_graph.ainvoke({"messages": [HumanMessage("Question one")]}, run_config)

    # Turn 2 in the same thread must not see turn 1's documents or critique.
    fake_llm.calls.clear()
    knowledge_base.search_results = []
    fake_llm.specialist_replies = [AIMessage("answer 2")]
    result = await thread_graph.ainvoke(
        {"messages": [HumanMessage("Question two")]}, run_config
    )

    (prompt,) = fake_llm.prompts("specialist")
    assert "User Question: Question two" in prompt
    assert "Turn one document" not in prompt
    assert "critique #" not in prompt
    assert result["retrieved_docs"] == []
    assert result["messages"][-1].content == "answer 2"


######################
# Loop guards (regression: runs crashed with GraphRecursionError)
######################


async def test_empty_search_results_do_not_loop(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    def search_whenever_told(prompt: str, tools: list[str]) -> AIMessage:
        if "You MUST use retrieve_documents" in prompt:
            return tool_call()
        return AIMessage("Answer from general knowledge")

    fake_llm.specialist_replies = [search_whenever_told, search_whenever_told]
    result = await ask("Explain entropy")

    assert len(knowledge_base.queries) == 1
    assert "found no relevant documents" in fake_llm.prompts("specialist")[1]
    assert result["messages"][-1].content == "Answer from general knowledge"


async def test_tool_rounds_are_capped(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    def always_search(prompt: str, tools: list[str]) -> AIMessage:
        return (
            tool_call(id=f"c{len(fake_llm.calls)}")
            if tools
            else AIMessage("Forced answer")
        )

    fake_llm.specialist_replies = [always_search] * 10
    result = await ask("Q")

    calls = fake_llm.specialist_calls()
    assert len(knowledge_base.queries) == graph_module.MAX_TOOL_ROUNDS
    assert calls[-1].tools == []  # tools are withdrawn after the cap
    assert "Tools are no longer available" in calls[-1].prompt
    assert result["messages"][-1].content == "Forced answer"


async def test_critique_feedback_is_passed_on_retry(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    fake_llm.specialist_replies = [AIMessage("weak draft"), AIMessage("better answer")]
    fake_llm.critique_decisions = ["retry", "respond"]
    result = await ask("Q")

    first, second = fake_llm.prompts("specialist")
    assert "Previous Critique Feedback (if any):\nNone" in first
    assert "critique #1" in second
    assert result["messages"][-1].content == "better answer"


async def test_critique_retries_are_capped(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    fake_llm.specialist_replies = [AIMessage(f"draft {i}") for i in range(10)]
    fake_llm.critique_decisions = ["retry"] * 10
    result = await ask("Q")

    assert len(fake_llm.prompts("specialist")) == graph_module.MAX_CRITIQUE_RETRIES + 1
    assert (
        result["messages"][-1].content == f"draft {graph_module.MAX_CRITIQUE_RETRIES}"
    )


######################
# Tools
######################


@dataclass
class FakeTavilyAPI:
    """Replaces the Tavily HTTP request; the real TavilySearch tool runs on top.

    Set ``response`` (the API's JSON) or ``error`` (raised by the request), then
    inspect ``requests`` (the search parameters) and ``api_keys``.
    """

    response: dict[str, Any] = field(default_factory=lambda: {"results": []})
    error: Exception | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)


@pytest.fixture
def tavily(monkeypatch: pytest.MonkeyPatch) -> FakeTavilyAPI:
    api = FakeTavilyAPI()

    async def raw_results_async(
        self: TavilySearchAPIWrapper, **params: Any
    ) -> dict[str, Any]:
        api.requests.append(params)
        api.api_keys.append(self.tavily_api_key.get_secret_value())
        if api.error:
            raise api.error
        return api.response

    def raw_results(self: TavilySearchAPIWrapper, **params: Any) -> dict[str, Any]:
        raise AssertionError("blocking HTTP call: use the async API")

    monkeypatch.setattr(TavilySearchAPIWrapper, "raw_results_async", raw_results_async)
    monkeypatch.setattr(TavilySearchAPIWrapper, "raw_results", raw_results)
    return api


async def run_tools(*calls: AIMessage, **configurable: Any) -> dict[str, Any]:
    state = State(messages=[HumanMessage("q"), *calls])
    return await graph_module.handle_tool_calls(state, config(**configurable))


async def test_web_search_results_become_documents(tavily: FakeTavilyAPI) -> None:
    tavily.response = {
        "query": "photosynthesis",
        "results": [
            {
                "title": "Photosynthesis",
                "url": "https://e.org/p",
                "content": "c" * 5000,
            },
            "not-a-dict",
        ],
    }
    out = await run_tools(
        tool_call("web_search", "photosynthesis"), web_search_api_key="tvly-1"
    )

    (doc,) = out["retrieved_docs"]
    assert doc.metadata == {
        "source": "https://e.org/p",
        "title": "Photosynthesis",
        "type": "web_search",
    }
    assert doc.page_content == "c" * 2000 + "... [WEB CONTENT TRUNCATED]"
    assert out["messages"][0].content.startswith("Found 1 relevant web result(s)")
    assert tavily.api_keys == ["tvly-1"]


async def test_web_search_sends_the_same_request_as_before(
    tavily: FakeTavilyAPI,
) -> None:
    # The parameters langchain-community's TavilySearchResults used to send.
    tavily.response = {"results": [{"url": "u", "title": "t", "content": "c"}]}
    await run_tools(tool_call("web_search", "photosynthesis"), web_search_api_key="k")
    (request,) = tavily.requests
    assert request["query"] == "photosynthesis"
    assert request["max_results"] == graph_module.WEB_SEARCH_MAX_RESULTS == 3
    assert request["search_depth"] == "advanced"
    assert request["include_answer"] is False
    assert request["include_raw_content"] is False
    assert request["include_images"] is False


async def test_web_search_falls_back_to_the_environment_key(
    tavily: FakeTavilyAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-env")
    tavily.response = {"results": [{"url": "u", "title": "t", "content": "c"}]}
    await run_tools(tool_call("web_search"), web_search_api_key="")
    assert tavily.api_keys == ["tvly-env"]


async def test_web_search_errors_are_reported(tavily: FakeTavilyAPI) -> None:
    # TavilySearch returns API and network failures as {"error": exception}.
    tavily.error = Exception("Error 401: Unauthorized")
    out = await run_tools(tool_call("web_search"), web_search_api_key="tvly-bad")
    assert out["messages"][0].content == (
        "Error during web search: Error 401: Unauthorized"
    )
    assert out["retrieved_docs"] == []


async def test_missing_web_search_key_is_reported(tavily: FakeTavilyAPI) -> None:
    out = await run_tools(tool_call("web_search"))
    content = out["messages"][0].content
    assert content.startswith("Error during web search: ")
    assert "tavily_api_key" in content
    assert tavily.requests == []


async def test_empty_web_search_reports_no_results(tavily: FakeTavilyAPI) -> None:
    # TavilySearch raises a ToolException for an empty result list.
    out = await run_tools(tool_call("web_search"), web_search_api_key="k")
    assert out["messages"][0].content == "No results found from web search."
    assert out["retrieved_docs"] == []


async def test_other_tool_errors_are_not_reported_as_no_results(
    tavily: FakeTavilyAPI,
) -> None:
    tavily.error = ToolException("Rate limit exceeded")
    out = await run_tools(tool_call("web_search"), web_search_api_key="k")
    assert out["messages"][0].content == "Error during web search: Rate limit exceeded"


async def test_web_search_can_be_disabled(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase, tavily: FakeTavilyAPI
) -> None:
    await ask("Q", enable_web_search=False)
    assert fake_llm.specialist_calls()[0].tools == ["retrieve_documents"]

    out = await run_tools(tool_call("web_search"), enable_web_search=False)
    assert out["messages"][0].content == "Web search is disabled in the configuration."
    assert tavily.requests == []


async def test_both_tools_are_offered_by_default(
    fake_llm: FakeLLM, knowledge_base: FakeKnowledgeBase
) -> None:
    await ask("Q")
    assert fake_llm.specialist_calls()[0].tools == ["retrieve_documents", "web_search"]


async def test_retrieval_errors_are_reported(knowledge_base: FakeKnowledgeBase) -> None:
    knowledge_base.error = ConnectionError("Elasticsearch is down")
    out = await run_tools(tool_call())
    assert out["messages"][0].content == "Error during retrieval: Elasticsearch is down"
    assert knowledge_base.closed == 1


async def test_empty_retrieval_suggests_web_search(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    out = await run_tools(tool_call())
    assert "No documents found" in out["messages"][0].content
    assert "web_search" in out["messages"][0].content
    out = await run_tools(tool_call(), enable_web_search=False)
    assert "web_search" not in out["messages"][0].content


async def test_unknown_tools_are_reported() -> None:
    out = await run_tools(tool_call("calculator"))
    assert out["messages"][0].content == "Unknown tool: calculator"


async def test_only_the_most_recent_documents_are_kept(
    knowledge_base: FakeKnowledgeBase,
) -> None:
    knowledge_base.search_results = [
        Document(page_content=f"doc {i}") for i in range(4)
    ]
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "retrieve_documents", "args": {"query": "a"}, "id": "1"},
            {"name": "retrieve_documents", "args": {"query": "b"}, "id": "2"},
        ],
    )
    out = await run_tools(message)
    assert len(out["retrieved_docs"]) == graph_module.MAX_STORED_DOCS
    assert len(out["messages"]) == 2


async def test_no_tool_calls_is_a_no_op() -> None:
    out = await graph_module.handle_tool_calls(
        State(messages=[HumanMessage("q")]), config()
    )
    assert out == {"messages": [], "retrieved_docs": []}
