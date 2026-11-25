"""Multi-agent retrieval graph.

Flow for every user turn::

    classify_query -> <subject>_agent <-> tools
                           |
                           v
                     critique_agent -> respond
                           |
                           +-> (retry) back to the same <subject>_agent

* ``classify_query`` routes the question to one of the specialist agents.
* A specialist agent answers, optionally calling the ``retrieve_documents`` /
  ``web_search`` tools first (executed by the ``tools`` node).
* ``critique_agent`` reviews the draft and either accepts it (``respond``) or sends
  it back to the same specialist with feedback.

To add a specialist, add a prompt in :mod:`retrieval_graph.prompts`, register it in
``SPECIALISTS`` below and extend the ``subject`` literal in ``QueryClassification``.
"""

from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from retrieval_graph import prompts, retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs_safe, get_message_text, load_chat_model

NodeUpdate = dict[str, Any]
"""A partial state update returned by a graph node."""


class QueryClassification(BaseModel):
    """Classification of user query into subject areas."""

    subject: Literal["science", "history", "literature", "general"]


class CritiqueDecision(BaseModel):
    """Decision from critique agent whether to retry or respond."""

    decision: Literal["respond", "retry", "improve_query"]
    reasoning: str


# Specialist registry: graph node name -> (subject, prompt template, description).
# The insertion order is the order nodes are added to the graph.
SPECIALISTS: dict[str, tuple[str, str, str]] = {
    "science_agent": (
        "science",
        prompts.SCIENCE_AGENT_PROMPT,
        "Science subject specialist agent.",
    ),
    "history_agent": (
        "history",
        prompts.HISTORY_AGENT_PROMPT,
        "History subject specialist agent.",
    ),
    "literature_agent": (
        "literature",
        prompts.LITERATURE_AGENT_PROMPT,
        "Literature subject specialist agent.",
    ),
    "general_agent": (
        "general",
        prompts.GENERAL_AGENT_PROMPT,
        "General knowledge specialist agent.",
    ),
}
DEFAULT_SPECIALIST = "general_agent"
_SUBJECT_TO_SPECIALIST = {
    subject: node for node, (subject, _, _) in SPECIALISTS.items()
}

# Loop guards. Without them an empty knowledge base, a failing search, or a critic
# that keeps asking for retries makes the run loop until LangGraph aborts it with a
# GraphRecursionError, and the user gets no answer at all.
MAX_TOOL_ROUNDS = 3
"""Tool rounds a specialist may request per user turn before it must answer."""
MAX_CRITIQUE_RETRIES = 2
"""Times the critic may send a draft back per user turn before it is accepted."""


######################
# Turn helpers
######################


def _current_turn(messages: Sequence[AnyMessage]) -> Sequence[AnyMessage]:
    """Return the messages produced after the latest human message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            return messages[i + 1 :]
    return messages


def _tool_rounds_this_turn(state: State) -> int:
    """Count the specialist messages that requested tools during this turn."""
    return sum(
        1
        for msg in _current_turn(state.messages)
        if isinstance(msg, AIMessage) and msg.tool_calls
    )


def _drafts_this_turn(state: State) -> int:
    """Count the specialist answers (messages without tool calls) this turn."""
    return sum(
        1
        for msg in _current_turn(state.messages)
        if isinstance(msg, AIMessage) and not msg.tool_calls
    )


def get_latest_user_question(state: State) -> str:
    """Return the text of the latest human message (the question of this turn).

    Falls back to the first message if the history contains no human message.
    """
    for msg in reversed(state.messages):
        if msg.type == "human":
            return get_message_text(msg)
    return get_message_text(state.messages[0]) if state.messages else ""


async def classify_query(state: State, config: RunnableConfig) -> NodeUpdate:
    """Classify the user's query into subject areas.

    This is the first node of every turn, so it also clears the per-turn fields:
    in a persistent thread (LangGraph Studio, the Touchstone UI) the previous
    question's documents and critique would otherwise leak into this one.
    """
    configuration = Configuration.from_runnable_config(config)
    user_question = get_latest_user_question(state)

    prompt = ChatPromptTemplate.from_template(prompts.CLASSIFICATION_SYSTEM_PROMPT)
    model = load_chat_model(configuration.query_model).with_structured_output(
        QueryClassification
    )

    message_value = await prompt.ainvoke({"question": user_question}, config)
    classification = cast(
        QueryClassification, await model.ainvoke(message_value, config)
    )
    return {
        "classification": classification.subject,
        "retrieved_docs": [],
        "agent_response": None,
        "critique_decision": None,
        "critique_feedback": None,
    }


######################
# Tools
######################

# These tools are bound to the specialist models so they can *request* a search.
# The searches themselves are executed by `handle_tool_calls` (the "tools" node),
# which also stores the results in `state.retrieved_docs`.


@tool
async def retrieve_documents(query: str) -> str:
    """Retrieve documents from the knowledge base using semantic search.

    Args:
        query: The search query to use for retrieval

    Returns:
        String representation of retrieved documents
    """
    return f"Searching knowledge base for: {query}"


@tool
async def web_search(query: str) -> str:
    """Search the web for information when local knowledge is insufficient.

    Args:
        query: The search query to use for web search

    Returns:
        String representation of web search results
    """
    return f"Searching web for: {query}"


tools = [retrieve_documents, web_search]

MAX_STORED_CHARS_PER_DOC = 3000
"""Characters of each retrieved document kept in state (~750 tokens)."""
MAX_WEB_RESULT_CHARS = 2000
"""Characters of each web search result kept in state."""
MAX_STORED_DOCS = 5
"""Only the most recent documents are kept in state."""
WEB_SEARCH_MAX_RESULTS = 3


async def _run_retrieval(
    query: str, config: RunnableConfig
) -> tuple[str, list[Document]]:
    """Search the knowledge base; return the tool message text and new documents."""
    try:
        async with retrieval.make_retriever(config) as retriever:
            docs = await retriever.ainvoke(query, config)
    except Exception as e:
        return f"Error during retrieval: {e!s}", []

    if not docs:
        content = "No documents found in the knowledge base for this query."
        # Suggest the web_search tool when local retrieval found nothing.
        if Configuration.from_runnable_config(config).enable_web_search:
            content += "\nYou may want to try using web_search for this query."
        return content, []

    # Truncate document content before storing to prevent token overflow.
    stored = []
    for doc in docs:
        content = doc.page_content
        if len(content) > MAX_STORED_CHARS_PER_DOC:
            content = (
                content[:MAX_STORED_CHARS_PER_DOC]
                + "... [TRUNCATED FOR TOKEN EFFICIENCY]"
            )
        stored.append(Document(page_content=content, metadata=doc.metadata))

    # Keep the tool response concise to prevent token overflow.
    summary = "\n\n".join(
        f"Document {i + 1}:\nContent: {doc.page_content[:300]}...\n"
        f"Source: {doc.metadata.get('source', 'Unknown')}"
        for i, doc in enumerate(docs)
    )
    return f"Retrieved {len(docs)} relevant document(s):\n\n{summary}", stored


async def _run_web_search(
    query: str, config: RunnableConfig
) -> tuple[str, list[Document]]:
    """Search the web with Tavily; return the tool message text and new documents."""
    configuration = Configuration.from_runnable_config(config)
    if not configuration.enable_web_search:
        return "Web search is disabled in the configuration.", []

    # Without a configured key, TavilySearch falls back to TAVILY_API_KEY.
    api_key: dict[str, Any] = (
        {"tavily_api_key": configuration.web_search_api_key}
        if configuration.web_search_api_key
        else {}
    )
    try:
        # Same request as the former langchain-community TavilySearchResults.
        search_tool = TavilySearch(
            max_results=WEB_SEARCH_MAX_RESULTS,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            **api_key,
        )
        # Async call: a sync HTTP request would block the event loop (and is
        # rejected by `langgraph dev`).
        response = await search_tool.ainvoke({"query": query})
        if isinstance(response, str):
            # TavilySearch raises a ToolException when nothing matched, and the
            # tool (handle_tool_error=True) returns its message instead.
            if response.startswith("No search results found"):
                return "No results found from web search.", []
            return f"Error during web search: {response}", []
        if "error" in response:  # API and network failures: {"error": exception}
            return f"Error during web search: {response['error']!s}", []

        web_docs = []
        for result in response.get("results", []):
            if isinstance(result, dict):
                content = result.get("content", "")
                if len(content) > MAX_WEB_RESULT_CHARS:
                    content = (
                        content[:MAX_WEB_RESULT_CHARS] + "... [WEB CONTENT TRUNCATED]"
                    )
                web_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": result.get("url", ""),
                            "title": result.get("title", ""),
                            "type": "web_search",
                        },
                    )
                )
    except Exception as e:
        return f"Error during web search: {e!s}", []

    if not web_docs:
        return "No results found from web search.", []

    summary = "\n\n".join(
        f"Result {i + 1}:\nTitle: {doc.metadata.get('title', 'No title')}\n"
        f"Content: {doc.page_content[:200]}...\n"
        f"Source: {doc.metadata.get('source', 'Unknown')}"
        for i, doc in enumerate(web_docs)
    )
    return f"Found {len(web_docs)} relevant web result(s):\n\n{summary}", web_docs


async def handle_tool_calls(state: State, config: RunnableConfig) -> NodeUpdate:
    """Execute the tool calls requested by the last specialist message."""
    messages_to_add: list[ToolMessage] = []
    retrieved_docs = list(state.retrieved_docs)

    last_message = state.messages[-1] if state.messages else None
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": messages_to_add, "retrieved_docs": retrieved_docs}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        query = tool_call["args"].get("query", "")

        if tool_name == "retrieve_documents":
            content, new_docs = await _run_retrieval(query, config)
        elif tool_name == "web_search":
            content, new_docs = await _run_web_search(query, config)
        else:
            content, new_docs = f"Unknown tool: {tool_name}", []

        retrieved_docs.extend(new_docs)
        messages_to_add.append(
            ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_name)
        )

    # Limit total documents to prevent token overflow in subsequent calls.
    retrieved_docs = retrieved_docs[-MAX_STORED_DOCS:]
    return {"messages": messages_to_add, "retrieved_docs": retrieved_docs}


######################
# Agents
######################


async def _run_specialist(
    state: State, config: RunnableConfig, prompt_template: str
) -> NodeUpdate:
    """Answer the user's question with a subject-specific prompt and tools."""
    configuration = Configuration.from_runnable_config(config)
    user_question = get_latest_user_question(state)
    tool_rounds = _tool_rounds_this_turn(state)
    can_use_tools = tool_rounds < MAX_TOOL_ROUNDS

    if state.retrieved_docs:
        retrieved_docs = format_docs_safe(state.retrieved_docs)
    elif tool_rounds == 0:
        retrieved_docs = (
            f"No documents currently available for the question: '{user_question}'. "
            "You MUST use retrieve_documents tool immediately with a search query "
            "based on this question."
        )
    else:
        # A search already ran this turn and found nothing. Repeating "you MUST
        # retrieve" here is what used to loop forever.
        retrieved_docs = (
            "The knowledge base search found no relevant documents for the "
            f"question: '{user_question}'. Do not search the knowledge base again; "
            "answer from your own knowledge and say so when you are unsure."
        )
    if not can_use_tools:
        retrieved_docs += (
            "\n\nTools are no longer available for this question: write your final "
            "answer now."
        )

    chat_model = load_chat_model(configuration.response_model)
    model: Runnable[Any, BaseMessage] = chat_model
    if can_use_tools:
        model = chat_model.bind_tools(
            tools if configuration.enable_web_search else [retrieve_documents]
        )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    message_value = await prompt.ainvoke(
        {
            "retrieved_docs": retrieved_docs,
            "question": user_question,
            "critique_feedback": state.critique_feedback or "None",
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    # Store the answer text for the critique agent. `get_message_text` also handles
    # providers (e.g. Anthropic) that return a list of content blocks.
    return {"messages": [response], "agent_response": get_message_text(response)}


def _make_specialist(
    prompt_template: str, description: str
) -> Callable[..., Awaitable[NodeUpdate]]:
    """Create a specialist agent node bound to one subject prompt."""

    async def specialist(state: State, config: RunnableConfig) -> NodeUpdate:
        return await _run_specialist(state, config, prompt_template)

    specialist.__doc__ = description
    return specialist


_SPECIALIST_NODES = {
    name: _make_specialist(prompt, description)
    for name, (_, prompt, description) in SPECIALISTS.items()
}
# Module-level names for the built-in specialists (kept for importers).
science_agent = _SPECIALIST_NODES["science_agent"]
history_agent = _SPECIALIST_NODES["history_agent"]
literature_agent = _SPECIALIST_NODES["literature_agent"]
general_agent = _SPECIALIST_NODES["general_agent"]


async def critique_agent(state: State, config: RunnableConfig) -> NodeUpdate:
    """Critique and evaluate the specialist agent's response."""
    configuration = Configuration.from_runnable_config(config)
    user_question = get_latest_user_question(state)
    retrieved_docs = (
        format_docs_safe(state.retrieved_docs)
        if state.retrieved_docs
        else "No documents available."
    )

    prompt = ChatPromptTemplate.from_template(prompts.CRITIQUE_SYSTEM_PROMPT)
    model = load_chat_model(configuration.response_model).with_structured_output(
        CritiqueDecision
    )
    message_value = await prompt.ainvoke(
        {
            "agent_response": state.agent_response or "",
            "user_question": user_question,
            "retrieved_docs": retrieved_docs,
        },
        config,
    )
    critique = cast(CritiqueDecision, await model.ainvoke(message_value, config))
    return {
        "critique_decision": critique.decision,
        "critique_feedback": critique.reasoning,
    }


async def respond(state: State, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Generate final response based on specialist agent output."""
    agent_response = (
        state.agent_response
        or "I apologize, but I couldn't generate a proper response."
    )
    return {"messages": [AIMessage(content=agent_response)]}


######################
# Routing
######################


def route_to_specialist(state: State) -> str:
    """Route to the appropriate specialist agent based on classification."""
    return _SUBJECT_TO_SPECIALIST.get(state.classification or "", DEFAULT_SPECIALIST)


def after_tools_routing(state: State) -> str:
    """Route back to the specialist that requested the tools."""
    return route_to_specialist(state)


def should_continue_to_tools(state: State) -> str:
    """Determine if we should continue to tools or move to critique."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "critique_agent"


def critique_router(state: State) -> str:
    """Route based on critique decision.

    ``retry`` / ``improve_query`` send the draft back to the same specialist (with
    the critique feedback) at most ``MAX_CRITIQUE_RETRIES`` times per turn; after
    that the latest draft is returned to the user.
    """
    wants_retry = state.critique_decision in ("retry", "improve_query")
    if wants_retry and _drafts_this_turn(state) <= MAX_CRITIQUE_RETRIES:
        return route_to_specialist(state)
    return "respond"


######################
# Graph
######################

# `config_schema` is deprecated in LangGraph 1.x in favour of `context_schema`, but
# switching changes how callers pass configuration (`configurable` -> `context`),
# so it is kept until a deliberate migration (see docs/development.md).
builder = StateGraph(  # type: ignore[call-arg]
    State, input_schema=InputState, config_schema=Configuration
)

builder.add_node("classify_query", classify_query)
for _name, _node in _SPECIALIST_NODES.items():
    builder.add_node(_name, _node)
builder.add_node("tools", handle_tool_calls)
builder.add_node("critique_agent", critique_agent)
builder.add_node("respond", respond)

_to_specialist: dict[Hashable, str] = {name: name for name in SPECIALISTS}
_to_tools_or_critique: dict[Hashable, str] = {
    "tools": "tools",
    "critique_agent": "critique_agent",
}
_from_critique: dict[Hashable, str] = {"respond": "respond", **_to_specialist}

builder.add_edge(START, "classify_query")
builder.add_conditional_edges("classify_query", route_to_specialist, _to_specialist)
for _name in SPECIALISTS:
    builder.add_conditional_edges(
        _name, should_continue_to_tools, _to_tools_or_critique
    )
builder.add_conditional_edges("tools", after_tools_routing, _to_specialist)
builder.add_conditional_edges("critique_agent", critique_router, _from_critique)
builder.add_edge("respond", END)

graph = builder.compile()
graph.name = "RetrievalGraph"
