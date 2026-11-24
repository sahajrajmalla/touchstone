"""State definitions for the indexing and retrieval graphs.

Classes:
    IndexState: Input/state of the indexing graph (documents to index).
    InputState: The public input of the retrieval graph (the conversation).
    State: The full internal state of the retrieval graph.

Functions:
    reduce_docs: Reducer that turns strings / dicts / Documents into Documents.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

############################  Doc Indexing State  #############################


def reduce_docs(
    existing: Sequence[Document] | None,
    new: Sequence[Document]
    | Sequence[dict[str, Any]]
    | Sequence[str]
    | str
    | Literal["delete"],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type.

    * ``"delete"`` clears the documents.
    * A string becomes a single Document.
    * A list/tuple *replaces* the documents; each item may be a Document, a
      ``Document``-shaped dict (``{"page_content": ..., "metadata": {...}}``) or a
      string.
    * Anything else leaves the existing documents unchanged.

    Strings get a random ``metadata["id"]``.

    Args:
        existing: The existing docs in the state, if any.
        new: The new input to process.
    """
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list | tuple):
        coerced: list[Document] = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            else:
                coerced.append(item)
        return coerced
    return existing or []


@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing.

    ``docs`` accepts plain text, ``http(s)://`` URLs (web pages and PDFs, which are
    downloaded and extracted) or Document-shaped dicts.
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


#############################  Agent State  ###################################


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    A restricted version of :class:`State`: callers only provide the messages.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """The conversation.

    New messages are appended; a message with the same ID as an existing one
    replaces it (see ``langgraph.graph.add_messages``). A typical turn looks like:

    1. HumanMessage - the user's question.
    2. AIMessage with ``tool_calls`` - a specialist requesting a search.
    3. ToolMessage(s) - the search results (steps 2-3 may repeat).
    4. AIMessage - the specialist's draft answer (repeats if the critic retries).
    5. AIMessage - the final answer, added by the ``respond`` node.
    """


@dataclass(kw_only=True)
class State(InputState):
    """The state of your graph / agent.

    The fields below are reset at the start of every user turn by
    ``classify_query``.
    """

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    classification: str | None = field(default=None)
    """Stores the classified subject area (science/history/literature/general)."""

    agent_response: str | None = field(default=None)
    """Stores the response from the specialist agent."""

    critique_decision: str | None = field(default=None)
    """Stores the decision from critique agent (respond/retry/improve_query)."""

    critique_feedback: str | None = field(default=None)
    """Stores the feedback/reasoning from critique agent."""
