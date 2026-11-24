"""Test doubles for the UI tests: realistic stream updates and a fake backend client."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from touchstone_ui.client import BackendError


def ai(
    content: Any = "", tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A message serialized the way the LangGraph server streams it."""
    return {
        "type": "ai",
        "content": content,
        "tool_calls": tool_calls or [],
        "id": "msg",
        "additional_kwargs": {},
        "response_metadata": {},
    }


def tool_message(content: str, name: str = "retrieve_documents") -> dict[str, Any]:
    return {"type": "tool", "content": content, "name": name, "tool_call_id": "c1"}


def document(content: str, **metadata: Any) -> dict[str, Any]:
    return {
        "type": "Document",
        "page_content": content,
        "metadata": metadata,
        "id": None,
    }


# One full question, as `runs.stream(..., stream_mode="updates")` yields it.
ANSWER = "Plants convert light into chemical energy [Source: biology.txt]."
STREAM: list[tuple[str, Any]] = [
    ("metadata", {"run_id": "run-1", "attempt": 1}),
    (
        "updates",
        {
            "classify_query": {
                "classification": "science",
                "retrieved_docs": [],
                "agent_response": None,
                "critique_decision": None,
                "critique_feedback": None,
            }
        },
    ),
    (
        "updates",
        {
            "science_agent": {
                "messages": [
                    ai(
                        tool_calls=[
                            {
                                "name": "retrieve_documents",
                                "args": {"query": "photosynthesis"},
                                "id": "c1",
                                "type": "tool_call",
                            }
                        ]
                    )
                ],
                "agent_response": "",
            }
        },
    ),
    (
        "updates",
        {
            "tools": {
                "messages": [
                    tool_message(
                        "Retrieved 1 relevant document(s):\n\nDocument 1:\nContent: ..."
                    )
                ],
                "retrieved_docs": [
                    document(
                        "Photosynthesis converts light energy into chemical energy.",
                        source="biology.txt",
                        user_id="alice",
                    )
                ],
            }
        },
    ),
    (
        "updates",
        {"science_agent": {"messages": [ai(ANSWER)], "agent_response": ANSWER}},
    ),
    (
        "updates",
        {
            "critique_agent": {
                "critique_decision": "respond",
                "critique_feedback": "Accurate and well cited.",
            }
        },
    ),
    ("updates", {"respond": {"messages": [ai(ANSWER)]}}),
]


class FakeClient:
    """Stands in for TouchstoneClient in Streamlit app tests.

    Configure the class attributes before running the app; inspect ``last`` after.
    """

    healthy = True
    stream: Sequence[tuple[str, Any]] = STREAM
    ask_error: str | None = None
    index_error: str | None = None
    last: "FakeClient | None" = None

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.threads = 0
        self.asked: list[tuple[str, str, dict[str, Any]]] = []
        self.indexed: list[tuple[list[str], dict[str, Any]]] = []
        FakeClient.last = self

    @classmethod
    def reset(cls) -> None:
        cls.healthy = True
        cls.stream = STREAM
        cls.ask_error = None
        cls.index_error = None
        cls.last = None

    def is_healthy(self) -> bool:
        return self.healthy

    def create_thread(self) -> str:
        self.threads += 1
        return f"thread-{self.threads}"

    def ask(
        self, thread_id: str, question: str, configurable: Mapping[str, Any]
    ) -> Iterator[tuple[str, Any]]:
        self.asked.append((thread_id, question, dict(configurable)))
        if self.ask_error:
            raise BackendError(self.ask_error)
        yield from self.stream

    def index(self, docs: Sequence[str], configurable: Mapping[str, Any]) -> None:
        if self.index_error:
            raise BackendError(self.index_error)
        self.indexed.append((list(docs), dict(configurable)))
