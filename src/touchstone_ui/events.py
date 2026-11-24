"""Turn the retrieval graph's stream updates into UI-friendly steps.

``RunTracker.consume`` is fed every ``(event, data)`` pair streamed by the LangGraph
server (``stream_mode="updates"``) and records what each agent did: routing,
searches, results, critique verdicts and the final answer. No Streamlit here, so
the mapping is easy to test.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

StepKind = Literal["route", "search", "results", "draft", "critique", "answer", "error"]

SUBJECTS: dict[str, tuple[str, str]] = {
    "science": ("Science", "🧪"),
    "history": ("History", "🏛️"),
    "literature": ("Literature", "📖"),
    "general": ("General", "💡"),
}
"""Subject -> (label, icon) for the specialist agents."""

_TOOLS: dict[str, tuple[str, str]] = {
    "retrieve_documents": ("Searching your knowledge base", "🔎"),
    "web_search": ("Searching the web", "🌐"),
}

SNIPPET_CHARS = 280


@dataclass(frozen=True)
class Step:
    """One thing the agents did while answering."""

    kind: StepKind
    title: str
    icon: str
    detail: str = ""


@dataclass(frozen=True)
class Source:
    """A document the answer could draw on."""

    title: str
    location: str
    snippet: str
    kind: str


def subject_label(subject: str | None) -> tuple[str, str]:
    """Return ``(label, icon)`` for a subject, with a neutral fallback."""
    if subject in SUBJECTS:
        return SUBJECTS[subject]
    return ((subject or "Unknown").title(), "🧭")


def message_text(message: Any) -> str:
    """Extract the text of a serialized message (string or content blocks)."""
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = [
            part if isinstance(part, str) else str(part.get("text") or "")
            for part in content
            if isinstance(part, str | Mapping)
        ]
        return "".join(parts).strip()
    return ""


def _shorten(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def to_source(doc: Any) -> Source:
    """Convert a serialized Document into a :class:`Source`."""
    doc = doc if isinstance(doc, Mapping) else {}
    metadata = doc.get("metadata") or {}
    location = str(metadata.get("source") or "")
    return Source(
        title=str(metadata.get("title") or location or "Knowledge base document"),
        location=location,
        snippet=_shorten(str(doc.get("page_content") or "")),
        kind=str(metadata.get("type") or "knowledge_base"),
    )


def _error_message(data: Any) -> str:
    if isinstance(data, Mapping):
        message = data.get("message") or data.get("error")
        if data.get("error") and data.get("message"):
            message = f"{data['error']}: {data['message']}"
        return str(message or "The run failed.")
    return str(data or "The run failed.")


@dataclass
class RunTracker:
    """Accumulates the steps, sources and answer of one question."""

    subject: str | None = None
    steps: list[Step] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None
    searches: int = 0
    critiques: int = 0

    def consume(self, event: str, data: Any) -> list[Step]:
        """Process one streamed ``(event, data)`` pair and return the new steps."""
        if event == "error":
            self.error = _error_message(data)
            return self._add([Step("error", "Something went wrong", "⚠️", self.error)])
        if event != "updates" or not isinstance(data, Mapping):
            return []  # metadata, heartbeats, ...
        steps: list[Step] = []
        for node, update in data.items():
            if isinstance(update, Mapping):
                steps.extend(self._handle(str(node), update))
        return self._add(steps)

    @property
    def summary(self) -> str:
        """A one-line description of how the answer was produced."""
        label, icon = subject_label(self.subject)
        parts = [f"{icon} {label} agent"]
        if self.searches:
            parts.append(f"{self.searches} search{'es' if self.searches > 1 else ''}")
        if self.critiques:
            rounds = "" if self.critiques == 1 else f" ({self.critiques} rounds)"
            parts.append(f"reviewed by the critic{rounds}")
        return " · ".join(parts)

    def _add(self, steps: list[Step]) -> list[Step]:
        self.steps.extend(steps)
        return steps

    def _handle(self, node: str, update: Mapping[str, Any]) -> list[Step]:
        messages = update.get("messages") or []
        if node == "classify_query":
            self.subject = update.get("classification")
            label, icon = subject_label(self.subject)
            return [Step("route", f"Routed to the {label} agent", icon)]

        if node == "tools":
            self.sources = [
                to_source(doc) for doc in update.get("retrieved_docs") or []
            ]
            return [self._tool_result(message) for message in messages]

        if node == "critique_agent":
            self.critiques += 1
            feedback = str(update.get("critique_feedback") or "")
            if update.get("critique_decision") == "respond":
                return [Step("critique", "Critic approved the answer", "✅", feedback)]
            return [
                Step("critique", "Critic asked for a better answer", "🔁", feedback)
            ]

        if node == "respond":
            self.answer = message_text(messages[-1]) if messages else ""
            return [Step("answer", "Answer ready", "🎓")]

        if node.endswith("_agent") and messages:  # a specialist
            tool_calls = messages[-1].get("tool_calls") or []
            if tool_calls:
                return [self._tool_request(call) for call in tool_calls]
            label, _ = subject_label(node.removesuffix("_agent"))
            return [Step("draft", f"{label} agent wrote a draft", "✍️")]
        return []

    def _tool_request(self, call: Mapping[str, Any]) -> Step:
        self.searches += 1
        name = str(call.get("name") or "tool")
        title, icon = _TOOLS.get(name, (f"Using {name}", "🛠️"))
        query = str((call.get("args") or {}).get("query") or "")
        return Step("search", title, icon, f"“{query}”" if query else "")

    @staticmethod
    def _tool_result(message: Any) -> Step:
        text = message_text(message)
        first_line = text.splitlines()[0] if text else "No output"
        if text.startswith(("Retrieved", "Found")):
            icon = "📚"
        elif text.startswith("Error"):
            icon = "⚠️"
        else:
            icon = "📭"
        return Step("results", first_line.rstrip(":"), icon)
