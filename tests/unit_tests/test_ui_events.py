from touchstone_ui.events import (
    RunTracker,
    Source,
    message_text,
    subject_label,
    to_source,
)

from .ui_fakes import ANSWER, STREAM, ai, document, tool_message


def run(stream):  # type: ignore[no-untyped-def]
    tracker = RunTracker()
    for event, data in stream:
        tracker.consume(event, data)
    return tracker


def test_full_run_is_summarised_step_by_step() -> None:
    tracker = run(STREAM)

    assert [(s.kind, s.title) for s in tracker.steps] == [
        ("route", "Routed to the Science agent"),
        ("search", "Searching your knowledge base"),
        ("results", "Retrieved 1 relevant document(s)"),
        ("draft", "Science agent wrote a draft"),
        ("critique", "Critic approved the answer"),
        ("answer", "Answer ready"),
    ]
    assert tracker.steps[1].detail == "“photosynthesis”"
    assert tracker.steps[4].detail == "Accurate and well cited."
    assert tracker.subject == "science"
    assert tracker.answer == ANSWER
    assert tracker.error is None
    assert tracker.sources == [
        Source(
            title="biology.txt",
            location="biology.txt",
            snippet="Photosynthesis converts light energy into chemical energy.",
            kind="knowledge_base",
        )
    ]
    assert tracker.summary == "🧪 Science agent · 1 search · reviewed by the critic"


def test_consume_returns_only_new_steps_and_ignores_metadata() -> None:
    tracker = RunTracker()
    assert tracker.consume("metadata", {"run_id": "r"}) == []
    assert tracker.consume("updates", "not-a-dict") == []
    (step,) = tracker.consume(
        "updates", {"classify_query": {"classification": "history"}}
    )
    assert step.title == "Routed to the History agent"
    assert step.icon == "🏛️"


def test_errors_are_captured() -> None:
    tracker = RunTracker()
    (step,) = tracker.consume(
        "error", {"error": "ValueError", "message": "user_id missing"}
    )
    assert step.kind == "error"
    assert tracker.error == "ValueError: user_id missing"
    assert run([("error", "boom")]).error == "boom"
    assert run([("error", {})]).error == "The run failed."


def test_retries_web_searches_and_failures() -> None:
    tracker = run(
        [
            ("updates", {"classify_query": {"classification": "general"}}),
            (
                "updates",
                {
                    "general_agent": {
                        "messages": [
                            ai(
                                tool_calls=[
                                    {
                                        "name": "web_search",
                                        "args": {"query": "news"},
                                        "id": "1",
                                    },
                                    {"name": "calculator", "args": {}, "id": "2"},
                                ]
                            )
                        ]
                    }
                },
            ),
            (
                "updates",
                {
                    "tools": {
                        "messages": [
                            tool_message("Error during web search: 401", "web_search"),
                            tool_message("Unknown tool: calculator", "calculator"),
                        ],
                        "retrieved_docs": [],
                    }
                },
            ),
            ("updates", {"general_agent": {"messages": [ai("draft")]}}),
            (
                "updates",
                {
                    "critique_agent": {
                        "critique_decision": "retry",
                        "critique_feedback": "Too vague",
                    }
                },
            ),
            (
                "updates",
                {
                    "general_agent": {
                        "messages": [ai([{"type": "text", "text": "Better"}])]
                    }
                },
            ),
            ("updates", {"critique_agent": {"critique_decision": "respond"}}),
            (
                "updates",
                {"respond": {"messages": [ai([{"type": "text", "text": "Better"}])]}},
            ),
        ]
    )
    titles = [(s.icon, s.title) for s in tracker.steps]
    assert ("🌐", "Searching the web") in titles
    assert ("🛠️", "Using calculator") in titles
    assert ("⚠️", "Error during web search: 401") in titles
    assert ("📭", "Unknown tool: calculator") in titles
    assert ("🔁", "Critic asked for a better answer") in titles
    assert tracker.answer == "Better"
    assert (
        tracker.summary
        == "💡 General agent · 2 searches · reviewed by the critic (2 rounds)"
    )


def test_helpers() -> None:
    assert subject_label("science") == ("Science", "🧪")
    assert subject_label("astrology") == ("Astrology", "🧭")
    assert subject_label(None) == ("Unknown", "🧭")
    assert message_text(ai("plain")) == "plain"
    assert (
        message_text(ai(["a", {"type": "text", "text": "b"}, {"type": "image"}]))
        == "ab"
    )
    assert message_text("not a message") == ""
    web = to_source(
        document("x" * 500, source="https://e.org", title="E", type="web_search")
    )
    assert (web.title, web.location, web.kind) == ("E", "https://e.org", "web_search")
    assert len(web.snippet) <= 280 and web.snippet.endswith("…")
    assert to_source({}).title == "Knowledge base document"
