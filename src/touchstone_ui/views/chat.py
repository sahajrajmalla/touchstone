"""The "Ask" page: chat with the multi-agent tutor and watch it work."""

from typing import Any

import streamlit as st

from touchstone_ui.client import BackendError
from touchstone_ui.events import RunTracker, Source, Step, subject_label
from touchstone_ui.views.common import (
    ERROR_HINT,
    START_BACKEND_HINT,
    Session,
    reset_conversation,
)

EXAMPLES: list[tuple[str, str]] = [
    ("🧪", "Explain the process of photosynthesis and its importance in ecosystems"),
    ("🏛️", "What were the main causes and consequences of the American Civil War?"),
    ("📖", "Analyze the theme of revenge in Shakespeare's Hamlet"),
    ("💡", "How can I improve my study habits for better academic performance?"),
]


def _md(text: str) -> str:
    """Escape characters that would turn untrusted text into markdown links."""
    return text.replace("[", r"\[").replace("]", r"\]")


def _write_step(container: Any, step: Step) -> None:
    container.markdown(f"{step.icon} **{step.title}**")
    if step.detail:
        container.caption(_md(step.detail))


def _render_sources(sources: list[Source]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon="📚"):
        for source in sources:
            icon = "🌐" if source.kind == "web_search" else "📄"
            title = _md(source.title)
            if source.location.startswith(("http://", "https://")):
                title = f"[{title}]({source.location})"
            st.markdown(f"{icon} **{title}**")
            if source.snippet:
                st.caption(_md(source.snippet))


def _show_error(message: str) -> None:
    st.error(message, icon="⚠️")
    st.caption(ERROR_HINT)


def _render_assistant(message: dict[str, Any]) -> None:
    if message.get("error"):
        _show_error(message["content"])
        return
    with st.status(message["summary"], state="complete", expanded=False):
        for step in message["steps"]:
            _write_step(st, step)
    st.markdown(message["content"])
    _render_sources(message["sources"])


def _render_history() -> None:
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                _render_assistant(message)


def _thread_id(session: Session) -> str:
    if not st.session_state.get("thread_id"):
        st.session_state["thread_id"] = session.client.create_thread()
    return str(st.session_state["thread_id"])


def _answer(session: Session, question: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎓"):
        tracker = RunTracker()
        status = st.status("Reading your question…", expanded=True)
        try:
            thread_id = _thread_id(session)
            for event, data in session.client.ask(thread_id, question, session.context):
                for step in tracker.consume(event, data):
                    _write_step(status, step)
                    if step.kind == "route":
                        label, icon = subject_label(tracker.subject)
                        status.update(label=f"{icon} The {label} agent is working…")
        except BackendError as e:
            tracker.error = str(e)

        if tracker.error or tracker.answer is None:
            error = tracker.error or "The run finished without an answer."
            status.update(label="Couldn't answer this time", state="error")
            _show_error(error)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error, "error": True}
            )
            return

        status.update(label=tracker.summary, state="complete", expanded=False)
        st.markdown(tracker.answer)
        _render_sources(tracker.sources)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": tracker.answer,
            "summary": tracker.summary,
            "steps": tracker.steps,
            "sources": tracker.sources,
        }
    )


def _example_picker() -> str | None:
    st.markdown("##### Try one of these")
    columns = st.columns(2)
    for i, (icon, text) in enumerate(EXAMPLES):
        if columns[i % 2].button(f"{icon} {text}", key=f"example-{i}", width="stretch"):
            return text
    return None


def render(session: Session) -> None:
    """Draw the chat page."""
    # A conversation belongs to one knowledge base: switching users starts afresh.
    if st.session_state.get("conversation_user") != session.user_id:
        st.session_state["conversation_user"] = session.user_id
        reset_conversation()

    st.title("Ask Touchstone")
    st.caption(
        "Your question goes to a science, history, literature or general specialist. "
        "Specialists can search your knowledge base and the web, and a critic "
        "reviews every answer before you see it."
    )

    if not session.connected:
        st.info(
            f"The Touchstone backend isn't running. {START_BACKEND_HINT}", icon="🔌"
        )
    elif not session.user_id:
        st.warning(
            "Enter a **User ID** in the sidebar to start. Answers draw on the "
            "documents you added under that ID.",
            icon="👤",
        )

    _render_history()

    question = None
    examples = st.empty()
    if session.ready and not st.session_state["messages"]:
        with examples.container():
            question = _example_picker()

    typed = st.chat_input("Ask a question…", disabled=not session.ready)
    question = typed or question
    if question and session.ready:
        examples.empty()
        _answer(session, question)
