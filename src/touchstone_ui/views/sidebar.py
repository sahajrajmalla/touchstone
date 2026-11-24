"""Sidebar: backend status, user id, conversation controls and settings."""

from typing import Any

import streamlit as st

from touchstone_ui.settings import UISettings, studio_url
from touchstone_ui.views.common import (
    START_BACKEND_HINT,
    Session,
    get_client,
    reset_conversation,
)


def _init_state(settings: UISettings) -> None:
    st.session_state.setdefault("user_id", settings.default_user_id)
    st.session_state.setdefault("enable_web_search", True)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)


def render(settings: UISettings) -> Session:
    """Draw the sidebar and return the resulting :class:`Session`."""
    _init_state(settings)
    client = get_client(settings.api_url)
    connected = client.is_healthy()

    with st.sidebar:
        st.markdown("### 🎓 Touchstone")
        st.caption("A team of AI tutors that research, answer and double-check.")

        if connected:
            st.success(f"Connected to `{settings.api_url}`", icon="🟢")
        else:
            st.error(
                f"Can't reach the backend at `{settings.api_url}`.\n\n{START_BACKEND_HINT}",
                icon="🔴",
            )
            st.button("Try again", width="stretch")  # Any click reruns the script.

        user_id = st.text_input(
            "User ID",
            key="user_id",
            placeholder="e.g. alice",
            help="Your knowledge base is private to this ID. Use the same ID "
            "when adding documents and when asking questions.",
        ).strip()

        # Always enabled: the sidebar renders before the chat page adds the first
        # answer, so a "disabled when empty" state would lag one interaction behind.
        st.button(
            "New conversation",
            icon="➕",
            width="stretch",
            on_click=reset_conversation,
        )

        with st.expander("Advanced settings", icon="⚙️"):
            response_model = st.text_input(
                "Answer model",
                key="response_model",
                placeholder="Server default",
                help="Model for the specialist and critic agents, as provider/model "
                "(e.g. anthropic/claude-sonnet-5-5).",
            )
            query_model = st.text_input(
                "Routing model",
                key="query_model",
                placeholder="Server default",
                help="Model that routes questions to a specialist.",
            )
            enable_web_search = st.toggle(
                "Allow web search",
                key="enable_web_search",
                help="Lets the agents search the web (needs TAVILY_API_KEY on the server).",
            )

        st.link_button(
            "Inspect in LangGraph Studio",
            studio_url(settings.api_url),
            icon="🔬",
            width="stretch",
            help="See the graph, state and every step of each run.",
        )

    overrides: dict[str, Any] = {"enable_web_search": enable_web_search}
    for key, value in (
        ("response_model", response_model),
        ("query_model", query_model),
    ):
        if value.strip():
            overrides[key] = value.strip()

    return Session(
        client=client,
        settings=settings,
        user_id=user_id,
        connected=connected,
        overrides=overrides,
    )
