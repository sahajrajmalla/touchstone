"""State shared by the pages: the backend client and the current session."""

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from touchstone_ui.client import TouchstoneClient
from touchstone_ui.settings import UISettings

START_BACKEND_HINT = (
    "Start it in another terminal with `make dev`, or run the backend and UI "
    "together with `make start`."
)
ERROR_HINT = (
    "Tip: run `make doctor` to check your API keys and services. A missing or "
    "empty ANTHROPIC_API_KEY shows up as a connection error."
)


@st.cache_resource(show_spinner=False)
def get_client(api_url: str) -> TouchstoneClient:
    """Return a client for ``api_url``, shared across reruns and pages."""
    return TouchstoneClient(api_url)


@dataclass(frozen=True)
class Session:
    """What every page needs to talk to the backend."""

    client: TouchstoneClient
    settings: UISettings
    user_id: str
    connected: bool
    overrides: dict[str, Any] = field(default_factory=dict)
    """Settings overrides for the retrieval graph (models, web search)."""

    @property
    def ready(self) -> bool:
        """True when questions can be asked and documents indexed."""
        return self.connected and bool(self.user_id)

    @property
    def context(self) -> dict[str, Any]:
        """The run context (settings) sent with every retrieval run."""
        return {"user_id": self.user_id, **self.overrides}


def reset_conversation() -> None:
    """Forget the current chat; the next question starts a new thread."""
    st.session_state["messages"] = []
    st.session_state["thread_id"] = None
