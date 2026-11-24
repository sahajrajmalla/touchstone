"""UI settings, read from the environment (and the project's ``.env`` file)."""

import os
from dataclasses import dataclass
from typing import Self

from dotenv import load_dotenv

DEFAULT_API_URL = "http://127.0.0.1:2024"
"""Where ``langgraph dev`` (``make dev``) serves the graphs."""
STUDIO_URL = "https://smith.langchain.com/studio/"


@dataclass(frozen=True)
class UISettings:
    """Settings for the Touchstone UI."""

    api_url: str = DEFAULT_API_URL
    """Base URL of the LangGraph server (``TOUCHSTONE_API_URL``)."""
    default_user_id: str = ""
    """Pre-filled user id (``USER_ID``)."""

    @classmethod
    def from_env(cls) -> Self:
        """Build settings from environment variables, loading ``.env`` if present."""
        load_dotenv(override=False)
        return cls(
            api_url=(os.getenv("TOUCHSTONE_API_URL") or DEFAULT_API_URL).rstrip("/"),
            default_user_id=os.getenv("USER_ID", "").strip(),
        )


def studio_url(api_url: str) -> str:
    """Return the LangGraph Studio URL that connects to ``api_url``."""
    return f"{STUDIO_URL}?baseUrl={api_url}"
