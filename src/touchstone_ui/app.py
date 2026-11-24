"""Touchstone web UI.

Run it with ``make ui`` (or ``make start`` to launch the backend too). The UI is a
client of the LangGraph server started by ``make dev`` - see docs/ui.md.
"""

import streamlit as st

from touchstone_ui.settings import UISettings
from touchstone_ui.views import chat, knowledge, sidebar


def main() -> None:
    """Configure the page, draw the sidebar and run the selected page."""
    st.set_page_config(
        page_title="Touchstone",
        page_icon="🎓",
        layout="centered",
        menu_items={
            "About": "Touchstone - multi-agent knowledge retrieval and reasoning "
            "for education, built on LangGraph."
        },
    )
    session = sidebar.render(UISettings.from_env())
    page = st.navigation(
        [
            st.Page(
                lambda: chat.render(session),
                title="Ask",
                icon="💬",
                url_path="ask",
                default=True,
            ),
            st.Page(
                lambda: knowledge.render(session),
                title="Knowledge base",
                icon="📚",
                url_path="knowledge",
            ),
        ]
    )
    page.run()


if __name__ == "__main__":
    main()
