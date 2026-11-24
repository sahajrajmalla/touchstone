"""The "Knowledge base" page: add notes, web pages and PDFs for a user."""

import streamlit as st

from touchstone_ui.client import BackendError
from touchstone_ui.documents import decode_uploads, parse_urls, split_text
from touchstone_ui.views.common import START_BACKEND_HINT, Session


def _index(session: Session, docs: list[str]) -> None:
    if not docs:
        st.warning("There is nothing to add yet.", icon="✏️")
        return
    noun = "document" if len(docs) == 1 else "documents"
    with st.spinner(f"Adding {len(docs)} {noun}… (web pages and PDFs take longer)"):
        try:
            session.client.index(docs, {"user_id": session.user_id})
        except BackendError as e:
            st.error(f"Couldn't add the {noun}: {e}", icon="⚠️")
            return
    st.success(
        f"Added {len(docs)} {noun} to **{session.user_id}**'s knowledge base. "
        "Ask about it on the **Ask** page.",
        icon="✅",
    )


def render(session: Session) -> None:
    """Draw the knowledge base page."""
    st.title("Knowledge base")
    st.caption(
        "Add study material - notes, web pages and PDFs. It is stored for the "
        "current user ID only, and the agents search it when answering."
    )
    if not session.connected:
        st.info(
            f"The Touchstone backend isn't running. {START_BACKEND_HINT}", icon="🔌"
        )
    elif not session.user_id:
        st.warning("Enter a **User ID** in the sidebar first.", icon="👤")
    else:
        st.markdown(f"Adding documents for **{session.user_id}**.")
    disabled = not session.ready

    text_tab, url_tab, file_tab = st.tabs(
        ["✍️ Paste text", "🔗 Web pages & PDFs", "📄 Upload files"]
    )

    with text_tab, st.form("index-text", border=False):
        text = st.text_area(
            "Text",
            height=220,
            placeholder="Photosynthesis is the process by which plants convert "
            "light energy into chemical energy…",
        )
        split = st.checkbox(
            "Treat each paragraph as a separate document",
            value=True,
            help="Smaller documents make search results more precise.",
        )
        if st.form_submit_button(
            "Add to knowledge base", type="primary", disabled=disabled
        ):
            _index(session, split_text(text, split_paragraphs=split))

    with url_tab, st.form("index-urls", border=False):
        urls_text = st.text_area(
            "URLs (one per line)",
            height=150,
            placeholder="https://en.wikipedia.org/wiki/Photosynthesis\n"
            "https://example.com/lecture-notes.pdf",
        )
        st.caption("Each page or PDF is downloaded and its text extracted.")
        if st.form_submit_button(
            "Add to knowledge base", type="primary", disabled=disabled
        ):
            urls, rejected = parse_urls(urls_text)
            if rejected:
                st.warning(
                    "Skipped lines that are not http(s) URLs: "
                    + ", ".join(f"`{line}`" for line in rejected),
                    icon="🔗",
                )
            _index(session, urls)

    with file_tab, st.form("index-files", border=False):
        files = st.file_uploader(
            "Text or Markdown files",
            type=["txt", "md"],
            accept_multiple_files=True,
            help="Each file becomes one document. For PDFs, use a URL.",
        )
        if st.form_submit_button(
            "Add to knowledge base", type="primary", disabled=disabled
        ):
            docs, unreadable = decode_uploads(
                (f.name, f.getvalue()) for f in files or []
            )
            if unreadable:
                st.warning(
                    "Skipped files that are not UTF-8 text: " + ", ".join(unreadable),
                    icon="📄",
                )
            _index(session, docs)
