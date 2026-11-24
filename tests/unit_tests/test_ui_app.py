"""Run the real Streamlit app headlessly (streamlit.testing) against a fake backend."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from touchstone_ui.views import common

from .ui_fakes import ANSWER, FakeClient

APP = str(Path(__file__).resolve().parents[2] / "src" / "touchstone_ui" / "app.py")


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakeClient]]:
    FakeClient.reset()
    monkeypatch.setattr(common, "TouchstoneClient", FakeClient)
    common.get_client.clear()
    yield FakeClient
    common.get_client.clear()


def start(user_id: str = "") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if user_id:
        at.sidebar.text_input(key="user_id").input(user_id).run()
    assert not at.exception, at.exception
    return at


def test_backend_down_explains_how_to_start_it() -> None:
    FakeClient.healthy = False
    at = start()
    assert "Can't reach the backend" in at.sidebar.error[0].value
    assert "make dev" in at.info[0].value
    assert at.chat_input[0].disabled


def test_user_id_is_required_before_asking() -> None:
    at = start()
    assert "User ID" in at.warning[0].value
    assert at.chat_input[0].disabled


def test_asking_a_question_shows_the_answer_trace_and_sources() -> None:
    at = start(user_id="alice")
    at.chat_input[0].set_value("What is photosynthesis?").run()
    assert not at.exception, at.exception

    client = FakeClient.last
    assert client is not None
    assert client.asked == [
        (
            "thread-1",
            "What is photosynthesis?",
            {"user_id": "alice", "enable_web_search": True},
        )
    ]
    assert at.chat_message[0].markdown[0].value == "What is photosynthesis?"
    assistant = at.chat_message[1]
    assert ANSWER in [m.value for m in assistant.markdown]
    assert (
        assistant.status[0].label
        == "🧪 Science agent · 1 search · reviewed by the critic"
    )
    assert "Sources (1)" in [e.label for e in assistant.expander]


def test_follow_up_questions_reuse_the_thread() -> None:
    at = start(user_id="alice")
    at.chat_input[0].set_value("First?").run()
    at.chat_input[0].set_value("Second?").run()
    client = FakeClient.last
    assert client is not None
    assert [thread for thread, _, _ in client.asked] == ["thread-1", "thread-1"]
    assert len(at.chat_message) == 4  # history is kept across reruns


def test_example_question_buttons() -> None:
    at = start(user_id="alice")
    at.button(key="example-2").click().run()
    client = FakeClient.last
    assert client is not None
    assert client.asked[0][1] == "Analyze the theme of revenge in Shakespeare's Hamlet"


def test_advanced_settings_are_sent_as_configuration() -> None:
    at = start(user_id="alice")
    at.sidebar.text_input(key="response_model").input("anthropic/custom").run()
    at.sidebar.toggle(key="enable_web_search").set_value(False).run()
    at.chat_input[0].set_value("Q?").run()
    client = FakeClient.last
    assert client is not None
    assert client.asked[0][2] == {
        "user_id": "alice",
        "enable_web_search": False,
        "response_model": "anthropic/custom",
    }


def test_backend_errors_are_shown() -> None:
    FakeClient.ask_error = "could not reach the Touchstone backend"
    at = start(user_id="alice")
    at.chat_input[0].set_value("Q?").run()
    assert not at.exception
    assert "could not reach the Touchstone backend" in at.chat_message[1].error[0].value


def test_new_conversation_and_user_switch_reset_the_chat() -> None:
    at = start(user_id="alice")
    at.chat_input[0].set_value("Q?").run()
    assert len(at.chat_message) == 2
    at.sidebar.button[0].click().run()  # "New conversation"
    assert len(at.chat_message) == 0

    at.chat_input[0].set_value("Q again?").run()
    at.sidebar.text_input(key="user_id").input("bob").run()
    assert len(at.chat_message) == 0


def submit_buttons(at: AppTest) -> list:  # type: ignore[type-arg]
    return [b for b in at.button if b.label == "Add to knowledge base"]


def knowledge_page() -> None:
    from touchstone_ui.settings import UISettings
    from touchstone_ui.views import knowledge, sidebar

    knowledge.render(sidebar.render(UISettings()))


def test_knowledge_page_indexes_pasted_text() -> None:
    at = AppTest.from_function(knowledge_page, default_timeout=30)
    at.session_state["user_id"] = "alice"
    at.run()
    # Form values are only sent with the submit click, so no run() in between.
    at.text_area[0].input("Para one.\n\nPara two.")
    submit_buttons(at)[0].click().run()
    assert not at.exception, at.exception

    client = FakeClient.last
    assert client is not None
    assert client.indexed == [(["Para one.", "Para two."], {"user_id": "alice"})]
    assert "Added 2 documents" in at.success[0].value


def test_knowledge_page_reports_bad_urls_and_errors() -> None:
    FakeClient.index_error = "Could not extract content"
    at = AppTest.from_function(knowledge_page, default_timeout=30)
    at.session_state["user_id"] = "alice"
    at.run()
    at.text_area[1].input("https://example.com\nnot-a-url")
    submit_buttons(at)[1].click().run()
    assert "not-a-url" in at.warning[0].value
    assert "Could not extract content" in at.error[0].value
