import pytest

from retrieval_graph.configuration import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_MODEL,
    DEFAULT_RESPONSE_MODEL,
    Configuration,
    IndexConfiguration,
    normalize_embedding_model,
)


def test_configuration_from_none() -> None:
    # Top-level keys are moved into `configurable` by LangChain's ensure_config.
    Configuration.from_runnable_config({"user_id": "foo"})


def test_reads_values_from_configurable() -> None:
    configuration = Configuration.from_runnable_config(
        {
            "configurable": {
                "user_id": "alice",
                "response_model": "anthropic/some-model",
                "enable_web_search": False,
                "search_kwargs": {"k": 3},
            }
        }
    )
    assert configuration.user_id == "alice"
    assert configuration.response_model == "anthropic/some-model"
    assert configuration.enable_web_search is False
    assert configuration.search_kwargs == {"k": 3}


def test_ignores_unknown_configurable_keys() -> None:
    # The LangGraph server adds keys such as thread_id / assistant_id.
    configuration = Configuration.from_runnable_config(
        {"configurable": {"user_id": "alice", "thread_id": "t-1", "assistant_id": "a"}}
    )
    assert configuration.user_id == "alice"


@pytest.mark.parametrize("user_id", ["", "default_user"])
def test_user_id_is_required(user_id: str) -> None:
    with pytest.raises(ValueError, match="valid user_id"):
        Configuration.from_runnable_config({"configurable": {"user_id": user_id}})


def test_user_id_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_ID", "bob")
    assert IndexConfiguration.from_runnable_config({}).user_id == "bob"


def test_web_search_api_key_defaults_to_tavily_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-123")
    configuration = Configuration.from_runnable_config(
        {"configurable": {"user_id": "a"}}
    )
    assert configuration.web_search_api_key == "tvly-123"


def test_defaults_use_currently_available_models() -> None:
    configuration = Configuration(user_id="alice")
    assert configuration.response_model == DEFAULT_RESPONSE_MODEL
    assert configuration.query_model == DEFAULT_QUERY_MODEL
    assert configuration.embedding_model == normalize_embedding_model(
        DEFAULT_EMBEDDING_MODEL
    )
    # These were retired by their providers and fail with "model not found".
    retired = (
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
        "text-embedding-004",
    )
    for value in (
        configuration.response_model,
        configuration.query_model,
        configuration.embedding_model,
    ):
        assert not value.endswith(retired)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("google/gemini-embedding-001", "google/models/gemini-embedding-001"),
        ("gemini-embedding-001", "google/models/gemini-embedding-001"),
        ("google/models/gemini-embedding-001", "google/models/gemini-embedding-001"),
        ("openai/text-embedding-3-small", "openai/text-embedding-3-small"),
    ],
)
def test_normalize_embedding_model(name: str, expected: str) -> None:
    assert normalize_embedding_model(name) == expected
    assert (
        IndexConfiguration(user_id="a", embedding_model=name).embedding_model
        == expected
    )


def test_empty_embedding_model_falls_back_to_default() -> None:
    configuration = IndexConfiguration(user_id="a", embedding_model="")
    assert configuration.embedding_model == normalize_embedding_model(
        DEFAULT_EMBEDDING_MODEL
    )
