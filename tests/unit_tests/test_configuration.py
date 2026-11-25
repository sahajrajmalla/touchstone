import warnings
from typing import Any

import pytest
from langgraph.runtime import Runtime

from retrieval_graph.configuration import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_MODEL,
    DEFAULT_RESPONSE_MODEL,
    Configuration,
    IndexConfiguration,
    normalize_embedding_model,
)

CONFIGURABLE_DEPRECATED = r"config\['configurable'\] is deprecated"


######################
# from_runtime: the run context first, `configurable` as a deprecated fallback
######################


def test_from_runtime_returns_the_context_itself() -> None:
    settings = Configuration(user_id="alice", response_model="anthropic/x")
    resolved = Configuration.from_runtime(
        Runtime(context=settings), {"configurable": {"user_id": "mallory"}}
    )
    assert resolved is settings


def test_from_runtime_accepts_a_mapping_and_ignores_unknown_keys() -> None:
    resolved = Configuration.from_runtime(
        Runtime(context={"user_id": "alice", "thread_id": "t-1", "query_model": "q/m"})
    )
    assert (resolved.user_id, resolved.query_model) == ("alice", "q/m")


def test_from_runtime_keeps_the_values_of_index_settings() -> None:
    index_settings = IndexConfiguration(user_id="alice", search_kwargs={"k": 4})
    resolved = Configuration.from_runtime(Runtime(context=index_settings))
    assert isinstance(resolved, Configuration)
    assert (resolved.user_id, resolved.search_kwargs) == ("alice", {"k": 4})
    assert resolved.response_model == DEFAULT_RESPONSE_MODEL


def test_index_settings_accept_the_retrieval_settings() -> None:
    settings = Configuration(user_id="alice")
    assert IndexConfiguration.from_runtime(Runtime(context=settings)) is settings


@pytest.mark.parametrize("runtime", [None, Runtime()], ids=["no runtime", "no context"])
def test_from_runtime_falls_back_to_configurable_with_a_warning(
    runtime: Runtime[Any] | None,
) -> None:
    config = {"configurable": {"user_id": "bob", "enable_web_search": False}}
    with pytest.warns(DeprecationWarning, match=CONFIGURABLE_DEPRECATED):
        resolved = Configuration.from_runtime(runtime, config)  # type: ignore[arg-type]
    assert (resolved.user_id, resolved.enable_web_search) == ("bob", False)


def test_from_runtime_does_not_warn_without_settings_in_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER_ID", "env-user")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        resolved = Configuration.from_runtime(
            Runtime(), {"configurable": {"thread_id": "t-1"}}
        )
    assert resolved.user_id == "env-user"


def test_from_runtime_rejects_other_context_types() -> None:
    with pytest.raises(TypeError, match="Configuration or a dict, got str"):
        Configuration.from_runtime(Runtime(context="alice"))


def test_from_runtime_still_requires_a_user_id() -> None:
    with pytest.raises(ValueError, match="valid user_id"):
        Configuration.from_runtime(Runtime(context={"query_model": "q/m"}))


######################
# Fields, defaults and from_runnable_config
######################


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
