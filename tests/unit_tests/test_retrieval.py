from typing import Any

import pytest
from elasticsearch import AsyncElasticsearch
from langchain_core.embeddings import DeterministicFakeEmbedding

from retrieval_graph import retrieval


@pytest.fixture
def elastic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unroutable URL: building the retriever must not perform any network I/O.
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ELASTICSEARCH_USER", "elastic")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "secret")


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    models: list[str] = []

    def make_text_encoder(model: str) -> DeterministicFakeEmbedding:
        models.append(model)
        return DeterministicFakeEmbedding(size=8)

    monkeypatch.setattr(retrieval, "make_text_encoder", make_text_encoder)
    return models


@pytest.fixture
def closed_clients(monkeypatch: pytest.MonkeyPatch) -> list[AsyncElasticsearch]:
    closed: list[AsyncElasticsearch] = []
    original_close = AsyncElasticsearch.close

    async def tracking_close(self: AsyncElasticsearch) -> None:
        closed.append(self)
        await original_close(self)

    monkeypatch.setattr(AsyncElasticsearch, "close", tracking_close)
    return closed


def test_make_text_encoder_supports_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    encoder = retrieval.make_text_encoder("google/models/gemini-embedding-001")
    assert type(encoder).__name__ == "GoogleGenerativeAIEmbeddings"
    assert encoder.model == "models/gemini-embedding-001"  # type: ignore[attr-defined]
    assert (
        retrieval.make_text_encoder("gemini-embedding-001").model
        == "gemini-embedding-001"
    )  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("", "cannot be empty"),
        ("openai/text-embedding-3-small", "Unsupported embedding provider: openai"),
    ],
)
def test_make_text_encoder_rejects_invalid_models(model: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        retrieval.make_text_encoder(model)


@pytest.mark.usefixtures("elastic_env")
async def test_make_retriever_scopes_search_to_user_and_closes_client(
    fake_embeddings: list[str], closed_clients: list[AsyncElasticsearch]
) -> None:
    search_kwargs: dict[str, Any] = {
        "k": 2,
        "filter": [{"term": {"metadata.year": 2024}}],
    }
    config = {"configurable": {"user_id": "alice", "search_kwargs": search_kwargs}}

    async with retrieval.make_retriever(config) as retriever:
        assert type(retriever.vectorstore).__name__ == "AsyncElasticsearchStore"
        assert retriever.search_kwargs == {
            "k": 2,
            "filter": [
                {"term": {"metadata.year": 2024}},
                {"term": {"metadata.user_id.keyword": "alice"}},
            ],
        }
        assert closed_clients == []

    assert len(closed_clients) == 1
    assert fake_embeddings == ["google/models/gemini-embedding-001"]
    # The caller's configuration is never mutated.
    assert search_kwargs == {"k": 2, "filter": [{"term": {"metadata.year": 2024}}]}


@pytest.mark.usefixtures("elastic_env", "fake_embeddings")
async def test_make_retriever_defaults_k_to_one() -> None:
    async with retrieval.make_retriever(
        {"configurable": {"user_id": "alice", "search_kwargs": {}}}
    ) as retriever:
        assert retriever.search_kwargs["k"] == 1


@pytest.mark.usefixtures("elastic_env", "fake_embeddings")
async def test_make_retriever_closes_client_when_block_raises(
    closed_clients: list[AsyncElasticsearch],
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with retrieval.make_retriever({"configurable": {"user_id": "alice"}}):
            raise RuntimeError("boom")
    assert len(closed_clients) == 1


@pytest.mark.usefixtures("fake_embeddings")
async def test_make_retriever_reports_missing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELASTICSEARCH_URL", "")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "")  # empty is allowed
    with pytest.raises(ValueError) as excinfo:
        async with retrieval.make_retriever({"configurable": {"user_id": "alice"}}):
            pass
    message = str(excinfo.value)
    assert "ELASTICSEARCH_URL, ELASTICSEARCH_USER" in message
    assert "ELASTICSEARCH_PASSWORD" not in message


@pytest.mark.usefixtures("elastic_env", "fake_embeddings")
async def test_make_retriever_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Expected one of: elastic-local"):
        async with retrieval.make_retriever(
            {"configurable": {"user_id": "alice", "retriever_provider": "pinecone"}}
        ):
            pass
