"""Per-user isolation against a real Elasticsearch (no LLM calls).

Guards the exact-match user filter: with the previous filter on the analyzed
``metadata.user_id`` field, mixed-case or hyphenated ids found nothing, and ids that
share a token could see each other's documents.
"""

import uuid

import pytest

from retrieval_graph import index_graph, retrieval
from retrieval_graph.configuration import IndexConfiguration

pytestmark = pytest.mark.integration


async def search(user_id: str, query: str) -> list[str]:
    configuration = IndexConfiguration(user_id=user_id, search_kwargs={"k": 10})
    async with retrieval.make_retriever(configuration) as retriever:
        docs = await retriever.ainvoke(query)
    return [doc.metadata["user_id"] for doc in docs]


async def test_search_is_scoped_to_the_exact_user_id() -> None:
    suffix = uuid.uuid4().hex[:8]
    owner = f"Student-{suffix}"  # mixed case and a hyphen
    lookalike = f"student-{suffix}-b"  # shares the analyzed tokens of `owner`

    await index_graph.ainvoke(
        {"docs": "Mitochondria are the powerhouse of the cell."},
        context={"user_id": owner},
    )
    await index_graph.ainvoke(
        {"docs": "Mitochondria produce most of the cell's ATP."},
        context={"user_id": lookalike},
    )

    assert set(await search(owner, "mitochondria")) == {owner}
    assert set(await search(lookalike, "mitochondria")) == {lookalike}
    assert await search(f"nobody-{suffix}", "mitochondria") == []
