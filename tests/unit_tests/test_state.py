from langchain_core.documents import Document

from retrieval_graph.state import reduce_docs


def test_delete_clears_documents() -> None:
    assert reduce_docs([Document(page_content="a")], "delete") == []


def test_string_becomes_one_document_with_id() -> None:
    (doc,) = reduce_docs(None, "hello")
    assert doc.page_content == "hello"
    assert doc.metadata["id"]


def test_sequence_replaces_documents_and_coerces_items() -> None:
    existing = [Document(page_content="old")]
    original = Document(page_content="doc", metadata={"source": "x"})
    docs = reduce_docs(
        existing,
        ["text", {"page_content": "dict", "metadata": {"k": 1}}, original],
    )
    assert [d.page_content for d in docs] == ["text", "dict", "doc"]
    assert docs[1].metadata == {"k": 1}
    assert docs[2] is original


def test_tuple_is_treated_like_a_list() -> None:
    docs = reduce_docs(None, ("a", "b"))
    assert [d.page_content for d in docs] == ["a", "b"]


def test_empty_list_clears_documents() -> None:
    assert reduce_docs([Document(page_content="a")], []) == []


def test_other_values_keep_existing_documents() -> None:
    existing = [Document(page_content="a")]
    assert reduce_docs(existing, None) == existing  # type: ignore[arg-type]
    assert reduce_docs(None, None) == []  # type: ignore[arg-type]
