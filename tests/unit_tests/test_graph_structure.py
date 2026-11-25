"""Guard the public contract of the graphs (what LangGraph Studio and clients see).

If one of these fails you changed node names, edges or schemas. That may be fine,
but it affects saved threads, Studio and API clients - update the docs too.
"""

import json
from pathlib import Path

from retrieval_graph import graph, index_graph
from retrieval_graph.configuration import Configuration, IndexConfiguration

ROOT = Path(__file__).resolve().parents[2]

SPECIALISTS = {"science_agent", "history_agent", "literature_agent", "general_agent"}


def edges(compiled) -> set[tuple[str, str, bool]]:  # type: ignore[no-untyped-def]
    return {
        (e.source, e.target, bool(e.conditional)) for e in compiled.get_graph().edges
    }


def test_retrieval_graph_topology() -> None:
    assert graph.name == "RetrievalGraph"
    assert set(graph.get_graph().nodes) == {
        "__start__",
        "__end__",
        "classify_query",
        "tools",
        "critique_agent",
        "respond",
        *SPECIALISTS,
    }
    expected = {
        ("__start__", "classify_query", False),
        ("respond", "__end__", False),
        *{("classify_query", s, True) for s in SPECIALISTS},
        *{(s, "tools", True) for s in SPECIALISTS},
        *{(s, "critique_agent", True) for s in SPECIALISTS},
        *{("tools", s, True) for s in SPECIALISTS},
        *{("critique_agent", s, True) for s in SPECIALISTS},
        ("critique_agent", "respond", True),
    }
    assert edges(graph) == expected


def test_index_graph_topology() -> None:
    assert index_graph.name == "IndexGraph"
    assert edges(index_graph) == {
        ("__start__", "index_urls", False),
        ("index_urls", "index_docs", False),
        ("index_docs", "__end__", False),
    }


def test_retrieval_graph_io_schemas() -> None:
    assert list(graph.get_input_jsonschema()["properties"]) == ["messages"]
    assert set(graph.get_output_jsonschema()["properties"]) == {
        "messages",
        "retrieved_docs",
        "classification",
        "agent_response",
        "critique_decision",
        "critique_feedback",
    }


def _context_fields(compiled) -> set[str]:  # type: ignore[no-untyped-def]
    return set(compiled.get_context_jsonschema()["properties"])


def test_run_settings_are_the_context_schema() -> None:
    # Studio's settings panel and the server's context filter both read this schema.
    assert graph.context_schema is Configuration
    assert index_graph.context_schema is IndexConfiguration


def test_configuration_schema() -> None:
    index_fields = {
        "user_id",
        "elasticsearch_host",
        "elasticsearch_port",
        "embedding_model",
        "retriever_provider",
        "search_kwargs",
    }
    assert _context_fields(index_graph) == index_fields
    assert _context_fields(graph) == index_fields | {
        "response_system_prompt",
        "response_model",
        "query_system_prompt",
        "query_model",
        "enable_web_search",
        "web_search_api_key",
        "min_retrieval_threshold",
    }


def test_langgraph_json_points_at_the_compiled_graphs() -> None:
    spec = json.loads((ROOT / "langgraph.json").read_text())
    assert spec["graphs"] == {
        "indexer": "./src/retrieval_graph/index_graph.py:graph",
        "retrieval_graph": "./src/retrieval_graph/graph.py:graph",
    }
    for target in spec["graphs"].values():
        path, _, attr = target.partition(":")
        assert (ROOT / path).is_file()
        assert attr == "graph"
