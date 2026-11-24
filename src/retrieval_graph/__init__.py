"""Touchstone: multi-agent knowledge retrieval and reasoning for education.

Two LangGraph graphs are exported (and registered in ``langgraph.json``):

``graph`` (``retrieval_graph``)
    Answers a question: a classifier routes it to a subject specialist (science,
    history, literature or general), which can search the user's knowledge base
    and the web before answering; a critique agent reviews the draft and may send
    it back for another attempt.

``index_graph`` (``indexer``)
    Adds documents - plain text, web pages or PDFs (by URL) - to a user's
    knowledge base in Elasticsearch.

Both graphs are configured through :class:`retrieval_graph.configuration.Configuration`
(``user_id`` is required).
"""

from retrieval_graph.graph import graph
from retrieval_graph.index_graph import graph as index_graph

__all__ = ["graph", "index_graph"]
