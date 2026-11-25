# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **Indexing and retrieval failed on fresh installs** with a `TypeError`:
  `langchain-elasticsearch` 1.x renamed `es_connection=` to `client=`. Dependency ranges are
  now pinned, with a `uv.lock`.
- **Retired default models**: `claude-3-5-sonnet-20240620`, `claude-3-haiku-20240307` and
  Google `text-embedding-004` now fail. Defaults are `anthropic/claude-sonnet-5-5`,
  `anthropic/claude-haiku-5-5` and `google/gemini-embedding-001`. **Re-index existing
  documents** ([how](docs/configuration.md#upgrading-from-earlier-versions)).
- **Plain-text documents were never indexed**: the URL step cleared every document before
  the text step ran.
- **Follow-up questions got the first question's answer**: the first human message of a
  thread was used instead of the latest one. Per-question state (documents, drafts,
  critique) no longer leaks into the next question.
- **Runs could loop until `GraphRecursionError`** when a search found nothing or the critic
  kept asking for retries. Each question now gets at most 3 search rounds and 2 critic
  retries, and agents stop being told to search after an empty result.
- **User isolation**: searches now match `user_id` exactly. Before, mixed-case or
  hyphenated IDs found nothing, and IDs sharing a word ("alice" / "alice-smith") could see
  each other's documents.
- Elasticsearch connections were never closed. `retrieval.make_retriever` is now an async
  context manager.
- Web search blocked the event loop (rejected by `langgraph dev`), ignored
  `enable_web_search` and `web_search_api_key`, and reported failures as "No results found".
- The `USER_ID` fallback crashed the indexer. Clearer errors for missing Elasticsearch
  settings and unknown retriever providers.
- URL downloads have a 60-second timeout, and PDF/HTML parsing runs off the event loop.

### Added

- **Web UI** (Streamlit, `make start`). Ask questions with a live trace of the agents' work
  and sources, and manage a per-user knowledge base (text, URLs, files).
- `make quickstart` and a self-documenting Makefile (`make help`). `make doctor` checks the
  setup; `make start` runs the backend and UI together.
- `.env.example`, `docker-compose.yml` for a local Elasticsearch (`make es-up`), GitHub
  Actions CI (Python 3.11-3.13), pre-commit hooks and `.editorconfig`.
- 133 offline unit tests (scripted models, in-memory store, headless UI runs) and an
  integration test for user isolation.
- Documentation in `docs/`: architecture, configuration, UI, Elasticsearch, development and
  troubleshooting.

### Changed

- **Run settings are now the run context.** They use LangGraph's `context_schema` in
  place of the deprecated `config_schema`, which LangGraph 2.0 removes.
  - Nodes read `runtime.context`, and the web UI sends `context=`.
  - Studio shows the same settings as before.
  - The graph topology and input/output schemas are unchanged.
  - The unit tests now fail on any LangGraph deprecation warning.
- `retrieval.make_retriever(...)` and `index_graph.ensure_docs_have_user_id(...)` now take
  an `IndexConfiguration` (or a `Configuration`) instead of a `RunnableConfig`.
- The four specialist agents share one implementation, driven by a `SPECIALISTS` registry.
  The graph topology and schemas are unchanged.
- Python distribution renamed to `touchstone` (import path `retrieval_graph` unchanged).
- `PyPDF2` replaced by `pypdf`. Unused `msgspec` and `tavily-python` removed.
  `langgraph-api` moved to the dev tools (`langgraph-cli[inmem]`).
- Strict mypy passes for the whole codebase.
- `make` targets renamed: `test-integration`, `test-watch` and others. The old names still
  work as aliases.

### Deprecated

- **Passing settings in `config["configurable"]`.** Python calls without a context still
  read settings from `configurable`, but they emit a `DeprecationWarning`. This fallback
  will be removed in a future release.

### Migrating callers

Details are in [Configuration](docs/configuration.md#migrating-from-configurable).

- **Python.** Replace `graph.ainvoke(inputs, {"configurable": {"user_id": "alice"}})` with
  `graph.ainvoke(inputs, context={"user_id": "alice"})`. Keep `thread_id` in
  `configurable`. If you pass both, the context wins.
- **LangGraph SDK.** Replace `config={"configurable": {...}}` with `context={...}`. The
  server still accepts `configurable` on its own, but it rejects a run that sends both
  (HTTP 400). Saved Studio assistants keep working.
- **Direct helper calls.** Use `make_retriever(IndexConfiguration(user_id="alice"))` and
  `ensure_docs_have_user_id(docs, IndexConfiguration(user_id="alice"))`.
- **Dict contexts.** A dict may only hold the fields of the graph you call: the indexer
  rejects retrieval-only fields with a `TypeError`. A `Configuration` instance works for
  both graphs.
- **Missing `user_id`.** A context without a valid `user_id` (and no `USER_ID` in the
  environment) now fails when the run starts, before the first node.

## [0.0.1]

- Initial release: multi-agent retrieval graph (classifier, science / history / literature /
  general specialists, critic) and indexing graph on LangGraph with Elasticsearch.
