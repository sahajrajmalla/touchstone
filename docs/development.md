# Development guide

## Setup

```bash
make setup     # uv installs Python 3.12 and all dependencies into .venv, creates .env
make hooks     # optional: format and lint automatically before each commit
make doctor    # check keys and services
```

Everything runs through [uv](https://docs.astral.sh/uv/). Use `uv run <command>` (or
`source .venv/bin/activate`) to run tools by hand.

## Daily workflow

| Task | Command |
| --- | --- |
| Run the graphs with hot reload + LangGraph Studio | `make dev` |
| Run the web UI against it | `make ui` |
| Re-run unit tests on every save | `make test-watch` |
| Run one test file | `make test TEST_FILE=tests/unit_tests/test_graph_flow.py` |
| Format and auto-fix | `make format` |
| Everything CI checks | `make check` |

`make dev` reloads the graphs when you save a file. In Studio you can inspect the state
after every node, edit it and re-run from any step.

## Quality gates

`make check` runs what CI runs on every pull request (Python 3.11, 3.12 and 3.13):

- **Ruff** - linting (pycodestyle, pyflakes, isort, pydocstyle with Google-style
  docstrings, pyupgrade, bugbear, flake8-async, simplify) and formatting.
- **mypy --strict** - for `src/` and `scripts/`.
- **pytest** - the offline unit tests.

The `flake8-async` rules matter here: `langgraph dev` rejects blocking I/O inside async
nodes. Use async clients (`aiohttp`, `ainvoke`) or `asyncio.to_thread`.

## Tests

| Suite | Needs | Run with |
| --- | --- | --- |
| `tests/unit_tests` | nothing (offline) | `make test` / `make coverage` |
| `tests/integration_tests` | API keys + Elasticsearch | `make test-integration` (skips with a reason otherwise) |

The unit tests run the **real compiled graphs** with test doubles from
`tests/unit_tests/conftest.py`:

- `fake_llm` replaces `load_chat_model` with a scripted model. Set `subject`, queue
  `specialist_replies` and `critique_decisions`, then inspect `calls`.
- `knowledge_base` replaces `retrieval.make_retriever` with an in-memory store
  (`search_results`, `added`, `queries`, and the `user_ids` it was opened for).
- `context(**settings)` builds the run settings passed as `context=`.
  `runtime(**settings)` builds the `runtime` argument for calling a node directly.

```python
async def test_example(fake_llm, knowledge_base):
    knowledge_base.search_results = [Document(page_content="Cats swim in bowls.")]
    fake_llm.specialist_replies = [
        tool_call("retrieve_documents", "cats"),
        AIMessage("In bowls."),
    ]
    result = await graph.ainvoke(
        {"messages": [HumanMessage("Where do cats swim?")]}, context=context()
    )
    assert result["messages"][-1].content == "In bowls."
```

`pyproject.toml` turns every `LangGraphDeprecationWarning` into a test failure, so the
graphs stay ready for LangGraph 2.0.

`test_graph_structure.py` pins the graphs' node names, edges and schemas. Studio, saved
threads and API clients depend on these. If you change the graph on purpose, update the
test and mention it in the changelog.

The UI tests run the Streamlit app headlessly (`streamlit.testing.v1.AppTest`) against a
fake backend (`tests/unit_tests/ui_fakes.py`).

## Dependencies

Dependencies are declared with version ranges in `pyproject.toml` and pinned in `uv.lock`
(commit both).

```bash
uv add <package>             # runtime dependency
uv add --group dev <package> # development tool
uv add --optional ui <pkg>   # web UI only
make upgrade                 # newest versions allowed by pyproject.toml, then re-sync
```

CI installs with `uv sync --locked` and fails if `uv.lock` is out of date. Run `make lock`
after editing `pyproject.toml` by hand.

## Extending Touchstone

### Add a specialist agent

Example: a mathematics specialist.

1. **Prompt** - add `MATH_AGENT_PROMPT` to `src/retrieval_graph/prompts.py`. It must use
   the `{question}`, `{retrieved_docs}` and `{critique_feedback}` placeholders. Copy an
   existing specialist prompt as a starting point.
2. **Register it** in `SPECIALISTS` in `src/retrieval_graph/graph.py`:

   ```python
   "mathematics_agent": ("mathematics", prompts.MATH_AGENT_PROMPT, "Mathematics specialist agent."),
   ```

   The node and all its edges (from the classifier, to and from `tools`, to and from the
   critic) are created from this registry.
3. **Let the classifier choose it** - add `"mathematics"` to `QueryClassification.subject`
   in `graph.py`, and describe the subject in `CLASSIFICATION_SYSTEM_PROMPT`.
4. **UI label** (optional) - add `"mathematics": ("Mathematics", "📐")` to `SUBJECTS` in
   `src/touchstone_ui/events.py`.
5. **Tests** - add the node to `SPECIALISTS` in `tests/unit_tests/test_graph_structure.py`
   and a routing case to `test_routes_to_the_matching_specialist`.

### Other extension points

| Change | Where |
| --- | --- |
| Agent behaviour, answer structure, critique criteria | `prompts.py` |
| Models and number of retrieved documents | Run configuration ([docs](configuration.md)) |
| Loop limits | `MAX_TOOL_ROUNDS`, `MAX_CRITIQUE_RETRIES` in `graph.py` |
| Document truncation and how many documents are kept | Constants at the top of the tools section in `graph.py` |
| URL text extraction (PDF, HTML) | `utils.py` |
| A new tool | Define it next to `retrieve_documents`, add it to `tools`, handle it in `handle_tool_calls` |

### Ideas for the future

These come from the original project vision and are **not implemented** yet: integration
with learning-management systems, quiz generation from indexed content, learning-path
recommendations, curriculum and grade-level alignment, multilingual embeddings, multi-modal
documents, caching and response-time monitoring.

## Known limitations and follow-ups

- **Settings in `configurable` are deprecated.** Python callers that pass
  `{"configurable": {...}}` without `context=` still work, with a `DeprecationWarning`.
  Remove that fallback (`IndexConfiguration.from_runtime`) in a later release. See
  [Run settings](architecture.md#run-settings-context_schema).
- **Web search uses `TavilySearchResults`** from `langchain-community`, which is being
  sunset. Migrating to `langchain_tavily.TavilySearch` (different response format) would
  also remove the `langchain-community` dependency. Validate it against the live Tavily API.
- **`score_threshold` in `search_kwargs` is not applied** by the default similarity search.
- **HTML extraction** joins adjacent block elements without a space in minified pages
  ("Heading" + "Text" -> "HeadingText").
- **Conversation history is not sent to the models.** Each question is answered on its own
  (see [Architecture](architecture.md#what-a-specialist-sees)).
- **The final answer appears twice in the thread's messages**: once as the specialist's
  draft and once from `respond`. The UI shows it once; Studio's message list shows both.
- **Only Elasticsearch with basic auth** is supported (`elastic-local`); no API-key or cloud
  ID auth yet.

## Releasing

Record user-visible changes under "Unreleased" in [CHANGELOG.md](../CHANGELOG.md), bump
`version` in `pyproject.toml` and tag the release.
