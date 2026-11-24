# Contributing to Touchstone

Thanks for helping! The short version:

```bash
make setup          # install everything
make hooks          # format + lint on every commit (recommended)
make test-watch     # tests re-run as you work
make check          # run before pushing: lint, strict types, tests
```

## Guidelines

- **Keep the educational focus.** Changes should make answers more accurate, clearer or
  easier to use for students and teachers.
- **Test without the network.** Unit tests must not need API keys or Elasticsearch; use the
  `fake_llm` and `knowledge_base` fixtures ([Development guide](docs/development.md#tests)).
  Anything that needs live services goes in `tests/integration_tests`.
- **Mind the graph's contract.** Node names, edges and schemas are used by LangGraph
  Studio, saved threads and API clients. `tests/unit_tests/test_graph_structure.py` fails
  when they change. Update it on purpose and note it in the changelog.
- **Never block the event loop** in graph nodes. Use async clients or `asyncio.to_thread`
  (`langgraph dev` rejects blocking calls).
- **Change prompts deliberately.** Prompt text goes to the models verbatim; describe
  behaviour changes in the pull request.
- **Document user-visible changes** in `docs/` and under "Unreleased" in `CHANGELOG.md`.

## Pull requests

1. Branch from `main`.
2. Keep each pull request focused, with tests for new behaviour and bug fixes.
3. Make sure `make check` passes. CI runs the same checks on Python 3.11-3.13.
4. Explain what changed and how you verified it.
