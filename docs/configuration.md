# Configuration

Touchstone has two kinds of settings:

- **Environment variables** (`.env`): keys and connections, read by the server process.
- **Run configuration**: per-run choices such as models and the user ID, passed with each
  run. They come from the UI sidebar, LangGraph Studio's configuration panel or
  `{"configurable": {...}}` in code.

Check your setup at any time with `make doctor`.

## Environment variables

Copy the template with `make env` (or `make setup`) and fill it in. `.env` is git-ignored.

| Variable | Required | Used for |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | yes (default models) | Claude models for the classifier, specialists and critic |
| `GOOGLE_API_KEY` | yes | Gemini embeddings, used when indexing and searching |
| `ELASTICSEARCH_URL` | yes | e.g. `http://localhost:9200` |
| `ELASTICSEARCH_USER` / `ELASTICSEARCH_PASSWORD` | yes | Basic auth; may be empty if security is disabled |
| `ELASTICSEARCH_BIND` | no | Interface `make es-up` publishes port 9200 on (default `127.0.0.1`) |
| `TAVILY_API_KEY` | no | Web search; without it the `web_search` tool reports an error |
| `USER_ID` | no | Default `user_id` when a run does not set one |
| `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | no | Tracing in [LangSmith](https://smith.langchain.com/) |
| `TOUCHSTONE_API_URL` | no | LangGraph server the UI talks to (default `http://127.0.0.1:2024`) |

`langgraph dev` loads `.env` automatically (see `"env"` in `langgraph.json`), and so do the
UI and `make doctor`.

## Run configuration

Defined in `src/retrieval_graph/configuration.py`. The indexer accepts the first group of
fields; the retrieval graph accepts all of them.

| Field | Default | Description |
| --- | --- | --- |
| `user_id` | `$USER_ID` | **Required.** Selects the user's knowledge base. Matched exactly (case-sensitive), up to 256 characters. |
| `embedding_model` | `google/gemini-embedding-001` | Embedding model as `provider/model`. Only Google models are supported. |
| `retriever_provider` | `elastic-local` | The only supported vector store. |
| `search_kwargs` | `{"k": 1, "score_threshold": 0.7}` | Passed to the similarity search. `k` is the number of documents per search. `score_threshold` is currently **not applied**. |
| `elasticsearch_host`, `elasticsearch_port` | from env | Informational only; the connection uses `ELASTICSEARCH_URL`. |
| `response_model` | `anthropic/claude-sonnet-5-5` | Specialists and critic. |
| `query_model` | `anthropic/claude-haiku-5-5` | Classifier. |
| `enable_web_search` | `true` | Offer the `web_search` tool to the specialists. |
| `web_search_api_key` | `$TAVILY_API_KEY` | Tavily key, if different from the environment. |
| `response_system_prompt`, `query_system_prompt`, `min_retrieval_threshold` | - | Reserved; not currently used. |

Example:

```python
config = {
    "configurable": {
        "user_id": "alice",
        "response_model": "anthropic/claude-opus-5-5",
        "search_kwargs": {"k": 3},
        "enable_web_search": False,
    }
}
await graph.ainvoke({"messages": [("user", "Explain entropy")]}, config)
```

## Choosing models

Models are given as `provider/model` and loaded with LangChain's
[`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/).

- **Anthropic** (default, installed): `anthropic/claude-sonnet-5-5`,
  `anthropic/claude-haiku-5-5`, `anthropic/claude-opus-5-5`.
- **Other providers**: install the integration package and set its key. For OpenAI, run
  `uv add langchain-openai`, set `OPENAI_API_KEY` and use, for example, `openai/gpt-4o`.

Embeddings are Google-only (`google/<model>`; the `models/` prefix is added for you).
Changing the embedding model changes the vectors, so re-index afterwards (below).

## Upgrading from earlier versions

- **Models.** `claude-3-5-sonnet-20240620`, `claude-3-haiku-20240307` and Google's
  `text-embedding-004` have been retired by their providers; requests to them fail. The
  defaults now use current models. If you saved old model names in a Studio assistant or
  in code, update them.
- **Re-index your documents.** Documents indexed with `text-embedding-004` cannot be
  searched with the new embedding model (different vectors and dimensions). Delete the
  index and add your documents again:

  ```bash
  make es-reset                         # Docker Elasticsearch from `make es-up`
  # or, for any cluster:
  curl -u "$ELASTICSEARCH_USER:$ELASTICSEARCH_PASSWORD" -X DELETE "$ELASTICSEARCH_URL/langchain_index"
  ```

- **User IDs are matched exactly.** Before, mixed-case or hyphenated IDs found nothing, and
  IDs sharing a word could see each other's documents.
- **`retrieval.make_retriever`** is now an async context manager:
  `async with make_retriever(config) as retriever:`.
- The Python distribution is now called `touchstone`. The import path is still
  `retrieval_graph`.
