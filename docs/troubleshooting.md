# Troubleshooting

Start with:

```bash
make doctor
```

It checks Python, the installed packages, your keys (without printing them),
Elasticsearch and the LangGraph server, and says how to fix each problem.

## Setup

| Problem | Fix |
| --- | --- |
| `uv is required but was not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv`. |
| `Docker is required for this command` | Install Docker, or point `ELASTICSEARCH_URL` at an existing cluster ([Elasticsearch](elasticsearch.md)). |
| Port 2024 or 8501 already in use | `make start API_PORT=2025 UI_PORT=8502` (and set `TOUCHSTONE_API_URL` if you run `make ui` on its own). |
| `uv.lock` out of date in CI | Run `make lock` and commit `uv.lock`. |

## Errors while answering

These appear in the UI's trace, in Studio and in the server log.

| Message | Cause and fix |
| --- | --- |
| `AnthropicConnectionError: Connection error` | Usually `ANTHROPIC_API_KEY` is **missing or empty**: the SDK reports it as a connection error. Set it in `.env` and restart the server. Otherwise check your network or proxy. |
| `AuthenticationError` / `401` from Anthropic | The key is invalid or revoked. |
| `not_found_error` / `model: ...` | A retired or misspelled model name in your run configuration or a saved Studio assistant. See [Configuration](configuration.md#choosing-models). |
| `Error during retrieval: Elasticsearch is not configured` | Add the `ELASTICSEARCH_*` variables to `.env`. |
| `Error during retrieval: ... API key not valid` | `GOOGLE_API_KEY` is missing or invalid (embeddings). |
| `Error during retrieval: ... Connection refused` | Elasticsearch isn't running: `make es-up`. |
| `Error during retrieval: ... dimensions` | Documents were indexed with a different embedding model. [Re-index](configuration.md#upgrading-from-earlier-versions). |
| `Error during web search: ... 401` or `Did not find tavily_api_key` | Set `TAVILY_API_KEY`, or turn web search off (`enable_web_search: false`). |
| `Please provide a valid user_id` | Set a user ID in the UI sidebar or Studio's configuration, or `USER_ID` in `.env`. |
| `GraphRecursionError` | Shouldn't happen any more because of the loop guards. If you raised `MAX_TOOL_ROUNDS` / `MAX_CRITIQUE_RETRIES`, also raise the run's `recursion_limit`. |
| `BlockingError ... synchronous blocking call` (`langgraph dev`) | Code you added does blocking I/O inside an async node. Use an async client or `await asyncio.to_thread(...)`. As a last resort, use `langgraph dev --allow-blocking`. |

## Answers and documents

| Problem | Fix |
| --- | --- |
| The agent never uses my documents | Index and ask with the **same `user_id`** (exact, case-sensitive). Check the trace: "Retrieved N relevant document(s)" means the search worked. |
| Only one document is used | `search_kwargs.k` defaults to 1. Raise it in the run configuration, e.g. `{"k": 3}`. |
| A URL fails to index | The page must be reachable without login and yield at least 200 characters of text. Scanned (image-only) PDFs have no extractable text. Paste the text instead. |
| Follow-up questions lose context | Each question is answered on its own. Restate the topic ("Tell me more about photosynthesis"). |

## UI and Studio

| Problem | Fix |
| --- | --- |
| UI: "Can't reach the backend" | Start it with `make dev`, or use `make start`. If the server runs elsewhere, set `TOUCHSTONE_API_URL`. |
| Studio can't connect to the local server | Some browsers (e.g. Safari) block `http://127.0.0.1` from an HTTPS page. Use Chrome or Firefox, or run `uv run langgraph dev --tunnel`. |
| Studio still shows old defaults | Restart `make dev`. Saved assistants keep their own configuration. Edit or recreate them. |
