# Elasticsearch

Touchstone stores every user's documents in one Elasticsearch index (`langchain_index`)
and filters each search by `user_id`. You need Elasticsearch 8.x (the Python client is
pinned below 9).

## Local Elasticsearch with Docker (recommended)

```bash
make es-up      # start and wait until healthy
make es-logs    # follow the logs
make es-down    # stop (documents are kept in a Docker volume)
make es-reset   # delete the container and ALL documents (asks first)
```

`docker-compose.yml` runs a single-node Elasticsearch 8.15 with security on and HTTP
(no TLS):

- the `elastic` user's password is `ELASTICSEARCH_PASSWORD` from `.env`;
- port 9200 is published on `127.0.0.1` only (set `ELASTICSEARCH_BIND=0.0.0.0` to change
  that - see below);
- 1 GB of heap. Docker needs about 2 GB of memory for it.

The defaults in `.env.example` already match:

```bash
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=change-me-please
```

Check it:

```bash
make doctor
# or
curl -u "elastic:$ELASTICSEARCH_PASSWORD" http://localhost:9200
```

## Using an existing cluster

Point `ELASTICSEARCH_URL`, `ELASTICSEARCH_USER` and `ELASTICSEARCH_PASSWORD` at it. Only
basic authentication is supported: API keys and Elastic Cloud IDs are not wired up yet.
The index is created automatically on first use.

## When the LangGraph server runs inside Docker

With `make dev` the server runs on your machine and `localhost:9200` just works. If you run
the server in a container instead - the LangGraph Studio desktop app, `langgraph up`, or
your own image - `localhost` inside that container is the container itself:

| Host OS | Set `ELASTICSEARCH_URL` to |
| --- | --- |
| macOS / Windows (Docker Desktop) | `http://host.docker.internal:9200` |
| Linux | the Docker bridge gateway, e.g. `http://172.17.0.1:9200` (find it with `docker network inspect bridge \| grep Gateway`) |

In both cases publish Elasticsearch on all interfaces by adding `ELASTICSEARCH_BIND=0.0.0.0`
to `.env` and re-running `make es-up`. This exposes port 9200 to your network, so keep a
strong password.

Alternatively, attach Elasticsearch to the server's Docker network and use its container
name:

```bash
docker network connect <server-network> touchstone-elasticsearch
# ELASTICSEARCH_URL=http://touchstone-elasticsearch:9200
```

After changing `.env`, restart the server so it picks up the new values. With the Studio
desktop app, restart the `*-langgraph-api-1` container and wait until `docker ps` shows it
as `(healthy)`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Elasticsearch is not configured: set ...` | Add the listed variables to `.env`. |
| `Connection refused` / `not reachable` | `make es-up`, or check `ELASTICSEARCH_URL`. |
| `NameResolutionError ... host.docker.internal` (Linux) | Use the gateway IP instead (table above). |
| HTTP 401 | `ELASTICSEARCH_USER` / `ELASTICSEARCH_PASSWORD` don't match the cluster. The Docker container keeps the password it was **first** created with: `make es-reset` to recreate it. |
| Container exits immediately | `make es-logs`. Usually too little memory for Docker, or `vm.max_map_count` too low on Linux (`sudo sysctl -w vm.max_map_count=262144`). |
| `different number of dimensions` / `dims` errors | The index was built with another embedding model. Re-index: `make es-reset` (or delete `langchain_index`). |
| No documents found | Index and ask with exactly the same `user_id` (case-sensitive). |

## Security

This setup is for development: TLS is off and the license is a trial. For production,
enable TLS, use a dedicated least-privilege user, and never commit `.env`.
