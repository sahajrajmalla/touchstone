# Touchstone - developer commands.
#
# New here? Run `make quickstart`. Run `make` (or `make help`) to list every command.
# Everything runs through uv (https://docs.astral.sh/uv/), which also installs a
# suitable Python version for you.

.DEFAULT_GOAL := help

UV        ?= uv
COMPOSE   ?= docker compose
API_PORT  ?= 2024
UI_PORT   ?= 8501
TEST_FILE ?= tests/unit_tests/

RUN := $(UV) run --all-extras

.PHONY: help quickstart setup env doctor start dev ui \
	es-up es-down es-logs es-reset \
	test test-watch test-integration coverage lint format typecheck check \
	lock upgrade hooks clean \
	tests test_watch extended_tests integration_tests \
	_require-uv _require-docker

##@ Getting started

help: ## Show this help
	@awk 'BEGIN { FS = ":.*## " } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

quickstart: setup ## One command: install, configure, check and start Touchstone
	@$(RUN) python scripts/doctor.py --quickstart
	@$(MAKE) --no-print-directory start

setup: _require-uv .env ## Install dependencies and create .env (safe to re-run)
	$(UV) sync --all-extras
	@echo ""
	@echo "Setup complete. Next:"
	@echo "  1. Add your API keys to .env"
	@echo "  2. make doctor    # check your configuration"
	@echo "  3. make start     # launch the backend and the web UI"

env: .env ## Create .env from .env.example (never overwrites an existing .env)

.env:
	@cp .env.example .env
	@echo "Created .env from .env.example - add your API keys to it."

doctor: _require-uv ## Check Python, API keys, Elasticsearch and the backend
	@$(RUN) python scripts/doctor.py

##@ Run

start: _require-uv .env ## Start the backend and the web UI together (Ctrl+C stops both)
	@$(RUN) python scripts/start.py --api-port $(API_PORT) --ui-port $(UI_PORT)

dev: _require-uv .env ## Start the LangGraph server (hot reload) and open LangGraph Studio
	$(RUN) langgraph dev --port $(API_PORT)

ui: _require-uv ## Start only the web UI (expects the backend from `make dev`)
	$(RUN) streamlit run src/touchstone_ui/app.py --server.port $(UI_PORT)

##@ Elasticsearch (Docker)

es-up: _require-docker .env ## Start Elasticsearch and wait until it is healthy
	@[ -n "$$ELASTICSEARCH_PASSWORD" ] || grep -Eq '^ELASTICSEARCH_PASSWORD=.+' .env || { \
		echo "Set ELASTICSEARCH_PASSWORD in .env first (any password - it becomes the 'elastic' user's)."; \
		exit 1; }
	$(COMPOSE) up -d --wait elasticsearch
	@echo "Elasticsearch is ready at http://localhost:9200"

es-down: _require-docker ## Stop Elasticsearch (indexed documents are kept)
	$(COMPOSE) down

es-logs: _require-docker ## Follow the Elasticsearch logs
	$(COMPOSE) logs -f elasticsearch

es-reset: _require-docker ## Delete Elasticsearch and ALL indexed documents (asks first)
	@printf "This permanently deletes every indexed document. Type 'yes' to continue: "; \
	read answer; [ "$$answer" = "yes" ] || { echo "Aborted."; exit 1; }
	$(COMPOSE) down -v

##@ Quality

test: _require-uv ## Run the unit tests (offline: no API keys or services needed)
	$(RUN) pytest $(TEST_FILE)

test-watch: _require-uv ## Re-run the unit tests whenever a file changes
	$(RUN) ptw --now . $(TEST_FILE)

test-integration: _require-uv ## Run the integration tests (needs API keys + Elasticsearch)
	$(RUN) pytest tests/integration_tests -rs

coverage: _require-uv ## Run the unit tests with a coverage report (htmlcov/index.html)
	$(RUN) pytest tests/unit_tests --cov --cov-report=term --cov-report=html

lint: _require-uv ## Lint, check formatting and type-check
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

format: _require-uv ## Format the code and fix lint issues automatically
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck: _require-uv ## Type-check with mypy (strict)
	$(RUN) mypy

check: lint test ## Run everything CI runs

##@ Maintenance

lock: _require-uv ## Refresh uv.lock after editing dependencies in pyproject.toml
	$(UV) lock

upgrade: _require-uv ## Upgrade dependencies to the newest versions allowed by pyproject.toml
	$(UV) lock --upgrade
	$(UV) sync --all-extras

hooks: _require-uv ## Install the git pre-commit hooks
	$(RUN) pre-commit install

clean: ## Remove caches, coverage reports and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist src/*.egg-info
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +

# Aliases for the previous target names.
tests: test
test_watch: test-watch
extended_tests integration_tests: test-integration

_require-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv is required but was not found. Install it with one of:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  brew install uv"; \
		echo "More options: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; }

_require-docker:
	@command -v docker >/dev/null 2>&1 || { \
		echo "Docker is required for this command: https://docs.docker.com/get-docker/"; \
		echo "Alternatively, point ELASTICSEARCH_URL in .env at an existing Elasticsearch."; \
		exit 1; }
