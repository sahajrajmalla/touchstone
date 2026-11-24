"""Check that a Touchstone setup is ready to run (``make doctor``).

Reports Python, installed packages, required keys in ``.env``, Elasticsearch and the
LangGraph server. Secret values are never printed - only whether they are set.
Exits with status 1 if a required check fails.

``--quickstart`` (used by ``make quickstart``) starts a local Elasticsearch with
Docker when it is not reachable, and treats a missing Elasticsearch as a warning:
the agents still answer questions, only knowledge-base search is unavailable.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

MIN_PYTHON = (3, 11)
Status = Literal["ok", "warn", "fail", "info"]
_SYMBOLS: dict[Status, tuple[str, str]] = {
    "ok": ("✔", "32"),
    "warn": ("!", "33"),
    "fail": ("✘", "31"),
    "info": ("·", "36"),
}


@dataclass
class Check:
    """The outcome of one check."""

    status: Status
    name: str
    message: str
    hint: str = ""


def _paint(text: str, color: str) -> str:
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
        return f"\033[{color}m{text}\033[0m"
    return text


def check_python() -> Check:
    """Check that Python is new enough."""
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info >= MIN_PYTHON:
        return Check("ok", "Python", version)
    return Check("fail", "Python", f"{version} (3.11+ required)", "Run `make setup`.")


def check_packages() -> list[Check]:
    """Check that the project and its dev/UI tools are installed."""
    checks = []
    for module, name in (
        ("retrieval_graph", "Touchstone package"),
        ("langgraph_cli", "LangGraph CLI (make dev)"),
        ("streamlit", "Streamlit (make ui)"),
    ):
        if importlib.util.find_spec(module):
            checks.append(Check("ok", name, "installed"))
        else:
            checks.append(Check("fail", name, "not installed", "Run `make setup`."))
    return checks


def check_keys() -> list[Check]:
    """Check that the required API keys and Elasticsearch settings are present."""
    checks = []
    if not ENV_FILE.exists():
        checks.append(
            Check("warn", ".env", "not found", "Run `make env` and add your keys.")
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        checks.append(Check("ok", "ANTHROPIC_API_KEY", "set"))
    elif os.environ.get("OPENAI_API_KEY"):
        checks.append(
            Check(
                "warn",
                "ANTHROPIC_API_KEY",
                "missing (OPENAI_API_KEY is set)",
                "The default models are Anthropic; configure OpenAI models in the "
                "UI or Studio and install langchain-openai.",
            )
        )
    else:
        checks.append(
            Check(
                "fail",
                "ANTHROPIC_API_KEY",
                "missing",
                "Needed by the default models: https://console.anthropic.com/",
            )
        )

    hints = {
        "GOOGLE_API_KEY": "Needed for embeddings: https://aistudio.google.com/app/apikey",
        "ELASTICSEARCH_URL": "e.g. http://localhost:9200",
        "ELASTICSEARCH_USER": "e.g. elastic",
        "ELASTICSEARCH_PASSWORD": "Any password; `make es-up` uses it.",
    }
    for name, hint in hints.items():
        # User/password may legitimately be empty (security disabled) but must be set.
        present = name in os.environ and (
            bool(os.environ[name])
            or name in ("ELASTICSEARCH_USER", "ELASTICSEARCH_PASSWORD")
        )
        checks.append(
            Check("ok", name, "set")
            if present
            else Check("fail", name, "missing", hint)
        )

    for name, purpose in (
        ("TAVILY_API_KEY", "web search"),
        ("LANGSMITH_API_KEY", "LangSmith tracing"),
        ("USER_ID", "a default user id"),
    ):
        if os.environ.get(name):
            checks.append(Check("ok", name, "set"))
        else:
            checks.append(Check("info", name, f"not set (optional: {purpose})"))
    return checks


def _get(url: str, *, user: str | None = None, password: str | None = None) -> bytes:
    request = urllib.request.Request(url)  # noqa: S310 - URLs come from the user's .env
    if user is not None:
        token = base64.b64encode(f"{user}:{password or ''}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
        return bytes(response.read())


def check_elasticsearch() -> Check:
    """Check that Elasticsearch answers with the configured credentials."""
    url = os.environ.get("ELASTICSEARCH_URL", "")
    if not url:
        return Check("fail", "Elasticsearch", "ELASTICSEARCH_URL is not set")
    try:
        body = _get(
            url,
            user=os.environ.get("ELASTICSEARCH_USER"),
            password=os.environ.get("ELASTICSEARCH_PASSWORD"),
        )
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return Check(
                "fail",
                "Elasticsearch",
                f"{url} rejected the credentials (HTTP 401)",
                "Check ELASTICSEARCH_USER / ELASTICSEARCH_PASSWORD in .env.",
            )
        return Check("warn", "Elasticsearch", f"{url} answered HTTP {e.code}")
    except (OSError, ValueError):
        return Check(
            "fail",
            "Elasticsearch",
            f"not reachable at {url}",
            "Start it with `make es-up` (needs Docker) or fix ELASTICSEARCH_URL.",
        )
    try:
        version = json.loads(body)["version"]["number"]
    except (ValueError, KeyError, TypeError):
        version = "unknown version"
    return Check("ok", "Elasticsearch", f"reachable at {url} ({version})")


def check_backend() -> Check:
    """Report whether the LangGraph server (``make dev``) is running (optional)."""
    url = (os.environ.get("TOUCHSTONE_API_URL") or "http://127.0.0.1:2024").rstrip("/")
    try:
        _get(f"{url}/ok")
    except (OSError, ValueError):
        return Check(
            "info", "LangGraph server", f"not running at {url} (start: make dev)"
        )
    return Check("ok", "LangGraph server", f"running at {url}")


def start_local_elasticsearch() -> bool:
    """Start the docker-compose Elasticsearch if the URL is local and Docker exists."""
    host = urlparse(os.environ.get("ELASTICSEARCH_URL", "")).hostname
    if host not in ("localhost", "127.0.0.1") or not shutil.which("docker"):
        return False
    print(_paint("Starting Elasticsearch with Docker (make es-up)…", "36"), flush=True)
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--wait", "elasticsearch"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    """Run all checks and print a report."""
    parser = argparse.ArgumentParser(description="Check your Touchstone setup.")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="start a local Elasticsearch if needed; only missing keys are fatal",
    )
    args = parser.parse_args()
    load_dotenv(ENV_FILE, override=False)

    checks = [check_python(), *check_packages(), *check_keys()]
    elasticsearch = check_elasticsearch()
    if (
        args.quickstart
        and elasticsearch.status == "fail"
        and "not reachable" in elasticsearch.message
    ):
        if start_local_elasticsearch():
            elasticsearch = check_elasticsearch()
        if elasticsearch.status == "fail":
            elasticsearch.status = "warn"
            elasticsearch.hint = (
                "Questions still work, but indexing and knowledge-base search need "
                "Elasticsearch: install Docker and run `make es-up`."
            )
    checks += [elasticsearch, check_backend()]

    print(_paint("\nTouchstone doctor\n", "1"))
    width = max(len(c.name) for c in checks)
    for check in checks:
        symbol, color = _SYMBOLS[check.status]
        print(f"  {_paint(symbol, color)} {check.name.ljust(width)}  {check.message}")
        if check.hint and check.status in ("fail", "warn"):
            print(f"    {' ' * width}  {_paint('→ ' + check.hint, color)}")

    failures = [c for c in checks if c.status == "fail"]
    if failures:
        print(
            _paint(f"\n{len(failures)} problem(s) to fix.", "31")
            + " Edit .env, then run this again."
        )
        return 1
    print(_paint("\nAll set!", "32") + " Run `make start` to launch Touchstone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
