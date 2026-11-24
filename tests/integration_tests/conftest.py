"""Integration tests run the real graphs against live services.

They need Elasticsearch plus Anthropic and Google API keys (see .env.example) and
are skipped - with the reason - when those are not available. Run them with::

    make test-integration
"""

import os
import urllib.request

import pytest
from dotenv import dotenv_values

REQUIRED_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "ELASTICSEARCH_URL",
    "ELASTICSEARCH_USER",
    "ELASTICSEARCH_PASSWORD",
)


def _elasticsearch_reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)  # noqa: S310 - local dev URL
    except urllib.error.HTTPError:
        return True  # Reachable (e.g. 401 without credentials).
    except OSError:
        return False
    return True


@pytest.fixture(autouse=True)
def _require_live_services(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load .env for these tests only, so it never leaks into the unit tests.
    for key, value in dotenv_values().items():
        if value is not None and key not in os.environ:
            monkeypatch.setenv(key, value)
    # User/password may be empty (security disabled); everything else must be set.
    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if name not in os.environ
        or (not os.environ[name] and not name.startswith("ELASTICSEARCH_"))
        or (name == "ELASTICSEARCH_URL" and not os.environ[name])
    ]
    if missing:
        pytest.skip(f"set {', '.join(missing)} in .env to run integration tests")
    url = os.environ["ELASTICSEARCH_URL"]
    if not _elasticsearch_reachable(url):
        pytest.skip(f"Elasticsearch is not reachable at {url} (try `make es-up`)")
