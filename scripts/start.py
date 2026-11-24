"""Run the LangGraph server and the Touchstone web UI together (``make start``).

The UI starts once the server is healthy. Ctrl+C - or either process exiting -
stops both. If a server is already running on the port (e.g. ``make dev`` in another
terminal), only the UI is started.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "touchstone_ui" / "app.py"
STUDIO_URL = "https://smith.langchain.com/studio/?baseUrl={api_url}"


def is_healthy(api_url: str) -> bool:
    """Return True if a LangGraph server answers at ``api_url``."""
    try:
        with urllib.request.urlopen(f"{api_url}/ok", timeout=2):  # noqa: S310 - local URL
            return True
    except OSError:
        return False


def wait_until_healthy(
    api_url: str, process: subprocess.Popen[bytes], timeout: float
) -> bool:
    """Wait for the server to answer; give up if it exits or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if is_healthy(api_url):
            return True
        time.sleep(0.5)
    return False


def stop(processes: list[subprocess.Popen[bytes]]) -> None:
    """Terminate the processes, killing any that do not exit within 10 seconds."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    """Start both processes and supervise them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=2024)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args()

    api_url = f"http://127.0.0.1:{args.api_port}"
    env = {**os.environ, "TOUCHSTONE_API_URL": api_url}
    processes: list[subprocess.Popen[bytes]] = []
    try:
        if is_healthy(api_url):
            print(
                f"A LangGraph server is already running at {api_url}; starting the UI only."
            )
        else:
            print("Starting the LangGraph server…", flush=True)
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "langgraph_cli",
                    "dev",
                    "--no-browser",
                    "--port",
                    str(args.api_port),
                ],
                cwd=ROOT,
                env=env,
            )
            processes.append(server)
            if not wait_until_healthy(api_url, server, args.startup_timeout):
                print(
                    f"\nThe LangGraph server did not start at {api_url}. "
                    "See the log above (or run `make doctor`).",
                    file=sys.stderr,
                )
                return 1

        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(APP),
                    "--server.port",
                    str(args.ui_port),
                ],
                cwd=ROOT,
                env=env,
            )
        )
        print(
            "\n🎓 Touchstone is running\n"
            f"   Web UI            http://localhost:{args.ui_port}\n"
            f"   LangGraph API     {api_url}\n"
            f"   LangGraph Studio  {STUDIO_URL.format(api_url=api_url)}\n"
            "   Press Ctrl+C to stop.\n",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        exited = next(process for process in processes if process.poll() is not None)
        return exited.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping Touchstone…")
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    sys.exit(main())
