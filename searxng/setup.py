#!/usr/bin/env python3
"""One-command SearXNG setup.

    python searxng/setup.py            start (idempotent) and verify
    python searxng/setup.py --stop     stop the container
    python searxng/setup.py --logs     tail the container logs
    python searxng/setup.py --check    verify only, change nothing

It generates the instance secret on first run, brings the container up, waits
for it to become healthy, and then proves the JSON API actually answers — which
is the step people skip, because SearXNG's JSON format is disabled by default
and fails with a bare 403.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252 and mangle the em-dashes below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
COMPOSE = HERE / "docker-compose.yml"
SETTINGS = HERE / "settings.yml"
URL = "http://localhost:8888"
PLACEHOLDER = "CHANGE_ME_RUN_SETUP"


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def docker_ok() -> bool:
    probe = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if probe.returncode != 0:
        print("Docker is not available or the daemon is not running.")
        print("  Install Docker Desktop, start it, then re-run this script.")
        print(f"  (docker said: {probe.stderr.strip()[:200]})")
        return False
    return True


def ensure_secret() -> None:
    text = SETTINGS.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        return
    secret = secrets.token_hex(32)
    SETTINGS.write_text(text.replace(PLACEHOLDER, secret), encoding="utf-8")
    print(f"generated instance secret in {SETTINGS.name}")


def compose(*args: str) -> subprocess.CompletedProcess:
    return run(["docker", "compose", "-f", str(COMPOSE), *args])


def wait_healthy(timeout: int = 120) -> bool:
    print("waiting for SearXNG to come up", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/healthz", timeout=3) as resp:
                if resp.status == 200:
                    print(" ok")
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" timed out")
    return False


def check_json_api() -> bool:
    """The test that matters: does a JSON query actually return results?"""
    url = f"{URL}/search?q=kubernetes+crashloopbackoff&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "kalam-datagen/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("\n  JSON API returned 403.")
            print("  `search.formats` in settings.yml must include `json`, and")
            print("  `server.limiter` must be false. Fix, then:")
            print("    python searxng/setup.py --restart")
        else:
            print(f"\n  JSON API returned HTTP {e.code}")
        return False
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        print(f"\n  JSON API unreachable: {e}")
        return False

    results = data.get("results", [])
    if not results:
        print("\n  JSON API answered but returned 0 results.")
        print("  Upstream engines may be rate-limiting a brand-new instance;")
        print("  wait a minute and re-run with --check.")
        return False

    print(f"  JSON API ok — {len(results)} results, engines: "
          f"{', '.join(sorted({r.get('engine', '?') for r in results}))[:80]}")
    print(f"  top hit: {results[0].get('title', '')[:70]}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Set up a local SearXNG for datagen")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--logs", action="store_true")
    p.add_argument("--check", action="store_true", help="verify only")
    args = p.parse_args()

    if not docker_ok():
        return 1

    if args.logs:
        subprocess.run(["docker", "compose", "-f", str(COMPOSE), "logs", "-f", "--tail", "80"])
        return 0

    if args.stop:
        print(compose("down").stdout or "stopped")
        return 0

    if args.check:
        return 0 if check_json_api() else 1

    if args.restart:
        compose("down")

    ensure_secret()

    print("starting SearXNG (first run pulls the image, this can take a minute)…")
    up = compose("up", "-d")
    if up.returncode != 0:
        print("docker compose failed:")
        print((up.stderr or up.stdout)[:800])
        return 1

    if not wait_healthy():
        print("\ncontainer did not become healthy. Logs:")
        print(compose("logs", "--tail", "30").stdout[-1500:])
        return 1

    print("verifying the JSON API…")
    if not check_json_api():
        return 1

    print(f"""
SearXNG is running at {URL}

  config.toml already points at it:
      [sources.keywords]
      engine      = "searxng"
      searxng_url = "{URL}"

  Try it:
      python -m datagen scrape "MLIS endpoint 503" --max-pages 3 --no-generate

  Stop it:  python searxng/setup.py --stop
  Logs:     python searxng/setup.py --logs
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
