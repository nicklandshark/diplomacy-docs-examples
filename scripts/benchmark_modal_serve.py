#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_COMMANDS = {
    "modal": ["modal", "serve", "scripts/serve_marimo_modal.py"],
    "uv-run": ["uv", "run", "modal", "serve", "scripts/serve_marimo_modal.py"],
}
_READY_PATTERNS = (
    "Serving... hit Ctrl-C to stop!",
    "Created web function",
)
_BUILD_RE = re.compile(r"Built image (?P<image_id>\S+) in (?P<seconds>\d+(?:\.\d+)?)s")
_PUBLIC_URL_RE = re.compile(r"=> (?P<url>https://\S+)")
_APP_URL_RE = re.compile(r"View app at (?P<url>https://\S+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how long `modal serve scripts/serve_marimo_modal.py` takes to become ready."
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark iterations.")
    parser.add_argument(
        "--launcher",
        choices=("modal", "uv-run", "both"),
        default="modal",
        help="Which local launcher path to benchmark.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Maximum wall-clock time to wait for one serve run to become ready.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser.parse_args()


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _run_once(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("MARIMO_TOKEN_PASSWORD", "benchmark-modal-serve")
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    started_at = time.monotonic()
    ready_at: float | None = None
    app_url: str | None = None
    public_url: str | None = None
    built_images: list[dict[str, Any]] = []
    output_lines: list[str] = []

    try:
        assert process.stdout is not None
        while True:
            if time.monotonic() - started_at > timeout_seconds:
                raise TimeoutError(
                    f"`{' '.join(command)}` did not become ready within {timeout_seconds} seconds."
                )

            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            text = line.rstrip("\n")
            output_lines.append(text)

            if match := _BUILD_RE.search(text):
                built_images.append(
                    {
                        "image_id": match.group("image_id"),
                        "seconds": float(match.group("seconds")),
                    }
                )
            if public_url is None and (match := _PUBLIC_URL_RE.search(text)):
                public_url = match.group("url")
            if app_url is None and (match := _APP_URL_RE.search(text)):
                app_url = match.group("url")
            if ready_at is None and any(pattern in text for pattern in _READY_PATTERNS):
                ready_at = time.monotonic()
                break

        if ready_at is None:
            raise RuntimeError(
                "`modal serve` exited before readiness. Last output:\n"
                + "\n".join(output_lines[-40:])
            )

        return {
            "command": command,
            "ready_seconds": round(ready_at - started_at, 3),
            "built_images": built_images,
            "app_url": app_url,
            "public_url": public_url,
            "tail_output": output_lines[-20:],
        }
    finally:
        _terminate_process_group(process)


def main() -> int:
    args = _parse_args()
    launcher_names = tuple(LAUNCHER_COMMANDS) if args.launcher == "both" else (args.launcher,)
    results = []
    for launcher_name in launcher_names:
        command = LAUNCHER_COMMANDS[launcher_name]
        for _ in range(args.runs):
            result = _run_once(command, args.timeout_seconds)
            result["launcher"] = launcher_name
            results.append(result)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for index, result in enumerate(results, start=1):
        print(f"Run {index} [{result['launcher']}]: ready in {result['ready_seconds']:.3f}s")
        print(f"  Command: {' '.join(result['command'])}")
        if result["public_url"]:
            print(f"  Public URL: {result['public_url']}")
        if result["app_url"]:
            print(f"  App URL: {result['app_url']}")
        if result["built_images"]:
            image_parts = ", ".join(
                f"{item['image_id']}={item['seconds']:.2f}s" for item in result["built_images"]
            )
            print(f"  Built images: {image_parts}")
        else:
            print("  Built images: none")

    if len(results) > 1:
        grouped: dict[str, list[float]] = {}
        for result in results:
            grouped.setdefault(result["launcher"], []).append(result["ready_seconds"])
        print("\nSummary:")
        for launcher_name, times in grouped.items():
            average = sum(times) / len(times)
            print(
                f"  {launcher_name}: avg={average:.3f}s"
                f" min={min(times):.3f}s max={max(times):.3f}s"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
