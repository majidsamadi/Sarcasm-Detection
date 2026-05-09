#!/usr/bin/env python3
"""Task-running helpers for the enhanced Streamlit dashboard.

This is intended for local use. Long training/evaluation tasks should not be run
from Hugging Face Spaces unless proper compute and persistence are configured.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_RUNS_DIR = PROJECT_ROOT / "reports" / "ui_runs"
UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunnerResult:
    return_code: int
    log_path: Path
    elapsed_seconds: float


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def command_for_runner(runner: str) -> List[str]:
    runner_path = PROJECT_ROOT / runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")
    if runner.endswith(".sh"):
        return ["bash", str(runner_path)]
    return [str(runner_path)]


def stream_command(
    command: List[str],
    log_name: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """Stream command output and write it to a local log file.

    Yields decoded output lines. At the end, yields a special status line.
    """
    UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = UI_RUNS_DIR / f"{safe_name(log_name)}_{timestamp()}.log"
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items() if v is not None})

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(shlex.quote(part) for part in command)}\n")
        log.write(f"Working directory: {PROJECT_ROOT}\n")
        log.write(f"Started: {time.ctime(started)}\n")
        log.write("=" * 80 + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            yield line.rstrip("\n")
        return_code = process.wait()
        elapsed = time.time() - started
        log.write("=" * 80 + "\n")
        log.write(f"Finished: {time.ctime()}\n")
        log.write(f"Return code: {return_code}\n")
        log.write(f"Elapsed seconds: {elapsed:.2f}\n")
        log.flush()

    yield f"__TASK_FINISHED__ return_code={return_code} log_path={log_path} elapsed_seconds={elapsed:.2f}"


def latest_logs(limit: int = 20) -> List[Path]:
    if not UI_RUNS_DIR.exists():
        return []
    logs = sorted(UI_RUNS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[:limit]
