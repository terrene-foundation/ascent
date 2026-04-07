# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Test all 80 exercise solutions run to completion without errors.

Each solution is executed in a subprocess with a timeout. Solutions are
parametrized by module and exercise number so failures are immediately
identifiable. Known-slow exercises (capstone/DL/AutoML) get a longer
timeout and are marked @pytest.mark.slow for selective skipping.

Run all:       uv run pytest tests/test_solutions.py -v
Run one module: uv run pytest tests/test_solutions.py -v -k "ascent01"
Skip slow:     uv run pytest tests/test_solutions.py -v -m "not slow"
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import SLOW_EXERCISES, discover_solutions, run_solution

_ALL_SOLUTIONS = discover_solutions()

# Standard timeout for most exercises
_TIMEOUT = 120
# Extended timeout for known-slow exercises (DL, AutoML, capstone)
_SLOW_TIMEOUT = 300


def _make_id(module: str, num: int) -> str:
    return f"{module}/ex_{num}"


def _is_slow(module: str, num: int) -> bool:
    return (module, num) in SLOW_EXERCISES


@pytest.mark.solution
@pytest.mark.parametrize(
    "module, num, path",
    _ALL_SOLUTIONS,
    ids=[_make_id(m, n) for m, n, _ in _ALL_SOLUTIONS],
)
def test_solution_runs(module: str, num: int, path: Path) -> None:
    """Exercise solution runs to completion with exit code 0."""
    timeout = _SLOW_TIMEOUT if _is_slow(module, num) else _TIMEOUT

    try:
        result = run_solution(path, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"TIMEOUT after {timeout}s — {module}/ex_{num} "
            f"({'known slow' if _is_slow(module, num) else 'unexpected'})"
        )

    if result.returncode != 0:
        # Extract the last meaningful traceback lines
        stderr_lines = result.stderr.strip().splitlines()
        # Find the last exception line
        error_summary = "\n".join(stderr_lines[-15:]) if stderr_lines else "(no stderr)"
        pytest.fail(
            f"Exit code {result.returncode}\n"
            f"─── stderr (last 15 lines) ───\n{error_summary}"
        )
