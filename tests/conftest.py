# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT test configuration — fixtures, markers, and environment setup."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# ── Custom markers ───────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "solution: marks exercise solution tests")
    config.addinivalue_line("markers", "dataset: marks dataset integrity tests")
    config.addinivalue_line(
        "markers", "consistency: marks three-format consistency tests"
    )
    config.addinivalue_line("markers", "slow: marks exercises known to take >60s")
    for i in range(1, 11):
        config.addinivalue_line(
            "markers", f"module{i:02d}: marks tests for ASCENT{i:02d}"
        )


# ── Module / exercise discovery ──────────────────────────────────────
MODULES = [f"ascent{i:02d}" for i in range(1, 11)]

# Exercises known to timeout (>120s) — heavy computation, not bugs
SLOW_EXERCISES: set[tuple[str, int]] = {
    ("ascent03", 7),  # HyperparameterSearch + TrainingPipeline + LightGBM
    ("ascent03", 8),  # Production Pipeline capstone
    ("ascent04", 1),  # Clustering with AutoML
    ("ascent04", 7),  # DL intro (torch training)
    ("ascent07", 8),  # DL Capstone (CNN + ONNX)
    ("ascent08", 7),  # NLP Capstone (transformer pipeline)
    ("ascent10", 2),  # Federated learning with DP (3 clients × 50 rounds)
    ("ascent10", 4),  # Compliance artefacts + bias audit + DriftMonitor
    ("ascent10", 6),  # Compliance automation with Nexus + DriftMonitor
    ("ascent10", 7),  # Enterprise governance registry (DataFlow + many CRUD)
    ("ascent10", 8),  # Regulated AI capstone (federated + adversarial + compliance)
}


def discover_solutions() -> list[tuple[str, int, Path]]:
    """Return (module, exercise_num, path) for every solution file."""
    results = []
    for mod in MODULES:
        sol_dir = REPO_ROOT / "modules" / mod / "solutions"
        if not sol_dir.exists():
            continue
        for f in sorted(sol_dir.glob("ex_*.py")):
            num = int(f.stem.split("_")[1])
            results.append((mod, num, f))
    return results


def discover_datasets() -> list[tuple[str, str, Path]]:
    """Return (module, filename, path) for every dataset file."""
    data_dir = REPO_ROOT / "data"
    results = []
    for mod_dir in sorted(data_dir.iterdir()):
        if not mod_dir.is_dir():
            continue
        for f in sorted(mod_dir.iterdir()):
            if f.suffix in (".csv", ".parquet", ".json"):
                results.append((mod_dir.name, f.name, f))
    return results


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return REPO_ROOT / "data"


@pytest.fixture(scope="session")
def python_exe() -> str:
    """Return the Python executable within the venv."""
    return sys.executable


def run_solution(path: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a solution file in a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach slow and module markers to parametrized solution tests."""
    import re

    pattern = re.compile(r"(ascent\d+)/ex_(\d+)")
    for item in items:
        if "test_solution_runs" not in item.name:
            continue
        match = pattern.search(item.nodeid)
        if not match:
            continue
        module = match.group(1)
        num = int(match.group(2))
        # Apply module marker
        item.add_marker(getattr(pytest.mark, f"module{module[-2:]}"))
        # Apply slow marker for known-slow exercises
        if (module, num) in SLOW_EXERCISES:
            item.add_marker(pytest.mark.slow)
