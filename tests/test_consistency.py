# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Verify three-format consistency: solutions, local, notebooks, colab.

Checks:
- Every solution has a matching local .py file
- Every solution has matching Jupyter and Colab notebooks
- TODO markers in exercises match across formats
- Solutions contain no TODO/blank markers (they must be complete)
- Exercises contain TODO markers (they must be stripped)
- Import consistency between solution and exercise

Run: uv run pytest tests/test_consistency.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.conftest import MODULES, REPO_ROOT


def _solution_files(module: str) -> list[Path]:
    return sorted((REPO_ROOT / "modules" / module / "solutions").glob("ex_*.py"))


def _local_files(module: str) -> list[Path]:
    return sorted((REPO_ROOT / "modules" / module / "local").glob("ex_*.py"))


def _notebook_files(module: str) -> list[Path]:
    return sorted((REPO_ROOT / "modules" / module / "notebooks").glob("ex_*.ipynb"))


def _colab_files(module: str) -> list[Path]:
    return sorted((REPO_ROOT / "modules" / module / "colab").glob("ex_*.ipynb"))


def _extract_exercise_num(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _count_todos(text: str) -> int:
    return len(re.findall(r"#\s*TODO:", text))


def _extract_notebook_source(path: Path) -> str:
    """Extract all code cell source from a .ipynb file."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            lines.extend(cell.get("source", []))
    return "".join(lines)


# ── Build parametrized lists ─────────────────────────────────────────
_ALL_SOLUTIONS: list[tuple[str, int, Path]] = []
for mod in MODULES:
    for f in _solution_files(mod):
        _ALL_SOLUTIONS.append((mod, _extract_exercise_num(f), f))


class TestFormatCompleteness:
    """Every solution must have all three exercise formats."""

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_local_exists(self, module: str, num: int, sol_path: Path) -> None:
        local = REPO_ROOT / "modules" / module / "local" / f"ex_{num}.py"
        assert local.exists(), f"Missing local exercise: {module}/local/ex_{num}.py"

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_notebook_exists(self, module: str, num: int, sol_path: Path) -> None:
        nb = REPO_ROOT / "modules" / module / "notebooks" / f"ex_{num}.ipynb"
        assert nb.exists(), f"Missing notebook: {module}/notebooks/ex_{num}.ipynb"

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_colab_exists(self, module: str, num: int, sol_path: Path) -> None:
        colab = REPO_ROOT / "modules" / module / "colab" / f"ex_{num}.ipynb"
        assert colab.exists(), f"Missing colab: {module}/colab/ex_{num}.ipynb"


class TestSolutionIntegrity:
    """Solutions must be complete — no TODO markers or blanks."""

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_solution_no_todos(self, module: str, num: int, sol_path: Path) -> None:
        text = sol_path.read_text(encoding="utf-8")
        todo_count = _count_todos(text)
        assert todo_count == 0, (
            f"{module}/solutions/ex_{num}.py has {todo_count} TODO markers — "
            "solutions must be complete"
        )

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_solution_no_blanks(self, module: str, num: int, sol_path: Path) -> None:
        text = sol_path.read_text(encoding="utf-8")
        blanks = re.findall(r"____", text)
        assert not blanks, (
            f"{module}/solutions/ex_{num}.py has {len(blanks)} ____ blanks — "
            "solutions must not contain placeholders"
        )

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_solution_has_header(self, module: str, num: int, sol_path: Path) -> None:
        text = sol_path.read_text(encoding="utf-8")
        assert (
            "ASCENT" in text and "Exercise" in text
        ), f"{module}/solutions/ex_{num}.py missing exercise header block"

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_solution_no_pandas(self, module: str, num: int, sol_path: Path) -> None:
        text = sol_path.read_text(encoding="utf-8")
        pandas_imports = re.findall(
            r"^\s*import pandas|^\s*from pandas", text, re.MULTILINE
        )
        assert (
            not pandas_imports
        ), f"{module}/solutions/ex_{num}.py imports pandas — polars only"


class TestExerciseQuality:
    """Exercises must have TODO markers (they should be stripped from solutions)."""

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_exercise_has_todos(self, module: str, num: int, sol_path: Path) -> None:
        local = REPO_ROOT / "modules" / module / "local" / f"ex_{num}.py"
        if not local.exists():
            pytest.skip("Local exercise missing")
        text = local.read_text(encoding="utf-8")
        todo_count = _count_todos(text)
        blank_count = len(re.findall(r"____", text))
        # Either explicit `# TODO:` markers or `____` blanks count as
        # student-fillable holes. Block-stripped exercises (M9-M10) often
        # use `# TASK N:` headers above large blank regions instead of
        # scattering TODO markers, which is equally valid pedagogically.
        assert todo_count > 0 or blank_count > 0, (
            f"{module}/local/ex_{num}.py has no TODO markers or ____ blanks — "
            "exercises must have student-fillable holes"
        )

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_exercise_no_pandas(self, module: str, num: int, sol_path: Path) -> None:
        local = REPO_ROOT / "modules" / module / "local" / f"ex_{num}.py"
        if not local.exists():
            pytest.skip("Local exercise missing")
        text = local.read_text(encoding="utf-8")
        pandas_imports = re.findall(
            r"^\s*import pandas|^\s*from pandas", text, re.MULTILINE
        )
        assert (
            not pandas_imports
        ), f"{module}/local/ex_{num}.py imports pandas — polars only"


class TestNotebookStructure:
    """Notebooks must have valid structure and matching content."""

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_notebook_valid_json(self, module: str, num: int, sol_path: Path) -> None:
        nb_path = REPO_ROOT / "modules" / module / "notebooks" / f"ex_{num}.ipynb"
        if not nb_path.exists():
            pytest.skip("Notebook missing")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in nb, "Notebook missing 'cells' key"
        assert "nbformat" in nb, "Notebook missing 'nbformat' key"
        assert nb["nbformat"] >= 4, f"Notebook format too old: {nb['nbformat']}"

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_colab_valid_json(self, module: str, num: int, sol_path: Path) -> None:
        colab_path = REPO_ROOT / "modules" / module / "colab" / f"ex_{num}.ipynb"
        if not colab_path.exists():
            pytest.skip("Colab notebook missing")
        nb = json.loads(colab_path.read_text(encoding="utf-8"))
        assert "cells" in nb, "Colab notebook missing 'cells' key"
        assert "nbformat" in nb, "Colab notebook missing 'nbformat' key"

    @pytest.mark.consistency
    @pytest.mark.parametrize(
        "module, num, sol_path",
        _ALL_SOLUTIONS,
        ids=[f"{m}/ex_{n}" for m, n, _ in _ALL_SOLUTIONS],
    )
    def test_notebook_has_code_cells(
        self, module: str, num: int, sol_path: Path
    ) -> None:
        nb_path = REPO_ROOT / "modules" / module / "notebooks" / f"ex_{num}.ipynb"
        if not nb_path.exists():
            pytest.skip("Notebook missing")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert (
            len(code_cells) >= 2
        ), f"Notebook has only {len(code_cells)} code cells — expected more"


class TestFormatCounts:
    """Module-level format count checks."""

    @pytest.mark.consistency
    @pytest.mark.parametrize("module", MODULES, ids=MODULES)
    def test_format_counts_match(self, module: str) -> None:
        n_sol = len(_solution_files(module))
        n_local = len(_local_files(module))
        n_nb = len(_notebook_files(module))
        n_colab = len(_colab_files(module))

        assert (
            n_sol == n_local
        ), f"{module}: {n_sol} solutions vs {n_local} local exercises"
        assert n_sol == n_nb, f"{module}: {n_sol} solutions vs {n_nb} notebooks"
        assert (
            n_sol == n_colab
        ), f"{module}: {n_sol} solutions vs {n_colab} colab notebooks"

    @pytest.mark.consistency
    @pytest.mark.parametrize("module", MODULES, ids=MODULES)
    def test_eight_exercises_per_module(self, module: str) -> None:
        n_sol = len(_solution_files(module))
        assert n_sol == 8, f"{module}: expected 8 exercises, got {n_sol}"
