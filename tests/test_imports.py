# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Verify all Kailash SDK packages import correctly at expected versions.

Quick smoke test that the venv is correctly configured before running
the full solution suite.

Run: uv run pytest tests/test_imports.py -v
"""

from __future__ import annotations

import importlib.metadata as md

import pytest

# (package_name, min_version, import_name)
KAILASH_PACKAGES: list[tuple[str, str, str]] = [
    ("kailash", "2.5.0", "kailash"),
    ("kailash-ml", "0.7.0", "kailash_ml"),
    ("kailash-dataflow", "1.8.0", "dataflow"),
    ("kailash-nexus", "1.9.0", "nexus"),
    ("kailash-kaizen", "2.5.0", "kaizen"),
    ("kaizen-agents", "0.7.0", "kaizen_agents"),
    ("kailash-pact", "0.8.0", "pact"),
    ("kailash-align", "0.3.1", "kailash_align"),
]

CORE_DEPS: list[tuple[str, str]] = [
    ("polars", "polars"),
    ("plotly", "plotly"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
    ("imbalanced-learn", "imblearn"),
    ("lightgbm", "lightgbm"),
]


def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3])


class TestKailashImports:
    """Verify all Kailash packages are installed and importable."""

    @pytest.mark.parametrize(
        "pkg_name, min_ver, import_name",
        KAILASH_PACKAGES,
        ids=[p[0] for p in KAILASH_PACKAGES],
    )
    def test_kailash_installed(
        self, pkg_name: str, min_ver: str, import_name: str
    ) -> None:
        version = md.version(pkg_name)
        assert _parse_version(version) >= _parse_version(
            min_ver
        ), f"{pkg_name} {version} < required {min_ver}"

    @pytest.mark.parametrize(
        "pkg_name, min_ver, import_name",
        KAILASH_PACKAGES,
        ids=[p[0] for p in KAILASH_PACKAGES],
    )
    def test_kailash_importable(
        self, pkg_name: str, min_ver: str, import_name: str
    ) -> None:
        try:
            __import__(import_name)
        except ImportError:
            pytest.fail(f"Cannot import {import_name} (package: {pkg_name})")


class TestCoreDeps:
    """Verify core third-party dependencies are importable."""

    @pytest.mark.parametrize(
        "pkg_name, import_name",
        CORE_DEPS,
        ids=[p[0] for p in CORE_DEPS],
    )
    def test_dep_importable(self, pkg_name: str, import_name: str) -> None:
        try:
            __import__(import_name)
        except ImportError:
            pytest.fail(f"Cannot import {import_name} (package: {pkg_name})")


class TestDataLoader:
    """Verify the shared data loader works."""

    def test_loader_importable(self) -> None:
        from shared import ASCENTDataLoader

        loader = ASCENTDataLoader()
        assert loader is not None

    def test_loader_loads_dataset(self) -> None:
        from shared import ASCENTDataLoader

        loader = ASCENTDataLoader()
        df = loader.load("ascent01", "sg_weather.csv")
        assert len(df) > 0
        assert len(df.columns) > 0
