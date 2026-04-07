# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Verify all datasets required by exercises exist and are loadable.

Checks:
- Every expected dataset file exists on disk
- Each file loads into a non-empty polars DataFrame
- Schema has expected minimum column count
- No fully-null columns (data generation sanity)

Run: uv run pytest tests/test_datasets.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.conftest import REPO_ROOT, discover_datasets

_ALL_DATASETS = discover_datasets()

# Expected dataset inventory: (module, filename, min_rows, min_cols)
EXPECTED_DATASETS: list[tuple[str, str, int, int]] = [
    # ascent01
    ("ascent01", "sg_weather.csv", 100, 3),
    ("ascent01", "hdb_resale.parquet", 10_000, 5),
    ("ascent01", "sg_taxi_trips.parquet", 10_000, 3),
    ("ascent01", "sg_cpi.csv", 50, 2),
    ("ascent01", "sg_employment.csv", 20, 2),
    ("ascent01", "sg_fx_rates.csv", 500, 2),
    ("ascent01", "economic_indicators.csv", 100, 3),
    ("ascent01", "ecommerce_ab_test.csv", 100, 3),
    # ascent02
    ("ascent02", "experiment_data.parquet", 10_000, 3),
    ("ascent02", "economic_indicators.csv", 50, 3),
    ("ascent02", "ecommerce_experiment.parquet", 100, 3),
    ("ascent02", "icu_patients.parquet", 50, 3),
    ("ascent02", "icu_admissions.parquet", 50, 3),
    ("ascent02", "icu_vitals.parquet", 50, 3),
    ("ascent02", "icu_labs.parquet", 50, 3),
    ("ascent02", "icu_medications.parquet", 50, 3),
    ("ascent02", "sg_cooling_measures.csv", 5, 2),
    # ascent03
    ("ascent03", "sg_credit_scoring.parquet", 10_000, 10),
    # ascent04
    ("ascent04", "ecommerce_customers.parquet", 5_000, 3),
    ("ascent04", "credit_card_fraud.parquet", 10_000, 5),
    ("ascent04", "sg_news_corpus.parquet", 500, 2),
    # ascent05
    ("ascent05", "documents.parquet", 100, 2),
    # ascent06
    ("ascent06", "preference_pairs.parquet", 100, 2),
    ("ascent06", "sg_domain_qa.parquet", 100, 2),
    # ascent07
    ("ascent07", "mnist_sample.parquet", 1_000, 2),
    ("ascent07", "fashion_mnist_sample.parquet", 1_000, 2),
    ("ascent07", "synthetic_spirals.parquet", 100, 2),
    ("ascent07", "hdb_resale_sample.parquet", 500, 2),
    # ascent08
    ("ascent08", "sg_news_articles.parquet", 500, 2),
    ("ascent08", "sg_parliament_speeches.parquet", 100, 2),
    ("ascent08", "sg_product_reviews.parquet", 500, 2),
    # ascent09
    ("ascent09", "sg_company_reports.parquet", 50, 2),
    ("ascent09", "sg_regulations.parquet", 50, 2),
    # ascent10
    ("ascent10", "inventory_demand.parquet", 50, 2),
    ("ascent10", "preference_pairs.parquet", 100, 2),
    ("ascent10", "sg_company_reports.parquet", 50, 2),
    ("ascent10", "sg_domain_qa.parquet", 100, 2),
]


def _read(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    return pl.read_parquet(path)


class TestDatasetInventory:
    """Verify all expected datasets exist on disk."""

    @pytest.mark.dataset
    @pytest.mark.parametrize(
        "module, filename, min_rows, min_cols",
        EXPECTED_DATASETS,
        ids=[f"{m}/{f}" for m, f, _, _ in EXPECTED_DATASETS],
    )
    def test_dataset_exists(
        self, module: str, filename: str, min_rows: int, min_cols: int
    ) -> None:
        path = REPO_ROOT / "data" / module / filename
        assert path.exists(), f"Missing dataset: data/{module}/{filename}"

    @pytest.mark.dataset
    @pytest.mark.parametrize(
        "module, filename, min_rows, min_cols",
        EXPECTED_DATASETS,
        ids=[f"{m}/{f}" for m, f, _, _ in EXPECTED_DATASETS],
    )
    def test_dataset_loadable(
        self, module: str, filename: str, min_rows: int, min_cols: int
    ) -> None:
        path = REPO_ROOT / "data" / module / filename
        if not path.exists():
            pytest.skip(f"Dataset missing: {module}/{filename}")
        df = _read(path)
        assert (
            len(df) >= min_rows
        ), f"{module}/{filename}: expected >={min_rows} rows, got {len(df)}"
        assert (
            len(df.columns) >= min_cols
        ), f"{module}/{filename}: expected >={min_cols} columns, got {len(df.columns)}"

    @pytest.mark.dataset
    @pytest.mark.parametrize(
        "module, filename, min_rows, min_cols",
        EXPECTED_DATASETS,
        ids=[f"{m}/{f}" for m, f, _, _ in EXPECTED_DATASETS],
    )
    def test_dataset_no_fully_null_columns(
        self, module: str, filename: str, min_rows: int, min_cols: int
    ) -> None:
        path = REPO_ROOT / "data" / module / filename
        if not path.exists():
            pytest.skip(f"Dataset missing: {module}/{filename}")
        df = _read(path)
        null_cols = [c for c in df.columns if df[c].null_count() == len(df)]
        assert not null_cols, f"{module}/{filename}: fully null columns: {null_cols}"


class TestDatasetCompleteness:
    """Verify no unexpected datasets exist (catch stale/orphan files)."""

    @pytest.mark.dataset
    def test_no_orphan_datasets(self) -> None:
        expected = {(m, f) for m, f, _, _ in EXPECTED_DATASETS}
        # Also include assessment datasets
        expected.add(("ascent_assessment", "mrt_stations.parquet"))
        expected.add(("ascent_assessment", "schools.parquet"))

        actual = {(m, f) for m, f, _ in _ALL_DATASETS}
        orphans = actual - expected
        assert not orphans, (
            f"Orphan datasets not in inventory: {sorted(orphans)}\n"
            "Add to EXPECTED_DATASETS or delete the file."
        )

    @pytest.mark.dataset
    def test_expected_count(self) -> None:
        actual_count = len(_ALL_DATASETS)
        expected_count = len(EXPECTED_DATASETS) + 2  # +2 for assessment
        assert actual_count == expected_count, (
            f"Dataset count mismatch: {actual_count} on disk vs "
            f"{expected_count} expected. Update EXPECTED_DATASETS."
        )
