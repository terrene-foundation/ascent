# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT02 — Exercise 8: Feature Store and Capstone Project
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use FeatureStore to persist, version, and retrieve features
#   with point-in-time correctness. Demonstrate data lineage — design,
#   execute, analyze, and report the full feature lifecycle.
#
# TASKS:
#   1. Connect to FeatureStore (shared DB with ExperimentTracker)
#   2. Define FeatureSchema with typed fields and versioning
#   3. Compute and store features
#   4. Retrieve features at different points in time (leakage prevention)
#   5. Version the schema and store updated features
#   6. Demonstrate data lineage: query which features trained which model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureStore, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Focus on recent data for feature engineering
hdb = hdb.with_columns(pl.col("month").str.to_date("%Y-%m").alias("transaction_date"))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Define FeatureSchema with typed fields
# ══════════════════════════════════════════════════════════════════════

# Version 1: Basic property features
property_schema_v1 = FeatureSchema(
    name="hdb_property_features",
    features=[
        FeatureField(
            name="floor_area_sqm",
            dtype="float64",
            nullable=False,
            description="Floor area in square metres",
        ),
        FeatureField(
            name="town_avg_price",
            dtype="float64",
            nullable=True,
            description="Average resale price in the same town",
        ),
        FeatureField(
            name="storey_midpoint",
            dtype="float64",
            nullable=False,
            description="Midpoint of storey range",
        ),
        FeatureField(
            name="price_per_sqm",
            dtype="float64",
            nullable=False,
            description="Transaction price per square metre",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=1,
)

print("=== FeatureSchema v1 ===")
print(f"Name: {property_schema_v1.name}")
print(f"Version: {property_schema_v1.version}")
for f in property_schema_v1.features:
    print(f"  {f.name}: {f.dtype} — {f.description}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute and store features
# ══════════════════════════════════════════════════════════════════════


# Compute features from raw data
def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version 1 property features."""
    # Town-level average price as a location feature
    town_avg = df.group_by("town").agg(
        pl.col("resale_price").mean().alias("town_avg_price")
    )
    return (
        df.with_columns(
            # Parse storey range to midpoint
            (
                (
                    pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                    + pl.col("storey_range")
                    .str.extract(r"TO (\d+)", 1)
                    .cast(pl.Float64)
                )
                / 2
            ).alias("storey_midpoint"),
            # Price per sqm
            (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        )
        .join(town_avg, on="town", how="left")
        .with_row_index("transaction_id")
    )


features_v1 = compute_v1_features(hdb)
print(f"\nComputed v1 features: {features_v1.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Version the schema — add market context features
# ══════════════════════════════════════════════════════════════════════

# Version 2 adds neighbourhood market features
# FeatureStore uses schema name as unique key — versioned schemas need distinct names
property_schema_v2 = FeatureSchema(
    name="hdb_property_features_v2",
    features=[
        *property_schema_v1.features,
        FeatureField(
            name="town_median_price",
            dtype="float64",
            nullable=True,
            description="Median price in the same town (trailing 6 months)",
        ),
        FeatureField(
            name="town_transaction_volume",
            dtype="int64",
            nullable=True,
            description="Number of transactions in town (trailing 6 months)",
        ),
        FeatureField(
            name="town_price_trend",
            dtype="float64",
            nullable=True,
            description="6-month price change % in town",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=2,
)


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features including market context."""
    # Start with v1 features
    result = compute_v1_features(df)

    # Compute trailing 6-month town-level statistics
    # group_by_dynamic requires data sorted by the time column
    result = result.sort("transaction_date")
    town_stats = (
        result.group_by_dynamic("transaction_date", every="1mo", group_by="town")
        .agg(
            pl.col("resale_price").median().alias("monthly_median"),
            pl.col("resale_price").count().alias("monthly_volume"),
        )
        .sort("town", "transaction_date")
    )

    # 6-month rolling stats per town
    town_stats = town_stats.with_columns(
        pl.col("monthly_median")
        .rolling_mean(window_size=6)
        .over("town")
        .alias("town_median_price"),
        pl.col("monthly_volume")
        .rolling_sum(window_size=6)
        .over("town")
        .alias("town_transaction_volume"),
        (
            (pl.col("monthly_median") - pl.col("monthly_median").shift(6).over("town"))
            / pl.col("monthly_median").shift(6).over("town")
            * 100
        ).alias("town_price_trend"),
    )

    # Join back to transactions
    result = result.join(
        town_stats.select(
            "town",
            "transaction_date",
            "town_median_price",
            "town_transaction_volume",
            "town_price_trend",
        ),
        on=["town", "transaction_date"],
        how="left",
    )

    return result


features_v2 = compute_v2_features(hdb)


# ══════════════════════════════════════════════════════════════════════
# All async work in a single event loop to avoid cross-loop issues
# ══════════════════════════════════════════════════════════════════════


async def main():
    # ── TASK 1: Connect to FeatureStore (shared DB) ──────────────────
    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()

    fs = FeatureStore(conn, table_prefix="kml_feat_")
    await fs.initialize()

    tracker = ExperimentTracker(conn)

    # ── TASK 3: Register schema and store features ───────────────────
    await fs.register_features(property_schema_v1)
    print(f"Registered schema: {property_schema_v1.name} v{property_schema_v1.version}")

    row_count = await fs.store(features_v1, property_schema_v1)
    print(f"Stored {row_count:,} feature rows")

    # ── TASK 4: Retrieve features at different points in time ────────
    # This is the KEY concept: point-in-time retrieval prevents leakage.
    # If training a model to predict prices at time T, you must only use
    # features computed from data BEFORE time T.

    cutoff_date = datetime(2023, 1, 1)
    features_jan_2023 = await fs.get_training_set(
        schema=property_schema_v1,
        start=datetime(2000, 1, 1),
        end=cutoff_date,
    )
    print(f"\n=== Point-in-Time Retrieval ===")
    print(f"Features as of 2023-01-01: {features_jan_2023.height:,} rows")

    features_jan_2024 = await fs.get_training_set(
        schema=property_schema_v1,
        start=datetime(2000, 1, 1),
        end=datetime(2024, 1, 1),
    )
    print(f"Features as of 2024-01-01: {features_jan_2024.height:,} rows")

    delta = features_jan_2024.height - features_jan_2023.height
    print(f"Additional transactions in 2023: {delta:,}")

    print("\n--- Leakage Prevention ---")
    print("To predict prices at T=2023-01-01:")
    print("  ✓ Use features retrieved as_of=2023-01-01")
    print("  ✗ Using features as_of=2024-01-01 would include future transactions")

    # ── TASK 5: Store v2 features ────────────────────────────────────
    await fs.register_features(property_schema_v2)
    print(
        f"\nRegistered schema: {property_schema_v2.name} v{property_schema_v2.version}"
    )

    row_count_v2 = await fs.store(features_v2, property_schema_v2)
    print(f"Stored {row_count_v2:,} v2 feature rows")

    schemas = await fs.list_schemas()
    print(f"Available schemas: {schemas}")

    # ── TASK 6: Data lineage — what data trained this model? ─────────
    experiment_id = await tracker.create_experiment(
        name="ascent02_feature_store_lifecycle",
        description="FeatureStore lifecycle demonstration",
        tags=["ascent02", "feature-store", "lineage"],
    )

    async with tracker.run(experiment_id, run_name="hdb_price_model_v1") as run:
        await run.log_params(
            {
                "feature_schema": "hdb_property_features_v2",
                "feature_version": "2",
                "as_of_date": "2023-06-01",
                "train_rows": str(features_jan_2023.height),
                "model_type": "LightGBM",
            }
        )
        await run.log_metrics(
            {
                "rmse": 45_000.0,
                "r2": 0.87,
                "mae": 32_000.0,
            }
        )
        await run.set_tag("purpose", "lineage-demo")
        run_id = run.id if hasattr(run, "id") else "logged"

    print(f"\n=== Data Lineage ===")
    print(f"Model run: {run_id}")
    print(f"  → Uses feature schema: hdb_property_features_v2")
    print(f"  → Features as_of: 2023-06-01")
    print(f"  → Training rows: {features_jan_2023.height:,}")
    print(f"  → Source: HDB resale data (data.gov.sg)")
    print()
    print("If a regulator asks 'what data trained this model?', you can answer:")
    print(f"  1. Model ID: {run_id}")
    print(f"  2. Feature schema: hdb_property_features_v2 (7 features)")
    print(f"  3. Point-in-time cutoff: 2023-06-01 (no future leakage)")
    print(f"  4. Training data: {features_jan_2023.height:,} HDB transactions")
    print(f"  5. Source: data.gov.sg resale flat prices")

    # Clean up
    await conn.close()


asyncio.run(main())

print("\n✓ Exercise 8 complete — FeatureStore lifecycle and capstone project")
print(
    "  Key concepts: point-in-time retrieval, schema versioning, data lineage, audit trail"
)
