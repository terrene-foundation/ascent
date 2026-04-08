# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT01 — Exercise 5: Window Functions and Trends
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use Polars window functions and lazy evaluation to compute
#   rolling averages, year-over-year changes, and rankings without
#   leaving the DataFrame — the foundation of time-series feature engineering.
#
# TASKS:
#   1. Build monthly price series per town using group_by + sort
#   2. Apply rolling_mean() with over() for per-group rolling windows
#   3. Compute year-over-year price change with shift() over partitions
#   4. Rank towns by recent price growth using lazy frames and collect()
#   5. Identify trend leaders and laggards across the Singapore market
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Parse dates and add derived columns used throughout
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build monthly price series per town
# ══════════════════════════════════════════════════════════════════════

# Before applying window functions we need a monthly aggregate.
# Each row in this DataFrame represents one (town, month) combination.
# Sorting by town then date is critical — rolling windows depend on order.

monthly_prices = (
    # TODO: Group by both "town" and "transaction_date"
    hdb.group_by(____, "transaction_date").agg(
        # TODO: Compute the median of price_per_sqm
        pl.col("price_per_sqm").____().alias("median_price_sqm"),
        pl.col("resale_price").median().alias("median_resale_price"),
        pl.len().alias("transaction_count"),
    )
    # TODO: Sort by "town" first, then "transaction_date" (window order matters)
    .sort(____, ____)
)

print(f"\n=== Monthly Price Series ===")
print(f"Shape: {monthly_prices.shape}  (one row per town per month)")
print(monthly_prices.filter(pl.col("town") == "BISHAN").head(6))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Rolling average with over() — per-group window functions
# ══════════════════════════════════════════════════════════════════════

# A window function computes a value for each row using a sliding window
# of surrounding rows — without collapsing the DataFrame like group_by does.
#
# rolling_mean(window_size=12) computes a 12-row moving average.
# .over("town") partitions the computation by town, so the window
# never crosses town boundaries. Each town gets its own independent window.
#
# Why 12 months? Seasonal smoothing: Singapore's rainy/dry season cycle
# affects transaction volume, so a 12-month window removes that noise.

monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    # TODO: Apply a 12-month rolling mean (window_size=12)
    .rolling_mean(window_size=____)
    .over("town")
    .alias("rolling_12m_price_sqm"),
)

# A shorter 3-month window for a more reactive signal
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=3)
    # TODO: Partition the window by "town" so it doesn't cross town boundaries
    .over(____)
    .alias("rolling_3m_price_sqm"),
)

print(f"\n=== Rolling Averages — Bishan (last 18 months) ===")
bishan = monthly_prices.filter(pl.col("town") == "BISHAN").tail(18)
print(
    bishan.select(
        "transaction_date",
        "median_price_sqm",
        "rolling_3m_price_sqm",
        "rolling_12m_price_sqm",
    )
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Year-over-year price change with shift()
# ══════════════════════════════════════════════════════════════════════

# shift(n) moves every value n positions forward, filling the first n
# rows with null. Combined with .over("town"), each town shifts independently.
#
# YoY change = (current - 12_months_ago) / 12_months_ago * 100
#
# shift(12) gives us the value from exactly 12 months earlier for each town.
# The first 12 rows per town will be null — that is correct and expected.

monthly_prices = monthly_prices.with_columns(
    # TODO: Shift median_price_sqm by 12 positions, partitioned by town
    pl.col("median_price_sqm")
    .shift(____)
    .over("town")
    .alias("price_sqm_12m_ago"),
)

monthly_prices = monthly_prices.with_columns(
    # TODO: YoY % = (current - 12_months_ago) / 12_months_ago * 100
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_12m_ago"))
        / pl.col("price_sqm_12m_ago")
        * ____
    ).alias("yoy_price_change_pct"),
)

print(f"\n=== Year-over-Year Price Change — Bishan (last 24 months) ===")
print(
    monthly_prices.filter(pl.col("town") == "BISHAN")
    .tail(24)
    .select(
        "transaction_date",
        "median_price_sqm",
        "price_sqm_12m_ago",
        "yoy_price_change_pct",
    )
)

# Market-wide YoY: which months had the strongest national price growth?
national_monthly = (
    # TODO: Group hdb by "transaction_date" only (national, not per-town)
    hdb.group_by(____)
    .agg(pl.col("price_per_sqm").median().alias("national_median_sqm"))
    .sort("transaction_date")
    .with_columns(
        # TODO: Shift the national series by 12 months
        pl.col("national_median_sqm")
        .shift(____)
        .alias("national_sqm_12m_ago"),
    )
    .with_columns(
        (
            (pl.col("national_median_sqm") - pl.col("national_sqm_12m_ago"))
            / pl.col("national_sqm_12m_ago")
            * 100
        ).alias("national_yoy_pct"),
    )
)

print(f"\n=== National YoY Price Change (top 10 months) ===")
print(
    national_monthly.drop_nulls("national_yoy_pct")
    .sort("national_yoy_pct", descending=True)
    .select("transaction_date", "national_median_sqm", "national_yoy_pct")
    .head(10)
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Lazy frames — defer execution until collect()
# ══════════════════════════════════════════════════════════════════════

# .lazy() converts a DataFrame into a LazyFrame.
# Nothing executes immediately — Polars builds a query plan instead.
# .collect() triggers execution. Polars optimises the plan before running:
#   - Predicate pushdown: filters are applied as early as possible
#   - Projection pushdown: unused columns are dropped before reading
#   - Query fusion: adjacent operations are combined
#
# For large datasets, lazy evaluation is significantly faster because
# Polars avoids materialising intermediate results.

# Find towns with consistently high YoY growth over the last 3 years
# Using lazy evaluation for the multi-step aggregation pipeline

recent_yoy = (
    # TODO: Convert monthly_prices to a LazyFrame by calling .lazy()
    monthly_prices.____()
    # Filter to last 3 years of data — lazy: no data moves yet
    # TODO: Filter to dates on or after Jan 1, 2021
    .filter(pl.col("transaction_date") >= pl.date(____, 1, 1))
    # Drop rows where YoY is null (first 12 months per town)
    .drop_nulls("yoy_price_change_pct")
    # Aggregate per town
    .group_by("town").agg(
        # TODO: Compute the mean YoY growth per town
        pl.col("yoy_price_change_pct").____().alias("mean_yoy_pct"),
        pl.col("yoy_price_change_pct").std().alias("std_yoy_pct"),
        pl.col("yoy_price_change_pct").max().alias("peak_yoy_pct"),
        pl.col("yoy_price_change_pct").min().alias("trough_yoy_pct"),
        pl.len().alias("months_of_data"),
    )
    # Sort by mean YoY descending
    .sort("mean_yoy_pct", descending=True)
    # TODO: Trigger execution by calling .collect() on the lazy plan
    .____()
)

print(f"\n=== Town YoY Growth Rankings (2021–present) ===")
print(f"Towns analysed: {recent_yoy.height}")
print(recent_yoy.head(10))


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Trend leaders and laggards
# ══════════════════════════════════════════════════════════════════════

# A rank window function assigns a position to each row within its partition.
# pl.col("mean_yoy_pct").rank() ranks all towns by growth rate.
# This is different from sort() — rank adds a column without reordering rows.

recent_yoy = recent_yoy.with_columns(
    # TODO: Rank by mean_yoy_pct with method="ordinal" and descending=True
    pl.col("mean_yoy_pct")
    .rank(method="ordinal", descending=____)
    .alias("growth_rank"),
)

# Classify towns as trend leaders, followers, or laggards
mean_growth = recent_yoy["mean_yoy_pct"].mean()
std_growth = recent_yoy["mean_yoy_pct"].std()

recent_yoy = recent_yoy.with_columns(
    pl.when(pl.col("mean_yoy_pct") > mean_growth + std_growth)
    # TODO: Mark high-growth towns as "leader"
    .then(pl.lit(____))
    .when(pl.col("mean_yoy_pct") < mean_growth - std_growth)
    .then(pl.lit("laggard"))
    .otherwise(pl.lit("follower"))
    .alias("trend_category"),
)

print(f"\n=== Trend Classification ===")
print(f"Mean YoY growth (all towns): {mean_growth:.2f}%")
print(f"Std dev: {std_growth:.2f}%")

category_counts = (
    recent_yoy.group_by("trend_category")
    .agg(pl.len().alias("count"), pl.col("mean_yoy_pct").mean().alias("avg_yoy"))
    .sort("avg_yoy", descending=True)
)
print(category_counts)

print(f"\n=== Trend Leaders (fastest-growing towns) ===")
print(
    recent_yoy.filter(pl.col("trend_category") == "leader").select(
        "town", "mean_yoy_pct", "peak_yoy_pct", "growth_rank"
    )
)

print(f"\n=== Trend Laggards (slowest-growing towns) ===")
print(
    recent_yoy.filter(pl.col("trend_category") == "laggard").select(
        "town", "mean_yoy_pct", "trough_yoy_pct", "growth_rank"
    )
)

print("\n✓ Exercise 5 complete — window functions, rolling averages, and YoY trends")
