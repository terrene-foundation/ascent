# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT01 — Exercise 6: Data Visualization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Produce interactive EDA charts with Plotly and ModelVisualizer
#   — learning when to reach for general-purpose Plotly (histograms, scatter,
#   heatmaps) vs the Kailash engine (training curves, metric comparisons).
#
# TASKS:
#   1. Understand Plotly Express vs ModelVisualizer
#   2. Visualise the HDB price distribution with a histogram
#   3. Explore price vs area with a scatter plot
#   4. Compare median prices across districts with a bar chart
#   5. Show correlation patterns with a heatmap
#   6. Plot price trends over time with ModelVisualizer.training_history
#   7. Export all figures as standalone HTML files
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Prepare derived columns used across all charts
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Understand Plotly Express vs ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

# Plotly Express (px) is a general-purpose charting library.
# It covers every chart type: histograms, scatter, bar, heatmap, line, etc.
# Every method returns a Figure object that you can:
#   - Display inline in Jupyter: fig.show()
#   - Save as standalone HTML:   fig.write_html("filename.html")
#   - Further customise:         fig.update_layout(title="...")
#
# ModelVisualizer is a Kailash engine for ML-specific charts:
#   viz.training_history()   — line chart for training/validation metrics
#   viz.metric_comparison()  — grouped bar chart for model comparisons
#   viz.confusion_matrix()   — heatmap for classification results
#   viz.feature_importance() — bar chart of model feature importances
#
# Rule of thumb: Plotly for EDA, ModelVisualizer for ML evaluation.

viz = ModelVisualizer()


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Histogram — price distribution
# ══════════════════════════════════════════════════════════════════════

# Histograms reveal the shape of a distribution:
# - Where is the peak? (modal price)
# - Is it symmetric or skewed? (long right tail = many cheap, few expensive)
# - Are there multiple peaks? (different market segments)

fig_hist = px.histogram(
    hdb.to_pandas(),  # Framework boundary: Plotly Express requires pandas — not student code
    x="resale_price",
    nbins=80,
    title="HDB Resale Price Distribution",
    labels={"resale_price": "Resale Price (S$)"},
)
fig_hist.update_layout(yaxis_title="Number of Transactions")
fig_hist.write_html("ex6_price_histogram.html")
print("Saved: ex6_price_histogram.html")

# Also show price per sqm distribution — normalises for flat size
fig_sqm = px.histogram(
    hdb.drop_nulls(
        "price_per_sqm"
    ).to_pandas(),  # Framework boundary: Plotly Express requires pandas — not student code
    x="price_per_sqm",
    nbins=80,
    title="HDB Price per sqm Distribution",
    labels={"price_per_sqm": "Price per sqm (S$)"},
)
fig_sqm.write_html("ex6_price_per_sqm_histogram.html")
print("Saved: ex6_price_per_sqm_histogram.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Scatter plot — price vs floor area
# ══════════════════════════════════════════════════════════════════════

# Scatter plots reveal relationships between two numeric variables.
# A positive slope here means larger flats cost more — but by how much?
# Outliers stand out as isolated points far from the main cluster.

# Sample for plotting speed — a scatter of 150k points is unreadable
hdb_sample = hdb.sample(n=min(5_000, hdb.height), seed=42)

fig_scatter = px.scatter(
    hdb_sample.to_pandas(),  # Framework boundary: Plotly Express requires pandas — not student code
    x="floor_area_sqm",
    y="resale_price",
    title="HDB Resale Price vs Floor Area",
    labels={"floor_area_sqm": "Floor Area (sqm)", "resale_price": "Resale Price (S$)"},
    opacity=0.4,
)
fig_scatter.write_html("ex6_price_vs_area_scatter.html")
print("Saved: ex6_price_vs_area_scatter.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bar chart — median price by district
# ══════════════════════════════════════════════════════════════════════

# Bar charts compare a single metric across categories.

# Aggregate: one row per district, sorted by median price
district_prices = (
    hdb.group_by("town")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.len().alias("transaction_count"),
    )
    .sort("median_price", descending=True)
)

fig_bar = px.bar(
    district_prices.to_pandas(),  # Framework boundary: Plotly Express requires pandas — not student code
    x="median_price",
    y="town",
    orientation="h",
    title="Median HDB Resale Price by Town",
    labels={"median_price": "Median Resale Price (S$)", "town": "Town"},
)
fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
fig_bar.write_html("ex6_median_price_by_town.html")
print("Saved: ex6_median_price_by_town.html")

# Also chart transaction volume — a different story from price
fig_volume = px.bar(
    district_prices.sort(
        "transaction_count", descending=True
    ).to_pandas(),  # Framework boundary: Plotly Express requires pandas — not student code
    x="transaction_count",
    y="town",
    orientation="h",
    title="HDB Transaction Volume by Town",
    labels={"transaction_count": "Number of Transactions", "town": "Town"},
)
fig_volume.update_layout(yaxis={"categoryorder": "total ascending"})
fig_volume.write_html("ex6_volume_by_town.html")
print("Saved: ex6_volume_by_town.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Heatmap — correlation between numeric features
# ══════════════════════════════════════════════════════════════════════

# A correlation heatmap shows whether pairs of variables move together.
# +1.0 = perfectly positively correlated (bigger X → bigger Y)
# -1.0 = perfectly negatively correlated (bigger X → smaller Y)
#  0.0 = no linear relationship

# Build the correlation matrix from numeric columns
numeric_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "year"]
hdb_numeric = hdb.select(numeric_cols).drop_nulls()

# Compute Pearson correlations using Polars
corr_data: list[list[float]] = []
for col_a in numeric_cols:
    row = []
    for col_b in numeric_cols:
        corr = hdb_numeric.select(pl.corr(col_a, col_b)).item()
        row.append(round(corr, 3))
    corr_data.append(row)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=corr_data,
        x=numeric_cols,
        y=numeric_cols,
        text=[[f"{v:.3f}" for v in row] for row in corr_data],
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
)
fig_heatmap.update_layout(
    title="Pearson Correlation Matrix — HDB Features",
)
fig_heatmap.write_html("ex6_correlation_heatmap.html")
print("Saved: ex6_correlation_heatmap.html")

# Print the correlation values as text too
print("\n=== Pearson Correlations ===")
header = f"{'':>20}" + "".join(f"{c:>16}" for c in numeric_cols)
print(header)
for col_a, row in zip(numeric_cols, corr_data):
    row_str = f"{col_a:>20}" + "".join(f"{v:>16.3f}" for v in row)
    print(row_str)


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Line chart — price trends over time (ModelVisualizer)
# ══════════════════════════════════════════════════════════════════════

# ModelVisualizer.training_history() is designed for ML training curves
# but works perfectly for any time series: pass a dict of {series_name: [values]}.

# Annual median price for the top 5 most-transacted towns
top_5_towns = (
    district_prices.sort("transaction_count", descending=True)["town"].head(5).to_list()
)

annual_prices = (
    hdb.filter(pl.col("town").is_in(top_5_towns))
    .group_by("year", "town")
    .agg(pl.col("resale_price").median().alias("median_price"))
    .sort("year")
)

# training_history() expects a dict of {series_name: [values]}
years = sorted(annual_prices["year"].unique().to_list())

price_series: dict[str, list[float]] = {}
for town in top_5_towns:
    town_data = annual_prices.filter(pl.col("town") == town).sort("year")
    # Align to full year range — fill missing years with 0
    town_prices_by_year = dict(
        zip(town_data["year"].to_list(), town_data["median_price"].to_list())
    )
    price_series[town] = [town_prices_by_year.get(y, 0.0) for y in years]

fig_line = viz.training_history(
    metrics=price_series,
    x_label="Year",
)
fig_line.update_layout(
    title="Annual Median HDB Price — Top 5 Towns",
    yaxis_title="Median Resale Price (S$)",
)
fig_line.write_html("ex6_price_trends.html")
print("Saved: ex6_price_trends.html")

# National trend (all towns)
national_annual = (
    hdb.group_by("year")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("year")
)

national_series = {
    "National Median Price": national_annual["median_price"].to_list(),
}
fig_national = viz.training_history(
    metrics=national_series,
    x_label="Year",
)
fig_national.update_layout(
    title="Singapore HDB National Price Trend",
    yaxis_title="Median Resale Price (S$)",
)
fig_national.write_html("ex6_national_price_trend.html")
print("Saved: ex6_national_price_trend.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Summary of all outputs
# ══════════════════════════════════════════════════════════════════════

outputs = [
    ("ex6_price_histogram.html", "Price distribution — shape of the market"),
    ("ex6_price_per_sqm_histogram.html", "Price/sqm distribution — normalised"),
    ("ex6_price_vs_area_scatter.html", "Price vs area — does size explain price?"),
    ("ex6_median_price_by_town.html", "Median price by town — where is expensive?"),
    ("ex6_volume_by_town.html", "Transaction volume by town — where is active?"),
    ("ex6_correlation_heatmap.html", "Correlation matrix — feature relationships"),
    ("ex6_price_trends.html", "Price trends — top 5 towns over time"),
    ("ex6_national_price_trend.html", "National price trend — macro view"),
]

print(f"\n{'=' * 60}")
print(f"  VISUALISATION OUTPUTS")
print(f"{'=' * 60}")
for filename, description in outputs:
    print(f"  {filename}")
    print(f"    → {description}")
print(f"{'=' * 60}")

print("\n✓ Exercise 6 complete — interactive EDA charts with Plotly + ModelVisualizer")
