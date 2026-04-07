"""
ASCENT Dataset Generator
======================
Generates all synthetic training datasets for the ASCENT course.

Run with:
    python3 scripts/generate_datasets.py

Writes to data/{module}/ directories. Uses polars for all DataFrame
operations. Fixed seed (42) for reproducibility.

Intentional messiness is documented inline — students are expected
to find and fix it.

Scale (as of regeneration):
  sg_credit_scoring.parquet  — 100,000 rows, 45 features
  ecommerce_customers.parquet — 50,000 customers
  sg_taxi_trips.parquet      — 50,000 rows (parquet, not csv)
  experiment_data.parquet    — 500,000 rows
  economic_indicators.csv    — 500+ rows (monthly 2000-2024 + quarterly)
  documents.parquet          — 500 rows
  sg_domain_qa.parquet       — 1,000 rows
  preference_pairs.parquet   — 500 rows
  mrt_stations.parquet       — ~150 rows (correct, actual MRT count)
  schools.parquet            — 350 rows
  hdb_resale.parquet         — 150,000 rows (HDB resale transactions 2017-2024)
  sg_cpi.csv                 — 180 rows (monthly CPI 2010-2024)
  sg_employment.csv          — 60 rows (quarterly employment 2010-2024)
  sg_fx_rates.csv            — ~3,900 rows (daily FX rates 2010-2024)
  credit_card_fraud.parquet  — 50,000 rows (PCA-transformed, 0.5% fraud rate)
  sg_news_corpus.parquet     — 5,000 rows (Singapore-themed news articles)
  hdb_resale_sample.parquet  — 2,000 rows (floor_area vs resale_price)
  mnist_sample.parquet       — 3,000 rows (synthetic MNIST-like 28x28 pixels)
  fashion_mnist_sample.parquet — 3,000 rows (synthetic Fashion-MNIST-like)
  synthetic_spirals.parquet  — 500 rows (two interleaved spirals)
"""

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "data"

DIRS = [
    DATA_ROOT / "ascent01",
    DATA_ROOT / "ascent02",
    DATA_ROOT / "ascent03",
    DATA_ROOT / "ascent04",
    DATA_ROOT / "ascent05",
    DATA_ROOT / "ascent06",
    DATA_ROOT / "ascent07",
    DATA_ROOT / "ascent08",
    DATA_ROOT / "ascent09",
    DATA_ROOT / "ascent10",
    DATA_ROOT / "ascent_assessment",
]

RNG = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nullify(arr: np.ndarray, rate: float) -> list:
    """Replace `rate` fraction of array values with None."""
    result = arr.tolist()
    n = len(result)
    indices = RNG.choice(n, size=int(n * rate), replace=False)
    for i in indices:
        result[i] = None
    return result


def _nullify_list(lst: list, rate: float) -> list:
    """Replace `rate` fraction of list values with None."""
    result = list(lst)
    n = len(result)
    indices = RNG.choice(n, size=int(n * rate), replace=False)
    for i in indices:
        result[i] = None
    return result


# ---------------------------------------------------------------------------
# 1. economic_indicators.csv  (ascent01)
#    Target: 500+ rows — monthly 2000-2024 with quarterly aggregates mixed in
# ---------------------------------------------------------------------------


def make_economic_indicators() -> pl.DataFrame:
    """
    Singapore economic indicators 2000-2024.
    Monthly rows (25 years * 12 = 300) plus quarterly summary rows (100),
    totalling ~400 rows. A duplicate row and mixed date formats push it over 500.

    Intentional messiness:
    - Mixed date formats: monthly uses "YYYY-MM", quarterly mixes "Q1 YYYY",
      "YYYY-Q2", and plain "2003-1"
    - Missing values in inflation_rate (~8%) and trade_balance (~6%)
    - GDP outlier: 2020 Q2 crash (-13% realistic but jarring)
    - tourist_arrivals as string with commas (e.g., "1,234,567") in some rows
      and plain integer strings in others
    - Duplicate row: 2019 Q2 appears twice
    - period_type column to distinguish monthly from quarterly
    """
    # --- Quarterly baseline (25 years x 4 = 100 rows) ---
    years = list(range(2000, 2025))
    quarters = [1, 2, 3, 4]
    q_rows = [(y, q) for y in years for q in quarters]
    n_q = len(q_rows)

    gdp_base_q = np.array(
        [
            5.0,
            5.2,
            4.8,
            3.9,  # 2000
            -1.2,
            3.1,
            2.8,
            2.5,  # 2001 recession
            4.2,
            4.5,
            5.0,
            5.3,  # 2002
            4.8,
            5.1,
            5.6,
            5.9,  # 2003
            8.4,
            8.7,
            8.2,
            8.0,  # 2004 boom
            7.4,
            7.0,
            6.8,
            6.5,  # 2005
            9.0,
            8.8,
            8.4,
            8.2,  # 2006
            8.9,
            8.5,
            8.0,
            7.8,  # 2007
            6.2,
            -0.4,
            -3.5,
            -1.1,  # 2008 GFC
            -4.2,
            -1.0,
            3.2,
            4.0,  # 2009
            14.5,
            13.2,
            11.8,
            12.0,  # 2010 rebound
            6.0,
            5.8,
            5.2,
            4.8,  # 2011
            3.5,
            3.2,
            2.8,
            2.5,  # 2012
            4.0,
            4.2,
            3.8,
            3.5,  # 2013
            3.9,
            3.5,
            2.8,
            2.5,  # 2014
            2.0,
            2.2,
            1.8,
            1.5,  # 2015
            2.4,
            2.0,
            1.8,
            2.0,  # 2016
            3.5,
            3.8,
            4.0,
            3.8,  # 2017
            3.2,
            3.4,
            3.0,
            2.8,  # 2018
            0.8,
            0.6,
            0.5,
            0.5,  # 2019 slowdown
            -3.3,
            -13.3,
            -5.8,
            -2.4,  # 2020 COVID
            -0.5,
            15.2,
            7.5,
            6.0,  # 2021 rebound
            4.5,
            4.2,
            3.8,
            2.5,  # 2022
            2.1,
            1.8,
            1.5,
            2.0,  # 2023
            2.5,
            2.8,
            3.0,
            3.2,  # 2024
        ],
        dtype=float,
    )

    gdp_growth_q = gdp_base_q + RNG.normal(0, 0.3, n_q)

    unemp_base_q = np.array(
        [
            2.4,
            2.5,
            2.6,
            2.7,  # 2000
            3.5,
            3.8,
            4.0,
            3.9,  # 2001
            3.6,
            3.5,
            3.3,
            3.2,  # 2002
            3.0,
            2.9,
            2.8,
            2.7,  # 2003
            2.5,
            2.4,
            2.3,
            2.2,  # 2004
            2.5,
            2.4,
            2.3,
            2.2,  # 2005
            2.4,
            2.3,
            2.2,
            2.1,  # 2006
            1.9,
            1.8,
            1.8,
            1.9,  # 2007
            2.0,
            2.5,
            3.2,
            3.3,  # 2008
            3.4,
            3.3,
            2.8,
            2.5,  # 2009
            2.2,
            2.0,
            1.9,
            1.9,  # 2010
            2.0,
            2.0,
            2.1,
            2.0,  # 2011
            1.9,
            1.9,
            1.8,
            1.8,  # 2012
            1.9,
            1.9,
            1.8,
            1.8,  # 2013
            1.9,
            2.0,
            2.0,
            1.9,  # 2014
            1.9,
            1.9,
            2.0,
            2.0,  # 2015
            2.2,
            2.1,
            2.1,
            2.0,  # 2016
            2.2,
            2.2,
            2.1,
            2.0,  # 2017
            2.1,
            2.1,
            2.0,
            2.0,  # 2018
            2.2,
            2.3,
            2.3,
            2.2,  # 2019
            2.6,
            4.0,
            4.5,
            4.1,  # 2020 COVID spike
            3.0,
            2.5,
            2.2,
            2.1,  # 2021
            2.0,
            2.0,
            1.9,
            1.9,  # 2022
            1.9,
            1.8,
            1.9,
            1.9,  # 2023
            1.8,
            1.8,
            1.9,
            1.9,  # 2024
        ],
        dtype=float,
    )

    unemp_q = np.clip(unemp_base_q + RNG.normal(0, 0.1, n_q), 1.0, 6.0)

    inflation_base_q = RNG.uniform(0.5, 3.5, n_q)
    inflation_base_q[80:84] = RNG.uniform(4.5, 6.5, 4)  # 2020
    inflation_base_q[84:88] = RNG.uniform(3.5, 5.5, 4)  # 2021
    inflation_base_q[88:92] = RNG.uniform(5.0, 7.0, 4)  # 2022 peak
    inflation_base_q[92:96] = RNG.uniform(3.0, 5.0, 4)  # 2023 easing
    inflation_raw_q = _nullify(np.round(inflation_base_q, 2), rate=0.08)

    trade_base_q = RNG.uniform(5.0, 22.0, n_q)
    trade_raw_q = _nullify(np.round(trade_base_q, 1), rate=0.06)

    ppi_q = np.zeros(n_q)
    ppi_q[0] = 100.0
    for i in range(1, n_q):
        shock = RNG.normal(0.008, 0.025)
        if 32 <= i <= 35:
            shock -= 0.04
        if 80 <= i <= 83:
            shock -= 0.01
        if 84 <= i <= 91:
            shock += 0.02
        ppi_q[i] = ppi_q[i - 1] * (1 + shock)

    tourists_q = np.zeros(n_q)
    tourists_q[0] = 5.1
    for i in range(1, n_q):
        shock = RNG.normal(0.015, 0.04)
        if 32 <= i <= 35:
            shock -= 0.08
        if 80 <= i <= 87:
            tourists_q[i] = tourists_q[i - 1] * 0.02
            continue
        if 88 <= i <= 95:
            shock += 0.25
        tourists_q[i] = max(0.1, tourists_q[i - 1] * (1 + shock))

    arrivals_int_q = (tourists_q * 1_000_000).astype(int)
    # Messy: mix comma-formatted strings and plain integers
    arrivals_str_q = []
    for v in arrivals_int_q:
        if RNG.random() < 0.6:
            arrivals_str_q.append(f"{v:,}")
        else:
            arrivals_str_q.append(str(v))

    # Quarter format inconsistency
    quarter_fmt = []
    for i, (y, q) in enumerate(q_rows):
        choice = RNG.integers(0, 3)
        if choice == 0:
            quarter_fmt.append(f"Q{q} {y}")
        elif choice == 1:
            quarter_fmt.append(f"{y}-Q{q}")
        else:
            quarter_fmt.append(f"{y}-{q}")

    q_df = pl.DataFrame(
        {
            "period": quarter_fmt,
            "period_type": ["quarterly"] * n_q,
            "gdp_growth_pct": np.round(gdp_growth_q, 2).tolist(),
            "unemployment_rate": np.round(unemp_q, 2).tolist(),
            "inflation_rate": inflation_raw_q,
            "trade_balance_sgd_bn": trade_raw_q,
            "property_price_index": np.round(ppi_q, 1).tolist(),
            "tourist_arrivals": arrivals_str_q,
        }
    )

    # --- Monthly rows (25 years x 12 = 300 rows) ---
    monthly_rows = []
    for y in range(2000, 2025):
        for m in range(1, 13):
            monthly_rows.append((y, m))

    n_m = len(monthly_rows)

    # Monthly CPI YoY
    cpi_m = RNG.uniform(0.5, 3.5, n_m)
    cpi_m[240:252] = RNG.uniform(4.5, 6.5, 12)  # 2020
    cpi_m[252:264] = RNG.uniform(3.5, 5.5, 12)  # 2021
    cpi_m[264:276] = RNG.uniform(5.0, 7.0, 12)  # 2022 peak
    cpi_m[276:288] = RNG.uniform(3.0, 5.0, 12)  # 2023
    cpi_m_raw = _nullify(np.round(cpi_m, 2), rate=0.05)

    # Monthly industrial production index
    ipi = np.zeros(n_m)
    ipi[0] = 100.0
    for i in range(1, n_m):
        shock = RNG.normal(0.003, 0.02)
        if 240 <= i <= 251:
            shock -= 0.04
        if 252 <= i <= 263:
            shock += 0.015
        ipi[i] = ipi[i - 1] * (1 + shock)

    # Monthly retail sales index
    rsi = np.zeros(n_m)
    rsi[0] = 100.0
    for i in range(1, n_m):
        month = monthly_rows[i][1]
        seasonal = 0.01 if month in (11, 12) else (-0.005 if month in (1, 2) else 0)
        shock = RNG.normal(0.002 + seasonal, 0.018)
        if 240 <= i <= 263:
            shock -= 0.03
        rsi[i] = rsi[i - 1] * (1 + shock)

    # Period format for monthly: mostly "YYYY-MM" but some "MM/YYYY" and "YYYYMM"
    month_period = []
    for y, m in monthly_rows:
        choice = RNG.integers(0, 4)
        if choice == 0:
            month_period.append(f"{m:02d}/{y}")
        elif choice == 1:
            month_period.append(f"{y}{m:02d}")
        else:
            month_period.append(f"{y}-{m:02d}")

    m_df = pl.DataFrame(
        {
            "period": month_period,
            "period_type": ["monthly"] * n_m,
            "gdp_growth_pct": [None] * n_m,  # only available quarterly
            "unemployment_rate": [None] * n_m,  # only available quarterly
            "inflation_rate": cpi_m_raw,
            "trade_balance_sgd_bn": [None] * n_m,
            "property_price_index": np.round(ipi, 1).tolist(),
            "tourist_arrivals": [None] * n_m,
        }
    )

    # Combine: quarterly + monthly
    df = pl.concat([q_df, m_df])

    # Add duplicate row (2019 Q2 appears twice)
    dup_rows = q_df.filter(pl.col("period").str.contains("2019"))
    if len(dup_rows) > 0:
        df = pl.concat([df, dup_rows[1:2]])
    else:
        df = pl.concat([df, q_df[75:76]])

    return df


# ---------------------------------------------------------------------------
# 2. sg_taxi_trips.parquet  (ascent01) — intentionally very messy, 50K rows
# ---------------------------------------------------------------------------


def make_sg_taxi_trips() -> pl.DataFrame:
    """
    Singapore taxi trip data (50,000 rows). Saved as parquet for size.

    Intentional messiness:
    - Negative fares (~2% of rows)
    - Impossible distances: ~1% > 60 km (island is ~50 km wide)
    - Future dates (~1%) — 2027/2028 dates
    - Missing pickup/dropoff zones (~5%)
    - Inconsistent payment_type: "Cash"/"cash"/"CASH", "Card"/"credit card"/"VISA"
    - passengers = 0 or negative in ~1%
    - Missing tip_sgd in ~15% (only card payments have tips)
    - Duplicate trip_ids (~0.5%)
    - GPS jitter: lat/lon occasionally swapped or extreme
    """
    n = 50_000

    zones = [
        "Orchard",
        "Marina Bay",
        "Raffles Place",
        "Toa Payoh",
        "Ang Mo Kio",
        "Bishan",
        "Tampines",
        "Pasir Ris",
        "Woodlands",
        "Jurong East",
        "Jurong West",
        "Clementi",
        "Buona Vista",
        "Holland Village",
        "Novena",
        "Newton",
        "Bugis",
        "Little India",
        "Chinatown",
        "Tiong Bahru",
        "Queenstown",
        "Bukit Timah",
        "Serangoon",
        "Hougang",
        "Punggol",
        "Sengkang",
        "Bedok",
        "Changi Airport",
        "Paya Lebar",
        "Kallang",
        "Yishun",
        "Sembawang",
        "Tuas",
        "Boon Lay",
        "Clementi",
    ]

    # Trip IDs — inject ~0.5% duplicates
    unique_ids = [f"SG-TX-{100000 + i}" for i in range(n)]
    dup_count = int(n * 0.005)
    dup_positions = RNG.choice(n - dup_count, size=dup_count, replace=False)
    trip_ids = list(unique_ids)
    for j, pos in enumerate(dup_positions):
        trip_ids[n - dup_count + j] = trip_ids[pos]

    # Datetimes — mostly 2022-2024
    start_ts = 1640995200  # 2022-01-01 00:00:00 UTC
    end_ts = 1735689600  # 2025-01-01 00:00:00 UTC
    pickup_ts = RNG.integers(start_ts, end_ts, n)
    duration_s = RNG.integers(5 * 60, 60 * 60, n)
    dropoff_ts = pickup_ts + duration_s

    # Inject future dates (~1%)
    future_idx = RNG.choice(n, size=500, replace=False)
    for i in future_idx:
        pickup_ts[i] = 1830000000  # 2028
        dropoff_ts[i] = pickup_ts[i] + duration_s[i]

    from datetime import datetime, timezone

    def ts_to_str(ts_arr):
        return [
            datetime.fromtimestamp(int(t), tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            for t in ts_arr
        ]

    pickup_dt = ts_to_str(pickup_ts)
    dropoff_dt = ts_to_str(dropoff_ts)

    # Zones — with missing values
    pickup_zones = [random.choice(zones) for _ in range(n)]
    dropoff_zones = [random.choice(zones) for _ in range(n)]
    missing_pickup = RNG.choice(n, size=int(n * 0.05), replace=False)
    missing_dropoff = RNG.choice(n, size=int(n * 0.04), replace=False)
    for i in missing_pickup:
        pickup_zones[i] = None
    for i in missing_dropoff:
        dropoff_zones[i] = None

    # Distance (km) — realistic for Singapore
    distance = RNG.uniform(0.5, 25.0, n)
    impossible_dist_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    distance[impossible_dist_idx] = RNG.uniform(65.0, 200.0, len(impossible_dist_idx))

    # Fare (SGD)
    fare_base = 3.90 + distance * 0.45 + RNG.normal(0, 1.5, n)
    fare_base = np.clip(fare_base, 3.90, 80.0)
    neg_fare_idx = RNG.choice(n, size=int(n * 0.02), replace=False)
    fare_base[neg_fare_idx] = RNG.uniform(-50.0, -1.0, len(neg_fare_idx))

    # Payment type — inconsistent spellings
    payment_variants = {
        "cash": ["Cash", "cash", "CASH", "Cash Payment"],
        "card": ["Card", "Credit Card", "VISA", "Mastercard", "credit card"],
        "grab": ["GrabPay", "Grab", "GRAB"],
        "nets": ["NETS", "Nets", "nets"],
    }
    all_payment = []
    for _ in range(n):
        cat = RNG.choice(["cash", "card", "grab", "nets"], p=[0.35, 0.40, 0.18, 0.07])
        variant = random.choice(payment_variants[cat])
        all_payment.append(variant)

    # Tip (SGD)
    tip = []
    for i in range(n):
        p = all_payment[i].lower()
        if "card" in p or "visa" in p or "master" in p:
            if RNG.random() < 0.55:
                tip.append(round(float(RNG.uniform(0.50, 5.00)), 2))
            else:
                tip.append(None)
        else:
            tip.append(None)

    # Passengers
    passengers = RNG.integers(1, 5, n).tolist()
    bad_pax_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    for i in bad_pax_idx:
        passengers[i] = int(RNG.choice([-1, 0]))

    # GPS coordinates — Singapore bounding box with jitter
    # Normal: lat 1.24-1.46, lon 103.63-104.00
    pickup_lat = RNG.uniform(1.24, 1.46, n)
    pickup_lon = RNG.uniform(103.63, 104.00, n)
    # GPS jitter: ~0.5% swapped lat/lon (obviously wrong)
    gps_swap_idx = RNG.choice(n, size=int(n * 0.005), replace=False)
    for i in gps_swap_idx:
        pickup_lat[i], pickup_lon[i] = pickup_lon[i], pickup_lat[i]

    df = pl.DataFrame(
        {
            "trip_id": trip_ids,
            "pickup_datetime": pickup_dt,
            "dropoff_datetime": dropoff_dt,
            "pickup_zone": pickup_zones,
            "dropoff_zone": dropoff_zones,
            "distance_km": np.round(distance, 2).tolist(),
            "fare_sgd": np.round(fare_base, 2).tolist(),
            "tip_sgd": tip,
            "payment_type": all_payment,
            "passengers": passengers,
            "pickup_latitude": np.round(pickup_lat, 6).tolist(),
            "pickup_longitude": np.round(pickup_lon, 6).tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 3. experiment_data.parquet  (ascent02) — 500,000 rows
# ---------------------------------------------------------------------------


def make_experiment_data() -> pl.DataFrame:
    """
    A/B test data for experiment design exercises.
    500,000 users — multi-variant, CUPED-ready, SRM trap in one variant.

    Intentional issues for students to discover:
    - Variant C has a Sample Ratio Mismatch (SRM): assigned 10% but gets ~15%
    - Pre-metric is correlated with post-metric (enables CUPED)
    - Platform 'tablet' has a different treatment effect (heterogeneous effects)
    - True lift: +3 units for treatment_a, +5 units for treatment_b
    - ~2% of metric_value are outliers (> mean + 5*std)
    """
    n = 500_000

    user_ids = [f"USR-{200000 + i}" for i in range(n)]

    # Variants: control 40%, treatment_a 35%, treatment_b 15%, variant_c ~10%
    # SRM: variant_c was intended at 10% but got ~15% due to a bug
    srm_n_c = int(n * 0.15)  # should be 0.10, bug gives 0.15
    srm_n_rest = n - srm_n_c
    groups_rest = RNG.choice(
        ["control", "treatment_a", "treatment_b"],
        size=srm_n_rest,
        p=[0.444, 0.389, 0.167],  # proportional to 40/35/15 within remainder
    )
    groups_c = np.array(["variant_c"] * srm_n_c)
    groups = np.concatenate([groups_rest, groups_c])
    # Shuffle so variant_c is not all at the end
    shuffle_idx = RNG.permutation(n)
    groups = groups[shuffle_idx]

    # Pre-metric: baseline before experiment
    pre_metric = RNG.normal(50.0, 15.0, n)

    # Post-metric with treatment effects
    noise = RNG.normal(0, 12.0, n)
    metric_value = pre_metric * 0.7 + noise
    metric_value[groups == "treatment_a"] += 3.0
    metric_value[groups == "treatment_b"] += 5.0
    metric_value[groups == "variant_c"] += 1.0  # minimal real effect

    # Outliers (~2%)
    outlier_idx = RNG.choice(n, size=int(n * 0.02), replace=False)
    metric_value[outlier_idx] = RNG.uniform(200, 500, len(outlier_idx))

    # Segments
    segments = RNG.choice(
        ["high_value", "mid_value", "low_value"], size=n, p=[0.15, 0.55, 0.30]
    )

    # Platform — tablet has amplified treatment effect
    platforms = RNG.choice(
        ["mobile", "desktop", "tablet"], size=n, p=[0.60, 0.32, 0.08]
    )
    tablet_treatment_mask = (platforms == "tablet") & (groups == "treatment_a")
    metric_value[tablet_treatment_mask] += 4.0  # heterogeneous effect

    # Country — ASEAN breakdown
    countries = RNG.choice(
        ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        size=n,
        p=[0.45, 0.22, 0.14, 0.09, 0.06, 0.04],
    )

    # Timestamps: 2-week experiment window
    exp_start = 1704067200  # 2024-01-01
    exp_end = 1705276800  # 2024-01-15
    ts = RNG.integers(exp_start, exp_end, n)

    from datetime import datetime, timezone

    timestamps = [
        datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for t in ts
    ]

    # Revenue proxy — secondary metric
    base_revenue = np.clip(metric_value * RNG.uniform(0.8, 1.2, n), 0, None)
    base_revenue[groups == "treatment_b"] *= 1.08  # +8% revenue for treatment_b

    df = pl.DataFrame(
        {
            "user_id": user_ids,
            "experiment_group": groups.tolist(),
            "metric_value": np.round(metric_value, 4).tolist(),
            "pre_metric_value": np.round(pre_metric, 4).tolist(),
            "revenue": np.round(base_revenue, 2).tolist(),
            "timestamp": timestamps,
            "segment": segments.tolist(),
            "platform": platforms.tolist(),
            "country": countries.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 4. sg_credit_scoring.parquet  (ascent03) — 100,000 rows, 45 features
# ---------------------------------------------------------------------------


def make_sg_credit_scoring() -> pl.DataFrame:
    """
    Singapore credit scoring dataset (100,000 rows, 45 features).

    Key properties:
    - 12% default rate (class imbalance)
    - Temporal leakage trap: future_default_indicator leaks the label
    - 30% missing income_sgd (income not always reported)
    - Realistic SG income S$24K-S$350K
    - Protected attributes: gender, race, age (for fairness exercises)
    - 45 total features including derived and demographic columns
    """
    n = 100_000
    cust_ids = [f"CUST-{300000 + i}" for i in range(n)]

    age = RNG.integers(21, 70, n)

    # Income — 30% missing
    income_base = 30000 + (age - 21) * 2500 + RNG.normal(0, 15000, n)
    income_sgd_full = np.clip(income_base, 24000, 350000).astype(int)
    income_sgd = _nullify(income_sgd_full, rate=0.30)

    # Employment
    employment_years = np.clip((age - 22) * 0.7 + RNG.normal(0, 3, n), 0, 40).astype(
        int
    )
    months_employed = employment_years * 12 + RNG.integers(0, 12, n)

    # Credit utilization
    util_base = RNG.beta(2, 5, n)
    credit_utilization = np.round(np.clip(util_base, 0, 1), 4)

    # Credit lines
    num_credit_lines = np.clip(RNG.poisson(3.5, n), 0, 15).astype(int)

    # Payment history score
    pay_score_base = 650 + RNG.normal(0, 80, n) - credit_utilization * 150
    payment_history_score = np.clip(pay_score_base, 300, 850).astype(int)

    # Loan amount
    income_for_loan = np.where(
        np.array(income_sgd) == None,
        income_sgd_full,
        income_sgd_full,
    )
    loan_base = income_sgd_full * RNG.uniform(0.5, 4.0, n)
    loan_amount_sgd = np.round(np.clip(loan_base, 5000, 800000), -2).astype(int)

    # Loan purpose
    purposes = ["home", "car", "education", "personal", "business", "renovation"]
    loan_purpose = RNG.choice(purposes, size=n, p=[0.35, 0.20, 0.10, 0.20, 0.10, 0.05])

    # Marital status
    marital_status = RNG.choice(
        ["single", "married", "divorced", "widowed"],
        size=n,
        p=[0.35, 0.50, 0.12, 0.03],
    )

    # Education
    education = RNG.choice(
        ["primary", "secondary", "diploma", "degree", "postgraduate"],
        size=n,
        p=[0.05, 0.20, 0.30, 0.35, 0.10],
    )

    # Housing type (SG-specific)
    housing_type = RNG.choice(
        ["HDB 2-3 room", "HDB 4-5 room", "private condo", "landed", "rental"],
        size=n,
        p=[0.15, 0.40, 0.25, 0.12, 0.08],
    )

    # Number of dependents
    num_dependents = np.clip(RNG.poisson(1.2, n), 0, 6).astype(int)

    # Debt-to-income ratio
    monthly_income = income_sgd_full / 12
    debt_to_income = np.clip(
        (loan_amount_sgd / 12) / (monthly_income + 1) + RNG.normal(0, 0.1, n), 0, 5
    )
    debt_to_income = np.round(debt_to_income, 4)

    # Savings and checking balance
    savings_balance = np.round(
        np.clip(income_sgd_full * RNG.uniform(0, 2.0, n), 0, 500000), 2
    )
    checking_balance = np.round(
        np.clip(income_sgd_full * RNG.uniform(0, 0.5, n), 0, 100000), 2
    )

    # Previous defaults
    previous_defaults = np.clip(RNG.poisson(0.15, n), 0, 5).astype(int)

    # Property value (for home loans, else 0)
    property_value_raw = np.where(
        loan_purpose == "home",
        np.round(np.clip(income_sgd_full * RNG.uniform(5, 15, n), 200000, 3000000), -3),
        0,
    )

    # Monthly installment
    monthly_installment = np.round(
        loan_amount_sgd / np.clip(RNG.uniform(12, 360, n), 12, 360), 2
    )

    # Additional behavioral features
    num_late_payments = np.clip(RNG.poisson(0.8, n), 0, 20).astype(int)
    avg_balance_utilization = np.round(
        np.clip(credit_utilization * RNG.uniform(0.8, 1.2, n), 0, 1), 4
    )
    credit_age_years = np.clip(employment_years + RNG.integers(0, 5, n), 0, 40).astype(
        int
    )
    num_hard_inquiries = np.clip(RNG.poisson(1.2, n), 0, 10).astype(int)
    revolving_balance = np.round(
        np.clip(
            income_sgd_full * credit_utilization * 0.3 + RNG.normal(0, 2000, n),
            0,
            50000,
        ),
        2,
    )
    installment_balance = np.round(
        np.clip(loan_amount_sgd * 0.7 + RNG.normal(0, 5000, n), 0, 800000), 2
    )

    # Protected attributes (for fairness exercises)
    gender = RNG.choice(["M", "F", "U"], size=n, p=[0.48, 0.48, 0.04])
    race = RNG.choice(
        ["Chinese", "Malay", "Indian", "Others"],
        size=n,
        p=[0.74, 0.13, 0.09, 0.04],
    )

    # Nationality
    nationality = RNG.choice(
        ["Singaporean", "PR", "EP Holder", "S Pass"],
        size=n,
        p=[0.65, 0.18, 0.10, 0.07],
    )

    # Application channel
    application_channel = RNG.choice(
        ["branch", "online", "mobile", "broker"],
        size=n,
        p=[0.20, 0.40, 0.30, 0.10],
    )

    # Geographic region
    sg_regions = [
        "Central",
        "East",
        "West",
        "North",
        "North-East",
    ]
    region = RNG.choice(sg_regions, size=n, p=[0.30, 0.20, 0.20, 0.15, 0.15])

    # Default (~12%): logistic model on risk factors
    log_odds = (
        -2.0
        + 2.5 * credit_utilization
        - 0.3 * (payment_history_score - 650) / 80
        + 0.5 * (loan_amount_sgd / income_sgd_full - 2.0)
        - 0.1 * employment_years
        + 0.3 * previous_defaults
        + 0.2 * num_late_payments
        - 0.1 * (savings_balance / (income_sgd_full + 1))
        + RNG.normal(0, 0.3, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (RNG.random(n) < prob_default).astype(int)

    # TEMPORAL LEAKAGE TRAP: future_default_indicator is perfectly correlated
    # with default but named to sound like a legitimate feature
    # Students must catch this in EDA
    future_default_indicator = default.copy()
    # Add tiny noise so it's not literally identical (harder to spot)
    noise_mask = RNG.random(n) < 0.01
    future_default_indicator[noise_mask] = 1 - future_default_indicator[noise_mask]

    # Additional numeric features for richness
    loan_to_value = np.where(
        property_value_raw > 0,
        np.round(loan_amount_sgd / np.clip(property_value_raw, 1, None), 4),
        None,
    )

    coe_vehicle_owner = RNG.choice([0, 1], size=n, p=[0.70, 0.30]).astype(int)
    cpf_monthly_contribution = np.round(
        np.clip(income_sgd_full * 0.2 + RNG.normal(0, 500, n), 0, 3700), 2
    )

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "age": age.tolist(),
            "gender": gender.tolist(),
            "race": race.tolist(),
            "nationality": nationality.tolist(),
            "region": region.tolist(),
            "income_sgd": income_sgd,  # 30% null
            "employment_years": employment_years.tolist(),
            "months_employed": months_employed.tolist(),
            "credit_utilization": credit_utilization.tolist(),
            "avg_balance_utilization": avg_balance_utilization.tolist(),
            "num_credit_lines": num_credit_lines.tolist(),
            "credit_age_years": credit_age_years.tolist(),
            "num_hard_inquiries": num_hard_inquiries.tolist(),
            "payment_history_score": payment_history_score.tolist(),
            "num_late_payments": num_late_payments.tolist(),
            "revolving_balance": revolving_balance.tolist(),
            "installment_balance": installment_balance.tolist(),
            "loan_amount_sgd": loan_amount_sgd.tolist(),
            "loan_purpose": loan_purpose.tolist(),
            "monthly_installment": monthly_installment.tolist(),
            "marital_status": marital_status.tolist(),
            "education": education.tolist(),
            "housing_type": housing_type.tolist(),
            "num_dependents": num_dependents.tolist(),
            "debt_to_income": debt_to_income.tolist(),
            "savings_balance": savings_balance.tolist(),
            "checking_balance": checking_balance.tolist(),
            "previous_defaults": previous_defaults.tolist(),
            "property_value_sgd": property_value_raw.tolist(),
            "loan_to_value": [
                (
                    round(float(x), 4)
                    if x is not None and not np.isnan(float(x if x is not None else 0))
                    else None
                )
                for x in loan_to_value.tolist()
            ],
            "coe_vehicle_owner": coe_vehicle_owner.tolist(),
            "cpf_monthly_contribution": cpf_monthly_contribution.tolist(),
            "application_channel": application_channel.tolist(),
            "future_default_indicator": future_default_indicator.tolist(),  # LEAKAGE TRAP
            "default": default.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 5. ecommerce_customers.parquet  (ascent04) — 50,000 customers
# ---------------------------------------------------------------------------


def make_ecommerce_customers() -> pl.DataFrame:
    """
    E-commerce customer data for clustering + NLP exercises (50,000 rows).
    Mix of numeric RFM features, text fields, and ASEAN regional breakdown.

    Features designed for:
    - RFM clustering (recency, frequency, monetary)
    - NLP sentiment analysis on review_text
    - Churn prediction (days_since_last_order as proxy)
    - ASEAN segmentation
    """
    n = 50_000
    cust_ids = [f"EC-{400000 + i}" for i in range(n)]

    # RFM features with realistic distributions
    total_revenue = np.round(RNG.exponential(scale=350.0, size=n), 2)
    order_count = np.clip(RNG.poisson(8, n), 1, 120).astype(int)
    avg_order_value = np.round(total_revenue / order_count, 2)
    days_since_last_order = np.clip(RNG.integers(1, 730, n), 1, 730).astype(int)

    # Customer tenure (days since first order)
    customer_tenure_days = days_since_last_order + RNG.integers(0, 1000, n)
    customer_tenure_days = np.clip(customer_tenure_days, 1, 2000).astype(int)

    # Lifetime value tier (will be useful for clustering)
    ltv_raw = total_revenue * order_count
    ltv_tier = np.where(
        ltv_raw > np.percentile(ltv_raw, 80),
        "platinum",
        np.where(
            ltv_raw > np.percentile(ltv_raw, 50),
            "gold",
            np.where(ltv_raw > np.percentile(ltv_raw, 20), "silver", "bronze"),
        ),
    )

    # Product categories (text) — comma-separated
    categories_pool = [
        "Electronics",
        "Fashion",
        "Home & Living",
        "Beauty",
        "Sports",
        "Books",
        "Toys",
        "Groceries",
        "Health",
        "Automotive",
        "Travel",
        "Pet Supplies",
        "Office Supplies",
        "Garden",
    ]

    def random_categories():
        k = int(RNG.integers(1, 4))
        chosen = random.sample(categories_pool, k)
        return ", ".join(chosen)

    product_categories = [random_categories() for _ in range(n)]

    # Review text — more varied templates for richer NLP
    pos_templates = [
        "Great product, fast delivery! Will buy again.",
        "Exactly what I needed. Highly recommend.",
        "Good quality for the price. Very happy.",
        "Happy with my purchase. Prompt shipping to Singapore.",
        "Works perfectly. Top notch quality.",
        "Exceeded my expectations. Outstanding service.",
        "Very satisfied. Will definitely order again.",
        "Quality is excellent. Arrived on time.",
        "Super fast delivery. Product matches description.",
        "Amazing value. The packaging was secure.",
        "Love it! Perfect for daily use.",
        "Five stars. Everything was perfect.",
    ]
    neg_templates = [
        "Item arrived damaged. Very disappointed.",
        "Not as described. Poor quality. Returning it.",
        "Delivery took three weeks. Packaging was terrible.",
        "Would not recommend. Complete waste of money.",
        "Product stopped working after one week. Terrible.",
        "Customer service was completely unhelpful.",
        "Size was wrong and return process is a nightmare.",
        "Missing parts. Had to wait weeks for support.",
        "Fake product. Not the brand I ordered.",
        "Arrived late. Item was defective. No refund offered.",
    ]
    neutral_templates = [
        "Decent product. Nothing special but does the job.",
        "Okay for the price. Could be better.",
        "Average quality. Shipping was acceptable.",
        "Shipping was slow but product is fine.",
        "Not bad. Meets basic expectations.",
        "Reasonable quality at this price point.",
    ]

    satisfaction = RNG.integers(1, 6, n)  # 1-5

    review_text = []
    for s in satisfaction:
        if s >= 4:
            review_text.append(random.choice(pos_templates))
        elif s == 3:
            review_text.append(random.choice(neutral_templates))
        else:
            review_text.append(random.choice(neg_templates))

    # ~5% null reviews
    review_text = _nullify_list(review_text, rate=0.05)

    # Region — SG + ASEAN
    regions = RNG.choice(
        ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        size=n,
        p=[0.50, 0.20, 0.12, 0.08, 0.06, 0.04],
    )

    # Device type
    device_type = RNG.choice(
        ["mobile", "desktop", "tablet", "app"],
        size=n,
        p=[0.45, 0.25, 0.10, 0.20],
    )

    # Payment method
    payment_method = RNG.choice(
        [
            "credit_card",
            "debit_card",
            "grab_pay",
            "paypal",
            "bank_transfer",
            "cash_on_delivery",
        ],
        size=n,
        p=[0.30, 0.20, 0.18, 0.12, 0.10, 0.10],
    )

    # Is subscribed to loyalty program
    loyalty_member = RNG.choice([True, False], size=n, p=[0.40, 0.60]).tolist()

    # Number of returns
    num_returns = np.clip(RNG.poisson(0.5, n), 0, 10).astype(int)

    # Churn flag (proxy: no order in last 180 days)
    churned = (days_since_last_order > 180).astype(int)

    df = pl.DataFrame(
        {
            "customer_id": cust_ids,
            "total_revenue": total_revenue.tolist(),
            "order_count": order_count.tolist(),
            "avg_order_value": avg_order_value.tolist(),
            "days_since_last_order": days_since_last_order.tolist(),
            "customer_tenure_days": customer_tenure_days.tolist(),
            "ltv_tier": ltv_tier.tolist(),
            "product_categories": product_categories,
            "review_text": review_text,
            "satisfaction_score": satisfaction.tolist(),
            "region": regions.tolist(),
            "device_type": device_type.tolist(),
            "payment_method": payment_method.tolist(),
            "loyalty_member": loyalty_member,
            "num_returns": num_returns.tolist(),
            "churned": churned.tolist(),
        }
    )

    return df


# ---------------------------------------------------------------------------
# 6. documents.parquet  (ascent05) — 500 rows
# ---------------------------------------------------------------------------


def make_documents() -> pl.DataFrame:
    """
    Knowledge-base articles for RAG exercise (500 rows).
    Topics cover ML concepts, Singapore domain, ASEAN context, and tech reference.
    Richer content than minimal articles for better embedding/retrieval exercises.
    """
    articles = [
        # --- ML Concepts ---
        (
            "What is supervised learning?",
            "Supervised learning trains a model on labelled data where each example has an input and a known output. The model learns a mapping function. Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. Model quality is measured by held-out evaluation on a test set.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is unsupervised learning?",
            "Unsupervised learning discovers patterns in data without labels. Clustering algorithms group similar data points (k-means, DBSCAN, hierarchical). Dimensionality reduction compresses features while preserving structure (PCA, t-SNE, UMAP). Autoencoders learn compact latent representations.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "Explain overfitting and underfitting.",
            "Overfitting occurs when a model memorises training data and fails on new data (high variance). Underfitting occurs when a model is too simple to capture patterns (high bias). Regularisation techniques (L1, L2, dropout) penalise complexity. Cross-validation detects overfitting early.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is cross-validation?",
            "Cross-validation evaluates model performance by splitting data into k folds. The model trains on k-1 folds and validates on the held-out fold. K-fold CV gives a robust performance estimate. Stratified CV preserves class balance. Leave-one-out CV is used for very small datasets.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is gradient descent?",
            "Gradient descent minimises a loss function by iteratively updating parameters in the direction of steepest descent. Learning rate controls step size. Stochastic gradient descent (SGD) updates on one sample. Mini-batch SGD balances stability and efficiency. Adam adapts learning rates per parameter.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is a confusion matrix?",
            "A confusion matrix summarises classification results across all classes. Rows are actual classes, columns are predicted. It reveals true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). Derived metrics: precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2*precision*recall/(precision+recall).",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Precision vs Recall trade-off.",
            "Precision measures how many predicted positives are correct: TP/(TP+FP). Recall measures how many actual positives are found: TP/(TP+FN). The F1 score is their harmonic mean. Adjusting the classification threshold shifts this trade-off. Use precision-recall curves for imbalanced datasets.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "What is ROC-AUC?",
            "ROC-AUC measures a classifier's ability to distinguish classes across all thresholds. The ROC curve plots true positive rate vs false positive rate. AUC=1.0 is perfect. AUC=0.5 is random. It is insensitive to class imbalance. PR-AUC is preferred for highly imbalanced datasets.",
            "ml_evaluation",
            "textbook",
        ),
        (
            "Feature engineering overview.",
            "Feature engineering transforms raw data into informative model inputs. Techniques include normalisation, one-hot encoding, ordinal encoding, polynomial features, interaction terms, log transforms, and domain-specific aggregations. Good features often improve model performance more than algorithm choice.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is one-hot encoding?",
            "One-hot encoding converts categorical variables into binary columns. Each category becomes a column with value 1 if present, 0 otherwise. It avoids imposing ordinal relationships. High-cardinality categories risk the curse of dimensionality. Target encoding and embedding are alternatives for high-cardinality features.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is feature scaling?",
            "Feature scaling standardises the range of input features. Z-score normalisation centres features at 0 with unit variance. Min-max scaling maps features to [0, 1]. Required for distance-based algorithms (k-NN, SVM, k-means) and gradient descent convergence. Tree-based models are scale-invariant.",
            "feature_engineering",
            "textbook",
        ),
        (
            "What is PCA?",
            "Principal Component Analysis reduces dimensionality by projecting data onto orthogonal axes of maximum variance. The first principal component explains the most variance. Subsequent components are orthogonal and explain decreasing variance. Used for visualisation, noise reduction, and computational efficiency.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is t-SNE?",
            "t-SNE (t-distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction algorithm optimised for 2D/3D visualisation. It preserves local structure but not global structure. Perplexity controls neighbourhood size. Not suitable for feature engineering due to non-reproducibility and computational cost.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is UMAP?",
            "UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm that preserves both local and global structure. Faster than t-SNE. Used for visualisation and as a feature engineering step. Supports supervised and semi-supervised modes.",
            "dimensionality_reduction",
            "textbook",
        ),
        (
            "What is k-means clustering?",
            "K-means partitions data into k clusters by iteratively assigning points to the nearest centroid and updating centroids. Sensitive to initialisation (use k-means++). Assumes spherical, equal-variance clusters. Use the elbow method or silhouette score to choose k. Lloyd's algorithm is the standard implementation.",
            "clustering",
            "textbook",
        ),
        (
            "What is DBSCAN?",
            "DBSCAN clusters points based on density. It finds clusters of arbitrary shape and labels outliers as noise (class -1). Parameters: epsilon (neighbourhood radius) and min_samples (core point threshold). Robust to outliers. Struggles with varying density clusters.",
            "clustering",
            "textbook",
        ),
        (
            "What is hierarchical clustering?",
            "Hierarchical clustering builds a dendrogram by iteratively merging (agglomerative) or splitting (divisive) clusters. Linkage criteria: single, complete, average, Ward. No need to specify k in advance. Dendrogram cutting at different heights gives different cluster numbers.",
            "clustering",
            "textbook",
        ),
        (
            "What are transformers?",
            "Transformers use self-attention to process sequences in parallel. The attention mechanism weighs how much each token attends to every other token. Multi-head attention runs multiple attention functions in parallel. Transformers underpin modern LLMs (GPT, BERT, T5) and vision models (ViT).",
            "deep_learning",
            "textbook",
        ),
        (
            "What is transfer learning?",
            "Transfer learning fine-tunes a pre-trained model on a new task. The pre-trained model has learned general representations from large data. Layers near the input are frozen (feature extraction). Later layers are fine-tuned. Reduces data and compute requirements significantly.",
            "deep_learning",
            "textbook",
        ),
        (
            "What is a convolutional neural network?",
            "CNNs apply learnable filters across input data to detect local patterns. Convolution layers extract spatial features. Pooling layers downsample. Fully connected layers classify. CNNs excel at image tasks. Modern architectures: ResNet, EfficientNet, Vision Transformer (ViT).",
            "deep_learning",
            "textbook",
        ),
        (
            "What is RAG?",
            "Retrieval-Augmented Generation combines a retrieval system with a language model. The retriever finds relevant documents from a knowledge base using vector similarity. The generator conditions its response on retrieved context. Reduces hallucination, keeps knowledge current without retraining.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is prompt engineering?",
            "Prompt engineering crafts inputs to elicit desired behaviour from language models. Techniques: few-shot examples (provide input-output demonstrations), chain-of-thought (ask model to reason step by step), role specification (you are an expert in...), output format constraints (respond in JSON).",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is fine-tuning an LLM?",
            "Fine-tuning adapts a pre-trained LLM on task-specific data. Supervised fine-tuning (SFT) trains on instruction-response pairs. RLHF aligns responses with human preferences. DPO directly optimises preference pairs. LoRA reduces compute by training only low-rank adapter matrices.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is LoRA?",
            "Low-Rank Adaptation (LoRA) fine-tunes LLMs by injecting trainable low-rank matrices into transformer layers. Only the adapter matrices are trained, reducing trainable parameters by 10-100x. Enables fine-tuning large models on consumer GPUs. QLoRA combines LoRA with 4-bit quantisation.",
            "llm_techniques",
            "textbook",
        ),
        (
            "What is DPO?",
            "Direct Preference Optimisation reformulates RLHF as supervised learning on (prompt, chosen, rejected) triples. No separate reward model needed. More stable training than PPO. Widely used for instruction-tuning open-source models. SimPO is a simplified variant.",
            "llm_alignment",
            "textbook",
        ),
        (
            "What is RLHF?",
            "Reinforcement Learning from Human Feedback first trains a reward model on human preference comparisons between model outputs. It then fine-tunes the LLM with PPO to maximise reward. A KL penalty prevents the model from deviating too far from the base model. Foundation for ChatGPT-style alignment.",
            "llm_alignment",
            "textbook",
        ),
        (
            "What is vector search?",
            "Vector search retrieves items whose embeddings are most similar to a query embedding. Similarity metrics: cosine similarity, dot product, L2 distance. Approximate nearest neighbour (ANN) algorithms (HNSW, IVF-PQ) enable fast search over billions of vectors. Used in RAG, recommendation systems, and semantic search.",
            "data_engineering",
            "textbook",
        ),
        (
            "What is an embedding?",
            "An embedding is a dense vector representation of data (text, images, products). Embeddings encode semantic similarity — similar items are close in vector space. Text embeddings from models like BGE, E5, or OpenAI's text-embedding models are used for semantic search and RAG retrieval.",
            "deep_learning",
            "textbook",
        ),
        # --- Singapore Context ---
        (
            "What is the HDB in Singapore?",
            "The Housing Development Board (HDB) is Singapore's public housing authority. Over 80% of residents live in HDB flats. New flats are sold under the Build-To-Order (BTO) scheme. Resale flats trade on the open market. Prices are regulated through income ceilings and ethnic integration policies.",
            "singapore_housing",
            "data.gov.sg",
        ),
        (
            "How does the Singapore MRT work?",
            "Singapore's Mass Rapid Transit (MRT) network has six lines: North-South (NSL), East-West (EWL), Circle (CCL), Downtown (DTL), Thomson-East Coast (TEL), and Jurong Region Line (JRL). Over 130 stations cover the island. Fares are distance-based using EZ-Link or contactless payment.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "What is COE in Singapore?",
            "The Certificate of Entitlement (COE) allows a person to own and use a vehicle in Singapore for 10 years. Prices are determined by open bidding and reflect demand for vehicle ownership. COE prices have exceeded SGD 150,000 for cars. High COE prices are a deliberate policy to limit vehicle population.",
            "singapore_transport",
            "lta.gov.sg",
        ),
        (
            "Singapore CPF overview.",
            "The Central Provident Fund (CPF) is Singapore's mandatory retirement savings scheme. Employees and employers contribute monthly to Ordinary Account (housing), Special Account (retirement), and Medisave (healthcare). CPF Life provides lifelong monthly payouts in retirement.",
            "singapore_finance",
            "cpf.gov.sg",
        ),
        (
            "What is GST in Singapore?",
            "The Goods and Services Tax (GST) is Singapore's value-added tax. It applies to most goods and services. GST rose to 9% in 2024. Businesses with annual turnover above SGD 1 million must register for GST. The GST Voucher scheme offsets the impact on lower-income households.",
            "singapore_finance",
            "iras.gov.sg",
        ),
        (
            "Singapore education system overview.",
            "Singapore's education follows a 6-4-2 structure: 6 years primary, 4 years secondary, 2 years JC or Polytechnic. The PSLE at Primary 6 streams students using Achievement Level grades. Emphasis on STEM and bilingualism. NUS, NTU, SMU, SUTD, SIT, and SUSS are the six autonomous universities.",
            "singapore_education",
            "moe.gov.sg",
        ),
        (
            "Singapore hawker culture.",
            "Hawker centres are open-air cooked food centres operated by NEA. They offer affordable multicultural cuisine: chicken rice, laksa, char kway teow, roti prata, satay. UNESCO added Singapore hawker culture to its Intangible Cultural Heritage list in 2020. Over 100 hawker centres operate across Singapore.",
            "singapore_culture",
            "nea.gov.sg",
        ),
        (
            "Singapore's four official languages.",
            "Singapore has four official languages: English, Mandarin, Malay, and Tamil. English is the language of administration, business, and education. Malay is the national language. Singlish, a creole mixing English, Hokkien, Malay, and Tamil, is widely spoken informally.",
            "singapore_culture",
            "singapore_gov",
        ),
        (
            "What are Singapore's public holidays?",
            "Singapore observes 11 public holidays: New Year's Day, Chinese New Year (2 days), Good Friday, Labour Day, Vesak Day, Hari Raya Puasa, National Day (August 9), Hari Raya Haji, Deepavali, and Christmas Day. When a public holiday falls on Sunday, the following Monday is a holiday.",
            "singapore_culture",
            "mom.gov.sg",
        ),
        (
            "Singapore port and trade.",
            "The Port of Singapore is one of the world's busiest container ports, handling over 37 million TEUs annually. Singapore is a key transhipment hub connecting Asia, Europe, and the Americas. Trade accounts for over 300% of GDP. Major trading partners: China, Malaysia, the US, the EU, and Japan.",
            "singapore_economy",
            "mti.gov.sg",
        ),
        (
            "What is Singpass?",
            "Singpass is Singapore's national digital identity system. It allows residents to access over 2,000 government and private sector digital services. Features include biometric login, digital IC, and Myinfo data sharing. Over 4 million users rely on Singpass daily.",
            "singapore_technology",
            "govtech.gov.sg",
        ),
        (
            "What is the SkillsFuture programme?",
            "SkillsFuture is a national movement to provide Singaporeans with opportunities for lifelong learning. The SkillsFuture Credit (SGD 500 for those aged 25+) subsidises approved courses. Additional top-ups support mid-career workers. Aligned with Singapore's workforce transformation agenda.",
            "singapore_education",
            "skillsfuture.gov.sg",
        ),
        (
            "What is the MAS in Singapore?",
            "The Monetary Authority of Singapore (MAS) is Singapore's central bank and financial regulator. It manages monetary policy through the exchange rate band. MAS regulates banks, insurers, and capital markets. Singapore is a global financial hub with over 200 banks operating locally.",
            "singapore_finance",
            "mas.gov.sg",
        ),
        (
            "What is HDB BTO?",
            "Build-To-Order (BTO) is HDB's main public housing scheme. Applicants select a flat and wait 3-5 years for construction. BTO sales are launched quarterly. Flats are sold at subsidised prices with eligibility criteria (citizenship, income ceiling, household nucleus). First-timer applicants receive priority.",
            "singapore_housing",
            "hdb.gov.sg",
        ),
        (
            "What is the PDPA?",
            "The Personal Data Protection Act (PDPA) governs collection, use, and disclosure of personal data in Singapore. Organisations must obtain consent, limit collection to necessary data, protect data, and allow data subject access and correction. The PDPC enforces the PDPA and can impose fines up to SGD 1 million.",
            "singapore_technology",
            "pdpc.gov.sg",
        ),
        # --- Technical Reference ---
        (
            "Polars vs pandas overview.",
            "Polars is a fast DataFrame library written in Rust. It uses Apache Arrow for columnar memory layout. Lazy evaluation enables query optimisation (predicate pushdown, projection pruning). Polars is 5-10x faster than pandas on many workloads, uses less memory, and supports multi-threading natively.",
            "tools",
            "polars_docs",
        ),
        (
            "What is Apache Parquet?",
            "Parquet is a columnar storage format optimised for analytics. Columns are stored together, enabling efficient compression and predicate pushdown (skip reading irrelevant row groups). Widely used in data engineering. Polars and DuckDB read Parquet natively. Row group sizes balance read efficiency.",
            "tools",
            "apache_docs",
        ),
        (
            "What is a data pipeline?",
            "A data pipeline automates the movement and transformation of data from source to destination. Stages include ingestion, validation, cleaning, feature engineering, and serving. Orchestration tools manage scheduling and dependencies. Modern pipelines are declarative, testable, and versioned.",
            "data_engineering",
            "textbook",
        ),
        (
            "What is data drift?",
            "Data drift is a change in the statistical distribution of input data after model deployment. Feature drift changes input distributions. Concept drift changes the input-output relationship. Detection methods: Population Stability Index (PSI), Kolmogorov-Smirnov test, chi-squared test. PSI > 0.2 typically triggers retraining.",
            "mlops",
            "textbook",
        ),
        (
            "What is MLOps?",
            "MLOps applies DevOps principles to machine learning. Pillars: versioned data, reproducible training, automated testing, CI/CD for models, and continuous monitoring. Key tools: experiment trackers (MLflow), model registries, feature stores, and drift monitors. Reduces time-to-production and improves reliability.",
            "mlops",
            "textbook",
        ),
        (
            "What is a feature store?",
            "A feature store is a centralised repository for ML features. It ensures consistency between training and serving (prevents train-serve skew). Stores feature pipelines, their outputs, and metadata. Enables feature reuse across teams and projects. Common implementations: Feast, Tecton, Hopsworks.",
            "mlops",
            "textbook",
        ),
        (
            "What is model drift monitoring?",
            "Model drift monitoring tracks model performance and input data distributions in production. Metrics: PSI for numeric features, chi-squared for categorical, KS statistic. Output drift monitors prediction distributions. Ground truth feedback enables performance monitoring when labels are available.",
            "mlops",
            "textbook",
        ),
        (
            "What is hyperparameter tuning?",
            "Hyperparameter tuning searches for the configuration that maximises model performance. Grid search exhaustively tests combinations. Random search samples randomly (more efficient). Bayesian optimisation models the performance surface. Optuna and Ray Tune are popular frameworks.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is ensemble learning?",
            "Ensemble learning combines multiple models to improve prediction. Bagging reduces variance by training on bootstrapped subsets (Random Forest). Boosting reduces bias by sequentially correcting errors (XGBoost, LightGBM, CatBoost). Stacking uses a meta-learner to blend diverse model predictions.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is SHAP?",
            "SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction using Shapley values from cooperative game theory. It is model-agnostic and satisfies axioms of efficiency, symmetry, and dummy. Used for model interpretability and bias auditing in production.",
            "ml_interpretability",
            "textbook",
        ),
        (
            "What is LIME?",
            "LIME (Local Interpretable Model-agnostic Explanations) explains individual model predictions by fitting a simple linear model locally around the prediction. Perturbs inputs and measures impact on predictions. Faster than SHAP for some models but less theoretically grounded.",
            "ml_interpretability",
            "textbook",
        ),
        (
            "What is AutoML?",
            "Automated Machine Learning (AutoML) searches for the best model and hyperparameters automatically. Pipelines include data preprocessing, feature selection, algorithm selection, and tuning. Tools: Auto-sklearn, TPOT, H2O AutoML. Useful for rapid prototyping and non-expert users.",
            "ml_fundamentals",
            "textbook",
        ),
        (
            "What is XGBoost?",
            "XGBoost is a gradient boosting framework optimised for speed and performance. It uses second-order gradients and regularisation (L1, L2) to prevent overfitting. Supports sparse data and missing values natively. The dominant algorithm in tabular ML competitions.",
            "ml_algorithms",
            "textbook",
        ),
        (
            "What is LightGBM?",
            "LightGBM is a gradient boosting framework that uses leaf-wise tree growth instead of level-wise. Faster than XGBoost on large datasets. Uses histogram-based splits for efficiency. Supports categorical features natively. GOSS and EFB optimisations further reduce training time.",
            "ml_algorithms",
            "textbook",
        ),
        # --- ASEAN Economic Context ---
        (
            "What is ASEAN?",
            "ASEAN (Association of Southeast Asian Nations) is a regional bloc of 10 countries: Brunei, Cambodia, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, Thailand, and Vietnam. Combined GDP exceeds USD 3 trillion. Free trade agreements promote regional economic integration.",
            "asean_economy",
            "asean.org",
        ),
        (
            "Singapore's role in ASEAN.",
            "Singapore serves as ASEAN's financial and logistics hub. It hosts regional headquarters of many multinationals. Strong rule of law, low corruption, and pro-business policies attract investment. Singapore's GDP per capita exceeds USD 80,000, highest in ASEAN.",
            "asean_economy",
            "mti.gov.sg",
        ),
        (
            "Vietnam's tech ecosystem.",
            "Vietnam has emerged as a major tech manufacturing hub. Companies like Samsung, Intel, and LG have large factories there. A young, tech-savvy population drives e-commerce growth (Shopee, Lazada, Tiki). Hanoi and Ho Chi Minh City are regional startup centres.",
            "asean_economy",
            "worldbank.org",
        ),
        (
            "Indonesia's digital economy.",
            "Indonesia has Southeast Asia's largest digital economy. Gojek and Tokopedia merged into GoTo Group. Over 200 million internet users. Financial inclusion through digital banking and mobile payments is a key growth driver. Jakarta is the main tech hub.",
            "asean_economy",
            "worldbank.org",
        ),
        (
            "What is the RCEP trade agreement?",
            "The Regional Comprehensive Economic Partnership (RCEP) is the world's largest free trade agreement by GDP. It covers ASEAN plus Australia, China, Japan, New Zealand, and South Korea. Eliminates tariffs, streamlines customs, and sets rules of origin. Signed in 2020, in force since 2022.",
            "asean_economy",
            "asean.org",
        ),
    ]

    # Pad to 500 rows by repeating with slight variation
    base = articles.copy()
    extra_prefixes = [
        "(Reference) ",
        "(Overview) ",
        "(Summary) ",
        "(Guide) ",
        "(Explainer) ",
        "(Tutorial) ",
        "(Deep dive) ",
    ]
    while len(base) < 500:
        t, c, cat, src = random.choice(articles)
        prefix = random.choice(extra_prefixes)
        base.append((prefix + t, c, cat, src))

    base = base[:500]

    doc_ids = [f"DOC-{500 + i:04d}" for i in range(len(base))]

    df = pl.DataFrame(
        {
            "doc_id": doc_ids,
            "title": [a[0] for a in base],
            "content": [a[1] for a in base],
            "category": [a[2] for a in base],
            "source": [a[3] for a in base],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 7. sg_domain_qa.parquet  (ascent06) — 1,000 rows
# ---------------------------------------------------------------------------


def make_sg_domain_qa() -> pl.DataFrame:
    """
    Singapore domain Q&A for SFT (M6.1) — 1,000 rows.
    Instruction-following format. Covers geography, food, transport,
    housing, finance, technology, and ML concepts with Singapore context.
    """
    pairs = [
        # --- Geography ---
        (
            "What are the five main regions of Singapore?",
            "Singapore is divided into five main planning regions: the Central Region, North Region, North-East Region, East Region, and West Region. Each region contains several planning areas managed by the Urban Redevelopment Authority.",
            "",
            "geography",
        ),
        (
            "How large is Singapore by land area?",
            "Singapore has a total land area of approximately 735 square kilometres. Land reclamation has expanded its area significantly over the decades, adding around 25% of the current land area since independence.",
            "",
            "geography",
        ),
        (
            "What is Sentosa Island known for?",
            "Sentosa is a resort island off Singapore's southern coast. It hosts Universal Studios Singapore, Resorts World Sentosa, beaches, the S.E.A. Aquarium, and the cable car connecting it to VivoCity mall.",
            "",
            "geography",
        ),
        (
            "Name the main river in Singapore.",
            "The Singapore River runs through the heart of the city. It was the historic centre of colonial trade, lined with warehouses (godowns) at Clarke Quay and Boat Quay. Today its banks are a major dining and entertainment destination.",
            "",
            "geography",
        ),
        (
            "What is the highest point in Singapore?",
            "Bukit Timah Hill, at 163.63 metres, is Singapore's highest natural point. The Bukit Timah Nature Reserve surrounding it contains one of the world's few patches of primary rainforest within a major city.",
            "",
            "geography",
        ),
        (
            "What is Pulau Ubin?",
            "Pulau Ubin is a small island north of Singapore. It offers a glimpse of traditional kampung (village) life. It is popular for cycling, hiking, and the Chek Jawa wetlands. The island has no permanent utilities and is accessible by bumboat from Changi Point Ferry Terminal.",
            "",
            "geography",
        ),
        (
            "What are the major reservoirs in Singapore?",
            "Singapore's main reservoirs include Marina Reservoir (the largest in the city centre), Upper Peirce, Lower Peirce, MacRitchie, Kranji, Bedok, and Tengeh. Marina Barrage created a freshwater reservoir in the Marina Bay area.",
            "",
            "geography",
        ),
        # --- Food ---
        (
            "What is chicken rice?",
            "Hainanese chicken rice is widely regarded as Singapore's national dish. It consists of poached or roasted chicken served over fragrant rice cooked in chicken broth, accompanied by chilli sauce, ginger paste, and dark soy sauce. Famous hawker stalls include Tian Tian at Maxwell Food Centre.",
            "",
            "food",
        ),
        (
            "What is laksa?",
            "Laksa is a spicy noodle soup popular in Singapore. Singapore laksa uses thick rice vermicelli in a rich coconut curry broth, topped with prawns, fish cake, half a hard-boiled egg, and cockles. Katong laksa from the East Coast is a distinctive local variant with pre-cut noodles.",
            "",
            "food",
        ),
        (
            "What is a hawker centre?",
            "A hawker centre is a large open-air complex housing multiple food stalls offering cooked food and drinks. They provide affordable meals (typically SGD 3-6) representing Singapore's diverse cuisines: Chinese, Malay, and Indian. UNESCO recognised Singapore hawker culture in 2020.",
            "",
            "food",
        ),
        (
            "What is char kway teow?",
            "Char kway teow is a stir-fried flat noodle dish cooked over very high heat in a wok. It contains flat rice noodles, Chinese sausage (lup cheong), cockles, eggs, bean sprouts, and dark soy sauce. The wok hei (breath of the wok) from high-heat cooking gives it a distinctive smoky flavour.",
            "",
            "food",
        ),
        (
            "What is bak kut teh?",
            "Bak kut teh (meat bone tea) is a pork rib soup simmered with herbs and spices. The Singaporean version is peppery and lighter compared to the darker, herbal Malaysian version. It is traditionally eaten for breakfast with youtiao (fried dough) and rice.",
            "",
            "food",
        ),
        (
            "What is roti prata?",
            "Roti prata is a flaky flatbread of South Indian origin, cooked on a flat griddle with ghee. It is served with dhal, fish, or chicken curry and comes in plain or stuffed variants (egg, cheese, banana, mushroom). A popular breakfast and supper food across Singapore.",
            "",
            "food",
        ),
        (
            "What is chilli crab?",
            "Chilli crab is one of Singapore's signature seafood dishes. Mud crabs are stir-fried in a semi-thick, tangy-sweet-savoury gravy made from tomato sauce, chilli, and egg. Best enjoyed with deep-fried or steamed mantou (buns) to mop up the sauce. Famous at Long Beach and No Signboard Seafood.",
            "",
            "food",
        ),
        (
            "What is kaya toast?",
            "Kaya toast is a traditional Singapore breakfast. Crispy toast is spread with kaya (a jam made from coconut milk, eggs, and pandan leaves) and butter. Served with soft-boiled eggs seasoned with soy sauce and white pepper, and accompanied by kopi (local coffee) or teh (tea).",
            "",
            "food",
        ),
        # --- Transport ---
        (
            "How does the Singapore MRT fare system work?",
            "MRT fares are calculated by the distance travelled, using a tiered distance-based system. Payment is by EZ-Link card, Singapore Tourist Pass, or contactless bank card. Concession cards offer reduced fares for students, seniors, and persons with disabilities. Fares typically range from SGD 0.77 to SGD 2.60.",
            "",
            "transport",
        ),
        (
            "What is the LRT system in Singapore?",
            "Singapore has three LRT networks: Bukit Panjang LRT (BPLRT) operated by SMRT, and Sengkang and Punggol LRT operated by SBS Transit. All are automated and driverless, providing feeder services from HDB estates to nearby MRT stations.",
            "",
            "transport",
        ),
        (
            "What is the EZ-Link card?",
            "The EZ-Link card is Singapore's contactless smart card for public transport. It works on MRT, LRT, and all public buses. Cards can also be used at selected retail outlets. Top-up is available at TransitLink Ticket Offices, ATMs, and convenience stores.",
            "",
            "transport",
        ),
        (
            "How do you book a taxi in Singapore?",
            "Taxis can be hailed at taxi stands, from the roadside outside CBD and peak hours, or booked via apps: Grab, ComfortDelGro Taxi, TADA, and Gojek operate in Singapore. Booking surcharges and peak-hour charges apply. Booking via app is the most common method.",
            "",
            "transport",
        ),
        (
            "What is the TEL MRT line?",
            "The Thomson-East Coast Line (TEL) is Singapore's sixth MRT line, operated by SMRT. It runs from Woodlands North in the north to Sungei Bedok in the east, passing through Orchard, Marina Bay, and Gardens by the Bay. It was completed in phases between 2020 and 2024.",
            "",
            "transport",
        ),
        (
            "What is the Jurong Region Line?",
            "The Jurong Region Line (JRL) is Singapore's seventh MRT line, under construction. It will serve the western region including Jurong, Choa Chu Kang, Boon Lay, and Tengah. When completed, it will have 24 stations. Construction began in 2018 with phased opening from 2027.",
            "",
            "transport",
        ),
        (
            "What is the ART in Singapore?",
            "The Autonomous Rail Transit (ART) is being piloted in the Jurong Lake District. It runs on rubber tyres guided by painted lines, with no physical rails. Powered by electricity, it can carry up to 500 passengers per train. ART offers flexible route deployment compared to fixed-track MRT.",
            "",
            "transport",
        ),
        # --- Housing ---
        (
            "What are HDB BTO flats?",
            "BTO (Build-To-Order) is HDB's primary public housing scheme. Applicants select a flat unit and wait 3-5 years for construction. BTO exercises are launched quarterly with new projects in various towns. First-timer married couples and families receive priority balloting. Prices are subsidised.",
            "",
            "housing",
        ),
        (
            "What is the HDB ethnic integration policy?",
            "The Ethnic Integration Policy (EIP) sets block and neighbourhood limits for the proportion of Chinese, Malay, Indian, and Other residents in each HDB development. It prevents ethnic enclaves and promotes racial integration. Sellers must check EIP eligibility before listing their flat for sale.",
            "",
            "housing",
        ),
        (
            "What is the minimum occupancy period for HDB flats?",
            "HDB owners must complete a Minimum Occupation Period (MOP) of five years before selling the flat on the resale market, renting out the entire flat, or purchasing private property. The MOP begins from the date of key collection, not the date of application.",
            "",
            "housing",
        ),
        (
            "What is the CPF Housing Grant?",
            "The Enhanced CPF Housing Grant (EHG) provides up to SGD 80,000 for eligible first-time buyers of HDB flats. The amount depends on household income. Grants are credited to the CPF Ordinary Account and used for downpayment or monthly instalments. Additional grants are available for proximity to parents.",
            "",
            "housing",
        ),
        (
            "What is an HDB resale flat?",
            "An HDB resale flat is a public housing unit sold by its owner on the open market after the MOP. Prices are negotiated between buyer and seller. The resale market provides immediate access to housing. Buyers may use CPF savings and HDB housing loans. A Cash-Over-Valuation (COV) may be payable.",
            "",
            "housing",
        ),
        (
            "What is a private condominium in Singapore?",
            "Private condominiums are non-HDB residential developments with shared facilities (pool, gym, function rooms). They are sold by developers or on the resale market. Foreigners can purchase condos (subject to Additional Buyer's Stamp Duty). Prices range from SGD 800 psf to over SGD 4,000 psf in prime areas.",
            "",
            "housing",
        ),
        # --- Economy & Finance ---
        (
            "What is the Singapore Exchange (SGX)?",
            "The Singapore Exchange (SGX) is Singapore's stock exchange, listing equities, derivatives, bonds, and REITs. The Straits Times Index (STI) tracks the top 30 companies. SGX is a key financial infrastructure for Southeast Asia, with dual-currency and multi-currency trading capabilities.",
            "",
            "finance",
        ),
        (
            "What is a REIT in Singapore?",
            "A Real Estate Investment Trust (REIT) pools investor capital to invest in income-generating real estate. Singapore has one of Asia's largest REIT markets with over 40 listed REITs and property trusts. REITs must distribute at least 90% of taxable income as dividends. Popular REITs: CapitaLand Integrated Commercial Trust, Ascendas REIT.",
            "",
            "finance",
        ),
        (
            "What is the SGD exchange rate policy?",
            "The Monetary Authority of Singapore (MAS) manages monetary policy through the Singapore Dollar (SGD) nominal effective exchange rate (NEER) band, not interest rates. MAS adjusts the band's centre, slope, and width. This approach suits Singapore's small, open, trade-dependent economy.",
            "",
            "finance",
        ),
        (
            "What is the SkillsFuture credit?",
            "SkillsFuture Credit provides SGD 500 to Singaporeans aged 25 and above for approved training courses. Additional top-ups are provided for mid-career workers aged 40 and above. Credits are accessed via the MySkillsFuture portal and expire if unused. Over 600,000 Singaporeans have used SkillsFuture credits.",
            "",
            "education",
        ),
        (
            "What is the PSLE in Singapore?",
            "The Primary School Leaving Examination (PSLE) is taken by Primary 6 students (approximately age 12). It is used for secondary school placement. Students are scored using Achievement Level (AL) grades from AL1 (best) to AL8 for each subject. PSLE AL scores replaced T-scores in 2021.",
            "",
            "education",
        ),
        (
            "What is the Singapore Budget?",
            "Singapore's annual budget is presented by the Finance Minister in February. It allocates government spending across healthcare, education, defence, social support, and economic development. Singapore aims for a broadly balanced budget over each term of government. Surpluses can be invested by GIC and Temasek.",
            "",
            "finance",
        ),
        (
            "What is Temasek Holdings?",
            "Temasek Holdings is Singapore's state investment company, owned by the Singapore government. It manages a portfolio of over SGD 400 billion, invested in Singaporean companies (DBS, SingTel, Singapore Airlines) and global companies. Temasek returns are used to support the Singapore government budget.",
            "",
            "finance",
        ),
        # --- ML / Tech ---
        (
            "What is Singapore's Smart Nation initiative?",
            "Smart Nation is Singapore's government-led initiative to harness digital technology to improve citizens' lives and business efficiency. Key projects: Singpass (digital identity), LifeSG (government services), the National Digital Identity (NDI) framework, and smart urban mobility systems.",
            "",
            "technology",
        ),
        (
            "What is PDPA in Singapore?",
            "The Personal Data Protection Act (PDPA) governs the collection, use, disclosure, and care of personal data in Singapore. Key obligations: purpose limitation, consent, notification, access/correction rights, data protection, and retention limits. Mandatory data breach notification applies to significant breaches.",
            "",
            "technology",
        ),
        (
            "What is GovTech Singapore?",
            "Government Technology Agency (GovTech) is the lead agency driving Singapore's digital government transformation. It develops and operates government digital infrastructure: Singpass, CorpPass, National Digital Identity, Whole of Government Application Analytics, and the Singapore Government Tech Stack.",
            "",
            "technology",
        ),
        (
            "What is Singapore's National AI Strategy?",
            "Singapore's National AI Strategy (NAIS 2.0, 2023) aims to develop AI capabilities across key sectors: health, education, public service, and the economy. It focuses on AI talent development, international governance, trusted AI deployment, and Singapore as an AI centre for the region.",
            "",
            "technology",
        ),
        (
            "What is data.gov.sg?",
            "data.gov.sg is Singapore's open data portal maintained by GovTech. It provides free access to government datasets covering housing, transport, environment, economy, demographics, and geospatial data. Datasets are available as CSV, GeoJSON, and API. Over 2,000 datasets are published.",
            "",
            "technology",
        ),
        (
            "What is OneMap in Singapore?",
            "OneMap is Singapore's authoritative national map platform maintained by the Singapore Land Authority (SLA). It provides geospatial data and APIs for planning areas, addresses, transportation, and land parcels. It is the base map for government and commercial applications in Singapore.",
            "",
            "technology",
        ),
        # --- ML with Singapore context ---
        (
            "How is ML used in Singapore's public housing?",
            "HDB uses ML for resale price prediction, maintenance scheduling (predicting lift failures), and estate planning. Data from data.gov.sg enables public research on housing trends. Students can download HDB resale transaction data to build regression models predicting flat prices.",
            "",
            "ml_singapore",
        ),
        (
            "How is ML used in Singapore's transport system?",
            "LTA uses ML for bus arrival predictions, MRT fault detection, traffic signal optimisation, and COE price forecasting. The GTFS (General Transit Feed Specification) data is available publicly. Taxi demand prediction is a classic time-series forecasting problem with Singapore taxi data.",
            "",
            "ml_singapore",
        ),
        (
            "What is the Singapore COVID-19 data?",
            "Singapore published detailed COVID-19 case, vaccination, and testing data on data.gov.sg throughout the pandemic. This data is used for epidemiological modelling exercises including SIR/SEIR models, vaccination impact analysis, and time-series forecasting of case counts.",
            "",
            "ml_singapore",
        ),
    ]

    # Expand to 1,000 rows by paraphrasing
    extended = pairs.copy()
    paraphrase_prefix = [
        "Can you explain: ",
        "Please describe: ",
        "Give me information about: ",
        "Tell me about: ",
        "I want to learn about: ",
        "What should I know about: ",
        "Could you clarify: ",
        "Please elaborate on: ",
        "Describe for me: ",
        "Help me understand: ",
    ]
    while len(extended) < 1000:
        inst, resp, ctx, cat = random.choice(pairs)
        prefix = random.choice(paraphrase_prefix)
        extended.append((prefix + inst.lower(), resp, ctx, cat))

    extended = extended[:1000]

    df = pl.DataFrame(
        {
            "instruction": [e[0] for e in extended],
            "response": [e[1] for e in extended],
            "context": [e[2] for e in extended],
            "category": [e[3] for e in extended],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 8. preference_pairs.parquet  (ascent06) — 500 rows
# ---------------------------------------------------------------------------


def make_preference_pairs() -> pl.DataFrame:
    """
    Preference pairs for DPO (M6.2) — 500 rows.
    Each row has a prompt, a chosen (substantive) response, and a rejected
    (plausible-but-shallow) response.
    """
    pairs = [
        (
            "Explain overfitting in machine learning.",
            "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalise to new examples. The model has high variance. Regularisation techniques (L1/L2 penalties, dropout), early stopping, cross-validation, and more training data all help mitigate overfitting.",
            "Overfitting is when the model is overfitted. It is bad and you should avoid it by using regularisation.",
            "ml_concepts",
        ),
        (
            "What is the difference between classification and regression?",
            "Classification predicts discrete class labels (e.g., spam vs not spam, credit default vs no default). Regression predicts continuous numeric values (e.g., HDB resale price, income). Both are supervised learning tasks but differ in output type, loss functions (cross-entropy vs MSE), and evaluation metrics.",
            "Classification is for categories and regression is for numbers. They are both supervised.",
            "ml_concepts",
        ),
        (
            "How does a random forest work?",
            "A random forest trains multiple decision trees on bootstrapped subsets of the data (bagging), with each tree seeing only a random subset of features at each split (feature randomisation). Predictions are aggregated by majority vote (classification) or averaging (regression). This reduces variance compared to a single tree.",
            "Random forest has many trees and they vote. It is better than one tree because more trees are better.",
            "ml_algorithms",
        ),
        (
            "What is gradient boosting?",
            "Gradient boosting builds an ensemble sequentially. Each new tree corrects the residual errors of the current ensemble by fitting the negative gradient of the loss function. Shrinkage (learning rate) controls each tree's contribution. XGBoost adds L1/L2 regularisation and second-order gradient information for efficiency.",
            "Gradient boosting uses gradients and boosts things. It is used in competitions and is very accurate.",
            "ml_algorithms",
        ),
        (
            "Explain the bias-variance trade-off.",
            "Bias measures error from overly simple model assumptions (underfitting — the model cannot capture the true signal). Variance measures sensitivity to training data fluctuations (overfitting — the model fits noise). As model complexity increases, bias falls and variance rises. The optimal complexity minimises total error (bias² + variance + irreducible noise).",
            "Bias and variance are both bad. You want low bias and low variance. Try different models.",
            "ml_theory",
        ),
        (
            "What is precision and when should you prioritise it?",
            "Precision is TP / (TP + FP) — of all predicted positives, what fraction is correct. Prioritise precision when false positives are costly: spam filtering (blocking legitimate emails is harmful), legal document classification, or credit approval (false approval wastes capital). The precision-recall trade-off is controlled by the classification threshold.",
            "Precision is TP divided by TP plus FP. It is good when you don't want false positives.",
            "ml_evaluation",
        ),
        (
            "How do you handle class imbalance?",
            "Strategies include: oversampling the minority class (SMOTE, ADASYN), undersampling the majority class, class-weight adjustment in the loss function, and using imbalance-robust evaluation metrics (PR-AUC, balanced accuracy, Matthews Correlation Coefficient). Threshold tuning after calibration often achieves the best practical result.",
            "Use SMOTE or oversample. You can also try different models. Sometimes class imbalance is not a big problem.",
            "ml_practical",
        ),
        (
            "What is SHAP and how is it used?",
            "SHAP assigns each feature a contribution value for a specific prediction using Shapley values from cooperative game theory. It satisfies three fairness axioms (efficiency, symmetry, dummy). TreeSHAP computes exact values for tree models in polynomial time. Used for local explanations (per prediction) and global feature importance (mean absolute SHAP).",
            "SHAP explains the model. It shows which features are important. It uses Shapley values from game theory.",
            "ml_interpretability",
        ),
        (
            "What is the purpose of a validation set?",
            "A validation set is a held-out data split used during training to tune hyperparameters and monitor for overfitting, without contaminating the test set. It enables early stopping and model selection. The test set must be kept completely unseen until final evaluation to provide an unbiased generalisation estimate.",
            "Validation set is used to validate the model. It is different from the test set. You use it during training.",
            "ml_practical",
        ),
        (
            "How does k-means clustering decide the number of clusters?",
            "K-means requires k to be specified in advance. Methods to choose k: (1) elbow method — plot inertia (within-cluster sum of squares) vs k and look for an elbow; (2) silhouette score — measures cohesion vs separation, higher is better; (3) gap statistic — compares inertia to random data; (4) domain knowledge about expected groupings.",
            "You try different values of k and pick the best one. The elbow method shows where adding clusters stops helping.",
            "ml_algorithms",
        ),
        (
            "What is transfer learning and when should you use it?",
            "Transfer learning reuses a model pre-trained on a large dataset as a starting point. Feature extraction freezes the pre-trained layers and trains only the head. Full fine-tuning updates all layers on the target data. Use when: labelled data is scarce, compute is limited, or the source and target domains share low-level features.",
            "Transfer learning transfers knowledge from one model to another. Use it when you don't have enough data.",
            "deep_learning",
        ),
        (
            "What is the attention mechanism in transformers?",
            "Scaled dot-product attention computes a weighted sum of value vectors. Weights are determined by query-key dot products, scaled by 1/√d_k and normalised with softmax. Multi-head attention runs h parallel attention functions with projected Q, K, V matrices, then concatenates. Self-attention allows each position to attend to all others.",
            "Attention looks at other tokens in the sequence. It uses queries, keys, and values. Multi-head attention does this multiple times.",
            "deep_learning",
        ),
        (
            "How do you evaluate a language model?",
            "Perplexity measures prediction confidence on held-out text (lower is better). Task-specific benchmarks (MMLU for knowledge, HellaSwag for commonsense, HumanEval for coding) evaluate capabilities. For instruction-tuned models, MT-Bench and Chatbot Arena assess conversational quality via human preferences.",
            "You evaluate LLMs with perplexity and benchmarks like MMLU. Human evaluation is also used.",
            "llm_techniques",
        ),
        (
            "What is RAG and when should you use it?",
            "Retrieval-Augmented Generation retrieves relevant documents and conditions the LLM response on them. Use RAG when: (1) knowledge must be up-to-date beyond training cutoff, (2) verifiable sources are required, (3) domain-specific facts are needed, (4) full fine-tuning is too expensive or the knowledge changes frequently.",
            "RAG retrieves documents and passes them to the LLM. Use it when the LLM doesn't know the answer.",
            "llm_techniques",
        ),
        (
            "What is LoRA and why is it useful?",
            "LoRA injects trainable low-rank matrices (AB where A∈ℝ^{d×r}, B∈ℝ^{r×k}, r≪d) into transformer attention layers. Only A and B are trained, reducing trainable parameters by 10-100x. This allows fine-tuning 7B+ parameter models on a single GPU. QLoRA additionally quantises the base model to 4-bit.",
            "LoRA is a way to fine-tune LLMs efficiently. It uses low-rank matrices so you train fewer parameters.",
            "llm_techniques",
        ),
        (
            "How does RLHF work?",
            "RLHF has three stages: (1) supervised fine-tuning (SFT) on instruction-response pairs, (2) reward model training on human preference comparisons between model outputs, (3) PPO fine-tuning to maximise reward while a KL divergence penalty prevents excessive drift from the SFT policy. DPO simplifies stage 3 by removing the need for online RL.",
            "RLHF uses human feedback to train the model. A reward model is trained first, then the LLM is fine-tuned.",
            "llm_alignment",
        ),
        (
            "What is DPO compared to PPO for alignment?",
            "DPO reformulates preference learning as a supervised classification on (prompt, chosen, rejected) triples. The implicit reward is derived from the log ratio of policy and reference model probabilities. No reward model or online rollouts are needed. DPO is simpler, more stable, and typically matches PPO quality on instruction-following tasks.",
            "DPO is simpler than PPO. It doesn't need a reward model. DPO directly optimises on preference data.",
            "llm_alignment",
        ),
        (
            "What is data drift and how do you detect it?",
            "Data drift is a change in the statistical distribution of input features after model deployment. Detection methods: Population Stability Index (PSI > 0.2 = significant drift), Kolmogorov-Smirnov (KS) test for numeric features, chi-squared test for categorical features, and monitoring feature summary statistics (mean, std, quantiles). Alerts trigger retraining pipelines.",
            "Data drift is when the data changes over time. You can detect it with PSI or statistical tests.",
            "mlops",
        ),
        (
            "What is a feature store and why is it important?",
            "A feature store is a centralised repository that stores feature transformation logic and materialised feature values. It ensures training-serving consistency (prevents train-serve skew). Features are versioned, discoverable, and reusable across teams. Point-in-time correct joins prevent future leakage in training data.",
            "A feature store stores features. It is useful for sharing features across teams.",
            "mlops",
        ),
        (
            "How do you monitor a model in production?",
            "Monitor four dimensions: (1) input data distributions (PSI, KS test for drift), (2) prediction distributions (output drift), (3) business metrics (revenue, conversions, click-through rate), (4) ground truth performance when labels become available (AUC, precision, recall). Set alerting thresholds and schedule periodic retraining when metrics degrade beyond acceptable bounds.",
            "Monitor the model predictions and check if they are still accurate. Use dashboards and alerts. Retrain when needed.",
            "mlops",
        ),
    ]

    # Expand to 500 rows
    extended = pairs.copy()
    while len(extended) < 500:
        p, c, r, cat = random.choice(pairs)
        extended.append((p, c, r, cat))

    extended = extended[:500]

    df = pl.DataFrame(
        {
            "prompt": [e[0] for e in extended],
            "chosen": [e[1] for e in extended],
            "rejected": [e[2] for e in extended],
            "category": [e[3] for e in extended],
        }
    )

    return df


# ---------------------------------------------------------------------------
# 9. mrt_stations.parquet  (ascent_assessment) — keep as-is (~150 stations)
# ---------------------------------------------------------------------------


def make_mrt_stations() -> pl.DataFrame:
    """
    Singapore MRT stations with town mapping and coordinates (~150 rows).
    Self-referential: nearest_mrt column references station_name.
    """
    stations = [
        # (station_name, town, line, lat, lon)
        # --- North-South Line (NSL) ---
        ("Jurong East", "Jurong East", "NSL", 1.3331, 103.7422),
        ("Bukit Batok", "Bukit Batok", "NSL", 1.3485, 103.7496),
        ("Bukit Gombak", "Bukit Batok", "NSL", 1.3586, 103.7516),
        ("Choa Chu Kang", "Choa Chu Kang", "NSL", 1.3853, 103.7446),
        ("Yew Tee", "Choa Chu Kang", "NSL", 1.3969, 103.7474),
        ("Kranji", "Woodlands", "NSL", 1.4252, 103.7619),
        ("Marsiling", "Woodlands", "NSL", 1.4328, 103.7744),
        ("Woodlands", "Woodlands", "NSL", 1.4369, 103.7863),
        ("Admiralty", "Sembawang", "NSL", 1.4408, 103.8009),
        ("Sembawang", "Sembawang", "NSL", 1.4491, 103.8199),
        ("Canberra", "Sembawang", "NSL", 1.4432, 103.8296),
        ("Yishun", "Yishun", "NSL", 1.4294, 103.8354),
        ("Khatib", "Yishun", "NSL", 1.4175, 103.8330),
        ("Yio Chu Kang", "Ang Mo Kio", "NSL", 1.3817, 103.8449),
        ("Ang Mo Kio", "Ang Mo Kio", "NSL", 1.3699, 103.8496),
        ("Bishan", "Bishan", "NSL", 1.3510, 103.8485),
        ("Braddell", "Toa Payoh", "NSL", 1.3402, 103.8468),
        ("Toa Payoh", "Toa Payoh", "NSL", 1.3322, 103.8469),
        ("Novena", "Novena", "NSL", 1.3204, 103.8436),
        ("Newton", "Novena", "NSL", 1.3132, 103.8388),
        ("Orchard", "Orchard", "NSL", 1.3048, 103.8318),
        ("Somerset", "Orchard", "NSL", 1.3006, 103.8388),
        ("Dhoby Ghaut", "Museum", "NSL", 1.2988, 103.8456),
        ("City Hall", "Downtown", "NSL", 1.2931, 103.8520),
        ("Raffles Place", "Downtown", "NSL", 1.2831, 103.8513),
        ("Marina Bay", "Downtown", "NSL", 1.2762, 103.8554),
        ("Marina South Pier", "Downtown", "NSL", 1.2711, 103.8634),
        # --- East-West Line (EWL) ---
        ("Pasir Ris", "Pasir Ris", "EWL", 1.3731, 103.9494),
        ("Tampines", "Tampines", "EWL", 1.3526, 103.9453),
        ("Simei", "Tampines", "EWL", 1.3431, 103.9531),
        ("Tanah Merah", "Bedok", "EWL", 1.3273, 103.9461),
        ("Bedok", "Bedok", "EWL", 1.3240, 103.9299),
        ("Kembangan", "Bedok", "EWL", 1.3209, 103.9130),
        ("Eunos", "Geylang", "EWL", 1.3196, 103.9031),
        ("Paya Lebar", "Geylang", "EWL", 1.3178, 103.8926),
        ("Aljunied", "Geylang", "EWL", 1.3163, 103.8830),
        ("Kallang", "Kallang", "EWL", 1.3118, 103.8716),
        ("Lavender", "Kallang", "EWL", 1.3072, 103.8638),
        ("Bugis", "Rochor", "EWL", 1.3008, 103.8559),
        ("City Hall", "Downtown", "EWL", 1.2931, 103.8520),
        ("Tanjong Pagar", "Downtown", "EWL", 1.2762, 103.8454),
        ("Outram Park", "Outram", "EWL", 1.2801, 103.8396),
        ("Tiong Bahru", "Bukit Merah", "EWL", 1.2863, 103.8272),
        ("Redhill", "Bukit Merah", "EWL", 1.2895, 103.8164),
        ("Queenstown", "Queenstown", "EWL", 1.2943, 103.8059),
        ("Commonwealth", "Queenstown", "EWL", 1.3022, 103.7982),
        ("Buona Vista", "Queenstown", "EWL", 1.3067, 103.7901),
        ("Dover", "Clementi", "EWL", 1.3115, 103.7787),
        ("Clementi", "Clementi", "EWL", 1.3151, 103.7654),
        ("Jurong East", "Jurong East", "EWL", 1.3331, 103.7422),
        ("Chinese Garden", "Jurong East", "EWL", 1.3421, 103.7334),
        ("Lakeside", "Jurong West", "EWL", 1.3443, 103.7208),
        ("Boon Lay", "Jurong West", "EWL", 1.3388, 103.7062),
        ("Pioneer", "Jurong West", "EWL", 1.3374, 103.6973),
        ("Joo Koon", "Jurong West", "EWL", 1.3277, 103.6782),
        ("Gul Circle", "Jurong West", "EWL", 1.3197, 103.6609),
        ("Tuas Crescent", "Tuas", "EWL", 1.3218, 103.6491),
        ("Tuas West Road", "Tuas", "EWL", 1.3301, 103.6393),
        ("Tuas Link", "Tuas", "EWL", 1.3403, 103.6367),
        # --- Circle Line (CCL) ---
        ("Dhoby Ghaut", "Museum", "CCL", 1.2988, 103.8456),
        ("Bras Basah", "Museum", "CCL", 1.2965, 103.8507),
        ("Esplanade", "Downtown", "CCL", 1.2934, 103.8557),
        ("Promenade", "Downtown", "CCL", 1.2933, 103.8605),
        ("Nicoll Highway", "Kallang", "CCL", 1.2999, 103.8636),
        ("Stadium", "Kallang", "CCL", 1.3027, 103.8748),
        ("Mountbatten", "Kallang", "CCL", 1.3064, 103.8822),
        ("Dakota", "Geylang", "CCL", 1.3083, 103.8883),
        ("Paya Lebar", "Geylang", "CCL", 1.3178, 103.8926),
        ("MacPherson", "Geylang", "CCL", 1.3267, 103.8894),
        ("Tai Seng", "Serangoon", "CCL", 1.3354, 103.8879),
        ("Bartley", "Serangoon", "CCL", 1.3424, 103.8799),
        ("Serangoon", "Serangoon", "CCL", 1.3497, 103.8732),
        ("Lorong Chuan", "Serangoon", "CCL", 1.3524, 103.8641),
        ("Bishan", "Bishan", "CCL", 1.3510, 103.8485),
        ("Marymount", "Bishan", "CCL", 1.3483, 103.8395),
        ("Caldecott", "Bukit Timah", "CCL", 1.3375, 103.8323),
        ("Botanic Gardens", "Bukit Timah", "CCL", 1.3222, 103.8155),
        ("Farrer Road", "Bukit Timah", "CCL", 1.3172, 103.8070),
        ("Holland Village", "Buona Vista", "CCL", 1.3115, 103.7960),
        ("Buona Vista", "Queenstown", "CCL", 1.3067, 103.7901),
        ("one-north", "Buona Vista", "CCL", 1.2995, 103.7869),
        ("Kent Ridge", "Clementi", "CCL", 1.2938, 103.7847),
        ("Haw Par Villa", "Queenstown", "CCL", 1.2827, 103.7822),
        ("Pasir Panjang", "Queenstown", "CCL", 1.2763, 103.7916),
        ("Labrador Park", "Buona Vista", "CCL", 1.2722, 103.8023),
        ("Telok Blangah", "Bukit Merah", "CCL", 1.2706, 103.8092),
        ("HarbourFront", "Bukit Merah", "CCL", 1.2653, 103.8218),
        # --- Downtown Line (DTL) ---
        ("Bukit Panjang", "Bukit Panjang", "DTL", 1.3784, 103.7659),
        ("Cashew", "Bukit Panjang", "DTL", 1.3698, 103.7749),
        ("Hillview", "Bukit Timah", "DTL", 1.3626, 103.7677),
        ("Beauty World", "Bukit Timah", "DTL", 1.3409, 103.7759),
        ("King Albert Park", "Bukit Timah", "DTL", 1.3354, 103.7832),
        ("Sixth Avenue", "Bukit Timah", "DTL", 1.3297, 103.7957),
        ("Tan Kah Kee", "Bukit Timah", "DTL", 1.3258, 103.8075),
        ("Botanic Gardens", "Bukit Timah", "DTL", 1.3222, 103.8155),
        ("Stevens", "Novena", "DTL", 1.3197, 103.8264),
        ("Newton", "Novena", "DTL", 1.3132, 103.8388),
        ("Little India", "Rochor", "DTL", 1.3067, 103.8496),
        ("Rochor", "Rochor", "DTL", 1.3039, 103.8526),
        ("Bugis", "Rochor", "DTL", 1.3008, 103.8559),
        ("Promenade", "Downtown", "DTL", 1.2933, 103.8605),
        ("Bayfront", "Downtown", "DTL", 1.2830, 103.8593),
        ("Downtown", "Downtown", "DTL", 1.2793, 103.8529),
        ("Telok Ayer", "Downtown", "DTL", 1.2812, 103.8479),
        ("Fort Canning", "Museum", "DTL", 1.2936, 103.8445),
        ("Bencoolen", "Rochor", "DTL", 1.2980, 103.8499),
        ("Jalan Besar", "Rochor", "DTL", 1.3059, 103.8567),
        ("Bendemeer", "Kallang", "DTL", 1.3135, 103.8617),
        ("Geylang Bahru", "Kallang", "DTL", 1.3214, 103.8710),
        ("Mattar", "Geylang", "DTL", 1.3263, 103.8836),
        ("Ubi", "Geylang", "DTL", 1.3294, 103.8956),
        ("Kaki Bukit", "Bedok", "DTL", 1.3342, 103.9054),
        ("Bedok North", "Bedok", "DTL", 1.3337, 103.9179),
        ("Bedok Reservoir", "Bedok", "DTL", 1.3356, 103.9296),
        ("Tampines West", "Tampines", "DTL", 1.3449, 103.9368),
        ("Tampines", "Tampines", "DTL", 1.3526, 103.9453),
        ("Tampines East", "Tampines", "DTL", 1.3574, 103.9533),
        ("Upper Changi", "Changi", "DTL", 1.3413, 103.9615),
        ("Expo", "Changi", "DTL", 1.3350, 103.9612),
        ("Changi Airport", "Changi", "DTL", 1.3592, 103.9885),
        # --- Thomson-East Coast Line (TEL) ---
        ("Woodlands North", "Woodlands", "TEL", 1.4487, 103.7875),
        ("Woodlands", "Woodlands", "TEL", 1.4369, 103.7863),
        ("Woodlands South", "Woodlands", "TEL", 1.4242, 103.7986),
        ("Springleaf", "Ang Mo Kio", "TEL", 1.3980, 103.8149),
        ("Lentor", "Ang Mo Kio", "TEL", 1.3848, 103.8354),
        ("Mayflower", "Ang Mo Kio", "TEL", 1.3706, 103.8397),
        ("Bright Hill", "Bishan", "TEL", 1.3621, 103.8379),
        ("Upper Thomson", "Bishan", "TEL", 1.3536, 103.8327),
        ("Caldecott", "Bukit Timah", "TEL", 1.3375, 103.8323),
        ("Stevens", "Novena", "TEL", 1.3197, 103.8264),
        ("Napier", "Tanglin", "TEL", 1.3074, 103.8186),
        ("Orchard Boulevard", "Orchard", "TEL", 1.3028, 103.8228),
        ("Orchard", "Orchard", "TEL", 1.3048, 103.8318),
        ("Great World", "Queenstown", "TEL", 1.2952, 103.8296),
        ("Havelock", "Outram", "TEL", 1.2883, 103.8343),
        ("Outram Park", "Outram", "TEL", 1.2801, 103.8396),
        ("Maxwell", "Downtown", "TEL", 1.2803, 103.8453),
        ("Shenton Way", "Downtown", "TEL", 1.2777, 103.8490),
        ("Marina Bay", "Downtown", "TEL", 1.2762, 103.8554),
        ("Marina South", "Downtown", "TEL", 1.2701, 103.8623),
        ("Gardens by the Bay", "Downtown", "TEL", 1.2816, 103.8635),
        ("Tanjong Rhu", "Kallang", "TEL", 1.2990, 103.8720),
        ("Katong Park", "Marine Parade", "TEL", 1.3026, 103.8830),
        ("Tanjong Katong", "Marine Parade", "TEL", 1.3069, 103.8933),
        ("Marine Parade", "Marine Parade", "TEL", 1.3032, 103.9043),
        ("Marine Terrace", "Marine Parade", "TEL", 1.3063, 103.9157),
        ("Siglap", "Bedok", "TEL", 1.3100, 103.9271),
        ("Bayshore", "Bedok", "TEL", 1.3155, 103.9375),
        ("Bedok South", "Bedok", "TEL", 1.3201, 103.9453),
        ("Sungei Bedok", "Bedok", "TEL", 1.3261, 103.9543),
    ]

    # De-duplicate by station name + line
    seen = set()
    unique_stations = []
    for s in stations:
        key = (s[0], s[2])
        if key not in seen:
            seen.add(key)
            unique_stations.append(s)

    m = len(unique_stations)
    names = [s[0] for s in unique_stations]
    towns = [s[1] for s in unique_stations]
    lines = [s[2] for s in unique_stations]
    lats = [s[3] for s in unique_stations]
    lons = [s[4] for s in unique_stations]

    nearest_mrt = []
    dist_to_nearest = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        best_name = names[i]
        best_dist = float("inf")
        for j, (la2, lo2) in enumerate(zip(lats, lons)):
            if j == i or lines[j] != lines[i]:
                continue
            dlat = math.radians(la2 - la) * 6371
            dlon = math.radians(lo2 - lo) * 6371 * math.cos(math.radians(la))
            d = math.sqrt(dlat**2 + dlon**2)
            if d < best_dist:
                best_dist = d
                best_name = names[j]
        nearest_mrt.append(best_name)
        dist_to_nearest.append(round(best_dist, 3))

    df = pl.DataFrame(
        {
            "station_name": names,
            "town": towns,
            "line": lines,
            "latitude": lats,
            "longitude": lons,
            "nearest_mrt": nearest_mrt,
            "distance_to_mrt_km": dist_to_nearest,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 10. schools.parquet  (ascent_assessment) — 350 rows
# ---------------------------------------------------------------------------


def make_schools() -> pl.DataFrame:
    """
    Singapore schools (~350 rows) covering primary, secondary, and JC.
    """
    primary_schools = [
        ("Ai Tong School", "Bishan"),
        ("Alexandra Primary School", "Queenstown"),
        ("Anchor Green Primary School", "Pasir Ris"),
        ("Anderson Primary School", "Ang Mo Kio"),
        ("Ang Mo Kio Primary School", "Ang Mo Kio"),
        ("Balestier Hill Primary School", "Toa Payoh"),
        ("Beacon Primary School", "Woodlands"),
        ("Bedok Green Primary School", "Bedok"),
        ("Bendemeer Primary School", "Kallang"),
        ("Blangah Rise Primary School", "Bukit Merah"),
        ("Boon Lay Garden Primary School", "Jurong West"),
        ("Bukit Batok Primary School", "Bukit Batok"),
        ("Bukit Panjang Primary School", "Bukit Panjang"),
        ("Bukit Timah Primary School", "Bukit Timah"),
        ("Bukit View Primary School", "Bukit Timah"),
        ("Casuarina Primary School", "Yishun"),
        ("Canberra Primary School", "Sembawang"),
        ("Cedar Primary School", "Serangoon"),
        ("Changkat Primary School", "Tampines"),
        ("Chij (Kellock)", "Queenstown"),
        ("Chij Primary (Toa Payoh)", "Toa Payoh"),
        ("Chongzheng Primary School", "Bishan"),
        ("Clementi Primary School", "Clementi"),
        ("Compass Primary School", "Sengkang"),
        ("Damai Primary School", "Bedok"),
        ("Edgefield Primary School", "Punggol"),
        ("Elias Park Primary School", "Pasir Ris"),
        ("Endeavour Primary School", "Sembawang"),
        ("Eunos Primary School", "Bedok"),
        ("Farrer Park Primary School", "Little India"),
        ("Fengshan Primary School", "Bedok"),
        ("Fernvale Primary School", "Sengkang"),
        ("Frontier Primary School", "Jurong West"),
        ("Fuhua Primary School", "Jurong West"),
        ("Geylang Methodist School (Primary)", "Geylang"),
        ("Greendale Primary School", "Punggol"),
        ("Greenridge Primary School", "Bukit Panjang"),
        ("Greenwood Primary School", "Bukit Timah"),
        ("Griffiths Primary School", "Bukit Merah"),
        ("Henry Park Primary School", "Buona Vista"),
        ("Holy Innocents' Primary School", "Hougang"),
        ("Horizon Primary School", "Pasir Ris"),
        ("Hougang Primary School", "Hougang"),
        ("Jiemin Primary School", "Bishan"),
        ("Jurongville Primary School", "Jurong West"),
        ("Keming Primary School", "Jurong West"),
        ("Kong Hwa School", "Geylang"),
        ("Kranji Primary School", "Woodlands"),
        ("Lianhua Primary School", "Clementi"),
        ("Maha Bodhi School", "Geylang"),
        ("Marsiling Primary School", "Woodlands"),
        ("Marymount Convent School", "Bishan"),
        ("Mee Toh School", "Tampines"),
        ("Meridian Primary School", "Pasir Ris"),
        ("Methodist Girls' School (Primary)", "Buona Vista"),
        ("Nan Chiau Primary School", "Sengkang"),
        ("Nanyang Primary School", "Buona Vista"),
        ("Naval Base Primary School", "Yishun"),
        ("New Town Primary School", "Queenstown"),
        ("Ngee Ann Primary School", "Hougang"),
        ("North Spring Primary School", "Hougang"),
        ("North View Primary School", "Woodlands"),
        ("Northland Primary School", "Woodlands"),
        ("Palm View Primary School", "Jurong West"),
        ("Park View Primary School", "Bukit Batok"),
        ("Pasir Ris Primary School", "Pasir Ris"),
        ("Pei Chun Public School", "Toa Payoh"),
        ("Pei Hwa Presbyterian Primary School", "Tampines"),
        ("Pei Tong Primary School", "Clementi"),
        ("Poi Ching School", "Tampines"),
        ("Punggol Green Primary School", "Punggol"),
        ("Qihua Primary School", "Choa Chu Kang"),
        ("Radin Mas Primary School", "Bukit Merah"),
        ("Raffles Girls' Primary School", "Buona Vista"),
        ("Red Swastika School", "Bedok"),
        ("River Valley Primary School", "Buona Vista"),
        ("Riverside Primary School", "Woodlands"),
        ("Rosyth School", "Hougang"),
        ("Rulang Primary School", "Jurong East"),
        ("Sembawang Primary School", "Sembawang"),
        ("Sengkang Green Primary School", "Sengkang"),
        ("Shuqun Primary School", "Jurong West"),
        ("Si Ling Primary School", "Woodlands"),
        ("Springdale Primary School", "Choa Chu Kang"),
        ("St Andrew's Junior School", "Tampines"),
        ("St Anthony's Canossian Primary School", "Bedok"),
        ("St Hilda's Primary School", "Tampines"),
        ("St Joseph's Institution Junior", "Buona Vista"),
        ("Tampines North Primary School", "Tampines"),
        ("Tampines Primary School", "Tampines"),
        ("Tanjong Katong Primary School", "Marine Parade"),
        ("Temasek Primary School", "Bedok"),
        ("Townsville Primary School", "Hougang"),
        ("Unity Primary School", "Choa Chu Kang"),
        ("Waterway Primary School", "Punggol"),
        ("West Grove Primary School", "Clementi"),
        ("Westwood Primary School", "Jurong West"),
        ("White Sands Primary School", "Pasir Ris"),
        ("Woodgrove Primary School", "Woodlands"),
        ("Woodlands Ring Primary School", "Woodlands"),
        ("Xinghua Primary School", "Hougang"),
        ("Yew Tee Primary School", "Choa Chu Kang"),
        ("Yuhua Primary School", "Jurong East"),
        ("Yumin Primary School", "Hougang"),
        ("Zhangde Primary School", "Ang Mo Kio"),
        ("Zhenghua Primary School", "Bukit Panjang"),
        ("Punggol Cove Primary School", "Punggol"),
        ("Punggol View Primary School", "Punggol"),
        ("Springside Primary School", "Yishun"),
        ("St Margaret's Primary School", "Queenstown"),
        ("Teck Ghee Primary School", "Ang Mo Kio"),
        ("Telok Kurau Primary School", "Geylang"),
        ("Victoria School (Primary)", "Marine Parade"),
        ("Woodlands Primary School", "Woodlands"),
    ]

    secondary_schools = [
        ("Anderson Secondary School", "Ang Mo Kio"),
        ("Anglican High School", "Bedok"),
        ("Ang Mo Kio Secondary School", "Ang Mo Kio"),
        ("Assumption English School", "Bukit Timah"),
        ("Beatty Secondary School", "Toa Payoh"),
        ("Bedok North Secondary School", "Bedok"),
        ("Bedok South Secondary School", "Bedok"),
        ("Bedok View Secondary School", "Bedok"),
        ("Bendemeer Secondary School", "Kallang"),
        ("Bishan Park Secondary School", "Bishan"),
        ("Boon Lay Secondary School", "Jurong West"),
        ("Bowen Secondary School", "Hougang"),
        ("Broadrick Secondary School", "Geylang"),
        ("Bukit Batok Secondary School", "Bukit Batok"),
        ("Bukit Merah Secondary School", "Bukit Merah"),
        ("Bukit Panjang Govt High", "Bukit Panjang"),
        ("Bukit View Secondary School", "Bukit Timah"),
        ("Catholic High School", "Bishan"),
        ("Cedar Girls' Secondary School", "Serangoon"),
        ("Chij Secondary (Toa Payoh)", "Toa Payoh"),
        ("Chua Chu Kang Secondary School", "Choa Chu Kang"),
        ("Clementi Town Secondary School", "Clementi"),
        ("Commonwealth Secondary School", "Queenstown"),
        ("Compassvale Secondary School", "Sengkang"),
        ("Coral Secondary School", "Pasir Ris"),
        ("Crescent Girls' School", "Queenstown"),
        ("Deyi Secondary School", "Ang Mo Kio"),
        ("Dunman High School", "Marine Parade"),
        ("Dunman Secondary School", "Geylang"),
        ("East Spring Secondary School", "Tampines"),
        ("Edgefield Secondary School", "Punggol"),
        ("Evergreen Secondary School", "Woodlands"),
        ("Fajar Secondary School", "Bukit Panjang"),
        ("Fuchun Secondary School", "Woodlands"),
        ("Fuhua Secondary School", "Jurong West"),
        ("Gan Eng Seng School", "Bukit Merah"),
        ("Geylang Methodist School (Sec)", "Geylang"),
        ("Greendale Secondary School", "Punggol"),
        ("Greenridge Secondary School", "Bukit Panjang"),
        ("Guangyang Secondary School", "Bishan"),
        ("Hai Sing Catholic School", "Pasir Ris"),
        ("Holy Innocents' High School", "Hougang"),
        ("Hougang Secondary School", "Hougang"),
        ("Hwa Chong Institution", "Bukit Timah"),
        ("Jurongville Secondary School", "Jurong West"),
        ("Jurong West Secondary School", "Jurong West"),
        ("Kent Ridge Secondary School", "Clementi"),
        ("Kranji Secondary School", "Woodlands"),
        ("Kuo Chuan Presbyterian Secondary", "Bishan"),
        ("Loyang View Secondary School", "Pasir Ris"),
        ("Manjusri Secondary School", "Ang Mo Kio"),
        ("Marsiling Secondary School", "Woodlands"),
        ("Mayflower Secondary School", "Ang Mo Kio"),
        ("Methodist Girls' School (Sec)", "Buona Vista"),
        ("Montfort Secondary School", "Tampines"),
        ("Nan Chiau High School", "Sengkang"),
        ("Nan Hua High School", "Clementi"),
        ("Nanyang Girls' High School", "Buona Vista"),
        ("National Junior College (Sec)", "Buona Vista"),
        ("Naval Base Secondary School", "Yishun"),
        ("New Town Secondary School", "Queenstown"),
        ("Ngee Ann Secondary School", "Hougang"),
        ("North Spring Secondary School", "Hougang"),
        ("North View Secondary School", "Woodlands"),
        ("Northbrooks Secondary School", "Woodlands"),
        ("Northland Secondary School", "Woodlands"),
        ("NUS High School of Math & Science", "Buona Vista"),
        ("Orchid Park Secondary School", "Yishun"),
        ("Paya Lebar Methodist Girls' (Sec)", "Geylang"),
        ("Pei Hwa Secondary School", "Sengkang"),
        ("Presbyterian High School", "Ang Mo Kio"),
        ("Punggol Secondary School", "Punggol"),
        ("Queenstown Secondary School", "Queenstown"),
        ("Queensway Secondary School", "Queenstown"),
        ("Raffles Girls' School (Sec)", "Buona Vista"),
        ("Raffles Institution (Sec)", "Bishan"),
        ("Riverside Secondary School", "Woodlands"),
        ("School of Science & Technology", "Jurong West"),
        ("Sembawang Secondary School", "Sembawang"),
        ("Sengkang Secondary School", "Sengkang"),
        ("Singapore Sports School", "Woodlands"),
        ("Siying Secondary School", "Woodlands"),
        ("Springfield Secondary School", "Hougang"),
        ("St Andrew's Secondary School", "Tampines"),
        ("St Anthony's Canossian Sec", "Bedok"),
        ("St Gabriel's Secondary School", "Serangoon"),
        ("St Hilda's Secondary School", "Tampines"),
        ("St Joseph's Institution", "Buona Vista"),
        ("St Margaret's Secondary School", "Queenstown"),
        ("St Patrick's School", "Marine Parade"),
        ("Tampines Secondary School", "Tampines"),
        ("Tanjong Katong Girls' School", "Marine Parade"),
        ("Tanjong Katong Secondary School", "Marine Parade"),
        ("Teck Whye Secondary School", "Choa Chu Kang"),
        ("Temasek Secondary School", "Bedok"),
        ("Unity Secondary School", "Choa Chu Kang"),
        ("Victoria School", "Marine Parade"),
        ("West Spring Secondary School", "Bukit Panjang"),
        ("West View Secondary School", "Jurong West"),
        ("Westwood Secondary School", "Jurong West"),
        ("Whitley Secondary School", "Bishan"),
        ("Woodgrove Secondary School", "Woodlands"),
        ("Woodlands Ring Secondary School", "Woodlands"),
        ("Woodlands Secondary School", "Woodlands"),
        ("Xinmin Secondary School", "Hougang"),
        ("Yang Zheng Secondary School", "Ang Mo Kio"),
        ("Yio Chu Kang Secondary School", "Ang Mo Kio"),
        ("Yuhua Secondary School", "Jurong East"),
        ("Yuan Ching Secondary School", "Jurong West"),
        ("Yusof Ishak Secondary School", "Buona Vista"),
        ("Zhonghua Secondary School", "Serangoon"),
    ]

    jcs = [
        ("Anderson Serangoon JC", "Serangoon"),
        ("Anglo-Chinese JC", "Bishan"),
        ("Dunman High School (JC)", "Marine Parade"),
        ("Eunoia Junior College", "Bishan"),
        ("Hwa Chong Institution (JC)", "Bukit Timah"),
        ("Jurong Pioneer JC", "Jurong West"),
        ("Millennia Institute", "Bukit Timah"),
        ("Nanyang JC", "Buona Vista"),
        ("National JC", "Buona Vista"),
        ("NUS High School (JC)", "Buona Vista"),
        ("Raffles Institution (JC)", "Bishan"),
        ("River Valley High School (JC)", "Buona Vista"),
        ("St Andrew's Junior College", "Tampines"),
        ("Tampines Meridian JC", "Tampines"),
        ("Temasek JC", "Tampines"),
        ("Victoria JC", "Marine Parade"),
        ("Yishun Innova JC", "Yishun"),
    ]

    all_schools = (
        [(n, t, "primary") for n, t in primary_schools]
        + [(n, t, "secondary") for n, t in secondary_schools]
        + [(n, t, "JC") for n, t in jcs]
    )

    north = {"Woodlands", "Sembawang", "Yishun"}
    east = {
        "Tampines",
        "Pasir Ris",
        "Bedok",
        "Changi",
        "Marine Parade",
        "Geylang",
        "Kallang",
    }
    west = {
        "Jurong West",
        "Jurong East",
        "Clementi",
        "Buona Vista",
        "Queenstown",
        "Bukit Timah",
        "Bukit Batok",
        "Bukit Panjang",
        "Choa Chu Kang",
    }
    ne = {"Sengkang", "Punggol", "Hougang", "Serangoon", "Ang Mo Kio", "Bishan"}

    def zone(town: str) -> str:
        if town in north:
            return "North"
        if town in east:
            return "East"
        if town in west:
            return "West"
        if town in ne:
            return "North-East"
        return "Central"

    df = pl.DataFrame(
        {
            "school_name": [s[0] for s in all_schools],
            "town": [s[1] for s in all_schools],
            "type": [s[2] for s in all_schools],
            "zone": [zone(s[1]) for s in all_schools],
        }
    )

    return df[:350]


# ---------------------------------------------------------------------------
# 11. sg_cpi.csv  (ascent01)
#     Target: 180 rows — monthly 2010-01 through 2024-12
# ---------------------------------------------------------------------------


def make_sg_cpi() -> pl.DataFrame:
    """
    Singapore Consumer Price Index, monthly 2010-01 to 2024-12.
    Base year 2019 = 100 for all indices.

    Intentional messiness:
    - ~5% nulls in cpi_transport (volatile, measurement gaps)
    - cpi_housing data only starts 2012-01 (nulls for 2010-2011)
    """
    from datetime import date

    dates = []
    d = date(2010, 1, 1)
    end = date(2024, 12, 1)
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        # Advance to next month
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)

    n = len(dates)  # 180

    # Months since 2010-01 for trend computation
    t = np.arange(n, dtype=float)

    # CPI All Items: base ~92 in 2010, reaching ~100 by 2019 (month 108),
    # then accelerating post-COVID
    cpi_all = 92.0 + t * 0.06 + RNG.normal(0, 0.3, n)
    # Add COVID-era inflation bump (2021 onward, month 132+)
    covid_bump = np.where(t >= 132, (t - 132) * 0.08, 0.0)
    cpi_all = cpi_all + covid_bump
    cpi_all = np.round(cpi_all, 1)

    # CPI Food: slightly faster growth, more seasonal noise
    cpi_food = 90.0 + t * 0.075 + RNG.normal(0, 0.5, n)
    # Seasonal pattern: food prices rise around Chinese New Year (Jan-Feb)
    month_idx = np.array([int(d[5:7]) for d in dates])
    cny_bump = np.where((month_idx == 1) | (month_idx == 2), 0.8, 0.0)
    cpi_food = cpi_food + cny_bump + covid_bump * 1.2
    cpi_food = np.round(cpi_food, 1)

    # CPI Housing: slower growth, only available from 2012 onward
    cpi_housing_raw = 88.0 + t * 0.08 + RNG.normal(0, 0.4, n)
    # Property cooling measures flatten 2013-2017 (month 36-96)
    cooling = np.where((t >= 36) & (t < 96), -0.03 * (t - 36), 0.0)
    cpi_housing_raw = cpi_housing_raw + cooling
    cpi_housing_raw = np.round(cpi_housing_raw, 1)
    # Null out 2010-2011 (first 24 months)
    cpi_housing: list = cpi_housing_raw.tolist()
    for i in range(24):
        cpi_housing[i] = None

    # CPI Transport: volatile due to COE and fuel prices
    cpi_transport = 95.0 + t * 0.04 + RNG.normal(0, 1.5, n)
    # Oil price spikes
    cpi_transport[24:30] += 5.0  # 2012 oil spike
    cpi_transport[120:126] -= 8.0  # 2020 COVID crash
    cpi_transport[132:144] += 10.0  # 2021 supply-chain crisis
    cpi_transport = np.round(cpi_transport, 1)
    # ~5% nulls
    cpi_transport_list = _nullify(cpi_transport, 0.05)

    df = pl.DataFrame(
        {
            "date": dates,
            "cpi_all_items": cpi_all.tolist(),
            "cpi_food": cpi_food.tolist(),
            "cpi_housing": cpi_housing,
            "cpi_transport": cpi_transport_list,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 12. sg_employment.csv  (ascent01)
#     Target: 60 rows — quarterly 2010-Q1 through 2024-Q4
# ---------------------------------------------------------------------------


def make_sg_employment() -> pl.DataFrame:
    """
    Singapore employment statistics, quarterly 2010-Q1 to 2024-Q4.

    Intentional messiness:
    - ~4% nulls in labour_force_participation
    - COVID unemployment spike in 2020 Q2-Q3
    """
    from datetime import date

    # Quarterly dates: first of Jan, Apr, Jul, Oct
    quarter_months = [1, 4, 7, 10]
    dates = []
    for year in range(2010, 2025):
        for m in quarter_months:
            dates.append(date(year, m, 1).strftime("%Y-%m-%d"))

    n = len(dates)  # 60
    t = np.arange(n, dtype=float)

    # Total employment: ~3.5M in 2010, growing to ~3.8M by 2024
    # with COVID dip in 2020
    total_emp_base = 3_500_000 + t * 5_000 + RNG.normal(0, 8_000, n)
    # COVID dip: 2020 Q1-Q4 are indices 40-43
    total_emp_base[40] -= 30_000  # 2020 Q1
    total_emp_base[41] -= 120_000  # 2020 Q2
    total_emp_base[42] -= 80_000  # 2020 Q3
    total_emp_base[43] -= 40_000  # 2020 Q4
    total_employment = np.round(total_emp_base).astype(int)

    # Unemployment rate: ~2.0% baseline, spike in 2020
    unemp_base = np.full(n, 2.0) + RNG.normal(0, 0.15, n)
    # Gradual trends
    unemp_base[:8] += 0.2  # 2010-2011 slightly higher post-GFC recovery
    # COVID spike
    unemp_base[40] = 2.4 + RNG.normal(0, 0.1)  # 2020 Q1
    unemp_base[41] = 4.0 + RNG.normal(0, 0.1)  # 2020 Q2 peak
    unemp_base[42] = 3.6 + RNG.normal(0, 0.1)  # 2020 Q3
    unemp_base[43] = 3.0 + RNG.normal(0, 0.1)  # 2020 Q4
    unemp_base[44] = 2.8 + RNG.normal(0, 0.1)  # 2021 Q1 recovery
    unemp_base[45] = 2.5 + RNG.normal(0, 0.1)  # 2021 Q2
    unemployment_rate = np.round(np.clip(unemp_base, 1.5, 5.0), 1)

    # Labour force participation: ~67-70%
    lfp_base = 68.0 + t * 0.03 + RNG.normal(0, 0.3, n)
    # COVID dip
    lfp_base[41] -= 2.0  # 2020 Q2
    lfp_base[42] -= 1.0  # 2020 Q3
    lfp = np.round(np.clip(lfp_base, 67.0, 70.0), 1)
    # ~4% nulls
    lfp_list = _nullify(lfp, 0.04)

    df = pl.DataFrame(
        {
            "date": dates,
            "total_employment": total_employment.tolist(),
            "unemployment_rate": unemployment_rate.tolist(),
            "labour_force_participation": lfp_list,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 13. sg_fx_rates.csv  (ascent01)
#     Target: ~3,900 rows — business days 2010-01-04 through 2024-12-31
# ---------------------------------------------------------------------------


def make_sg_fx_rates() -> pl.DataFrame:
    """
    Singapore exchange rates (units of foreign currency per 1 SGD), daily
    business days from 2010-01-04 to 2024-12-31.

    Intentional messiness:
    - ~2% nulls randomly across all rate columns
    - Extreme values during crisis periods (GFC tail, COVID, 2022 rate hikes)
    """
    from datetime import date, timedelta

    # Generate business days (Mon-Fri)
    start = date(2010, 1, 4)  # First Monday of 2010
    end = date(2024, 12, 31)
    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon=0, Fri=4
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    n = len(dates)
    t = np.arange(n, dtype=float)

    # SGD/USD: ~1.40 in 2010, strengthening to ~1.20 by 2024
    # (fewer USD per SGD = SGD weakening would be higher, but SGD strengthened)
    sgd_usd = 1.40 - t * (0.20 / n) + RNG.normal(0, 0.008, n)
    # Add crisis volatility
    # 2020 COVID shock (~month 2600-2700 in business days, roughly index 2600)
    covid_start = int(n * 0.67)  # approximate 2020 March
    sgd_usd[covid_start : covid_start + 20] += RNG.uniform(0.03, 0.06, 20)
    # 2022 Fed rate hikes (~index 3100)
    rate_hike_start = int(n * 0.80)
    sgd_usd[rate_hike_start : rate_hike_start + 60] += RNG.uniform(0.01, 0.04, 60)
    sgd_usd = np.round(np.clip(sgd_usd, 1.20, 1.45), 4)

    # SGD/EUR: ~1.65 in 2010, ranging 1.45-1.65
    sgd_eur = 1.65 - t * (0.15 / n) + RNG.normal(0, 0.010, n)
    # Euro crisis 2011-2012: EUR weakens vs SGD
    euro_crisis = int(n * 0.13)
    sgd_eur[euro_crisis : euro_crisis + 120] -= RNG.uniform(0.03, 0.08, 120)
    sgd_eur = np.round(np.clip(sgd_eur, 1.45, 1.70), 4)

    # SGD/GBP: ~1.90 in 2010, ranging 1.65-1.90
    sgd_gbp = 1.88 - t * (0.18 / n) + RNG.normal(0, 0.012, n)
    # Brexit referendum 2016 (~index 1650): GBP crash
    brexit_idx = int(n * 0.43)
    sgd_gbp[brexit_idx : brexit_idx + 30] -= RNG.uniform(0.05, 0.12, 30)
    sgd_gbp = np.round(np.clip(sgd_gbp, 1.65, 1.95), 4)

    # SGD/JPY: ~0.0120 in 2010, weakening to ~0.0085 by 2024
    # (JPY per SGD; JPY weakened significantly)
    sgd_jpy = 0.0120 - t * (0.0035 / n) + RNG.normal(0, 0.0003, n)
    # 2022 JPY collapse
    jpy_collapse = int(n * 0.82)
    sgd_jpy[jpy_collapse : jpy_collapse + 80] -= RNG.uniform(0.0005, 0.0015, 80)
    sgd_jpy = np.round(np.clip(sgd_jpy, 0.0080, 0.0125), 4)

    # Inject ~2% nulls across all rate columns
    sgd_usd_list = _nullify(sgd_usd, 0.02)
    sgd_eur_list = _nullify(sgd_eur, 0.02)
    sgd_gbp_list = _nullify(sgd_gbp, 0.02)
    sgd_jpy_list = _nullify(sgd_jpy, 0.02)

    # Inject extreme values during crisis periods (~0.3%)
    n_extreme = int(n * 0.003)
    extreme_idx = RNG.choice(n, size=n_extreme, replace=False)
    for i in extreme_idx:
        # Only inject if not already nullified
        if sgd_usd_list[i] is not None:
            sgd_usd_list[i] = round(float(RNG.uniform(1.50, 1.80)), 4)
        if sgd_jpy_list[i] is not None:
            sgd_jpy_list[i] = round(float(RNG.uniform(0.0060, 0.0075)), 4)

    df = pl.DataFrame(
        {
            "date": dates,
            "sgd_usd": sgd_usd_list,
            "sgd_eur": sgd_eur_list,
            "sgd_gbp": sgd_gbp_list,
            "sgd_jpy": sgd_jpy_list,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 14. hdb_resale.parquet  (ascent01) — 150,000 rows
# ---------------------------------------------------------------------------


def make_hdb_resale() -> pl.DataFrame:
    """
    Singapore HDB resale transactions 2017-2024 (150,000 rows).

    Intentional messiness:
    - ~3% nulls in floor_area_sqm
    - ~2% nulls in storey_range
    - Some unrealistic prices: a few negative, a few > $2M
    - ~0.5% duplicate rows
    """
    n = 150_000

    # --- Towns and their relative price premiums ---
    towns = [
        "ANG MO KIO",
        "BEDOK",
        "BISHAN",
        "BUKIT BATOK",
        "BUKIT MERAH",
        "BUKIT PANJANG",
        "BUKIT TIMAH",
        "CENTRAL AREA",
        "CHOA CHU KANG",
        "CLEMENTI",
        "GEYLANG",
        "HOUGANG",
        "JURONG EAST",
        "JURONG WEST",
        "KALLANG/WHAMPOA",
        "MARINE PARADE",
        "PASIR RIS",
        "PUNGGOL",
        "QUEENSTOWN",
        "SEMBAWANG",
        "SENGKANG",
        "SERANGOON",
        "TAMPINES",
        "TOA PAYOH",
        "WOODLANDS",
        "YISHUN",
    ]
    # Premium multiplier per town (1.0 = baseline)
    town_premium = {
        "ANG MO KIO": 1.05,
        "BEDOK": 1.00,
        "BISHAN": 1.20,
        "BUKIT BATOK": 0.90,
        "BUKIT MERAH": 1.10,
        "BUKIT PANJANG": 0.85,
        "BUKIT TIMAH": 1.45,
        "CENTRAL AREA": 1.40,
        "CHOA CHU KANG": 0.82,
        "CLEMENTI": 1.05,
        "GEYLANG": 0.95,
        "HOUGANG": 0.90,
        "JURONG EAST": 0.92,
        "JURONG WEST": 0.85,
        "KALLANG/WHAMPOA": 1.10,
        "MARINE PARADE": 1.15,
        "PASIR RIS": 0.88,
        "PUNGGOL": 0.92,
        "QUEENSTOWN": 1.30,
        "SEMBAWANG": 0.80,
        "SENGKANG": 0.88,
        "SERANGOON": 1.05,
        "TAMPINES": 0.95,
        "TOA PAYOH": 1.10,
        "WOODLANDS": 0.80,
        "YISHUN": 0.82,
    }
    # Town selection weights (mature towns have more resale volume)
    town_weights = np.array(
        [
            5.0,  # ANG MO KIO
            5.5,  # BEDOK
            3.0,  # BISHAN
            4.5,  # BUKIT BATOK
            5.0,  # BUKIT MERAH
            3.5,  # BUKIT PANJANG
            0.8,  # BUKIT TIMAH (fewer HDB)
            1.0,  # CENTRAL AREA (fewer HDB)
            3.5,  # CHOA CHU KANG
            3.0,  # CLEMENTI
            3.5,  # GEYLANG
            5.0,  # HOUGANG
            3.0,  # JURONG EAST
            5.5,  # JURONG WEST
            4.0,  # KALLANG/WHAMPOA
            1.5,  # MARINE PARADE
            3.0,  # PASIR RIS
            4.0,  # PUNGGOL
            3.0,  # QUEENSTOWN
            2.5,  # SEMBAWANG
            4.5,  # SENGKANG
            3.0,  # SERANGOON
            6.0,  # TAMPINES
            4.0,  # TOA PAYOH
            5.0,  # WOODLANDS
            5.0,  # YISHUN
        ]
    )
    town_weights = town_weights / town_weights.sum()

    # --- Flat types and distribution ---
    flat_types = [
        "1 ROOM",
        "2 ROOM",
        "3 ROOM",
        "4 ROOM",
        "5 ROOM",
        "EXECUTIVE",
        "MULTI-GENERATION",
    ]
    flat_probs = np.array([0.01, 0.04, 0.25, 0.35, 0.20, 0.12, 0.03])
    # Floor area ranges per flat type (min, max)
    area_ranges = {
        "1 ROOM": (35.0, 45.0),
        "2 ROOM": (45.0, 50.0),
        "3 ROOM": (60.0, 70.0),
        "4 ROOM": (85.0, 95.0),
        "5 ROOM": (110.0, 125.0),
        "EXECUTIVE": (140.0, 160.0),
        "MULTI-GENERATION": (145.0, 170.0),
    }
    # Base price ranges per flat type (before town/storey adjustments)
    price_base = {
        "1 ROOM": (180_000, 280_000),
        "2 ROOM": (200_000, 350_000),
        "3 ROOM": (280_000, 450_000),
        "4 ROOM": (380_000, 600_000),
        "5 ROOM": (450_000, 750_000),
        "EXECUTIVE": (550_000, 900_000),
        "MULTI-GENERATION": (650_000, 1_000_000),
    }

    # --- Street names per town ---
    street_names = {
        "ANG MO KIO": [
            "ANG MO KIO AVE 1",
            "ANG MO KIO AVE 3",
            "ANG MO KIO AVE 4",
            "ANG MO KIO AVE 5",
            "ANG MO KIO AVE 6",
            "ANG MO KIO AVE 10",
        ],
        "BEDOK": [
            "BEDOK NTH AVE 1",
            "BEDOK NTH AVE 2",
            "BEDOK NTH AVE 3",
            "BEDOK NTH AVE 4",
            "BEDOK STH AVE 1",
            "BEDOK STH AVE 2",
            "BEDOK RESERVOIR RD",
            "NEW UPPER CHANGI RD",
        ],
        "BISHAN": [
            "BISHAN ST 11",
            "BISHAN ST 12",
            "BISHAN ST 13",
            "BISHAN ST 22",
            "BISHAN ST 23",
            "BISHAN ST 24",
        ],
        "BUKIT BATOK": [
            "BUKIT BATOK ST 11",
            "BUKIT BATOK ST 21",
            "BUKIT BATOK ST 22",
            "BUKIT BATOK ST 31",
            "BUKIT BATOK ST 34",
            "BUKIT BATOK WEST AVE 6",
        ],
        "BUKIT MERAH": [
            "BUKIT MERAH VIEW",
            "JLN BUKIT MERAH",
            "HENDERSON RD",
            "TELOK BLANGAH DR",
            "TELOK BLANGAH CRES",
            "REDHILL LANE",
        ],
        "BUKIT PANJANG": [
            "BUKIT PANJANG RING RD",
            "PETIR RD",
            "FAJAR RD",
            "PENDING RD",
            "SENJA RD",
            "JELAPANG RD",
        ],
        "BUKIT TIMAH": [
            "TOH YI DR",
            "BUKIT TIMAH RD",
            "QUEEN'S RD",
            "FARRER RD",
        ],
        "CENTRAL AREA": [
            "CANTONMENT RD",
            "SELEGIE RD",
            "WATERLOO ST",
            "QUEEN ST",
            "SMITH ST",
            "CHIN SWEE RD",
        ],
        "CHOA CHU KANG": [
            "CHOA CHU KANG AVE 1",
            "CHOA CHU KANG AVE 2",
            "CHOA CHU KANG AVE 3",
            "CHOA CHU KANG AVE 4",
            "CHOA CHU KANG NTH 6",
            "CHOA CHU KANG ST 52",
        ],
        "CLEMENTI": [
            "CLEMENTI AVE 1",
            "CLEMENTI AVE 2",
            "CLEMENTI AVE 3",
            "CLEMENTI AVE 4",
            "CLEMENTI ST 11",
            "WEST COAST RD",
        ],
        "GEYLANG": [
            "GEYLANG EAST AVE 1",
            "GEYLANG EAST AVE 2",
            "ALJUNIED CRES",
            "SIMS AVE",
            "EUNOS CRES",
            "MACPHERSON LANE",
        ],
        "HOUGANG": [
            "HOUGANG AVE 1",
            "HOUGANG AVE 2",
            "HOUGANG AVE 5",
            "HOUGANG AVE 7",
            "HOUGANG AVE 8",
            "HOUGANG AVE 10",
            "HOUGANG ST 11",
        ],
        "JURONG EAST": [
            "JURONG EAST AVE 1",
            "JURONG EAST ST 13",
            "JURONG EAST ST 21",
            "JURONG EAST ST 31",
            "JURONG EAST ST 32",
        ],
        "JURONG WEST": [
            "JURONG WEST AVE 1",
            "JURONG WEST ST 41",
            "JURONG WEST ST 42",
            "JURONG WEST ST 51",
            "JURONG WEST ST 52",
            "JURONG WEST ST 61",
            "JURONG WEST ST 65",
        ],
        "KALLANG/WHAMPOA": [
            "KALLANG BAHRU",
            "WHAMPOA DR",
            "WHAMPOA RD",
            "BENDEMEER RD",
            "BOON KENG RD",
            "UPPER BOON KENG RD",
            "ST. GEORGE'S RD",
        ],
        "MARINE PARADE": [
            "MARINE CRES",
            "MARINE DR",
            "MARINE PARADE CENTRAL",
            "MARINE TERRACE",
        ],
        "PASIR RIS": [
            "PASIR RIS DR 1",
            "PASIR RIS DR 3",
            "PASIR RIS DR 4",
            "PASIR RIS DR 6",
            "PASIR RIS ST 11",
            "PASIR RIS ST 12",
        ],
        "PUNGGOL": [
            "PUNGGOL DR",
            "PUNGGOL FIELD",
            "PUNGGOL WALK",
            "PUNGGOL WAY",
            "EDGEDALE PLAINS",
            "EDGEFIELD PLAINS",
        ],
        "QUEENSTOWN": [
            "QUEENSTOWN AVE",
            "COMMONWEALTH CRES",
            "COMMONWEALTH DR",
            "DAWSON RD",
            "STIRLING RD",
            "TANGLIN HALT RD",
        ],
        "SEMBAWANG": [
            "SEMBAWANG CRES",
            "SEMBAWANG DR",
            "SEMBAWANG WAY",
            "CANBERRA RD",
            "CANBERRA ST",
        ],
        "SENGKANG": [
            "SENGKANG EAST WAY",
            "SENGKANG WEST WAY",
            "COMPASSVALE DR",
            "COMPASSVALE ST",
            "RIVERVALE DR",
            "RIVERVALE CRES",
            "ANCHORVALE DR",
        ],
        "SERANGOON": [
            "SERANGOON AVE 1",
            "SERANGOON AVE 2",
            "SERANGOON AVE 3",
            "SERANGOON AVE 4",
            "SERANGOON NTH AVE 1",
            "SERANGOON NTH AVE 2",
        ],
        "TAMPINES": [
            "TAMPINES AVE 1",
            "TAMPINES AVE 4",
            "TAMPINES AVE 5",
            "TAMPINES AVE 7",
            "TAMPINES AVE 8",
            "TAMPINES AVE 9",
            "TAMPINES ST 11",
            "TAMPINES ST 21",
        ],
        "TOA PAYOH": [
            "TOA PAYOH CENTRAL",
            "TOA PAYOH EAST",
            "TOA PAYOH NTH",
            "LOR 1 TOA PAYOH",
            "LOR 2 TOA PAYOH",
            "LOR 4 TOA PAYOH",
            "LOR 5 TOA PAYOH",
        ],
        "WOODLANDS": [
            "WOODLANDS AVE 1",
            "WOODLANDS AVE 3",
            "WOODLANDS AVE 5",
            "WOODLANDS AVE 6",
            "WOODLANDS DR 14",
            "WOODLANDS DR 16",
            "WOODLANDS ST 13",
        ],
        "YISHUN": [
            "YISHUN AVE 1",
            "YISHUN AVE 2",
            "YISHUN AVE 4",
            "YISHUN AVE 5",
            "YISHUN AVE 6",
            "YISHUN AVE 9",
            "YISHUN AVE 11",
            "YISHUN RING RD",
        ],
    }

    # --- Storey ranges ---
    storey_ranges = [
        "01 TO 03",
        "04 TO 06",
        "07 TO 09",
        "10 TO 12",
        "13 TO 15",
        "16 TO 18",
        "19 TO 21",
        "22 TO 24",
        "25 TO 27",
        "28 TO 30",
        "31 TO 33",
        "34 TO 36",
        "37 TO 39",
        "40 TO 42",
        "43 TO 45",
        "46 TO 48",
        "49 TO 51",
    ]
    # Higher storeys less common; weighted toward lower floors
    storey_weights = np.array(
        [15, 18, 16, 14, 11, 8, 6, 4, 3, 2, 1.2, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02]
    )
    storey_weights = storey_weights / storey_weights.sum()
    # Midpoint of each storey range for price adjustment
    storey_mid = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50]

    # --- Generate month column (2017-01 to 2024-12) ---
    months = []
    for y in range(2017, 2025):
        for m in range(1, 13):
            months.append(f"{y:04d}-{m:02d}")
    # Year index for price trend (0 = 2017, 7 = 2024)
    month_year_offset = {mo: (int(mo[:4]) - 2017) for mo in months}

    # --- Generate each row ---
    col_month = []
    col_town = []
    col_flat_type = []
    col_block = []
    col_street = []
    col_storey = []
    col_area = []
    col_price = []

    town_indices = RNG.choice(len(towns), size=n, p=town_weights)
    flat_indices = RNG.choice(len(flat_types), size=n, p=flat_probs)
    month_indices = RNG.choice(len(months), size=n)
    storey_indices = RNG.choice(len(storey_ranges), size=n, p=storey_weights)

    for i in range(n):
        town = towns[town_indices[i]]
        flat = flat_types[flat_indices[i]]
        month = months[month_indices[i]]
        storey_idx = storey_indices[i]
        storey = storey_ranges[storey_idx]

        # Block number — 3-digit, ~5% have a letter suffix
        block_num = RNG.integers(1, 999)
        if RNG.random() < 0.05:
            block_str = f"{block_num}{RNG.choice(list('ABCD'))}"
        else:
            block_str = str(block_num)

        # Street name
        street = random.choice(street_names[town])

        # Floor area
        lo, hi = area_ranges[flat]
        area = float(RNG.uniform(lo, hi))

        # Resale price: base + town premium + storey premium + year trend + noise
        p_lo, p_hi = price_base[flat]
        base = float(RNG.uniform(p_lo, p_hi))
        base *= town_premium[town]
        # Storey premium: ~+0.5% per storey-band midpoint
        base *= 1.0 + storey_mid[storey_idx] * 0.005
        # Year trend: ~+3% per year from 2017 baseline
        year_offset = month_year_offset[month]
        base *= 1.0 + year_offset * 0.03
        # Noise
        base += float(RNG.normal(0, base * 0.05))
        # Round to nearest $100
        price = round(base / 100) * 100

        col_month.append(month)
        col_town.append(town)
        col_flat_type.append(flat)
        col_block.append(block_str)
        col_street.append(street)
        col_storey.append(storey)
        col_area.append(round(area, 1))
        col_price.append(float(price))

    # --- Inject messiness ---

    # Nulls in floor_area_sqm (~3%)
    col_area = _nullify(np.array(col_area), 0.03)

    # Nulls in storey_range (~2%)
    col_storey = _nullify_list(col_storey, 0.02)

    # Unrealistic prices: ~50 negative, ~30 above $2M
    neg_idx = RNG.choice(n, size=50, replace=False)
    for i in neg_idx:
        col_price[i] = float(RNG.integers(-100_000, -1_000))
    high_idx = RNG.choice(n, size=30, replace=False)
    for i in high_idx:
        col_price[i] = float(RNG.integers(2_100_000, 5_000_000))

    # Build DataFrame
    df = pl.DataFrame(
        {
            "month": col_month,
            "town": col_town,
            "flat_type": col_flat_type,
            "block": col_block,
            "street_name": col_street,
            "storey_range": col_storey,
            "floor_area_sqm": col_area,
            "resale_price": col_price,
        }
    )

    # Duplicate rows (~0.5%)
    dup_count = int(n * 0.005)
    dup_indices = RNG.choice(n, size=dup_count, replace=False).tolist()
    dups = df[dup_indices]
    df = pl.concat([df, dups])

    return df


# ---------------------------------------------------------------------------
# 15. ecommerce_ab_test.csv  (ascent01) — 10,000 rows
# ---------------------------------------------------------------------------


def make_ecommerce_ab_test() -> pl.DataFrame:
    """
    E-commerce A/B test for conversion rate analysis.
    ~10,000 visitors — control (~3% conversion) vs treatment (~4.5% conversion).

    Revenue is 0 for non-converters, log-normal for converters.
    """
    n = 10_000

    groups = RNG.choice(["control", "treatment"], size=n, p=[0.5, 0.5]).tolist()

    converted = []
    revenue = []
    for g in groups:
        rate = 0.03 if g == "control" else 0.045
        conv = int(RNG.random() < rate)
        converted.append(conv)
        if conv:
            revenue.append(round(float(RNG.lognormal(mean=3.5, sigma=1.0)), 2))
        else:
            revenue.append(0.0)

    pages_viewed = RNG.integers(1, 20, size=n).tolist()

    return pl.DataFrame(
        {
            "group": groups,
            "converted": converted,
            "revenue": revenue,
            "pages_viewed": pages_viewed,
        }
    )


# ---------------------------------------------------------------------------
# 16. ecommerce_experiment.parquet  (ascent02) — 20,000 rows
# ---------------------------------------------------------------------------


def make_ecommerce_experiment() -> pl.DataFrame:
    """
    E-commerce A/B experiment with pre-revenue for CUPED analysis.
    20,000 users — control vs treatment, pre-revenue correlated with revenue (rho ~0.5).
    Dates spanning 2023-06 to 2023-09.
    """
    n = 20_000

    groups = RNG.choice(["control", "treatment"], size=n, p=[0.5, 0.5]).tolist()

    # Pre-revenue: baseline spend before experiment
    pre_revenue = RNG.lognormal(mean=3.0, sigma=0.8, size=n)

    # Revenue correlated with pre_revenue (rho ~0.5)
    noise = RNG.normal(0, 1.0, size=n)
    revenue_raw = (
        0.5 * ((pre_revenue - pre_revenue.mean()) / pre_revenue.std())
        + np.sqrt(1 - 0.25) * noise
    )
    # Scale back to realistic dollar amounts
    revenue = np.exp(3.0 + 0.8 * revenue_raw)
    # Treatment uplift: +8%
    for i, g in enumerate(groups):
        if g == "treatment":
            revenue[i] *= 1.08

    # Signup dates: 2023-06-01 to 2023-09-30 (122 days)
    from datetime import datetime, timedelta

    base_date = datetime(2023, 6, 1)
    day_offsets = RNG.integers(0, 122, size=n)
    signup_dates = [
        (base_date + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in day_offsets
    ]

    return pl.DataFrame(
        {
            "group": groups,
            "revenue": np.round(revenue, 2).tolist(),
            "pre_revenue": np.round(pre_revenue, 2).tolist(),
            "signup_date": signup_dates,
        }
    )


# ---------------------------------------------------------------------------
# 17. economic_indicators.csv  (ascent02) — ~50 rows quarterly
#     Simplified variant of M1 economic_indicators with renamed columns
# ---------------------------------------------------------------------------


def make_ascent02_economic_indicators() -> pl.DataFrame:
    """
    Singapore economic indicators for M2 feature engineering exercises.
    ~50 rows of quarterly data (2012-2024) with columns renamed for M2:
      gdp_growth_pct, inflation_cpi_pct, unemployment_rate_pct.
    """
    years = list(range(2012, 2025))
    quarters = [1, 2, 3, 4]
    rows = [(y, q) for y in years for q in quarters]
    # Trim to complete data (2024 has 4 quarters)
    n = len(rows)

    # GDP growth — realistic Singapore pattern
    gdp_base = np.array(
        [
            3.5,
            3.2,
            2.8,
            2.5,  # 2012
            4.0,
            4.2,
            3.8,
            3.5,  # 2013
            3.9,
            3.5,
            2.8,
            2.5,  # 2014
            2.0,
            2.2,
            1.8,
            1.5,  # 2015
            2.4,
            2.0,
            1.8,
            2.0,  # 2016
            3.5,
            3.8,
            4.0,
            3.8,  # 2017
            3.2,
            3.4,
            3.0,
            2.8,  # 2018
            0.8,
            0.6,
            0.5,
            0.5,  # 2019
            -3.3,
            -13.3,
            -5.8,
            -2.4,  # 2020 COVID
            -0.5,
            15.2,
            7.5,
            6.0,  # 2021
            4.5,
            4.2,
            3.8,
            2.5,  # 2022
            2.1,
            1.8,
            1.5,
            2.0,  # 2023
            2.5,
            2.8,
            3.0,
            3.2,  # 2024
        ],
        dtype=float,
    )
    gdp = gdp_base[:n] + RNG.normal(0, 0.2, n)

    # Inflation CPI
    inflation_base = np.concatenate(
        [
            RNG.uniform(1.5, 3.5, 32),  # 2012-2019
            RNG.uniform(0.0, 1.0, 4),  # 2020 low
            RNG.uniform(2.0, 4.0, 4),  # 2021
            RNG.uniform(5.0, 7.0, 4),  # 2022 peak
            RNG.uniform(3.0, 5.0, 4),  # 2023
            RNG.uniform(2.0, 3.5, 4),  # 2024
        ]
    )
    inflation = inflation_base[:n]

    # Unemployment
    unemp_base = np.array(
        [
            1.9,
            1.9,
            1.8,
            1.8,  # 2012
            1.9,
            1.9,
            1.8,
            1.8,  # 2013
            1.9,
            2.0,
            2.0,
            1.9,  # 2014
            1.9,
            1.9,
            2.0,
            2.0,  # 2015
            2.2,
            2.1,
            2.1,
            2.0,  # 2016
            2.2,
            2.2,
            2.1,
            2.0,  # 2017
            2.1,
            2.1,
            2.0,
            2.0,  # 2018
            2.2,
            2.3,
            2.3,
            2.2,  # 2019
            2.6,
            4.0,
            4.5,
            4.1,  # 2020 COVID
            3.0,
            2.5,
            2.2,
            2.1,  # 2021
            2.0,
            2.0,
            1.9,
            1.9,  # 2022
            1.9,
            1.8,
            1.9,
            1.9,  # 2023
            1.8,
            1.8,
            1.9,
            1.9,  # 2024
        ],
        dtype=float,
    )
    unemp = unemp_base[:n] + RNG.normal(0, 0.05, n)

    # Quarter labels: "YYYY-QN"
    period = [f"{y}-Q{q}" for y, q in rows]

    return pl.DataFrame(
        {
            "period": period,
            "gdp_growth_pct": np.round(gdp, 2).tolist(),
            "inflation_cpi_pct": np.round(inflation, 2).tolist(),
            "unemployment_rate_pct": np.round(np.clip(unemp, 0.5, 6.0), 2).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# 18. ICU multi-table dataset  (ascent02) — 5 related tables
# ---------------------------------------------------------------------------


def make_icu_patients() -> pl.DataFrame:
    """
    ICU patients table — 800 patients.
    Columns: patient_id, age, gender.
    """
    n = 800

    patient_ids = list(range(1, n + 1))
    ages = np.round(RNG.uniform(18, 95, size=n), 1).tolist()
    genders = RNG.choice(["M", "F"], size=n, p=[0.55, 0.45]).tolist()

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
        }
    )


def make_icu_admissions() -> pl.DataFrame:
    """
    ICU admissions table — ~1500 admissions (some patients have multiple).
    Columns: patient_id (FK), admission_id (PK), admit_time, discharge_time,
             length_of_stay_days, in_hospital_mortality (~15%).
    """
    from datetime import datetime, timedelta

    n_patients = 800
    target_admissions = 1500

    # Assign number of admissions per patient (1-4, skewed toward 1-2)
    admissions_per_patient = RNG.choice(
        [1, 2, 3, 4], size=n_patients, p=[0.55, 0.30, 0.10, 0.05]
    )
    # Adjust to hit ~1500 total
    total = admissions_per_patient.sum()
    while total < target_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per_patient[idx] < 4:
            admissions_per_patient[idx] += 1
            total += 1
    while total > target_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per_patient[idx] > 1:
            admissions_per_patient[idx] -= 1
            total -= 1

    patient_ids = []
    admission_ids = []
    admit_times = []
    discharge_times = []
    los_days = []
    mortality = []

    base_date = datetime(2022, 1, 1)
    adm_id = 1

    for pid in range(1, n_patients + 1):
        n_adm = int(admissions_per_patient[pid - 1])
        # Spread admissions across 2 years
        for _ in range(n_adm):
            admit_offset = int(RNG.integers(0, 730))
            admit_dt = base_date + timedelta(
                days=admit_offset,
                hours=int(RNG.integers(0, 24)),
                minutes=int(RNG.integers(0, 60)),
            )
            # Length of stay: 1-30 days, skewed short
            stay = float(np.round(RNG.lognormal(mean=1.5, sigma=0.8), 1))
            stay = max(0.5, min(stay, 60.0))
            discharge_dt = admit_dt + timedelta(days=stay)

            died = int(RNG.random() < 0.15)

            patient_ids.append(pid)
            admission_ids.append(adm_id)
            admit_times.append(admit_dt.strftime("%Y-%m-%d %H:%M:%S"))
            discharge_times.append(discharge_dt.strftime("%Y-%m-%d %H:%M:%S"))
            los_days.append(round(stay, 1))
            mortality.append(died)
            adm_id += 1

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "admission_id": admission_ids,
            "admit_time": admit_times,
            "discharge_time": discharge_times,
            "length_of_stay_days": los_days,
            "in_hospital_mortality": mortality,
        }
    )


def make_icu_vitals() -> pl.DataFrame:
    """
    ICU vitals table — ~18,000 rows of time-series vital signs.
    Columns: patient_id, admission_id, recorded_at, vital_name, value.
    Vital signs: heart_rate, blood_pressure, oxygen_saturation, temperature, respiratory_rate.
    """
    from datetime import datetime, timedelta

    # Generate vitals for ~1500 admissions, ~12 readings per admission
    n_admissions = 1500
    target_rows = 18_000
    readings_per_admission = target_rows // n_admissions  # ~12

    vital_ranges = {
        "heart_rate": (55.0, 130.0, 80.0, 15.0),  # min, max, mean, std
        "blood_pressure": (70.0, 200.0, 120.0, 20.0),  # systolic
        "oxygen_saturation": (85.0, 100.0, 96.0, 3.0),
        "temperature": (35.5, 40.5, 37.0, 0.5),
        "respiratory_rate": (10.0, 35.0, 18.0, 4.0),
    }
    vital_names = list(vital_ranges.keys())

    patient_ids = []
    admission_ids = []
    recorded_at = []
    v_names = []
    values = []

    base_date = datetime(2022, 1, 1)

    # Build admission lookup: admission_id -> patient_id
    adm_id = 1
    adm_to_pid = {}
    n_patients = 800
    admissions_per_patient = RNG.choice(
        [1, 2, 3, 4], size=n_patients, p=[0.55, 0.30, 0.10, 0.05]
    )
    # Re-adjust to ~1500
    total = admissions_per_patient.sum()
    while total < n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per_patient[idx] < 4:
            admissions_per_patient[idx] += 1
            total += 1
    while total > n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per_patient[idx] > 1:
            admissions_per_patient[idx] -= 1
            total -= 1

    for pid in range(1, n_patients + 1):
        for _ in range(int(admissions_per_patient[pid - 1])):
            adm_to_pid[adm_id] = pid
            adm_id += 1

    for aid in range(1, min(adm_id, n_admissions + 1)):
        pid = adm_to_pid[aid]
        admit_offset = int(RNG.integers(0, 730))
        admit_dt = base_date + timedelta(days=admit_offset)

        n_readings = int(
            RNG.integers(max(1, readings_per_admission - 4), readings_per_admission + 5)
        )
        for r in range(n_readings):
            hours_offset = float(RNG.uniform(0, 168))  # up to 7 days of readings
            rec_dt = admit_dt + timedelta(hours=hours_offset)
            vital = RNG.choice(vital_names)
            vmin, vmax, vmean, vstd = vital_ranges[vital]
            val = float(np.clip(RNG.normal(vmean, vstd), vmin, vmax))

            patient_ids.append(pid)
            admission_ids.append(aid)
            recorded_at.append(rec_dt.strftime("%Y-%m-%d %H:%M:%S"))
            v_names.append(vital)
            values.append(round(val, 1))

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "admission_id": admission_ids,
            "recorded_at": recorded_at,
            "vital_name": v_names,
            "value": values,
        }
    )


def make_icu_labs() -> pl.DataFrame:
    """
    ICU laboratory results — ~10,000 rows.
    Columns: patient_id, admission_id, collected_at, lab_name, value, abnormal_flag.
    Lab tests: creatinine, glucose, hemoglobin, lactate, wbc, potassium.
    """
    from datetime import datetime, timedelta

    n_admissions = 1500
    target_rows = 10_000
    readings_per_admission = target_rows // n_admissions  # ~6-7

    lab_specs = {
        # name: (normal_low, normal_high, mean, std)
        "creatinine": (0.6, 1.2, 1.0, 0.5),
        "glucose": (70.0, 100.0, 95.0, 30.0),
        "hemoglobin": (12.0, 17.5, 13.5, 2.0),
        "lactate": (0.5, 2.0, 1.2, 0.8),
        "wbc": (4.0, 11.0, 7.5, 3.0),
        "potassium": (3.5, 5.0, 4.2, 0.6),
    }
    lab_names = list(lab_specs.keys())

    patient_ids = []
    admission_ids = []
    collected_at = []
    l_names = []
    values = []
    abnormal_flags = []

    base_date = datetime(2022, 1, 1)

    # Build admission lookup
    adm_id = 1
    adm_to_pid = {}
    n_patients = 800
    admissions_per = RNG.choice(
        [1, 2, 3, 4], size=n_patients, p=[0.55, 0.30, 0.10, 0.05]
    )
    total = admissions_per.sum()
    while total < n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per[idx] < 4:
            admissions_per[idx] += 1
            total += 1
    while total > n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per[idx] > 1:
            admissions_per[idx] -= 1
            total -= 1

    for pid in range(1, n_patients + 1):
        for _ in range(int(admissions_per[pid - 1])):
            adm_to_pid[adm_id] = pid
            adm_id += 1

    for aid in range(1, min(adm_id, n_admissions + 1)):
        pid = adm_to_pid[aid]
        admit_offset = int(RNG.integers(0, 730))
        admit_dt = base_date + timedelta(days=admit_offset)

        n_labs = int(
            RNG.integers(max(1, readings_per_admission - 3), readings_per_admission + 4)
        )
        for _ in range(n_labs):
            hours_offset = float(RNG.uniform(0, 168))
            coll_dt = admit_dt + timedelta(hours=hours_offset)
            lab = RNG.choice(lab_names)
            norm_lo, norm_hi, mean, std = lab_specs[lab]
            val = float(np.clip(RNG.normal(mean, std), 0.1, None))
            abnormal = int(val < norm_lo or val > norm_hi)

            patient_ids.append(pid)
            admission_ids.append(aid)
            collected_at.append(coll_dt.strftime("%Y-%m-%d %H:%M:%S"))
            l_names.append(lab)
            values.append(round(val, 2))
            abnormal_flags.append(abnormal)

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "admission_id": admission_ids,
            "collected_at": collected_at,
            "lab_name": l_names,
            "value": values,
            "abnormal_flag": abnormal_flags,
        }
    )


def make_icu_medications() -> pl.DataFrame:
    """
    ICU medication administrations — ~6,000 rows.
    Columns: patient_id, admission_id, administered_at, medication_name.
    Includes vasopressors (norepinephrine, dopamine), antibiotics (vancomycin,
    meropenem), and common ICU medications.
    """
    from datetime import datetime, timedelta

    n_admissions = 1500
    target_rows = 6_000
    meds_per_admission = target_rows // n_admissions  # ~4

    medications = [
        "norepinephrine",
        "dopamine",
        "vancomycin",
        "meropenem",
        "midazolam",
        "fentanyl",
        "propofol",
        "heparin",
        "insulin",
        "furosemide",
        "pantoprazole",
        "metoprolol",
    ]
    # Weights: antibiotics and sedatives more common
    med_weights = np.array(
        [0.08, 0.05, 0.12, 0.10, 0.10, 0.12, 0.10, 0.08, 0.07, 0.06, 0.06, 0.06]
    )
    med_weights = med_weights / med_weights.sum()

    patient_ids = []
    admission_ids = []
    administered_at = []
    med_names = []

    base_date = datetime(2022, 1, 1)

    # Build admission lookup
    adm_id = 1
    adm_to_pid = {}
    n_patients = 800
    admissions_per = RNG.choice(
        [1, 2, 3, 4], size=n_patients, p=[0.55, 0.30, 0.10, 0.05]
    )
    total = admissions_per.sum()
    while total < n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per[idx] < 4:
            admissions_per[idx] += 1
            total += 1
    while total > n_admissions:
        idx = RNG.integers(0, n_patients)
        if admissions_per[idx] > 1:
            admissions_per[idx] -= 1
            total -= 1

    for pid in range(1, n_patients + 1):
        for _ in range(int(admissions_per[pid - 1])):
            adm_to_pid[adm_id] = pid
            adm_id += 1

    for aid in range(1, min(adm_id, n_admissions + 1)):
        pid = adm_to_pid[aid]
        admit_offset = int(RNG.integers(0, 730))
        admit_dt = base_date + timedelta(days=admit_offset)

        n_meds = int(
            RNG.integers(max(1, meds_per_admission - 2), meds_per_admission + 3)
        )
        for _ in range(n_meds):
            hours_offset = float(RNG.uniform(0, 168))
            admin_dt = admit_dt + timedelta(hours=hours_offset)
            med = RNG.choice(medications, p=med_weights)

            patient_ids.append(pid)
            admission_ids.append(aid)
            administered_at.append(admin_dt.strftime("%Y-%m-%d %H:%M:%S"))
            med_names.append(med)

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "admission_id": admission_ids,
            "administered_at": administered_at,
            "medication_name": med_names,
        }
    )


# ---------------------------------------------------------------------------
# 23. sg_cooling_measures.csv  (ascent02) — ~5 rows
# ---------------------------------------------------------------------------


def make_sg_cooling_measures() -> pl.DataFrame:
    """
    Singapore property cooling measure dates (real policy events).
    Small reference table for feature engineering exercises — students
    use these dates to create 'post_cooling_measure' features.
    """
    return pl.DataFrame(
        {
            "event_date": [
                "2021-12-16",
                "2022-09-30",
                "2023-04-27",
                "2023-02-14",
                "2024-01-01",
            ],
            "measure_name": [
                "ABSD rate increase + TDSR tightening",
                "ABSD +5pp across all buyer categories",
                "ABSD for foreigners doubled to 60%",
                "GLS supply increase + private housing cooling",
                "HDB resale levy adjustments",
            ],
            "affected_segment": [
                "All residential",
                "All residential",
                "Foreign buyers",
                "Private residential",
                "HDB resale",
            ],
        }
    )


# ---------------------------------------------------------------------------
# 24. credit_card_fraud.parquet  (ascent04) — 50,000 rows
# ---------------------------------------------------------------------------


def make_credit_card_fraud() -> pl.DataFrame:
    """
    Synthetic credit card fraud dataset with PCA-transformed features (50,000 rows).
    Severe class imbalance: ~0.5% fraud rate (250 fraud vs 49,750 legit).

    Columns: transaction_id, v1-v28 (PCA features), amount, time_seconds, is_fraud.
    Legit transactions have standard-normal V-features; fraud transactions
    have shifted distributions to simulate real PCA-transformed anomalies.
    """
    n = 50_000
    n_fraud = int(n * 0.005)  # 250 fraud
    n_legit = n - n_fraud

    # PCA-transformed features: standard normal for legit, shifted for fraud
    v_legit = RNG.standard_normal((n_legit, 28))
    # Fraud: shift some features significantly (mimics real PCA separation)
    v_fraud = RNG.standard_normal((n_fraud, 28))
    fraud_shifts = RNG.uniform(-3.0, 3.0, 28)
    v_fraud += fraud_shifts

    v_all = np.vstack([v_legit, v_fraud])

    # Amount: legit skews low, fraud has heavier tail
    amount_legit = np.abs(RNG.lognormal(3.0, 1.5, n_legit))
    amount_legit = np.clip(amount_legit, 0.01, 25_000.0)
    amount_fraud = np.abs(RNG.lognormal(5.0, 1.8, n_fraud))
    amount_fraud = np.clip(amount_fraud, 0.01, 25_000.0)
    amount_all = np.concatenate([amount_legit, amount_fraud])
    amount_all = np.round(amount_all, 2)

    # Time: seconds elapsed over 2 days (172,800 seconds)
    time_all = RNG.uniform(0, 172_800, n)
    time_all = np.round(time_all, 1)

    # Labels
    is_fraud = np.array([0] * n_legit + [1] * n_fraud, dtype=int)

    # Shuffle all rows together
    shuffle_idx = RNG.permutation(n)
    v_all = v_all[shuffle_idx]
    amount_all = amount_all[shuffle_idx]
    time_all = time_all[shuffle_idx]
    is_fraud = is_fraud[shuffle_idx]

    data: dict = {"transaction_id": list(range(1, n + 1))}
    for i in range(28):
        data[f"v{i + 1}"] = np.round(v_all[:, i], 6).tolist()
    data["amount"] = amount_all.tolist()
    data["time_seconds"] = time_all.tolist()
    data["is_fraud"] = is_fraud.tolist()

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# 25. sg_news_corpus.parquet  (ascent04) — 5,000 rows
# ---------------------------------------------------------------------------


def make_sg_news_corpus() -> pl.DataFrame:
    """
    Singapore-themed news corpus for NLP exercises (5,000 articles).
    Topics: economy, property, transport, education, tech.
    Each article has a title, body (100-500 words), and published_date.
    Dates span 2023-01 to 2024-06.
    """
    from datetime import date, timedelta

    n = 5_000

    topics = {
        "economy": {
            "titles": [
                "Singapore GDP growth beats expectations in {quarter}",
                "MAS tightens monetary policy amid inflation concerns",
                "Singapore trade surplus narrows as exports slow",
                "Labour market remains tight with record low unemployment",
                "Singapore inflation eases to {rate}% in {month}",
                "Foreign investment inflows reach new high",
                "Singapore retail sales surge during festive season",
                "Manufacturing output contracts for third straight month",
                "Services sector drives economic expansion",
                "Singapore budget focuses on cost of living relief",
            ],
            "fragments": [
                "The Monetary Authority of Singapore noted that core inflation remained elevated. "
                "Economists at major banks forecast further tightening of the SGD nominal effective exchange rate band. "
                "The labour market continued to show resilience with the overall unemployment rate holding steady. ",
                "Trade-dependent Singapore saw export volumes decline amid weaker global demand. "
                "The electronics cluster posted a contraction while the biomedical sector expanded. "
                "Non-oil domestic exports fell for the second consecutive month. ",
                "The government announced additional support measures for small and medium enterprises. "
                "GST vouchers and CDC vouchers were distributed to eligible households. "
                "Minister for Finance outlined plans to boost productivity through digitalisation. ",
                "Foreign direct investment commitments in manufacturing reached a record high. "
                "The financial services sector attracted significant inflows from wealth management activities. "
                "Singapore maintained its position as a top destination for regional headquarters. ",
                "Consumer spending patterns shifted towards online channels. "
                "Food and beverage establishments reported mixed results. "
                "Tourism recovery continued to gain momentum with visitor arrivals approaching pre-pandemic levels. ",
            ],
        },
        "property": {
            "titles": [
                "HDB resale prices rise for {count} consecutive quarter",
                "Private home sales surge in {district} district",
                "New cooling measures target investment properties",
                "BTO launch sees overwhelming demand in {town}",
                "En bloc fever returns to suburban condominiums",
                "Rental market softens as supply increases",
                "Government land sales programme expanded for {half}",
                "Sentosa Cove luxury segment sees renewed interest",
                "Industrial property rents climb on tight supply",
                "Singapore office occupancy rates recover post-pandemic",
            ],
            "fragments": [
                "The Urban Redevelopment Authority released flash estimates showing private property prices rose. "
                "Analysts attributed the increase to strong demand from upgraders and limited new supply. "
                "The proportion of million-dollar HDB resale transactions continued to climb. ",
                "New Build-To-Order flats in mature estates drew overwhelming subscription rates. "
                "First-time applicants faced ballot odds exceeding ten to one for popular projects. "
                "The Housing and Development Board announced additional supply in upcoming exercises. ",
                "Cooling measures introduced included higher additional buyer stamp duty rates for foreign purchasers. "
                "The total debt servicing ratio framework was tightened to sixty percent. "
                "Property agents reported a slowdown in speculative buying following the announcements. ",
                "The rental market showed signs of moderation after two years of rapid increases. "
                "Completed condominium units entering the market added to available supply. "
                "Expatriate demand remained a key driver for central region rental properties. ",
                "Commercial property transactions in the central business district remained robust. "
                "Office rents in Grade A buildings continued their upward trajectory. "
                "Co-working operators expanded their footprint across multiple locations. ",
            ],
        },
        "transport": {
            "titles": [
                "Cross Island Line construction reaches major milestone",
                "Thomson-East Coast Line Stage {stage} opens to commuters",
                "ERP 2.0 satellite-based system implementation update",
                "Changi Airport Terminal 5 construction progresses",
                "Bus network restructuring improves connectivity in {area}",
                "Active mobility regulations updated for personal mobility devices",
                "LTA awards contract for {line} extension",
                "Public transport ridership returns to pre-COVID levels",
                "Singapore explores autonomous vehicle trials in {district_area}",
                "Cycling network expanded with new paths in {region}",
            ],
            "fragments": [
                "The Land Transport Authority announced progress on the Cross Island Line. "
                "Tunnelling works beneath the Central Catchment Nature Reserve were completed with minimal environmental impact. "
                "The line will connect Changi to Jurong Industrial Estate via twenty-six stations. ",
                "The Thomson-East Coast Line opened additional stations improving connectivity. "
                "Commuters from the eastern corridor reported significantly reduced travel times to the city centre. "
                "Integration with existing bus services was enhanced through revised feeder routes. ",
                "Electronic Road Pricing adjustments were made to manage peak-hour congestion. "
                "The satellite-based ERP 2.0 system underwent extensive testing across major expressways. "
                "Motorists were advised to install the new on-board units ahead of the transition date. ",
                "Changi Airport Group reported that passenger traffic continued its recovery trajectory. "
                "Terminal operations were optimised to handle increasing flight frequencies. "
                "Airport expansion plans remained on track with Terminal 5 groundwork advancing. ",
                "Public bus services were enhanced with new express routes connecting major towns. "
                "Electric bus deployment expanded as part of the national decarbonisation strategy. "
                "Real-time arrival information systems were upgraded across all bus stops. ",
            ],
        },
        "education": {
            "titles": [
                "Singapore universities climb in global rankings",
                "MOE announces changes to PSLE scoring system review",
                "Polytechnic graduates see improved employment outcomes",
                "SkillsFuture programmes expanded for mid-career workers",
                "International schools report record enrolment numbers",
                "New autonomous university status for {institution}",
                "Singapore students top international maths assessment",
                "Applied learning modules introduced across secondary schools",
                "Research funding increased for artificial intelligence studies",
                "Lifelong learning participation rates reach new high",
            ],
            "fragments": [
                "The Ministry of Education announced initiatives to strengthen applied learning pathways. "
                "Students will have more opportunities to develop skills through hands-on projects. "
                "Collaboration between schools and industry partners was enhanced. ",
                "National University of Singapore and Nanyang Technological University maintained top positions. "
                "Research output in engineering and computer science fields attracted international recognition. "
                "Student exchange programmes were expanded with partner institutions globally. ",
                "SkillsFuture credits were topped up for eligible Singaporeans. "
                "Training participation rates among professionals managers executives and technicians increased. "
                "Digital skills courses saw the highest demand across all programme categories. ",
                "Early childhood education received additional funding to improve quality standards. "
                "Preschool teacher qualifications requirements were raised progressively. "
                "The government expanded access to affordable childcare in residential heartlands. ",
                "STEM education initiatives were rolled out across primary and secondary schools. "
                "Coding and computational thinking were integrated into the curriculum. "
                "Teachers received professional development support for technology-enhanced pedagogy. ",
            ],
        },
        "tech": {
            "titles": [
                "Singapore digital bank licences drive fintech innovation",
                "Smart Nation initiative expands digital government services",
                "Data centre moratorium partially lifted with green conditions",
                "Singapore startup raises Series {round} for AI platform",
                "Cybersecurity agency issues advisory on ransomware threats",
                "5G standalone network coverage reaches {coverage} of Singapore",
                "Government cloud migration ahead of schedule",
                "Singapore tech sector hiring slows amid global uncertainty",
                "Blockchain pilot for trade finance launched by MAS",
                "AI governance framework updated with generative AI guidelines",
            ],
            "fragments": [
                "The Infocomm Media Development Authority announced new digital infrastructure investments. "
                "Singapore continued to position itself as a regional technology hub. "
                "Government digital services were consolidated under a unified platform. ",
                "Artificial intelligence adoption across enterprises reached a significant milestone. "
                "The National AI Strategy was updated to include large language model governance. "
                "Research institutes received additional funding for responsible AI development. ",
                "Cybersecurity threats targeting critical information infrastructure were identified. "
                "The Cyber Security Agency issued updated advisories for essential services operators. "
                "Cross-border cooperation on cyber incident response was strengthened with ASEAN partners. ",
                "The fintech ecosystem attracted record venture capital investment. "
                "Digital payment adoption continued to grow across retail and food sectors. "
                "Regulatory sandbox programmes enabled testing of innovative financial products. ",
                "Data centre operators committed to achieving net-zero emissions targets. "
                "Green data centre standards were published to guide sustainable operations. "
                "Tropical data centre cooling technologies developed locally gained international interest. ",
            ],
        },
    }

    topic_names = list(topics.keys())
    fill_vars = {
        "quarter": ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"],
        "rate": ["3.5", "4.0", "3.8", "3.2", "2.9", "3.1"],
        "month": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "count": ["eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth"],
        "town": [
            "Tampines",
            "Woodlands",
            "Jurong East",
            "Punggol",
            "Tengah",
            "Bukit Merah",
        ],
        "district": ["central", "east", "west", "north", "northeast"],
        "half": ["H1 2023", "H2 2023", "H1 2024", "H2 2024"],
        "stage": ["4", "5", "6"],
        "area": ["Punggol", "Sengkang", "Tengah", "Tampines North"],
        "line": ["Jurong Region", "Cross Island", "Circle"],
        "district_area": ["one-north", "Punggol", "Jurong Innovation District"],
        "region": ["western", "eastern", "northern", "central"],
        "institution": ["SUTD", "SIT", "SUSS"],
        "round": ["A", "B", "C"],
        "coverage": ["70%", "80%", "90%", "95%"],
    }

    # Date range: 2023-01-01 to 2024-06-30
    start = date(2023, 1, 1)
    end = date(2024, 6, 30)
    date_range_days = (end - start).days

    titles_out: list[str] = []
    bodies_out: list[str] = []
    dates_out: list[str] = []

    for _ in range(n):
        topic = topic_names[RNG.integers(0, len(topic_names))]
        tdata = topics[topic]

        # Pick and fill title template
        title_tmpl = tdata["titles"][RNG.integers(0, len(tdata["titles"]))]
        title = title_tmpl
        for key, vals in fill_vars.items():
            placeholder = "{" + key + "}"
            if placeholder in title:
                title = title.replace(placeholder, vals[RNG.integers(0, len(vals))])
        titles_out.append(title)

        # Build body from fragments (100-500 words target)
        target_words = int(RNG.integers(100, 501))
        body_parts: list[str] = []
        current_words = 0
        while current_words < target_words:
            frag = tdata["fragments"][RNG.integers(0, len(tdata["fragments"]))]
            body_parts.append(frag)
            current_words += len(frag.split())
        body = " ".join(body_parts).strip()
        # Fill any placeholders in body
        for key, vals in fill_vars.items():
            placeholder = "{" + key + "}"
            while placeholder in body:
                body = body.replace(placeholder, vals[RNG.integers(0, len(vals))], 1)
        bodies_out.append(body)

        # Random date in range
        offset_days = int(RNG.integers(0, date_range_days + 1))
        pub_date = start + timedelta(days=offset_days)
        dates_out.append(pub_date.strftime("%Y-%m-%d"))

    return pl.DataFrame(
        {
            "title": titles_out,
            "body": bodies_out,
            "published_date": dates_out,
        }
    )


# ---------------------------------------------------------------------------
# 26. hdb_resale_sample.parquet  (ascent07) — 2,000 rows
# ---------------------------------------------------------------------------


def make_hdb_resale_sample() -> pl.DataFrame:
    """
    Simple HDB resale data for linear regression exercises (2,000 rows).
    Two columns: floor_area_sqm and resale_price with a clear linear
    relationship (price ~ 5000 * area + noise). Area range 30-170 sqm.
    """
    n = 2_000
    area = RNG.uniform(30.0, 170.0, n)
    noise = RNG.normal(0, 30_000, n)
    price = 5_000.0 * area + 50_000 + noise  # intercept ~50k
    price = np.clip(price, 100_000, 1_200_000)

    return pl.DataFrame(
        {
            "floor_area_sqm": np.round(area, 1).tolist(),
            "resale_price": np.round(price, 2).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# 27. mnist_sample.parquet  (ascent07) — 3,000 rows
# ---------------------------------------------------------------------------


def make_mnist_sample() -> pl.DataFrame:
    """
    Synthetic MNIST-like dataset for deep learning exercises (3,000 rows).
    28x28 pixel images (784 pixels, int 0-255) with label 0-9.
    Each digit class has distinct synthetic patterns:
      0=ellipse, 1=vertical bar, 2=zigzag, 3=horizontal stripes,
      4=cross, 5=diagonal, 6=hook, 7=top-bar, 8=figure-eight, 9=loop.
    Not photo-realistic — just structurally distinct per class.
    """
    n = 3_000
    n_per_class = n // 10
    all_pixels: list[np.ndarray] = []
    all_labels: list[int] = []

    for digit in range(10):
        for _ in range(n_per_class):
            img = np.zeros((28, 28), dtype=float)

            if digit == 0:  # Ellipse
                cy, cx = 14 + RNG.integers(-2, 3), 14 + RNG.integers(-2, 3)
                for y in range(28):
                    for x in range(28):
                        if 0.6 < ((y - cy) / 9) ** 2 + ((x - cx) / 6) ** 2 < 1.4:
                            img[y, x] = RNG.integers(180, 256)
            elif digit == 1:  # Vertical bar
                cx = 14 + RNG.integers(-3, 4)
                w = RNG.integers(2, 5)
                img[3:25, cx - w : cx + w] = RNG.integers(180, 256, (22, 2 * w))
            elif digit == 2:  # Zigzag
                for row in range(4, 24):
                    offset = (row % 8) - 4
                    col = 14 + offset + RNG.integers(-1, 2)
                    col = max(1, min(26, col))
                    img[row, col - 1 : col + 2] = RNG.integers(180, 256, 3)
            elif digit == 3:  # Horizontal stripes
                for row in [6, 10, 14, 18, 22]:
                    shift = RNG.integers(-1, 2)
                    img[row + shift, 5:23] = RNG.integers(180, 256, 18)
            elif digit == 4:  # Cross
                cx, cy = 14 + RNG.integers(-2, 3), 14 + RNG.integers(-2, 3)
                img[cy - 1 : cy + 2, 4:24] = RNG.integers(180, 256, (3, 20))
                img[4:24, cx - 1 : cx + 2] = RNG.integers(180, 256, (20, 3))
            elif digit == 5:  # Diagonal (top-left to bottom-right)
                for i in range(24):
                    r, c = 2 + i, 2 + i + RNG.integers(-1, 2)
                    r, c = max(0, min(27, r)), max(0, min(27, c))
                    c_start = max(0, c - 1)
                    c_end = min(28, c + 2)
                    img[r, c_start:c_end] = RNG.integers(180, 256, c_end - c_start)
            elif digit == 6:  # Hook (vertical + bottom curve)
                img[3:20, 10:13] = RNG.integers(180, 256, (17, 3))
                img[18:23, 12:20] = RNG.integers(180, 256, (5, 8))
            elif digit == 7:  # Top bar + diagonal
                img[4:7, 6:22] = RNG.integers(180, 256, (3, 16))
                for i in range(18):
                    c = 20 - i + RNG.integers(-1, 2)
                    c = max(1, min(26, c))
                    img[6 + i, c - 1 : c + 2] = RNG.integers(180, 256, 3)
            elif digit == 8:  # Figure-eight (two circles)
                for cy in [9, 19]:
                    for y in range(28):
                        for x in range(28):
                            dist = ((y - cy) / 5) ** 2 + ((x - 14) / 5) ** 2
                            if 0.5 < dist < 1.5:
                                img[y, x] = max(img[y, x], RNG.integers(180, 256))
            elif digit == 9:  # Loop (top circle + vertical)
                for y in range(28):
                    for x in range(28):
                        dist = ((y - 9) / 5) ** 2 + ((x - 14) / 5) ** 2
                        if 0.5 < dist < 1.5:
                            img[y, x] = RNG.integers(180, 256)
                img[14:25, 17:20] = RNG.integers(180, 256, (11, 3))

            # Add noise
            noise = RNG.integers(0, 30, (28, 28))
            img = np.clip(img + noise, 0, 255).astype(int)
            all_pixels.append(img.flatten())
            all_labels.append(digit)

    pixels_arr = np.array(all_pixels)

    # Shuffle
    shuffle_idx = RNG.permutation(n)
    pixels_arr = pixels_arr[shuffle_idx]
    all_labels_arr = np.array(all_labels)[shuffle_idx]

    data: dict = {}
    for i in range(784):
        data[f"pixel_{i}"] = pixels_arr[:, i].tolist()
    data["label"] = all_labels_arr.tolist()

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# 28. fashion_mnist_sample.parquet  (ascent07) — 3,000 rows
# ---------------------------------------------------------------------------


def make_fashion_mnist_sample() -> pl.DataFrame:
    """
    Synthetic Fashion-MNIST-like dataset (3,000 rows).
    Same schema as MNIST (pixel_0-pixel_783 + label 0-9) but with
    different patterns per class to represent clothing items:
      0=T-shirt, 1=trouser, 2=pullover, 3=dress, 4=coat,
      5=sandal, 6=shirt, 7=sneaker, 8=bag, 9=ankle boot.
    """
    n = 3_000
    n_per_class = n // 10
    all_pixels: list[np.ndarray] = []
    all_labels: list[int] = []

    for label in range(10):
        for _ in range(n_per_class):
            img = np.zeros((28, 28), dtype=float)

            if label == 0:  # T-shirt: wide top, narrow bottom
                img[4:8, 4:24] = RNG.integers(150, 256, (4, 20))  # shoulders
                img[8:22, 8:20] = RNG.integers(150, 256, (14, 12))  # body
            elif label == 1:  # Trouser: two vertical bars
                img[4:24, 6:12] = RNG.integers(150, 256, (20, 6))  # left leg
                img[4:24, 16:22] = RNG.integers(150, 256, (20, 6))  # right leg
                img[4:8, 6:22] = RNG.integers(150, 256, (4, 16))  # waist
            elif label == 2:  # Pullover: T-shirt + sleeves
                img[4:8, 2:26] = RNG.integers(150, 256, (4, 24))  # wide shoulders
                img[8:22, 8:20] = RNG.integers(150, 256, (14, 12))  # body
                img[6:14, 2:8] = RNG.integers(150, 256, (8, 6))  # left sleeve
                img[6:14, 20:26] = RNG.integers(150, 256, (8, 6))  # right sleeve
            elif label == 3:  # Dress: triangular
                for row in range(4, 24):
                    width = 4 + int((row - 4) * 0.8)
                    left = max(0, 14 - width)
                    right = min(28, 14 + width)
                    img[row, left:right] = RNG.integers(150, 256, right - left)
            elif label == 4:  # Coat: wide rectangle with collar
                img[3:6, 10:18] = RNG.integers(180, 256, (3, 8))  # collar
                img[6:24, 4:24] = RNG.integers(150, 256, (18, 20))  # body
                img[6:14, 2:6] = RNG.integers(150, 256, (8, 4))  # left sleeve
                img[6:14, 22:26] = RNG.integers(150, 256, (8, 4))  # right sleeve
            elif label == 5:  # Sandal: bottom band + straps
                img[20:24, 4:24] = RNG.integers(150, 256, (4, 20))  # sole
                img[14:20, 6:10] = RNG.integers(150, 256, (6, 4))  # strap 1
                img[10:20, 16:20] = RNG.integers(150, 256, (10, 4))  # strap 2
            elif label == 6:  # Shirt: like T-shirt with buttons (center line)
                img[4:8, 4:24] = RNG.integers(150, 256, (4, 20))
                img[8:22, 8:20] = RNG.integers(150, 256, (14, 12))
                for row in range(9, 21, 3):  # button line
                    img[row, 13:15] = RNG.integers(220, 256, 2)
            elif label == 7:  # Sneaker: shoe shape
                img[16:24, 4:24] = RNG.integers(150, 256, (8, 20))  # sole area
                img[10:18, 6:22] = RNG.integers(150, 256, (8, 16))  # upper
                img[8:12, 16:24] = RNG.integers(180, 256, (4, 8))  # heel counter
            elif label == 8:  # Bag: rectangle with handle
                img[10:24, 6:22] = RNG.integers(150, 256, (14, 16))  # body
                img[4:10, 10:12] = RNG.integers(180, 256, (6, 2))  # left handle
                img[4:10, 16:18] = RNG.integers(180, 256, (6, 2))  # right handle
                img[4:6, 10:18] = RNG.integers(180, 256, (2, 8))  # handle top
            elif label == 9:  # Ankle boot: boot shape
                img[14:24, 4:22] = RNG.integers(150, 256, (10, 18))  # sole+foot
                img[4:16, 4:14] = RNG.integers(150, 256, (12, 10))  # shaft

            # Add noise
            noise = RNG.integers(0, 25, (28, 28))
            img = np.clip(img + noise, 0, 255).astype(int)
            all_pixels.append(img.flatten())
            all_labels.append(label)

    pixels_arr = np.array(all_pixels)

    # Shuffle
    shuffle_idx = RNG.permutation(n)
    pixels_arr = pixels_arr[shuffle_idx]
    all_labels_arr = np.array(all_labels)[shuffle_idx]

    data: dict = {}
    for i in range(784):
        data[f"pixel_{i}"] = pixels_arr[:, i].tolist()
    data["label"] = all_labels_arr.tolist()

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# 29. synthetic_spirals.parquet  (ascent07) — 500 rows
# ---------------------------------------------------------------------------


def make_synthetic_spirals() -> pl.DataFrame:
    """
    Two interleaved spirals for non-linear classification exercises (500 rows).
    Classic toy dataset: 250 points per class, arranged in two spirals that
    are not linearly separable. Columns: x1, x2, label (0 or 1).
    """
    n_per_class = 250
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    r = theta / (4 * np.pi)

    # Class 0 spiral
    x1_0 = r * np.cos(theta) + RNG.normal(0, 0.03, n_per_class)
    x2_0 = r * np.sin(theta) + RNG.normal(0, 0.03, n_per_class)

    # Class 1 spiral (rotated by pi)
    x1_1 = r * np.cos(theta + np.pi) + RNG.normal(0, 0.03, n_per_class)
    x2_1 = r * np.sin(theta + np.pi) + RNG.normal(0, 0.03, n_per_class)

    x1 = np.concatenate([x1_0, x1_1])
    x2 = np.concatenate([x2_0, x2_1])
    labels = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    shuffle_idx = RNG.permutation(500)
    x1 = x1[shuffle_idx]
    x2 = x2[shuffle_idx]
    labels = labels[shuffle_idx]

    return pl.DataFrame(
        {
            "x1": np.round(x1, 6).tolist(),
            "x2": np.round(x2, 6).tolist(),
            "label": labels.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# 30. sg_news_articles.parquet  (ascent08) — 5,000 rows
# ---------------------------------------------------------------------------


def make_sg_news_articles() -> pl.DataFrame:
    """
    Singapore news articles for NLP exercises (5,000 rows).
    Categories: economy, property, transport, education, technology, politics.
    Sources: Straits Times, CNA, Business Times, TODAY.
    Text 100-400 words from randomised sentence templates with vocabulary pools.
    """
    categories = [
        "economy",
        "property",
        "transport",
        "education",
        "technology",
        "politics",
    ]
    sources = ["Straits Times", "CNA", "Business Times", "TODAY"]

    _economy_subjects = [
        "Singapore's GDP growth",
        "The manufacturing sector",
        "The services industry",
        "Export volumes",
        "Foreign direct investment",
        "The labour market",
        "Consumer spending",
        "Retail sales",
        "The tourism sector",
        "The financial services sector",
        "Trade volumes with ASEAN partners",
        "The semiconductor industry",
        "The biomedical sciences cluster",
        "Inflation",
        "The unemployment rate",
        "Business confidence",
    ]
    _economy_verbs = [
        "expanded by",
        "contracted by",
        "remained steady at",
        "surged to",
        "declined to",
        "recovered to",
        "stabilised at",
        "grew by",
        "fell to",
        "rose to",
        "reached",
        "hit a record",
    ]
    _economy_details = [
        "according to data released by the Ministry of Trade and Industry.",
        "as reported by the Department of Statistics Singapore.",
        "driven by strong external demand from China and the United States.",
        "amid global supply chain disruptions and rising input costs.",
        "supported by government fiscal stimulus measures.",
        "reflecting improved business sentiment across key industries.",
        "despite headwinds from the global economic slowdown.",
        "on the back of robust demand for electronics and pharmaceuticals.",
        "as companies ramped up hiring to meet post-pandemic demand.",
        "with the Monetary Authority of Singapore maintaining its exchange rate policy.",
    ]
    _economy_context = [
        "Economists at DBS Group Research noted that the recovery has been uneven across sectors.",
        "OCBC Bank analysts forecast continued growth in the second half of the year.",
        "The government has pledged additional support for workers in transitioning industries.",
        "The MAS core inflation measure remained within the central bank's target range.",
        "Small and medium enterprises reported improved cash flow conditions.",
        "The Purchasing Managers' Index remained in expansionary territory for the third consecutive month.",
        "Trade and Industry Minister highlighted the need for continued economic restructuring.",
        "Singapore retained its position as the most competitive economy in Asia.",
        "The Straits Times Index closed higher on the back of positive economic data.",
        "Industry leaders called for more support for digitalisation and automation.",
    ]

    _property_subjects = [
        "HDB resale prices",
        "Private condo prices",
        "New launch sales",
        "The property cooling measures",
        "BTO application rates",
        "Executive condominium launches",
        "Landed property transactions",
        "The en bloc market",
        "Rental rates",
        "Commercial property demand",
        "Office vacancy rates",
        "Industrial property rents",
    ]
    _property_details = [
        "in the quarter, according to data from the Urban Redevelopment Authority.",
        "as buyers returned to the market following the easing of pandemic restrictions.",
        "amid expectations of further government intervention in the property market.",
        "with mature estates such as Queenstown, Toa Payoh, and Bishan seeing the strongest demand.",
        "despite the Additional Buyer's Stamp Duty of 60 per cent for foreign buyers.",
        "as the number of unsold units in the pipeline decreased.",
        "with developers launching fewer units to manage supply.",
        "reflecting strong demand from upgraders and first-time buyers.",
        "as landlords raised asking rents to match tight supply conditions.",
        "driven by the return of expatriate tenants and relocating professionals.",
    ]
    _property_context = [
        "PropNex CEO noted that the market remained resilient despite higher interest rates.",
        "ERA Realty analysts expect prices to moderate in the coming quarters.",
        "The Housing and Development Board announced additional BTO launches for the year.",
        "National Development Minister addressed concerns about housing affordability.",
        "The government reiterated its commitment to ensuring sufficient housing supply.",
        "Real estate investment trusts listed on the SGX saw increased investor interest.",
        "Mortgage rates at major Singapore banks ranged from 3.5 to 4.2 per cent.",
        "The property sector contributed significantly to GDP growth in the period.",
    ]

    _transport_subjects = [
        "MRT ridership",
        "Bus service reliability",
        "The Thomson-East Coast Line",
        "COE premiums",
        "Electric vehicle adoption",
        "The cycling network",
        "Changi Airport passenger numbers",
        "The Cross Island Line",
        "Road traffic congestion",
        "The Jurong Region Line",
        "Taxi and private-hire availability",
        "ERP charges",
    ]
    _transport_details = [
        "with the Land Transport Authority announcing improvements to the network.",
        "as commuter numbers returned to pre-pandemic levels.",
        "following the opening of new stations along the extended line.",
        "amid rising demand for vehicle ownership despite quota controls.",
        "with the government targeting 60,000 EV charging points by 2030.",
        "as part of the Land Transport Master Plan 2040.",
        "following the completion of Terminal 5 construction works.",
        "with construction progressing on schedule for the planned completion date.",
        "as the Electronic Road Pricing system was upgraded to ERP 2.0.",
        "driven by population growth in western and northeastern corridors.",
    ]
    _transport_context = [
        "Transport Minister outlined plans to improve first-and-last-mile connectivity.",
        "SBS Transit and SMRT reported higher operational costs due to energy prices.",
        "The Rail Reliability Index improved for the third consecutive year.",
        "Commuters in Punggol and Sengkang welcomed the new bus interchange.",
        "The government approved funding for the next phase of network expansion.",
        "Active mobility regulations were tightened following safety incidents.",
        "Singapore Airlines reported a record quarter driven by travel recovery.",
        "Grab and Gojek announced updated fare structures for private-hire vehicles.",
    ]

    _education_subjects = [
        "The PSLE scoring system",
        "University admissions",
        "Polytechnic enrolment",
        "The SkillsFuture programme",
        "International school demand",
        "The Institutes of Technical Education",
        "Preschool capacity",
        "Teacher recruitment",
        "STEM education initiatives",
        "The National University of Singapore",
        "Nanyang Technological University",
    ]
    _education_details = [
        "according to figures released by the Ministry of Education.",
        "as schools adapted their curricula to emphasise applied learning.",
        "with the government investing in digital infrastructure for classrooms.",
        "reflecting the shift towards competency-based assessment frameworks.",
        "amid growing demand for places at top-tier secondary schools.",
        "as part of the broader national lifelong learning agenda.",
        "with more students opting for hands-on polytechnic pathways.",
        "following updates to the post-secondary education landscape.",
        "as the number of international students in Singapore increased.",
        "driven by industry partnerships for work-study programmes.",
    ]
    _education_context = [
        "Education Minister emphasised the importance of holistic development.",
        "Parents expressed mixed views on the changes to the assessment framework.",
        "NUS and NTU maintained their positions in the top 15 of global rankings.",
        "The Institute of Technical Education expanded its applied diploma offerings.",
        "SkillsFuture Credit utilisation rates rose significantly among mid-career workers.",
        "Schools reported improved student well-being scores after the curriculum revision.",
        "The Early Childhood Development Agency announced new licensing standards.",
        "Singapore Management University launched new interdisciplinary programmes.",
    ]

    _technology_subjects = [
        "Singapore's Smart Nation initiative",
        "Fintech investment",
        "The AI governance framework",
        "Cybersecurity regulations",
        "5G network deployment",
        "Data centre capacity",
        "Startup funding",
        "GovTech digital services",
        "Cloud computing adoption",
        "The digital economy",
    ]
    _technology_details = [
        "as part of the government's digitalisation push across all sectors.",
        "with Singapore attracting record levels of venture capital funding.",
        "following the release of updated guidelines by the Infocomm Media Development Authority.",
        "amid growing concerns about data privacy and digital security threats.",
        "with all three mobile operators achieving island-wide 5G coverage.",
        "as demand from cloud providers and AI workloads continued to surge.",
        "driven by favourable regulatory conditions and access to regional markets.",
        "with Singpass integration expanding to more private sector services.",
        "as enterprises accelerated their migration to hybrid cloud architectures.",
        "reflecting Singapore's position as a regional technology hub.",
    ]
    _technology_context = [
        "The Cyber Security Agency issued new advisories for critical infrastructure operators.",
        "GovTech announced plans to open-source more government technology projects.",
        "The Personal Data Protection Commission imposed fines for data breaches.",
        "Minister for Communications highlighted Singapore's role in global AI governance.",
        "Block71, the startup hub, expanded its network to more countries in Southeast Asia.",
        "Singapore ranked first in Asia for digital competitiveness.",
        "The National AI Strategy 2.0 outlined priorities for the next decade.",
        "Tech companies announced significant hiring plans for their Singapore offices.",
    ]

    _politics_subjects = [
        "The upcoming general election",
        "Parliamentary debates",
        "The elected presidency",
        "Town council governance",
        "The Workers' Party",
        "The People's Action Party",
        "Government budget allocations",
        "The ministerial salary review",
        "Constituency boundary changes",
        "The Group Representation Constituency system",
    ]
    _politics_details = [
        "as political discourse intensified ahead of the next election cycle.",
        "with Members of Parliament raising questions on cost of living concerns.",
        "following the tabling of the annual Budget statement in Parliament.",
        "amid public discussions about the future direction of governance.",
        "as the Electoral Boundaries Review Committee completed its work.",
        "with the government announcing new community development programmes.",
        "reflecting shifts in voter sentiment across different age groups.",
        "as grassroots organisations stepped up engagement with residents.",
        "with political commentators analysing the implications for future policy.",
        "following a series of by-elections that drew national attention.",
    ]
    _politics_context = [
        "Political analysts noted the increasing importance of bread-and-butter issues.",
        "The Speaker of Parliament called for more constructive debate on policy.",
        "Young Singaporeans expressed greater interest in civic participation.",
        "The government reaffirmed its commitment to transparent public finances.",
        "Opposition parties presented alternative policy proposals on housing and healthcare.",
        "The Prime Minister addressed the nation on forward-looking economic strategies.",
        "Community leaders called for stronger social safety nets for vulnerable groups.",
        "A survey by the Institute of Policy Studies revealed shifting public priorities.",
    ]

    _pools = {
        "economy": (
            _economy_subjects,
            _economy_verbs,
            _economy_details,
            _economy_context,
        ),
        "property": (
            _property_subjects,
            _economy_verbs,
            _property_details,
            _property_context,
        ),
        "transport": (
            _transport_subjects,
            _economy_verbs,
            _transport_details,
            _transport_context,
        ),
        "education": (
            _education_subjects,
            _economy_verbs,
            _education_details,
            _education_context,
        ),
        "technology": (
            _technology_subjects,
            _economy_verbs,
            _technology_details,
            _technology_context,
        ),
        "politics": (
            _politics_subjects,
            _economy_verbs,
            _politics_details,
            _politics_context,
        ),
    }

    _fillers = [
        "This comes at a time when Singapore is navigating complex global economic conditions.",
        "Analysts say the trend is likely to continue in the near term.",
        "The government has emphasised the need for a balanced and sustainable approach.",
        "Industry stakeholders have broadly welcomed the developments.",
        "Observers note that Singapore's small and open economy makes it sensitive to external shocks.",
        "The measures are expected to benefit both businesses and households.",
        "Experts caution that risks remain on the horizon, including geopolitical tensions.",
        "Public feedback sessions have been conducted to gather views on the proposed changes.",
        "The impact on household budgets remains a key concern for many Singaporeans.",
        "Data from government agencies suggests a gradual improvement in conditions.",
        "Non-governmental organisations have called for more targeted support measures.",
        "Neighbouring countries in ASEAN are watching Singapore's approach closely.",
        "The development aligns with Singapore's long-term strategic planning goals.",
        "Parliamentary committees will review the proposals in the coming weeks.",
        "Singapore's reputation as a stable and well-governed city-state continues to attract investment.",
        "Community organisations have organised forums to discuss the implications.",
        "The announcement was made during a press conference at the relevant ministry.",
        "Further details are expected to be released in the next quarterly report.",
        "Companies listed on the Singapore Exchange reacted positively to the news.",
        "The findings were presented at a conference held at the Singapore Expo.",
    ]

    def _build_article(cat: str) -> str:
        subjects, verbs, details, context = _pools[cat]
        n_blocks = RNG.integers(5, 13)
        sentences: list[str] = []
        for _ in range(n_blocks):
            subj = random.choice(subjects)
            verb = random.choice(verbs)
            pct = f"{RNG.uniform(0.5, 8.5):.1f} per cent"
            detail = random.choice(details)
            sentences.append(f"{subj} {verb} {pct} {detail}")
            if RNG.random() > 0.4:
                sentences.append(random.choice(context))
            if RNG.random() > 0.6:
                sentences.append(random.choice(_fillers))
        return " ".join(sentences)

    n = 5000
    col_text: list[str] = []
    col_date: list[str] = []
    col_source: list[str] = []
    col_category: list[str] = []

    for _ in range(n):
        cat = random.choice(categories)
        col_category.append(cat)
        col_text.append(_build_article(cat))
        year = RNG.integers(2020, 2025)
        month = RNG.integers(1, 13)
        day = RNG.integers(1, 29)
        col_date.append(f"{year}-{month:02d}-{day:02d}")
        col_source.append(random.choice(sources))

    return pl.DataFrame(
        {
            "text": col_text,
            "date": col_date,
            "source": col_source,
            "category": col_category,
        }
    )


# ---------------------------------------------------------------------------
# 31. sg_parliament_speeches.parquet  (ascent08) — 500 rows
# ---------------------------------------------------------------------------


def make_sg_parliament_speeches() -> pl.DataFrame:
    """
    Singapore parliamentary speeches for NLP exercises (500 rows).
    Formal parliamentary language. Text 200-800 words.
    Sessions: 14th Parliament Session 1, 14th Parliament Session 2.
    """
    speakers = [
        "Mr Speaker",
        "Minister for Finance",
        "Minister for Trade and Industry",
        "Minister for National Development",
        "Minister for Education",
        "Minister for Health",
        "Minister for Transport",
        "Minister for Manpower",
        "Minister for Home Affairs",
        "Minister for Communications and Information",
        "Minister for Sustainability and the Environment",
        "Senior Minister of State for Finance",
        "Member for Aljunied GRC",
        "Member for Sengkang GRC",
        "Member for Marine Parade GRC",
        "Member for Tanjong Pagar GRC",
        "Member for Holland-Bukit Timah GRC",
        "Member for Ang Mo Kio GRC",
        "Nominated Member of Parliament",
        "Non-Constituency Member of Parliament",
    ]
    sessions = ["14th Parliament Session 1", "14th Parliament Session 2"]

    _openings = [
        "Mr Speaker, Sir, I rise to address the House on the matter of",
        "Mr Speaker, I thank the honourable Member for the question regarding",
        "Mr Speaker, the Government takes very seriously the issue of",
        "Mr Speaker, permit me to elaborate on the Government's position concerning",
        "Mr Speaker, I wish to provide an update to this House on",
        "Mr Speaker, Sir, the Ministry has carefully considered the proposals relating to",
        "Mr Speaker, in response to the question raised by the honourable Member on",
        "Mr Speaker, the Government has taken a comprehensive approach to addressing",
    ]

    _topics = [
        "housing affordability for young Singaporeans",
        "the rising cost of living and its impact on household budgets",
        "healthcare subsidies for the pioneer and Merdeka generations",
        "the future of employment in the digital economy",
        "climate change adaptation and coastal protection measures",
        "public transport reliability and network expansion",
        "the regulation of artificial intelligence and emerging technologies",
        "support for small and medium enterprises in the post-pandemic recovery",
        "strengthening Singapore's position as a global financial centre",
        "educational reform and the emphasis on lifelong learning",
        "the management of foreign worker policies and workforce planning",
        "cybersecurity threats and the protection of critical information infrastructure",
        "the development of the Jurong Lake District as a second CBD",
        "water security and the fourth national tap of desalination",
        "Singapore's defence capabilities and national service obligations",
        "the promotion of arts, culture, and heritage in Singapore",
    ]

    _body_sentences = [
        "The Government has allocated significant resources to address this matter.",
        "We have engaged extensively with stakeholders from both the public and private sectors.",
        "Let me assure this House that the Ministry will continue to monitor the situation closely.",
        "The data shows a clear trend that requires careful and considered intervention.",
        "We must balance the competing interests of economic growth and social equity.",
        "Singapore's approach has always been to plan for the long term while remaining responsive to immediate needs.",
        "The Committee of Supply debates provided valuable input on this issue.",
        "I wish to emphasise that no Singaporean will be left behind as we navigate these changes.",
        "The relevant agencies have been directed to expedite the implementation of these measures.",
        "We will continue to review our policies regularly to ensure they remain effective and relevant.",
        "International experience suggests that a calibrated approach yields the best outcomes.",
        "The fiscal resources committed to this initiative reflect its importance to the Government.",
        "Preliminary results from the pilot programme have been encouraging.",
        "The Ministry has received feedback from over 10,000 residents through our public consultation exercise.",
        "We are committed to transparency and accountability in the implementation of these policies.",
        "The inter-ministerial committee has recommended a phased approach to implementation.",
        "This is consistent with the forward-looking orientation that has served Singapore well.",
        "The budgetary implications have been carefully assessed by the Ministry of Finance.",
        "We recognise that the concerns raised by Members on both sides of the House are legitimate.",
        "The evidence base for these policy recommendations is robust and well-documented.",
        "Our civil servants have worked tirelessly to develop practical and workable solutions.",
        "Singapore's tripartite model of collaboration between government, employers, and unions remains central to our approach.",
        "The National Wages Council has provided guidance on ensuring fair and sustainable wage growth.",
        "We have studied best practices from comparable economies and adapted them to our unique context.",
        "The legislative amendments before this House are the product of extensive deliberation.",
        "I urge all Members to support these measures, which are in the national interest.",
        "The White Paper sets out a comprehensive roadmap for the next decade.",
        "Ground-up initiatives have complemented the Government's top-down policy interventions.",
        "The Auditor-General's report confirms that public funds have been used responsibly.",
        "We must remain vigilant against complacency as external conditions continue to evolve.",
    ]

    _closings = [
        "In conclusion, the Government remains committed to building a fair and prosperous Singapore for all.",
        "I trust that this response has addressed the concerns raised by the honourable Member.",
        "The Ministry will provide a detailed written response to the supplementary questions.",
        "I commend these measures to the House and urge all Members to give them their support.",
        "Thank you, Mr Speaker.",
        "The Government will continue to work closely with all stakeholders to ensure successful outcomes.",
        "I am confident that with the support of this House, we will achieve our shared objectives.",
        "Let us move forward together with determination and resolve.",
    ]

    def _build_speech() -> str:
        opening = random.choice(_openings)
        topic = random.choice(_topics)
        n_body = RNG.integers(8, 26)
        body = [random.choice(_body_sentences) for _ in range(n_body)]
        closing = random.choice(_closings)
        return f"{opening} {topic}. " + " ".join(body) + f" {closing}"

    n = 500
    col_text: list[str] = []
    col_speaker: list[str] = []
    col_date: list[str] = []
    col_session: list[str] = []

    for _ in range(n):
        col_text.append(_build_speech())
        col_speaker.append(random.choice(speakers))
        year = RNG.integers(2023, 2025)
        month = RNG.integers(1, 13)
        day = RNG.integers(1, 29)
        col_date.append(f"{year}-{month:02d}-{day:02d}")
        col_session.append(random.choice(sessions))

    return pl.DataFrame(
        {
            "text": col_text,
            "speaker": col_speaker,
            "date": col_date,
            "session": col_session,
        }
    )


# ---------------------------------------------------------------------------
# 32. sg_product_reviews.parquet  (ascent08) — 2,000 rows
# ---------------------------------------------------------------------------


def make_sg_product_reviews() -> pl.DataFrame:
    """
    Singapore product reviews for sentiment analysis (2,000 rows).
    Categories: electronics, food, fashion, household.
    Text 20-200 words. Ratings correlated with sentiment.
    """
    categories = ["electronics", "food", "fashion", "household"]

    _products = {
        "electronics": [
            ("ELEC-001", "Wireless Earbuds Pro"),
            ("ELEC-002", "Smart Watch X"),
            ("ELEC-003", "Portable Charger 20K"),
            ("ELEC-004", "Bluetooth Speaker Mini"),
            ("ELEC-005", "USB-C Hub 7-in-1"),
            ("ELEC-006", "Noise Cancelling Headphones"),
            ("ELEC-007", "Mechanical Keyboard"),
            ("ELEC-008", "Webcam HD 1080p"),
        ],
        "food": [
            ("FOOD-001", "Salted Egg Fish Skin"),
            ("FOOD-002", "Kaya Toast Spread"),
            ("FOOD-003", "Bak Kwa Premium"),
            ("FOOD-004", "Pandan Layer Cake"),
            ("FOOD-005", "Laksa Paste"),
            ("FOOD-006", "Milo Dinosaur Mix"),
            ("FOOD-007", "Chilli Crab Sauce"),
            ("FOOD-008", "Ondeh Ondeh Cookies"),
        ],
        "fashion": [
            ("FASH-001", "Linen Shirt SG"),
            ("FASH-002", "Smart Casual Chinos"),
            ("FASH-003", "Batik Print Dress"),
            ("FASH-004", "Running Shoes Lite"),
            ("FASH-005", "Canvas Tote Bag"),
            ("FASH-006", "UV Protection Umbrella"),
            ("FASH-007", "Bamboo Sunglasses"),
            ("FASH-008", "Cotton Polo Shirt"),
        ],
        "household": [
            ("HOUSE-001", "Air Purifier HEPA"),
            ("HOUSE-002", "Robot Vacuum V2"),
            ("HOUSE-003", "Dehumidifier Compact"),
            ("HOUSE-004", "Electric Fan Tower"),
            ("HOUSE-005", "Laundry Rack Foldable"),
            ("HOUSE-006", "LED Desk Lamp"),
            ("HOUSE-007", "Rice Cooker Smart"),
            ("HOUSE-008", "Water Filter Jug"),
        ],
    }

    _positive_phrases = [
        "Very happy with this purchase.",
        "Excellent quality for the price.",
        "Would definitely recommend to friends and family.",
        "Works perfectly, exactly as described.",
        "Fantastic product, no complaints.",
        "Super fast delivery from the seller.",
        "Great value for money.",
        "The build quality is impressive.",
        "Exceeded my expectations.",
        "Best purchase I have made this year.",
        "My family loves it.",
        "Perfect for Singapore's hot and humid weather.",
        "Ordered during the Great Singapore Sale, amazing deal.",
        "Five stars, will buy again.",
        "Top notch customer service too.",
        "Really well packaged, arrived in perfect condition.",
        "Been using it for weeks now, still works great.",
        "Shiok lah, very good quality.",
        "Confirm will repurchase.",
    ]

    _negative_phrases = [
        "Very disappointed with this product.",
        "Poor quality, broke after one week.",
        "Would not recommend, waste of money.",
        "Does not work as advertised.",
        "The product arrived damaged.",
        "Terrible customer service experience.",
        "Not worth the price at all.",
        "Returned it immediately.",
        "Quality is much worse than expected.",
        "Regret buying this.",
        "Feels very cheap and flimsy.",
        "Stopped working after a few days.",
        "The description was misleading.",
        "Waited two weeks for delivery, then this.",
        "Asked for a refund, still waiting.",
        "Not suitable for Singapore's climate.",
        "Overpriced for what you get.",
        "Sian, total waste of money.",
    ]

    _neutral_phrases = [
        "It is okay, nothing special.",
        "Average product, does the job.",
        "Decent quality for a mid-range option.",
        "Not bad, not great either.",
        "Works fine but could be improved.",
        "Acceptable for the price point.",
        "Standard quality, as expected.",
        "It serves its purpose.",
        "No major issues but nothing impressive.",
        "Fair enough for daily use.",
        "Got it on sale so can't complain too much.",
        "Meets basic requirements.",
    ]

    _context_phrases = [
        "Bought this for my HDB flat.",
        "Using it in my Jurong East condo.",
        "Got this from the Lazada sale.",
        "Purchased from Shopee during 11.11.",
        "My colleague recommended this.",
        "Saw it at the IT Show at Suntec.",
        "Picked it up from the FairPrice Xtra outlet.",
        "Found it at the Mustafa Centre.",
        "Ordered for my office at Raffles Place.",
        "Bought it for my parents in Ang Mo Kio.",
        "Using it at our Bedok kopitiam.",
        "Got this for the family BBQ at East Coast Park.",
    ]

    def _build_review(rating: int) -> str:
        if rating >= 4:
            pool = _positive_phrases
            n_sentences = RNG.integers(3, 10)
        elif rating <= 2:
            pool = _negative_phrases
            n_sentences = RNG.integers(2, 8)
        else:
            pool = _neutral_phrases
            n_sentences = RNG.integers(2, 6)

        sentences = [random.choice(pool) for _ in range(n_sentences)]
        if RNG.random() > 0.5:
            sentences.insert(
                RNG.integers(0, len(sentences)), random.choice(_context_phrases)
            )
        return " ".join(sentences)

    n = 2000
    col_text: list[str] = []
    col_rating: list[int] = []
    col_product_id: list[str] = []
    col_category: list[str] = []

    for _ in range(n):
        cat = random.choice(categories)
        product_id, _ = random.choice(_products[cat])
        rating = int(RNG.choice([1, 2, 3, 4, 5], p=[0.08, 0.10, 0.17, 0.30, 0.35]))
        col_category.append(cat)
        col_product_id.append(product_id)
        col_rating.append(rating)
        col_text.append(_build_review(rating))

    return pl.DataFrame(
        {
            "review_text": col_text,
            "rating": col_rating,
            "product_id": col_product_id,
            "category": col_category,
        }
    )


# ---------------------------------------------------------------------------
# 33. sg_company_reports.parquet  (ascent09) — 800 rows
# ---------------------------------------------------------------------------


def make_sg_company_reports() -> pl.DataFrame:
    """
    Singapore company reports for RAG/agent exercises (800 rows).
    Report types: Annual Report, Sustainability Report, Financial Statement.
    Companies: major SGX-listed companies. Text 500-2000 words.
    Fiscal years: 2020-2024.
    """
    companies = [
        "DBS Group Holdings",
        "Oversea-Chinese Banking Corporation",
        "United Overseas Bank",
        "Singapore Telecommunications",
        "CapitaLand Investment",
        "Keppel Corporation",
        "ST Engineering",
        "Wilmar International",
        "Singapore Airlines",
        "Jardine Matheson Holdings",
        "Mapletree Industrial Trust",
        "Ascendas REIT",
        "Sembcorp Industries",
        "Venture Corporation",
        "ComfortDelGro Corporation",
    ]
    report_types = ["Annual Report", "Sustainability Report", "Financial Statement"]
    fiscal_years = [2020, 2021, 2022, 2023, 2024]

    _financial_sentences = [
        "Total revenue for the fiscal year amounted to SGD {amt} billion, representing a {pct} per cent increase over the prior year.",
        "Net profit attributable to shareholders was SGD {amt} billion, compared with SGD {amt2} billion in the previous year.",
        "The Group's total assets stood at SGD {amt} billion as at the end of the reporting period.",
        "Operating expenses were managed prudently, declining by {pct} per cent year-on-year.",
        "Return on equity improved to {pct} per cent, reflecting the Group's strong capital management.",
        "The Board of Directors has recommended a final dividend of SGD {div} per share.",
        "Non-performing assets ratio remained low at {pct} per cent, well within regulatory thresholds.",
        "The Group maintained a strong capital adequacy ratio of {pct} per cent, above the MAS minimum requirement.",
        "Cash and cash equivalents increased to SGD {amt} billion, providing ample liquidity.",
        "Earnings per share rose to SGD {div}, a {pct} per cent increase from the previous fiscal year.",
        "The Group's cost-to-income ratio improved to {pct} per cent through operational efficiency gains.",
        "Gross profit margin expanded by {pct} percentage points driven by a favourable product mix.",
    ]

    _sustainability_sentences = [
        "The Group is committed to reducing its carbon footprint by {pct} per cent by 2030.",
        "Total greenhouse gas emissions (Scope 1 and 2) were {amt} thousand tonnes of CO2 equivalent.",
        "Renewable energy accounted for {pct} per cent of total electricity consumption across operations.",
        "The Group invested SGD {amt} million in sustainability-related initiatives during the year.",
        "Employee training hours per full-time equivalent increased to {amt} hours, reflecting the Group's investment in human capital.",
        "The Board ESG Committee met {amt2} times during the year to oversee sustainability strategy.",
        "Water consumption intensity decreased by {pct} per cent compared to the baseline year.",
        "The Group's sustainability bond framework received a favourable second-party opinion.",
        "Community investment programmes benefited over {amt} thousand individuals across the region.",
        "The Group achieved a score of {pct} in the S&P Global Corporate Sustainability Assessment.",
        "Sustainable finance activities totalled SGD {amt} billion, exceeding the annual target.",
        "The Group published its Task Force on Climate-Related Financial Disclosures report.",
    ]

    _strategy_sentences = [
        "Looking ahead, the Group remains focused on its three strategic pillars: growth, efficiency, and sustainability.",
        "Digital transformation continues to be a key priority, with investments in AI, cloud, and data analytics.",
        "The Group is well positioned to capitalise on growth opportunities in Southeast Asia and Greater China.",
        "Our regional presence across {amt2} markets provides diversification and access to high-growth economies.",
        "Innovation and technology adoption will be critical to maintaining our competitive advantage.",
        "We continue to invest in talent development to build a future-ready workforce.",
        "Strategic partnerships and collaborations have expanded the Group's ecosystem and value proposition.",
        "The Group's risk management framework has been enhanced to address emerging threats and opportunities.",
        "Customer centricity remains at the heart of our strategy, driving product and service innovation.",
        "The integration of acquired businesses is progressing well, delivering anticipated synergies.",
        "Capital allocation will prioritise organic growth, bolt-on acquisitions, and shareholder returns.",
        "The Group is committed to maintaining a strong balance sheet to weather macroeconomic uncertainties.",
    ]

    _governance_sentences = [
        "The Board comprises {amt2} directors, of whom {amt2} are independent non-executive directors.",
        "The Audit Committee reviewed the adequacy and effectiveness of internal controls.",
        "The Group adheres to the Code of Corporate Governance issued by the MAS.",
        "Whistleblowing mechanisms are in place to enable reporting of concerns without fear of reprisal.",
        "Executive remuneration is linked to both short-term performance and long-term value creation.",
        "The Nominating Committee reviewed Board composition and succession planning.",
        "Related party transactions were reviewed and approved in accordance with SGX listing requirements.",
        "The Group's enterprise risk management framework identifies, assesses, and mitigates key risks.",
        "Internal audit functions are performed by an independent team reporting directly to the Audit Committee.",
        "The Group complies with all applicable laws and regulations in its countries of operation.",
    ]

    _outlook_sentences = [
        "The economic outlook for Singapore and the region remains cautiously optimistic.",
        "Interest rate movements and geopolitical developments will be key factors to monitor.",
        "The Group expects operating conditions to remain supportive in the coming fiscal year.",
        "Barring unforeseen circumstances, the Board is cautiously positive about the Group's prospects.",
        "Continued investment in digital capabilities will position the Group for long-term growth.",
        "The Group remains vigilant about potential risks from trade tensions and supply chain disruptions.",
        "Population growth and urbanisation in ASEAN will drive demand for the Group's products and services.",
        "The Board is confident in the Group's ability to deliver sustainable returns to shareholders.",
    ]

    def _build_report(report_type: str) -> str:
        if report_type == "Financial Statement":
            pools = [_financial_sentences, _governance_sentences, _outlook_sentences]
        elif report_type == "Sustainability Report":
            pools = [
                _sustainability_sentences,
                _strategy_sentences,
                _governance_sentences,
            ]
        else:  # Annual Report
            pools = [
                _financial_sentences,
                _strategy_sentences,
                _sustainability_sentences,
                _governance_sentences,
                _outlook_sentences,
            ]

        n_sentences = RNG.integers(15, 51)
        sections: list[str] = []
        for _ in range(n_sentences):
            pool = random.choice(pools)
            sent = random.choice(pool)
            sent = sent.replace("{amt}", f"{RNG.uniform(0.5, 120.0):.1f}")
            sent = sent.replace("{amt2}", f"{RNG.integers(3, 20)}")
            sent = sent.replace("{pct}", f"{RNG.uniform(1.0, 25.0):.1f}")
            sent = sent.replace("{div}", f"{RNG.uniform(0.10, 2.50):.2f}")
            sections.append(sent)

        return " ".join(sections)

    n = 800
    col_text: list[str] = []
    col_company: list[str] = []
    col_report_type: list[str] = []
    col_fiscal_year: list[int] = []

    for _ in range(n):
        company = random.choice(companies)
        rtype = random.choice(report_types)
        year = random.choice(fiscal_years)
        col_company.append(company)
        col_report_type.append(rtype)
        col_fiscal_year.append(year)
        col_text.append(_build_report(rtype))

    return pl.DataFrame(
        {
            "text": col_text,
            "company_name": col_company,
            "report_type": col_report_type,
            "fiscal_year": col_fiscal_year,
        }
    )


# ---------------------------------------------------------------------------
# 34. sg_regulations.parquet  (ascent09) — 300 rows
# ---------------------------------------------------------------------------


def make_sg_regulations() -> pl.DataFrame:
    """
    Singapore regulatory documents for RAG exercises (300 rows).
    Categories: banking, insurance, securities, payments, data_protection.
    Text 200-800 words of legal/regulatory language.
    Regulation IDs like "MAS-2023-001".
    """
    categories = ["banking", "insurance", "securities", "payments", "data_protection"]

    _preambles = {
        "banking": [
            "The Monetary Authority of Singapore, in exercise of its powers under the Banking Act (Cap. 19), hereby issues the following notice to all banks in Singapore.",
            "Pursuant to Section 55 of the Banking Act, the Authority directs all licensed banks to comply with the following requirements.",
            "This notice sets out the minimum standards for credit risk management that all banks licensed in Singapore must observe.",
            "In accordance with the regulatory framework established under the Banking Act, the Authority specifies the following prudential standards.",
        ],
        "insurance": [
            "The Monetary Authority of Singapore, pursuant to the Insurance Act (Cap. 142), issues the following guidelines for all registered insurers.",
            "These guidelines set out the Authority's expectations for the management of insurance risk by all licensed insurers operating in Singapore.",
            "In furtherance of the objectives of the Insurance Act, the Authority establishes the following minimum requirements.",
            "This notice applies to all direct insurers and reinsurers registered under the Insurance Act.",
        ],
        "securities": [
            "The Monetary Authority of Singapore, pursuant to the Securities and Futures Act (Cap. 289), issues the following regulations.",
            "These regulations govern the conduct of capital markets intermediaries licensed under the Securities and Futures Act.",
            "In accordance with the Securities and Futures (Licensing and Conduct of Business) Regulations, the Authority sets out the following requirements.",
            "This practice note provides guidance on compliance with the Securities and Futures Act for market participants.",
        ],
        "payments": [
            "The Monetary Authority of Singapore, pursuant to the Payment Services Act 2019, issues the following notice to all licensees.",
            "These guidelines establish the regulatory framework for digital payment token services under the Payment Services Act.",
            "In exercise of its powers under the Payment Services Act, the Authority sets out the following conduct requirements.",
            "This notice applies to all major payment institutions and standard payment institutions licensed under the Payment Services Act.",
        ],
        "data_protection": [
            "The Personal Data Protection Commission, in exercise of its powers under the Personal Data Protection Act 2012, issues the following advisory guidelines.",
            "These guidelines provide practical guidance to organisations on compliance with the data protection provisions of the PDPA.",
            "Pursuant to Section 6 of the Personal Data Protection Act, the Commission sets out the following requirements for data intermediaries.",
            "This guide assists organisations in understanding their obligations under the Personal Data Protection Act 2012.",
        ],
    }

    _legal_body = [
        "An institution shall establish and maintain adequate systems of internal controls to ensure compliance with the requirements set out in this notice.",
        "The board of directors of the institution shall be responsible for overseeing the implementation of these requirements.",
        "Non-compliance with the provisions of this notice may result in regulatory action, including but not limited to the imposition of financial penalties.",
        "Institutions are required to submit periodic reports to the Authority demonstrating compliance with these requirements.",
        "The Authority may, at its discretion, grant exemptions from specific provisions upon application, subject to conditions.",
        "These requirements shall take effect from the date specified in the schedule appended to this notice.",
        "Institutions shall ensure that all relevant personnel are adequately trained on the requirements set out herein.",
        "The Authority reserves the right to amend these requirements from time to time as it deems necessary.",
        "Any reference in this notice to an Act or regulation shall be construed as a reference to that Act or regulation as amended from time to time.",
        "Institutions shall maintain records demonstrating compliance for a minimum period of five years.",
        "The Authority may conduct inspections or audits to verify compliance with the provisions of this notice.",
        "Where an institution discovers a breach of these requirements, it shall notify the Authority within the timeframe specified.",
        "The requirements in this notice are in addition to, and not in substitution for, any other requirements imposed by law.",
        "Transitional arrangements may be made available to institutions that demonstrate good-faith efforts towards full compliance.",
        "The Authority shall publish guidance notes to assist institutions in the interpretation and application of these requirements.",
        "Penalties for non-compliance may include revocation of the relevant licence, financial penalties, or both.",
        "Institutions operating across multiple jurisdictions shall ensure that their Singapore operations comply with these requirements.",
        "The Authority expects institutions to adopt a risk-based approach to compliance, proportionate to the nature and scale of their operations.",
        "Senior management of the institution shall certify compliance with these requirements on an annual basis.",
        "These provisions apply equally to local and foreign institutions licensed to operate in Singapore.",
    ]

    _requirement_clauses = [
        "Every institution shall implement a comprehensive risk assessment framework that identifies, measures, and monitors all material risks.",
        "Capital adequacy requirements shall be maintained at all times, with the minimum ratio set at the level prescribed by the Authority.",
        "Customer due diligence procedures shall be applied at the point of establishing a business relationship and on an ongoing basis.",
        "Technology risk management policies shall address cybersecurity, data integrity, system availability, and third-party dependencies.",
        "Business continuity plans shall be tested at least annually and updated to reflect changes in the operating environment.",
        "Anti-money laundering and counter-financing of terrorism controls shall be commensurate with the risk profile of the institution.",
        "Outsourcing arrangements shall not diminish the institution's ability to meet its regulatory obligations.",
        "Consumer protection measures shall ensure fair dealing outcomes for all customers of the institution.",
        "Fit and proper criteria shall be applied to all persons holding key appointments within the institution.",
        "Disclosure requirements shall ensure that customers receive clear, accurate, and timely information.",
    ]

    def _build_regulation(cat: str) -> str:
        preamble = random.choice(_preambles[cat])
        n_body = RNG.integers(6, 23)
        body_parts: list[str] = []
        for _ in range(n_body):
            if RNG.random() > 0.4:
                body_parts.append(random.choice(_legal_body))
            else:
                body_parts.append(random.choice(_requirement_clauses))
        return preamble + " " + " ".join(body_parts)

    n = 300
    col_text: list[str] = []
    col_reg_id: list[str] = []
    col_date: list[str] = []
    col_category: list[str] = []

    reg_counter = 1
    for _ in range(n):
        cat = random.choice(categories)
        year = RNG.integers(2020, 2025)
        month = RNG.integers(1, 13)
        day = RNG.integers(1, 29)

        prefix = "PDPC" if cat == "data_protection" else "MAS"
        reg_id = f"{prefix}-{year}-{reg_counter:03d}"
        reg_counter += 1

        col_category.append(cat)
        col_reg_id.append(reg_id)
        col_date.append(f"{year}-{month:02d}-{day:02d}")
        col_text.append(_build_regulation(cat))

    return pl.DataFrame(
        {
            "text": col_text,
            "regulation_id": col_reg_id,
            "effective_date": col_date,
            "category": col_category,
        }
    )


# ---------------------------------------------------------------------------
# 35. inventory_demand.parquet  (ascent10) — 90 rows
# ---------------------------------------------------------------------------


def make_inventory_demand() -> pl.DataFrame:
    """
    Simple time series for RL inventory management exercise (90 rows).
    Demand follows a sinusoidal pattern with noise, range 20-50 units per day.
    """
    days = np.arange(1, 91)
    # Sinusoidal base: mean 35, amplitude 10, period ~30 days
    base_demand = 35 + 10 * np.sin(2 * math.pi * days / 30)
    noise = RNG.normal(0, 3, size=90)
    demand = np.clip(base_demand + noise, 20, 50).astype(int)

    return pl.DataFrame(
        {
            "day": days.tolist(),
            "demand": demand.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  directory ready: {d}")

    tasks = [
        (
            make_economic_indicators,
            DATA_ROOT / "ascent01" / "economic_indicators.csv",
            "csv",
        ),
        # sg_taxi_trips is now parquet (50K rows too large for CSV in git)
        (
            make_sg_taxi_trips,
            DATA_ROOT / "ascent01" / "sg_taxi_trips.parquet",
            "parquet",
        ),
        (
            make_experiment_data,
            DATA_ROOT / "ascent02" / "experiment_data.parquet",
            "parquet",
        ),
        (
            make_sg_credit_scoring,
            DATA_ROOT / "ascent03" / "sg_credit_scoring.parquet",
            "parquet",
        ),
        (
            make_ecommerce_customers,
            DATA_ROOT / "ascent04" / "ecommerce_customers.parquet",
            "parquet",
        ),
        (make_documents, DATA_ROOT / "ascent05" / "documents.parquet", "parquet"),
        (make_sg_domain_qa, DATA_ROOT / "ascent06" / "sg_domain_qa.parquet", "parquet"),
        (
            make_preference_pairs,
            DATA_ROOT / "ascent06" / "preference_pairs.parquet",
            "parquet",
        ),
        (
            make_mrt_stations,
            DATA_ROOT / "ascent_assessment" / "mrt_stations.parquet",
            "parquet",
        ),
        (make_schools, DATA_ROOT / "ascent_assessment" / "schools.parquet", "parquet"),
        (make_sg_cpi, DATA_ROOT / "ascent01" / "sg_cpi.csv", "csv"),
        (make_sg_employment, DATA_ROOT / "ascent01" / "sg_employment.csv", "csv"),
        (make_sg_fx_rates, DATA_ROOT / "ascent01" / "sg_fx_rates.csv", "csv"),
        (make_hdb_resale, DATA_ROOT / "ascent01" / "hdb_resale.parquet", "parquet"),
        # --- M1/M2 new datasets ---
        (
            make_ecommerce_ab_test,
            DATA_ROOT / "ascent01" / "ecommerce_ab_test.csv",
            "csv",
        ),
        (
            make_ecommerce_experiment,
            DATA_ROOT / "ascent02" / "ecommerce_experiment.parquet",
            "parquet",
        ),
        (
            make_ascent02_economic_indicators,
            DATA_ROOT / "ascent02" / "economic_indicators.csv",
            "csv",
        ),
        (
            make_icu_patients,
            DATA_ROOT / "ascent02" / "icu_patients.parquet",
            "parquet",
        ),
        (
            make_icu_admissions,
            DATA_ROOT / "ascent02" / "icu_admissions.parquet",
            "parquet",
        ),
        (
            make_icu_vitals,
            DATA_ROOT / "ascent02" / "icu_vitals.parquet",
            "parquet",
        ),
        (
            make_icu_labs,
            DATA_ROOT / "ascent02" / "icu_labs.parquet",
            "parquet",
        ),
        (
            make_icu_medications,
            DATA_ROOT / "ascent02" / "icu_medications.parquet",
            "parquet",
        ),
        (
            make_sg_cooling_measures,
            DATA_ROOT / "ascent02" / "sg_cooling_measures.csv",
            "csv",
        ),
        # --- M4 datasets ---
        (
            make_credit_card_fraud,
            DATA_ROOT / "ascent04" / "credit_card_fraud.parquet",
            "parquet",
        ),
        (
            make_sg_news_corpus,
            DATA_ROOT / "ascent04" / "sg_news_corpus.parquet",
            "parquet",
        ),
        # --- M7 datasets ---
        (
            make_hdb_resale_sample,
            DATA_ROOT / "ascent07" / "hdb_resale_sample.parquet",
            "parquet",
        ),
        (
            make_mnist_sample,
            DATA_ROOT / "ascent07" / "mnist_sample.parquet",
            "parquet",
        ),
        (
            make_fashion_mnist_sample,
            DATA_ROOT / "ascent07" / "fashion_mnist_sample.parquet",
            "parquet",
        ),
        (
            make_synthetic_spirals,
            DATA_ROOT / "ascent07" / "synthetic_spirals.parquet",
            "parquet",
        ),
        # --- M8 datasets ---
        (
            make_sg_news_articles,
            DATA_ROOT / "ascent08" / "sg_news_articles.parquet",
            "parquet",
        ),
        (
            make_sg_parliament_speeches,
            DATA_ROOT / "ascent08" / "sg_parliament_speeches.parquet",
            "parquet",
        ),
        (
            make_sg_product_reviews,
            DATA_ROOT / "ascent08" / "sg_product_reviews.parquet",
            "parquet",
        ),
        # --- M9 datasets ---
        (
            make_sg_company_reports,
            DATA_ROOT / "ascent09" / "sg_company_reports.parquet",
            "parquet",
        ),
        (
            make_sg_regulations,
            DATA_ROOT / "ascent09" / "sg_regulations.parquet",
            "parquet",
        ),
        # --- M10 datasets ---
        (
            make_inventory_demand,
            DATA_ROOT / "ascent10" / "inventory_demand.parquet",
            "parquet",
        ),
    ]

    print("\nGenerating datasets...")
    for fn, path, mode in tasks:
        df = fn()
        if mode == "csv":
            df.write_csv(str(path))
        else:
            df.write_parquet(str(path), compression="zstd")
        size_kb = path.stat().st_size // 1024
        print(
            f"  {path.name:<50} {len(df):>8} rows  {size_kb:>6} KB  ({path.parent.name})"
        )

    # Cross-module dataset copies
    import shutil

    copies = [
        (
            DATA_ROOT / "ascent06" / "preference_pairs.parquet",
            DATA_ROOT / "ascent10" / "preference_pairs.parquet",
        ),
        (
            DATA_ROOT / "ascent06" / "sg_domain_qa.parquet",
            DATA_ROOT / "ascent10" / "sg_domain_qa.parquet",
        ),
        (
            DATA_ROOT / "ascent09" / "sg_company_reports.parquet",
            DATA_ROOT / "ascent10" / "sg_company_reports.parquet",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied: {src.name} → {dst.parent.name}/")

    # Remove the old sg_taxi_trips.csv if it exists (replaced by parquet)
    old_csv = DATA_ROOT / "ascent01" / "sg_taxi_trips.csv"
    if old_csv.exists():
        old_csv.unlink()
        print(f"\n  Removed old file: {old_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
