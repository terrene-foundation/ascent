# Chapter 12: DriftMonitor

## Overview

Production models degrade when the data they receive drifts from the data they were trained on. `DriftMonitor` detects this using two statistical tests -- Population Stability Index (PSI) and the Kolmogorov-Smirnov (KS) test -- applied per feature. It also monitors performance degradation by comparing current metrics to a stored baseline. This chapter covers reference data setup, drift checking, performance monitoring, scheduled monitoring, and the full report structure.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Polars installed (polars-native data handling)
- Completed [Chapter 11: ModelRegistry](11_model_registry.md)
- Understanding of ConnectionManager from [Core Chapter 10](../00-core/10_connection_manager.md)

## Concepts

### Concept 1: Reference Distribution

Before checking for drift, you must store the training data distribution as a reference. `set_reference()` computes per-feature statistics (histograms for PSI, empirical CDF for KS) and persists them. All subsequent `check_drift()` calls compare current data against this reference.

- **What**: A stored statistical fingerprint of the training data distribution per feature
- **Why**: Drift is defined relative to training data -- without a reference, there is nothing to compare against
- **How**: `set_reference(model_name, reference_data, feature_columns)` computes and stores distribution statistics
- **When**: Once after training (or retraining), before deploying the model

### Concept 2: PSI and KS-Test

PSI (Population Stability Index) measures how much the overall distribution shape has changed by comparing binned frequencies. KS (Kolmogorov-Smirnov) measures the maximum distance between two cumulative distribution functions and provides a p-value for statistical significance. Both are computed per feature.

- **What**: Two complementary statistical tests for distribution shift
- **Why**: PSI captures overall distribution change; KS captures the specific point of maximum divergence and provides a significance test
- **How**: PSI bins both distributions and sums `(p - q) * ln(p/q)`; KS computes `max|F_ref(x) - F_cur(x)|`
- **When**: On every `check_drift()` call -- both tests run automatically

### Concept 3: Performance Degradation

Beyond feature drift, DriftMonitor tracks whether the model's actual performance has degraded by comparing current predictions against actuals. `check_performance()` computes metrics (accuracy, F1, etc.) and compares them to a stored baseline, flagging degradation when the drop exceeds the threshold.

- **What**: Comparison of current prediction quality against a baseline
- **Why**: Feature drift does not always cause performance degradation, and performance can degrade without feature drift (e.g., label shift)
- **How**: `check_performance(model_name, predictions, actuals, baseline_metrics=...)` computes deltas
- **When**: When you have ground truth labels -- either from human review, delayed feedback, or validation sets

### Concept 4: Scheduled Monitoring

`schedule_monitoring()` creates a background asyncio task that calls a user-provided data function at regular intervals and runs drift checks automatically. `cancel_monitoring()` stops it. `shutdown()` cancels all active schedules.

- **What**: Automated periodic drift checks running as background tasks
- **Why**: Production models need continuous monitoring, not just one-time checks
- **How**: `schedule_monitoring(model_name, interval, data_fn, spec=...)` starts a repeating task
- **When**: After deploying a model to production, set up scheduled monitoring with an appropriate interval

## Key API

| Method                  | Parameters                                                       | Returns                        | Description                      |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------ | -------------------------------- |
| `DriftMonitor()`        | `conn`, `psi_threshold`, `ks_threshold`, `performance_threshold` | `DriftMonitor`                 | Create with thresholds           |
| `set_reference()`       | `model_name`, `reference_data: pl.DataFrame`, `feature_columns`  | `None`                         | Store training distribution      |
| `check_drift()`         | `model_name`, `current_data: pl.DataFrame`                       | `DriftReport`                  | Compare current vs reference     |
| `check_performance()`   | `model_name`, `predictions`, `actuals`, `baseline_metrics=None`  | `PerformanceDegradationReport` | Check metric degradation         |
| `get_drift_history()`   | `model_name`, `limit=10`                                         | `list[DriftReport]`            | Retrieve stored drift reports    |
| `schedule_monitoring()` | `model_name`, `interval`, `data_fn`, `spec=None`                 | `None`                         | Start periodic background checks |
| `cancel_monitoring()`   | `model_name`                                                     | `bool`                         | Stop scheduled monitoring        |
| `shutdown()`            | --                                                               | `None`                         | Cancel all active schedules      |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
from datetime import timedelta

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.drift_monitor import (
    DriftMonitor,
    DriftReport,
    DriftSpec,
    FeatureDriftResult,
    PerformanceDegradationReport,
)


async def main() -> None:
    # ── 1. Set up ConnectionManager + DriftMonitor ──────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    monitor = DriftMonitor(
        conn,
        psi_threshold=0.2,   # PSI > 0.2 triggers drift alert
        ks_threshold=0.05,   # KS p-value < 0.05 triggers drift alert
        performance_threshold=0.1,
    )

    # ── 2. Create reference data (training distribution) ────────────

    reference_df = pl.DataFrame(
        {
            "age": [25.0 + (i % 50) for i in range(500)],
            "income": [30000.0 + i * 100.0 for i in range(500)],
            "tenure": [float(1 + i % 24) for i in range(500)],
        }
    )

    # ── 3. set_reference() — store per-feature distribution ─────────

    await monitor.set_reference(
        model_name="churn_model",
        reference_data=reference_df,
        feature_columns=["age", "income", "tenure"],
    )

    # ── 4. check_drift() — no drift (same distribution) ────────────

    stable_df = pl.DataFrame(
        {
            "age": [26.0 + (i % 50) for i in range(300)],
            "income": [30500.0 + i * 100.0 for i in range(300)],
            "tenure": [float(2 + i % 24) for i in range(300)],
        }
    )

    report = await monitor.check_drift("churn_model", stable_df)

    assert isinstance(report, DriftReport)
    assert report.model_name == "churn_model"
    assert len(report.feature_results) == 3
    assert report.sample_size_reference == 500
    assert report.sample_size_current == 300

    # Each feature has drift statistics
    for fr in report.feature_results:
        assert isinstance(fr, FeatureDriftResult)
        assert fr.psi >= 0.0
        assert 0.0 <= fr.ks_pvalue <= 1.0
        assert fr.drift_type in ("none", "moderate", "severe")

    # ── 5. check_drift() — significant drift ───────────────────────

    drifted_df = pl.DataFrame(
        {
            "age": [80.0 + (i % 10) for i in range(300)],
            "income": [150000.0 + i * 500.0 for i in range(300)],
            "tenure": [float(50 + i % 5) for i in range(300)],
        }
    )

    drift_report = await monitor.check_drift("churn_model", drifted_df)

    assert drift_report.overall_drift_detected is True
    assert drift_report.overall_severity in ("moderate", "severe")
    assert len(drift_report.drifted_features) > 0

    # ── 6. get_drift_history() — stored reports ─────────────────────

    history = await monitor.get_drift_history("churn_model", limit=10)
    assert len(history) >= 2

    # ── 7. check_performance() — baseline comparison ────────────────

    predictions = pl.DataFrame({"pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    actuals = pl.DataFrame({"actual": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    perf_report = await monitor.check_performance(
        "churn_model", predictions, actuals,
    )

    assert isinstance(perf_report, PerformanceDegradationReport)
    assert perf_report.current_metrics["accuracy"] == 1.0
    assert perf_report.degraded is False

    # ── 8. check_performance() — with degradation ──────────────────

    degraded_preds = pl.DataFrame(
        {"pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    )
    degraded_actuals = pl.DataFrame(
        {"actual": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )

    degraded_report = await monitor.check_performance(
        "churn_model",
        degraded_preds,
        degraded_actuals,
        baseline_metrics={"accuracy": 0.95, "f1": 0.93},
    )

    assert degraded_report.degraded is True
    assert degraded_report.degradation["accuracy"] > 0

    # ── 9. DriftSpec — custom thresholds ────────────────────────────

    spec = DriftSpec(
        feature_columns=["age", "income"],
        psi_threshold=0.15,
        ks_threshold=0.01,
    )

    assert spec.psi_threshold == 0.15
    assert spec.feature_columns == ["age", "income"]

    # ── 10. schedule_monitoring() + cancel_monitoring() ─────────────

    async def get_current_data() -> pl.DataFrame:
        return stable_df

    await monitor.schedule_monitoring(
        model_name="churn_model",
        interval=timedelta(seconds=60),
        data_fn=get_current_data,
        spec=spec,
    )

    assert "churn_model" in monitor.active_schedules

    cancelled = await monitor.cancel_monitoring("churn_model")
    assert cancelled is True
    assert "churn_model" not in monitor.active_schedules

    # ── 11. shutdown() — cancel all schedules ───────────────────────

    await monitor.schedule_monitoring(
        "churn_model", timedelta(seconds=60), get_current_data
    )
    await monitor.shutdown()
    assert len(monitor.active_schedules) == 0

    # ── 12. Edge cases ──────────────────────────────────────────────

    # check_drift without reference
    try:
        await monitor.check_drift("no_reference_model", stable_df)
    except ValueError as e:
        assert "no reference" in str(e).lower()

    # Non-finite threshold values
    try:
        DriftMonitor(conn, psi_threshold=float("nan"))
    except ValueError:
        pass  # Expected

    await conn.close()

asyncio.run(main())
```

### Step-by-Step Explanation

1. **Setup**: DriftMonitor takes a ConnectionManager and three thresholds: `psi_threshold` (PSI above this triggers alert), `ks_threshold` (KS p-value below this triggers alert), and `performance_threshold` (metric drop above this flags degradation).

2. **Reference data**: A Polars DataFrame representing the training distribution. DriftMonitor computes per-feature statistics from this data.

3. **set_reference()**: Stores the statistical fingerprint. This must be called before any `check_drift()` call for that model.

4. **Stable check**: When current data follows the same distribution as the reference, `overall_severity` is `"none"` or `"moderate"`. Each `FeatureDriftResult` includes PSI, KS statistic, KS p-value, and drift type.

5. **Drifted check**: Dramatically different distributions (age 80+ instead of 25-74, income 150K+ instead of 30K-80K) trigger clear drift. `overall_drift_detected` is `True` and PSI values are high.

6. **History**: All drift reports are persisted. `get_drift_history()` retrieves them for trend analysis.

7. **Performance check (no degradation)**: Perfect predictions (pred == actual for all rows) produce accuracy 1.0. The first call stores this as the baseline.

8. **Performance check (with degradation)**: Comparing current metrics against an explicit baseline of 0.95 accuracy reveals degradation when the model predicts all zeros.

9. **DriftSpec**: Custom per-check configuration that overrides the monitor's default thresholds and selects specific features.

10. **Scheduled monitoring**: `schedule_monitoring()` starts a background task. `cancel_monitoring()` stops it. The data function is called at each interval to fetch fresh data.

11. **Shutdown**: `shutdown()` cancels all active schedules at once -- use during application teardown.

12. **Edge cases**: Checking drift without a reference raises `ValueError`. Non-finite thresholds are rejected at construction.

## Common Mistakes

| Mistake                                          | Correct Pattern                            | Why                                                                   |
| ------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------- |
| Calling `check_drift()` before `set_reference()` | Always call `set_reference()` first        | There is no reference to compare against; raises `ValueError`         |
| Using too-small reference data                   | Use 500+ rows for reliable statistics      | Small samples produce noisy PSI/KS values that trigger false alarms   |
| Setting PSI threshold too low                    | Start with 0.2 (industry standard)         | Thresholds below 0.1 flag minor natural variation as drift            |
| Forgetting to call `shutdown()` on exit          | Call `shutdown()` in your cleanup/teardown | Background tasks keep the event loop alive, preventing clean shutdown |

## Exercises

1. Create a reference distribution with 1000 rows and three features. Generate current data by shifting one feature by 2 standard deviations while keeping the others stable. Verify that only the shifted feature shows drift.

2. Set up scheduled monitoring with a 5-second interval and a data function that alternates between stable and drifted data. Cancel after 15 seconds and inspect the drift history.

3. Implement a complete monitoring pipeline: register a model in ModelRegistry, set a drift reference, check drift, check performance, and decide whether to trigger retraining based on the reports.

## Key Takeaways

- DriftMonitor detects feature distribution shift using PSI and KS-test per feature
- `set_reference()` must be called before any drift checks -- it stores the training distribution
- `check_drift()` returns a `DriftReport` with per-feature results, overall severity, and drifted feature list
- `check_performance()` compares current prediction metrics against a stored baseline
- `DriftSpec` enables per-check custom thresholds and feature selection
- Scheduled monitoring runs drift checks as background tasks at configurable intervals
- Non-finite thresholds and missing references are caught with clear errors

## Next Chapter

[Chapter 13: InferenceServer](13_inference_server.md) -- Serve model predictions via InferenceServer with single-record and batch prediction, caching, and model info queries.
