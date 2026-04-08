# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT02 — Exercise 4: Bootstrap and Resampling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master bootstrap resampling methods — BCa intervals,
#   parametric vs non-parametric approaches, and distribution-free
#   inference. Compare bootstrap runs with ExperimentTracker.
#
# TASKS:
#   1. Load experiment data and compute basic sample statistics
#   2. Implement percentile, basic, and BCa bootstrap intervals
#   3. Parametric bootstrap — simulate from fitted distribution
#   4. Distribution-free methods: sign test, Wilcoxon signed-rank
#   5. Compare bootstrap run configurations with ExperimentTracker
#   6. Visualise bootstrap distributions with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml import ExperimentTracker
from scipy import stats

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
exp_data = loader.load("ascent02", "experiment_data.parquet")

print("=== Experiment Data ===")
print(f"Shape: {exp_data.shape}")
print(f"Columns: {exp_data.columns}")
print(exp_data.head(8))

# Primary metric: treatment effect (revenue lift per user)
# Subsample to 2000 per group — bootstrap is computationally expensive
SAMPLE_N = 2_000

control_revenue = (
    exp_data.filter(pl.col("experiment_group") == "control")
    .sample(n=SAMPLE_N, seed=42)["revenue"]
    .to_numpy()
    .astype(np.float64)
)
treatment_revenue = (
    exp_data.filter(pl.col("experiment_group") == "treatment_a")
    .sample(n=SAMPLE_N, seed=42)["revenue"]
    .to_numpy()
    .astype(np.float64)
)
lift = treatment_revenue.mean() - control_revenue.mean()

n_control = len(control_revenue)
n_treatment = len(treatment_revenue)

print(
    f"\nControl:   n={n_control:,}, mean=${control_revenue.mean():.2f}, std=${control_revenue.std():.2f}"
)
print(
    f"Treatment: n={n_treatment:,}, mean=${treatment_revenue.mean():.2f}, std=${treatment_revenue.std():.2f}"
)
print(f"Observed lift: ${treatment_revenue.mean() - control_revenue.mean():.2f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Why bootstrap? — when theory fails
# ══════════════════════════════════════════════════════════════════════

# Demonstrate where Normal theory breaks down: median
sample_median = np.median(treatment_revenue)
# Normal theory SE for median (only valid for large n and symmetric dist)
normal_se_median = treatment_revenue.std() * np.sqrt(np.pi / (2 * n_treatment))
normal_ci_median = (
    sample_median - 1.96 * normal_se_median,
    sample_median + 1.96 * normal_se_median,
)

print(f"\n=== Why Bootstrap? ===")
print(f"Sample median: ${sample_median:.2f}")
print(f"Normal-theory 95% CI: [${normal_ci_median[0]:.2f}, ${normal_ci_median[1]:.2f}]")
print(f"(Valid only if distribution is symmetric and n is large)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Three bootstrap interval methods
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=42)
n_resamples = 10_000
alpha = 0.05

# ─────── Percentile bootstrap ────────────────────────────────────────
# 1. Resample with replacement
# 2. Compute statistic on each resample
# 3. Use α/2 and 1-α/2 quantiles as CI

# TODO: Build the bootstrap distribution of the median (10K resamples)
boot_medians = np.array(
    [
        ____  # Hint: np.median(rng.choice(treatment_revenue, size=n_treatment, replace=True))
        for _ in range(n_resamples)
    ]
)

# TODO: Compute the percentile CI using np.percentile at alpha/2 and 1-alpha/2
pct_ci = (
    ____,  # Hint: np.percentile(boot_medians, 100 * alpha / 2)
    ____,  # Hint: np.percentile(boot_medians, 100 * (1 - alpha / 2))
)

print(f"\n=== Bootstrap CI Methods for Median ===")
print(
    f"Percentile CI:    [${pct_ci[0]:.2f}, ${pct_ci[1]:.2f}]  width=${pct_ci[1]-pct_ci[0]:.2f}"
)

# ─────── Basic (pivot) bootstrap ─────────────────────────────────────
# Inverts the bootstrap distribution to correct for bias.
# CI: [2θ̂ - Q_{1-α/2}, 2θ̂ - Q_{α/2}]

# TODO: Compute the basic (pivot) CI
basic_ci = (
    ____,  # Hint: 2 * sample_median - np.percentile(boot_medians, 100 * (1 - alpha / 2))
    ____,  # Hint: 2 * sample_median - np.percentile(boot_medians, 100 * alpha / 2)
)
print(
    f"Basic (pivot) CI: [${basic_ci[0]:.2f}, ${basic_ci[1]:.2f}]  width=${basic_ci[1]-basic_ci[0]:.2f}"
)

# ─────── BCa (Bias-Corrected and accelerated) bootstrap ──────────────
# The gold standard. Corrects for bias (z₀) and acceleration (a).

# TODO: Use scipy.stats.bootstrap with method="BCa" on the median
bca_result = ____  # Hint: stats.bootstrap((treatment_revenue,), statistic=np.median, n_resamples=n_resamples, confidence_level=1 - alpha, method="BCa", random_state=42)
bca_ci = (bca_result.confidence_interval.low, bca_result.confidence_interval.high)
print(
    f"BCa CI:           [${bca_ci[0]:.2f}, ${bca_ci[1]:.2f}]  width=${bca_ci[1]-bca_ci[0]:.2f}"
)


# Manual BCa computation to show the machinery
def bca_manual(data: np.ndarray, statistic, n_boot: int = 5000, alpha: float = 0.05):
    """Manual BCa interval — shows z0 and acceleration computation."""
    n = len(data)
    theta_hat = statistic(data)

    # Bootstrap replicates
    boot_stats = np.array(
        [statistic(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    )

    # TODO: Bias correction z0 = Φ⁻¹(proportion of boot stats below theta_hat)
    z0 = ____  # Hint: stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # Acceleration a: based on jackknife influence values
    jackknife_stats = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jackknife_stats.mean()
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    # Adjusted quantiles
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    alpha_lo = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
    alpha_hi = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))

    ci_lo = np.percentile(boot_stats, 100 * alpha_lo)
    ci_hi = np.percentile(boot_stats, 100 * alpha_hi)

    return ci_lo, ci_hi, z0, a, boot_stats


bca_lo, bca_hi, z0_manual, a_manual, boot_manual = bca_manual(
    treatment_revenue, np.median, n_boot=5000, alpha=alpha
)
print(f"\nManual BCa: z0={z0_manual:.4f}, a={a_manual:.4f}")
print(f"  z0 > 0 → bootstrap distribution is biased above true median (upward skew)")
print(f"  a ≠ 0 → SE changes with location (acceleration)")
print(f"Manual BCa CI: [${bca_lo:.2f}, ${bca_hi:.2f}]")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Parametric bootstrap
# ══════════════════════════════════════════════════════════════════════
# Fit a distribution, then simulate from it.

# Fit Normal to treatment revenue
norm_mu, norm_sigma = treatment_revenue.mean(), treatment_revenue.std(ddof=1)

# TODO: Generate parametric bootstrap samples from N(norm_mu, norm_sigma)
# and compute the median of each simulated sample
param_boot_medians = np.array(
    [
        ____  # Hint: np.median(rng.normal(loc=norm_mu, scale=norm_sigma, size=n_treatment))
        for _ in range(n_resamples)
    ]
)

param_pct_ci = (
    np.percentile(param_boot_medians, 100 * alpha / 2),
    np.percentile(param_boot_medians, 100 * (1 - alpha / 2)),
)

# TODO: Run a Kolmogorov-Smirnov test of the Normal fit
ks_stat, ks_p = (
    ____  # Hint: stats.kstest(treatment_revenue, "norm", args=(norm_mu, norm_sigma))
)

print(f"\n=== Parametric Bootstrap ===")
print(f"Fitted Normal: μ={norm_mu:.2f}, σ={norm_sigma:.2f}")
print(f"KS test (Normal fit): D={ks_stat:.4f}, p={ks_p:.4f}")
if ks_p > 0.05:
    print("Cannot reject Normal fit — parametric bootstrap is appropriate")
else:
    print("Normal fit rejected — non-parametric bootstrap is safer")
print(f"Parametric boot CI: [${param_pct_ci[0]:.2f}, ${param_pct_ci[1]:.2f}]")
print(f"Non-parametric BCa: [${bca_ci[0]:.2f}, ${bca_ci[1]:.2f}]")

# Bootstrap variance comparison
np_boot_var = np.var(boot_medians)
param_boot_var = np.var(param_boot_medians)
print(f"\nBootstrap variance (non-param): {np_boot_var:.4f}")
print(f"Bootstrap variance (parametric): {param_boot_var:.4f}")
print(
    f"Parametric is {'more' if param_boot_var < np_boot_var else 'less'} efficient (lower variance)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Distribution-free tests — no distributional assumption
# ══════════════════════════════════════════════════════════════════════

# ─────── Sign test (one-sample) ───────────────────────────────────────
# Count how many treatment observations exceed the control mean.
individual_diffs = treatment_revenue - control_revenue.mean()
k_positive = np.sum(individual_diffs > 0)
n_nonzero = np.sum(individual_diffs != 0)
sign_p = 2 * min(
    stats.binom.cdf(k_positive, n_nonzero, 0.5),
    1 - stats.binom.cdf(k_positive - 1, n_nonzero, 0.5),
)

print(f"\n=== Distribution-Free Tests ===")
print(f"Sign test (H0: treatment median = control mean):")
print(f"  Treatment values above control mean: {k_positive}/{n_nonzero}")
print(f"  p-value: {sign_p:.6f}")

# ─────── Mann-Whitney U (two independent samples) ──────────────────────
# TODO: Run Mann-Whitney U (two-sided)
mw_stat, mw_p = (
    ____  # Hint: stats.mannwhitneyu(treatment_revenue, control_revenue, alternative="two-sided")
)
auc = mw_stat / (n_treatment * n_control)  # Probability of superiority

print(f"\nMann-Whitney U test (treatment vs control):")
print(f"  U-statistic: {mw_stat:.1f}")
print(f"  p-value: {mw_p:.6f}")
print(f"  P(treatment > control) = {auc:.4f}")

# Comparison of all test results
print(f"\n=== Test Summary ===")
t_stat, t_p = stats.ttest_ind(treatment_revenue, control_revenue, equal_var=False)
print(f"{'Test':<25} {'p-value':>10} {'Decision':>15}")
print("─" * 55)
for test_name, p_val in [
    ("Welch's t-test", t_p),
    ("Mann-Whitney U", mw_p),
    ("Sign test", sign_p),
]:
    decision = "SIGNIFICANT" if p_val < 0.05 else "not significant"
    print(f"{test_name:<25} {p_val:>10.6f} {decision:>15}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare bootstrap configurations with ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def compare_bootstrap_runs():
    """Log and compare different bootstrap configurations as experiment runs."""
    conn = ConnectionManager("sqlite:///ascent02_bootstrap.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)

    # TODO: Create an experiment with tags for the M2 bootstrap comparison
    exp_id = await tracker.create_experiment(
        name=____,  # Hint: "ascent02_bootstrap_comparison"
        description="Bootstrap CI methods comparison — percentile, basic, BCa, parametric",
        tags=["ascent02", "bootstrap", "inference"],
    )

    # Run 1: Non-parametric percentile bootstrap
    async with tracker.run(exp_id, run_name="nonparam_percentile") as run:
        await run.log_params(
            {
                "method": "percentile",
                "parametric": "False",
                "n_resamples": str(n_resamples),
                "statistic": "median",
                "random_seed": "42",
            }
        )
        await run.log_metrics(
            {
                "ci_lower": float(pct_ci[0]),
                "ci_upper": float(pct_ci[1]),
                "ci_width": float(pct_ci[1] - pct_ci[0]),
                "boot_std": float(np.std(boot_medians)),
                "point_estimate": float(sample_median),
            }
        )
        await run.set_tag("method_type", "nonparametric")

    # Run 2: BCa bootstrap
    async with tracker.run(exp_id, run_name="nonparam_bca") as run:
        await run.log_params(
            {
                "method": "BCa",
                "parametric": "False",
                "n_resamples": str(n_resamples),
                "statistic": "median",
                "bias_correction_z0": str(float(z0_manual)),
                "acceleration_a": str(float(a_manual)),
            }
        )
        await run.log_metrics(
            {
                "ci_lower": float(bca_ci[0]),
                "ci_upper": float(bca_ci[1]),
                "ci_width": float(bca_ci[1] - bca_ci[0]),
                "boot_std": float(np.std(boot_manual)),
                "point_estimate": float(sample_median),
                "bias_z0": float(z0_manual),
                "acceleration_a": float(a_manual),
            }
        )
        await run.set_tag("method_type", "nonparametric")

    # Run 3: Parametric bootstrap
    async with tracker.run(exp_id, run_name="parametric_normal") as run:
        await run.log_params(
            {
                "method": "percentile",
                "parametric": "True",
                "distribution": "Normal",
                "n_resamples": str(n_resamples),
                "statistic": "median",
                "fitted_mu": str(float(norm_mu)),
                "fitted_sigma": str(float(norm_sigma)),
            }
        )
        await run.log_metrics(
            {
                "ci_lower": float(param_pct_ci[0]),
                "ci_upper": float(param_pct_ci[1]),
                "ci_width": float(param_pct_ci[1] - param_pct_ci[0]),
                "boot_std": float(np.std(param_boot_medians)),
                "point_estimate": float(sample_median),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }
        )
        await run.set_tag("method_type", "parametric")

    print(f"\n=== ExperimentTracker: Bootstrap Run Comparison ===")
    print(f"  Experiment: {exp_id}")
    print(f"  Logged 3 runs: nonparam_percentile, nonparam_bca, parametric_normal")
    print(
        f"  CI widths — percentile: ${pct_ci[1]-pct_ci[0]:.2f}, "
        f"BCa: ${bca_ci[1]-bca_ci[0]:.2f}, "
        f"parametric: ${param_pct_ci[1]-param_pct_ci[0]:.2f}"
    )

    await conn.close()
    return exp_id


exp_id = asyncio.run(compare_bootstrap_runs())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise bootstrap distributions with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Bootstrap distribution convergence --
convergence_sizes = [100, 500, 1000, 2000, 5000, n_resamples]
convergence_results = {}

rng2 = np.random.default_rng(seed=42)
all_boot = np.array(
    [
        np.median(rng2.choice(treatment_revenue, size=n_treatment, replace=True))
        for _ in range(n_resamples)
    ]
)

for size in convergence_sizes:
    subset = all_boot[:size]
    convergence_results[f"n={size}"] = {
        "boot_mean": float(subset.mean()),
        "ci_width": float(np.percentile(subset, 97.5) - np.percentile(subset, 2.5)),
        "boot_std": float(subset.std()),
    }

fig_convergence = viz.metric_comparison(convergence_results)
fig_convergence.update_layout(
    title="Bootstrap Estimate Convergence vs Number of Resamples"
)
fig_convergence.write_html("ex4_bootstrap_convergence.html")
print("\nSaved: ex4_bootstrap_convergence.html")

# -- Plot 2: Method comparison --
method_metrics = {
    "Percentile": {
        "ci_lower": float(pct_ci[0]),
        "ci_upper": float(pct_ci[1]),
        "ci_width": float(pct_ci[1] - pct_ci[0]),
    },
    "Basic": {
        "ci_lower": float(basic_ci[0]),
        "ci_upper": float(basic_ci[1]),
        "ci_width": float(basic_ci[1] - basic_ci[0]),
    },
    "BCa": {
        "ci_lower": float(bca_ci[0]),
        "ci_upper": float(bca_ci[1]),
        "ci_width": float(bca_ci[1] - bca_ci[0]),
    },
    "Parametric": {
        "ci_lower": float(param_pct_ci[0]),
        "ci_upper": float(param_pct_ci[1]),
        "ci_width": float(param_pct_ci[1] - param_pct_ci[0]),
    },
}

fig_methods = viz.metric_comparison(method_metrics)
fig_methods.update_layout(title="Bootstrap CI Methods: Median Revenue")
fig_methods.write_html("ex4_bootstrap_methods.html")
print("Saved: ex4_bootstrap_methods.html")

# -- Plot 3: Bootstrap distribution training history --
running_ci_widths = []
step = max(1, n_resamples // 200)
for i in range(step, n_resamples + 1, step):
    subset = all_boot[:i]
    width = np.percentile(subset, 97.5) - np.percentile(subset, 2.5)
    running_ci_widths.append(float(width))

history_metrics = {"CI Width (95%)": running_ci_widths}
fig_history = viz.training_history(
    history_metrics, x_label=f"Bootstrap Resamples (×{step})"
)
fig_history.update_layout(title="Bootstrap CI Width Convergence")
fig_history.write_html("ex4_bootstrap_history.html")
print("Saved: ex4_bootstrap_history.html")

print(f"\n✓ Exercise 4 complete — bootstrap and distribution-free inference")
print(
    f"  Key concepts: BCa intervals, parametric vs non-parametric, ExperimentTracker comparison"
)
