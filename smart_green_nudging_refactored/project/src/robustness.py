"""
src/robustness.py
─────────────────
All robustness and falsification checks for the replication.

Public API
----------
    run_permutation_test(Y, T, X, cfg)  → dict
    run_bandwidth_sensitivity(df, feature_cols, cfg)  → pd.DataFrame
    run_placebo_test(Y, T, X, cfg)  → dict
    run_sutva_sensitivity(Y_train, T_train, ate_reference, cfg)  → pd.DataFrame
    run_all_robustness_checks(df_train, Y, T, X, cfg)  → RobustnessResults
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config

# Import existing ate helpers (already in project)
from ate import run_ate_suite, permutation_test, bandwidth_sensitivity


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class RobustnessResults:
    permutation:        dict
    bandwidth:          pd.DataFrame
    placebo:            dict
    sutva:              pd.DataFrame

    def print_summary(self) -> None:
        perm = self.permutation
        plac = self.placebo

        print("═" * 60)
        print("  ROBUSTNESS CHECK SUMMARY")
        print("═" * 60)

        # Permutation test
        sig = "✓ Significant" if perm["p_value"] < 0.05 else "✗ Not significant"
        print(f"\n  Permutation test")
        print(f"    Observed ATE : {perm['observed_ate']:+.4f}")
        print(f"    Empirical p  : {perm['p_value']:.4f}  ({sig} at 5%)")

        # Placebo test
        plac_sig = "⚠ Unexpected!" if plac["p_value"] < 0.05 else "✓ As expected (n.s.)"
        print(f"\n  Placebo / falsification test")
        print(f"    Placebo ATE  : {plac['observed_ate']:+.4f}")
        print(f"    Empirical p  : {plac['p_value']:.4f}  ({plac_sig})")

        # Bandwidth
        n_sig = self.bandwidth["sig"].str.strip().ne("").sum()
        print(f"\n  Bandwidth sensitivity")
        print(f"    Bandwidths tested : {len(self.bandwidth)}")
        print(f"    Significant ATEs  : {n_sig} / {len(self.bandwidth)}")

        # SUTVA
        print(f"\n  SUTVA sensitivity  (attenuation from social spillover)")
        print(self.sutva.to_string(index=False))
        print("═" * 60)


# ── Individual checks ──────────────────────────────────────────────────────────

def run_permutation_test(
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    cfg: Config,
) -> dict:
    """
    Non-parametric permutation test.
    Shuffles treatment labels `cfg.robustness.n_permutations` times and
    recomputes the ATE each time to build the null distribution.

    Returns a dict with keys: observed_ate, null_mean, null_std, p_value, null_dist.
    """
    return permutation_test(
        Y, T, X,
        n_permutations=cfg.robustness.n_permutations,
        seed=cfg.random_seed,
    )


def run_bandwidth_sensitivity(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    Re-estimate the ATE on progressively restricted sub-samples.
    A robust effect should be stable across bandwidths.

    Returns a DataFrame with columns: bandwidth, n, ate, ci_lo, ci_hi, pvalue, sig.
    """
    return bandwidth_sensitivity(
        df,
        Y_col=cfg.columns.outcome,
        T_col=cfg.columns.treatment,
        feature_cols=feature_cols,
    )


def run_placebo_test(
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    cfg: Config,
) -> dict:
    """
    Falsification test: apply the permutation test to a *pre-treatment*
    pseudo-outcome (simulated here as an independent Bernoulli draw with the
    same mean as Y).  The nudge cannot have caused past behaviour, so the
    ATE on this placebo should be statistically indistinguishable from zero.

    Returns the same dict structure as run_permutation_test.
    """
    rng = np.random.default_rng(cfg.random_seed + cfg.robustness.placebo_seed_offset)
    Y_placebo = pd.Series(
        (rng.random(len(Y)) < float(Y.mean())).astype(float),
        index=Y.index,
    )
    result = permutation_test(
        Y_placebo, T, X,
        n_permutations=cfg.robustness.n_permutations // 2,  # faster
        seed=cfg.random_seed,
    )
    result["is_placebo"] = True
    return result


def run_sutva_sensitivity(
    Y_train:       pd.Series,
    T_train:       pd.Series,
    ate_reference: float,
    cfg:           Config,
) -> pd.DataFrame:
    """
    Quantify how SUTVA violations (social spillover to control units) would
    bias the ATE estimate toward zero.

    For each contamination rate r, a fraction r of control-group observations
    are assigned a partial nudge effect (50% of ATE), and the resulting
    'biased' ATE is compared to the reference.

    Returns a DataFrame: contamination_rate, biased_ate, true_ate, bias.
    """
    rows = []
    for cr in cfg.robustness.sutva_contamination_rates:
        rng = np.random.default_rng(cfg.random_seed)
        Y_contaminated = Y_train.copy()
        ctrl_idx = np.where(T_train == 0)[0]
        n_contaminate = int(cr * len(ctrl_idx))
        contaminated  = rng.choice(ctrl_idx, n_contaminate, replace=False)
        # Partial nudge effect: control outcomes shift by 50% of ATE
        Y_contaminated.iloc[contaminated] = (
            Y_contaminated.iloc[contaminated] * (1 + ate_reference * 0.5)
        )
        biased_ate = (
            Y_train[T_train == 1].mean()
            - Y_contaminated[T_train == 0].mean()
        )
        rows.append({
            "contamination_rate": f"{cr:.0%}",
            "biased_ate":  round(biased_ate, 4),
            "true_ate":    round(ate_reference, 4),
            "bias":        round(biased_ate - ate_reference, 4),
        })
    return pd.DataFrame(rows)


# ── Convenience wrapper ────────────────────────────────────────────────────────

def run_all_robustness_checks(
    df_train:     pd.DataFrame,
    Y_train:      pd.Series,
    T_train:      pd.Series,
    X_train:      pd.DataFrame,
    ate_reference: float,
    feature_cols: list[str],
    cfg:          Config,
) -> RobustnessResults:
    """
    Run all four robustness checks and return a RobustnessResults object.
    Call `.print_summary()` on the result for a formatted overview.
    """
    print("Running permutation test …")
    perm = run_permutation_test(Y_train, T_train, X_train, cfg)

    print("Running bandwidth sensitivity …")
    bw = run_bandwidth_sensitivity(df_train, feature_cols, cfg)

    print("Running placebo / falsification test …")
    placebo = run_placebo_test(Y_train, T_train, X_train, cfg)

    print("Running SUTVA sensitivity …")
    sutva = run_sutva_sensitivity(Y_train, T_train, ate_reference, cfg)

    print("Done.\n")
    return RobustnessResults(
        permutation=perm,
        bandwidth=bw,
        placebo=placebo,
        sutva=sutva,
    )
