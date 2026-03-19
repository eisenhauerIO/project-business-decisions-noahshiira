"""
src/reporting.py
────────────────
Assembles the replication summary table from all result objects.

Public API
----------
    build_summary_table(ate_results, robustness, cf, rate_res, policy, qini)
        → pd.DataFrame
    print_summary(summary_df)
"""

from __future__ import annotations

import pandas as pd


def build_summary_table(
    ate_results,       # ATEResults from ate.py
    robustness,        # RobustnessResults from src/robustness.py
    cf,                # CausalForestResults from causal_forest.py
    rate_res:  dict,   # from compute_rate()
    policy,            # PolicyResults from src/policy.py
    qini:      dict,   # from compute_qini()
    pct_benefit: float,
) -> pd.DataFrame:
    """
    Collect all key metrics into a single comparison DataFrame.

    The 'Paper reports' column allows direct comparison against published values.
    """
    perm  = robustness.permutation
    placebo = robustness.placebo

    rows = [
        # ── ATE estimates ──────────────────────────────────────────────────────
        ("ATE — OLS naive",
         f"{ate_results.ate[0]:+.4f}  (p={ate_results.pvalue[0]:.4f})",
         "−0.026 (p < 0.03)"),

        ("ATE — OLS + controls (HC3)",
         f"{ate_results.ate[1]:+.4f}  (p={ate_results.pvalue[1]:.4f})",
         "−0.026 (p < 0.03)"),

        ("ATE — IPW",
         f"{ate_results.ate[2]:+.4f}  (p={ate_results.pvalue[2]:.4f})",
         "N/A"),

        ("ATE — AIPW (doubly-robust)",
         f"{ate_results.ate[3]:+.4f}  (p={ate_results.pvalue[3]:.4f})",
         "N/A"),

        # ── Robustness ─────────────────────────────────────────────────────────
        ("Permutation test p-value",
         f"{perm['p_value']:.4f}",
         "< 0.05"),

        ("Placebo falsification p-value",
         f"{placebo['p_value']:.4f}  (should be > 0.05)",
         "n.s."),

        # ── Causal forest ──────────────────────────────────────────────────────
        ("Mean CATE (causal forest)",
         f"{cf.cate.mean():+.4f}  ({cf.cate.mean()*100:+.2f} pp)",
         "−0.026 (universal)"),

        ("CATE std (effect heterogeneity)",
         f"{cf.cate.std():.4f}",
         "—"),

        ("% customers CATE < 0 (benefit)",
         f"{pct_benefit:.1%}",
         "~95%"),

        ("Confidence score",
         f"{cf.conf_score:.1%}",
         "—"),

        ("RATE (targeting quality)",
         f"{rate_res['rate']:+.4f}",
         "Positive (TOC)"),

        ("AUQC (QINI curve area)",
         f"{qini['auqc']:.4f}",
         "—"),

        # ── Policy ─────────────────────────────────────────────────────────────
        ("Optimal targeting share",
         f"{policy.best_frac:.1%}",
         "~95%"),

        ("Max profit — smart targeting",
         f"€{policy.max_profit_smart:.2f}",
         "~2× universal"),

        ("Max profit — universal nudging",
         f"€{policy.max_profit_univ:.2f}",
         "—"),

        ("Value of personalisation",
         f"€{policy.personalisation_value:.2f}",
         "—"),

        ("Personalisation gain (%)",
         f"{policy.personalisation_gain_pct:.1f}%",
         "~100%"),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Replication", "Paper reports"])


def print_summary(summary_df: pd.DataFrame) -> None:
    """Pretty-print the summary table."""
    print("═" * 85)
    print("  REPLICATION SUMMARY — Smart Green Nudging (von Zahn et al., 2024)")
    print("═" * 85)
    print(summary_df.to_string(index=False))
    print("═" * 85)
