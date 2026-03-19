"""
src/policy.py
─────────────
Nudge targeting policy, profit analysis, QINI curve, and CATE segmentation.

Public API
----------
    compute_policy(df_test, ate_reference, cfg)  → PolicyResults
    compute_qini(df_test, Y_test, T_test)  → dict
    compute_cate_segments(df_test, cfg)  → pd.DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config

# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class PolicyResults:
    fracs:          np.ndarray   # [0, 1] targeting share grid
    profit_smart:   np.ndarray   # smart targeting profit at each frac
    profit_univ:    np.ndarray   # universal nudging profit at each frac
    best_frac:      float        # optimal targeting share
    df_pol:         pd.DataFrame # test set sorted by CATE (ascending)

    @property
    def max_profit_smart(self) -> float:
        return float(self.profit_smart.max())

    @property
    def max_profit_univ(self) -> float:
        return float(self.profit_univ.max())

    @property
    def personalisation_value(self) -> float:
        return self.max_profit_smart - self.max_profit_univ

    @property
    def personalisation_gain_pct(self) -> float:
        base = max(1.0, self.max_profit_univ)
        return self.personalisation_value / base * 100

    def print_summary(self) -> None:
        print("═" * 55)
        print("  TARGETING POLICY RESULTS")
        print("═" * 55)
        print(f"  Optimal targeting share       : {self.best_frac:.1%}")
        print(f"  Max profit — smart targeting  : €{self.max_profit_smart:.2f}")
        print(f"  Max profit — universal nudging: €{self.max_profit_univ:.2f}")
        print(f"  Value of personalisation      : €{self.personalisation_value:.2f}")
        print(f"  Personalisation gain (%)      : {self.personalisation_gain_pct:.1f}%")
        print("═" * 55)


# ── Core computation ───────────────────────────────────────────────────────────

def compute_policy(
    df_test:       pd.DataFrame,
    ate_reference: float,
    cfg:           Config,
) -> PolicyResults:
    """
    Compute the profit-maximising smart targeting policy.

    The optimal rule is: nudge customer i iff
        -CATE_i * return_cost > nudge_cost
    which reduces to: nudge the customers with the most negative CATE first,
    stopping when the marginal benefit falls below nudge_cost.

    Parameters
    ----------
    df_test : must contain a 'cate' column (from causal forest predictions).
    ate_reference : the ATE estimate used for the universal nudging baseline.
    cfg : experiment config.
    """
    rc    = cfg.policy.return_cost
    nc    = cfg.policy.nudge_cost
    fracs = np.linspace(0, 1, cfg.policy.n_fracs)

    df_pol = df_test.sort_values("cate").reset_index(drop=True)
    n      = len(df_pol)

    profit_smart, profit_univ = [], []
    for frac in fracs:
        k = int(frac * n)
        # Smart: nudge top-k most receptive customers
        smart_benefit = (-df_pol["cate"].iloc[:k].clip(upper=0)).sum() * rc
        profit_smart.append(smart_benefit - k * nc)
        # Universal: nudge same k customers but assume everyone has ATE effect
        univ_benefit = frac * n * max(0.0, -ate_reference) * rc
        profit_univ.append(univ_benefit - k * nc)

    profit_smart = np.array(profit_smart)
    profit_univ  = np.array(profit_univ)
    best_frac    = fracs[np.argmax(profit_smart)]

    return PolicyResults(
        fracs=fracs,
        profit_smart=profit_smart,
        profit_univ=profit_univ,
        best_frac=best_frac,
        df_pol=df_pol,
    )


def compute_qini(
    df_test: pd.DataFrame,
    Y_test:  pd.Series,
    T_test:  pd.Series,
) -> dict:
    """
    Compute the QINI curve (standard uplift model evaluation metric).

    Customers are sorted from most-to-least receptive (ascending CATE).
    The curve traces cumulative net uplift as the fraction targeted increases.
    The area under the QINI curve (AUQC) summarises targeting discrimination.

    Returns
    -------
    dict with keys: fracs, qini_vals, auqc
    """
    cate_sorted_idx = df_test["cate"].argsort().values
    Y_sorted = Y_test.values[cate_sorted_idx]
    T_sorted = T_test.values[cate_sorted_idx]

    n_test           = len(Y_sorted)
    n_treated_total  = T_sorted.sum()
    n_control_total  = (1 - T_sorted).sum()
    ratio            = n_treated_total / max(1, n_control_total)

    qini_vals = []
    cum_uplift = 0.0
    for i in range(n_test):
        if T_sorted[i] == 1:
            cum_uplift += Y_sorted[i]
        else:
            cum_uplift -= Y_sorted[i] * ratio
        qini_vals.append(cum_uplift)

    fracs = np.linspace(0, 1, n_test)
    auqc  = float(np.trapz(qini_vals, fracs) / n_test)

    return {"fracs": fracs, "qini_vals": np.array(qini_vals), "auqc": auqc}


def compute_cate_segments(
    df_test:  pd.DataFrame,
    cfg:      Config,
    outcome_col: str | None = None,
) -> pd.DataFrame:
    """
    Segment customers into CATE quartiles and summarise return rates per segment.

    Returns a DataFrame indexed by quartile label with columns:
        n, return_rate, mean_cate, ci_width
    Also attaches a 'cate_quartile' column to df_test in-place.
    """
    Y = outcome_col or cfg.columns.outcome

    df_test["cate_quartile"] = pd.qcut(
        df_test["cate"],
        q=4,
        labels=["Q1\n(most\nreceptive)", "Q2", "Q3", "Q4\n(least\nreceptive)"],
    )

    seg = (
        df_test
        .groupby("cate_quartile", observed=True)
        .agg(
            n=          (Y,      "count"),
            return_rate=(Y,      "mean"),
            mean_cate=  ("cate", "mean"),
            ci_width=   ("cate", lambda x: x.std() * 1.96 / np.sqrt(len(x))),
        )
        .round(4)
    )
    return seg
