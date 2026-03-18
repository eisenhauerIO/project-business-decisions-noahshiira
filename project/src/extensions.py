"""
src/extensions.py
─────────────────
Section 8 extension analyses: profit lever sensitivity, CATE DGP simulation,
cross-domain translation table, multi-arm nudge, and dynamic targeting.

Public API
----------
    profit_lever_analysis(df_pol, fracs, ate_reference, cfg)
    cate_dgp_simulation(cfg)
    domain_translation_table()  → pd.DataFrame
    multi_arm_simulation(cfg)
    dynamic_targeting_simulation(cfg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.config import Config


# ── 8.1 Profit lever sensitivity ──────────────────────────────────────────────

def profit_lever_analysis(
    df_pol:        pd.DataFrame,
    fracs:         np.ndarray,
    ate_reference: float,
    cfg:           Config,
) -> pd.DataFrame:
    """
    Sweep return_cost × nudge_cost combinations and compute the value of
    personalisation (smart profit − universal profit) for each pair.

    Produces two plots: a heatmap and a break-even curve.
    Returns the full grid as a DataFrame.
    """
    rc_list = cfg.extensions.profit_lever.return_costs
    nc_list = cfg.extensions.profit_lever.nudge_costs
    n       = len(df_pol)
    c       = cfg.plots.colors

    # ── Heatmap grid ──────────────────────────────────────────────────────────
    rows = []
    for rc in rc_list:
        for nc in nc_list:
            p_smart, p_univ = [], []
            for frac in fracs:
                k = int(frac * n)
                p_smart.append(
                    (-df_pol["cate"].iloc[:k].clip(upper=0)).sum() * rc - k * nc
                )
                p_univ.append(
                    frac * n * max(0.0, -ate_reference) * rc - k * nc
                )
            p_smart_arr = np.array(p_smart)
            p_univ_arr  = np.array(p_univ)
            rows.append({
                "return_cost":            rc,
                "nudge_cost":             nc,
                "smart_gain":             p_smart_arr.max(),
                "univ_gain":              p_univ_arr.max(),
                "personalisation_value":  p_smart_arr.max() - p_univ_arr.max(),
                "optimal_frac":           fracs[np.argmax(p_smart_arr)],
            })

    grid_df = pd.DataFrame(rows)
    pivot   = grid_df.pivot(
        index="return_cost", columns="nudge_cost", values="personalisation_value"
    )

    fig, ax = plt.subplots(figsize=cfg.plots.figsize_square)
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlGn", ax=ax,
        cbar_kws={"label": "€ gain: smart vs. universal nudging"},
    )
    ax.set_title("Value of Personalisation (€) — Return Cost vs. Nudge Cost")
    ax.set_xlabel("Nudge cost per customer (€)")
    ax.set_ylabel("Return cost per item (€)")
    plt.tight_layout()
    plt.show()

    # ── Break-even curve ──────────────────────────────────────────────────────
    rc_sweep    = np.linspace(1, cfg.extensions.profit_lever.breakeven_sweep_max, 100)
    nc_fixed    = cfg.policy.nudge_cost
    smart_maxes, univ_maxes = [], []

    for rc in rc_sweep:
        ps = [
            (-df_pol["cate"].iloc[:int(f * n)].clip(upper=0)).sum() * rc
            - int(f * n) * nc_fixed
            for f in fracs
        ]
        pu = [
            f * n * max(0.0, -ate_reference) * rc - int(f * n) * nc_fixed
            for f in fracs
        ]
        smart_maxes.append(max(ps))
        univ_maxes.append(max(pu))

    fig, ax = plt.subplots(figsize=cfg.plots.figsize_wide)
    ax.plot(rc_sweep, smart_maxes, color=c["smart"],     linewidth=2.5, label="Smart targeting")
    ax.plot(rc_sweep, univ_maxes,  color=c["universal"], linewidth=2.5, linestyle="--",
            label="Universal nudging")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.fill_between(
        rc_sweep, smart_maxes, univ_maxes,
        where=[s > u for s, u in zip(smart_maxes, univ_maxes)],
        alpha=0.15, color=c["smart"], label="Smart outperforms",
    )
    ax.set_xlabel("Return cost per item (€)")
    ax.set_ylabel("Max achievable profit gain (€)")
    ax.set_title(
        f"Break-Even Analysis: Smart vs. Universal Nudging\n"
        f"(nudge cost fixed at €{nc_fixed})"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    return grid_df


# ── 8.2 CATE DGP simulation ────────────────────────────────────────────────────

def cate_dgp_simulation(cfg: Config) -> None:
    """
    Simulate CATE distributions under four data-generating processes to
    illustrate when the causal forest adds value over a simple ATE.
    """
    rc   = cfg.policy.return_cost
    nc   = cfg.policy.nudge_cost
    c    = cfg.plots.colors
    ext  = cfg.extensions.cate_dgp_simulation
    rng  = np.random.default_rng(cfg.random_seed)

    if not ext.scenarios:
        # Defaults if YAML had empty scenarios dict
        scenarios_data = {
            "Homogeneous (σ=0.01)":        rng.normal(-0.05, 0.01, ext.n_sim),
            "Low heterogeneity (σ=0.03)":  rng.normal(-0.05, 0.03, ext.n_sim),
            "Moderate — paper (σ=0.07)":   rng.normal(-0.05, 0.07, ext.n_sim),
            "High heterogeneity (σ=0.15)": rng.normal(-0.03, 0.15, ext.n_sim),
        }
    else:
        scenarios_data = {
            label: rng.normal(params[0], params[1], ext.n_sim)
            for label, params in ext.scenarios.items()
        }

    colors = [c["universal"], c["smart"], c["accent"], c["purple"]]
    fig, axes = plt.subplots(1, len(scenarios_data), figsize=(15, 3.5), sharey=False)

    for ax, (label, cates), color in zip(axes, scenarios_data.items(), colors):
        # Economic value
        smart_profit = (-cates[cates < 0]).sum() * rc - (cates < 0).sum() * nc
        univ_profit  = max(0.0, -cates.mean()) * len(cates) * rc - len(cates) * nc
        uplift_eur   = smart_profit - univ_profit

        ax.hist(cates, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0,            color="black", linestyle="-",  linewidth=0.8, alpha=0.5)
        ax.axvline(cates.mean(), color="red",   linestyle="--", linewidth=1.5,
                   label=f"μ={cates.mean():.3f}")
        pct_benefit = (cates < 0).mean()
        ax.set_title(
            f"{label}\n{pct_benefit:.0%} benefit | uplift €{uplift_eur:.0f}",
            fontsize=8.5,
        )
        ax.set_xlabel("CATE")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Count")
    fig.suptitle(
        "Simulated CATE Distributions: When Does Personalisation Add Value?",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.show()

    print("\nKey insight:")
    print("  Homogeneous DGP  → ATE is sufficient; causal forest adds little")
    print("  High heterogeneity → targeting substantially beats universal nudging")
    print("  The €uplift shown per panel = gain of smart vs. universal strategy")


# ── 8.3 Cross-domain translation table ────────────────────────────────────────

def domain_translation_table() -> pd.DataFrame:
    """
    Return a DataFrame mapping the paper's methodology to four other domains.
    """
    rows = [
        {
            "Domain":          "E-commerce returns (original)",
            "Treatment T":     "Green nudge (binary)",
            "Outcome Y":       "Item returned (binary)",
            "Key features X":  "Past return rate, cart value, session",
            "Nuisance model":  "GBM classifier",
            "ID assumption":   "RCT",
        },
        {
            "Domain":          "Healthcare — medication adherence",
            "Treatment T":     "SMS reminder (binary)",
            "Outcome Y":       "Days adherent (count)",
            "Key features X":  "Age, diagnosis, prior adherence",
            "Nuisance model":  "Poisson / GBM",
            "ID assumption":   "Unconfoundedness",
        },
        {
            "Domain":          "EdTech — course completion",
            "Treatment T":     "Personalised study plan (binary)",
            "Outcome Y":       "Course completed (binary)",
            "Key features X":  "Prior quiz scores, login frequency",
            "Nuisance model":  "Logistic / GBM",
            "ID assumption":   "RCT (A/B test)",
        },
        {
            "Domain":          "Credit — default prevention",
            "Treatment T":     "Early intervention call (binary)",
            "Outcome Y":       "Default <90d (binary)",
            "Key features X":  "Credit score, payment history, LTV",
            "Nuisance model":  "GBM classifier",
            "ID assumption":   "Unconfoundedness",
        },
        {
            "Domain":          "Energy — peak-load nudge",
            "Treatment T":     "Peak-hour alert (binary)",
            "Outcome Y":       "kWh in peak window (continuous)",
            "Key features X":  "House size, past consumption, temp",
            "Nuisance model":  "GBM regressor",
            "ID assumption":   "RCT (randomised rollout)",
        },
    ]
    return pd.DataFrame(rows)


# ── 8.4 Multi-arm nudge simulation ────────────────────────────────────────────

def multi_arm_simulation(cfg: Config) -> dict:
    """
    Simulate three competing nudge frames and illustrate personalised arm routing.

    Returns a dict with per-arm mean CATEs and the gain from personalisation.
    """
    ext   = cfg.extensions.multi_arm
    rng   = np.random.default_rng(cfg.random_seed)
    n_sim = ext.n_sim
    c     = cfg.plots.colors

    past_return_rate  = rng.beta(2, 5, n_sim)
    env_sensitivity   = rng.uniform(0, 1, n_sim)
    price_sensitivity = rng.uniform(0, 1, n_sim)

    cate_env  = -0.02 - 0.10 * env_sensitivity   * past_return_rate + rng.normal(0, 0.02, n_sim)
    cate_norm = -0.03 - 0.04 * past_return_rate                     + rng.normal(0, 0.02, n_sim)
    cate_fin  = -0.01 - 0.12 * price_sensitivity * past_return_rate + rng.normal(0, 0.02, n_sim)

    cate_stack = np.stack([cate_env, cate_norm, cate_fin], axis=1)
    best_arm   = np.argmin(cate_stack, axis=1)
    best_cate  = cate_stack[np.arange(n_sim), best_arm]

    arm_cates  = [cate_env, cate_norm, cate_fin]
    arm_colors = ext.arm_colors

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for cate_arr, name, color in zip(arm_cates, ext.arms, arm_colors):
        axes[0].hist(
            cate_arr, bins=50, alpha=0.5,
            label=f"{name} (μ={cate_arr.mean():.3f})",
            color=color, edgecolor="white",
        )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_title("CATE Distributions by Nudge Frame")
    axes[0].set_xlabel("CATE"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

    arm_shares = pd.Series(best_arm).value_counts().sort_index()
    arm_shares.index = ext.arms
    arm_shares.plot(kind="bar", ax=axes[1], color=arm_colors, edgecolor="white")
    axes[1].set_title("Optimal Arm Allocation\n(personalised routing)")
    axes[1].set_xlabel(""); axes[1].set_ylabel("N customers")
    axes[1].set_xticklabels(ext.arms, rotation=0)

    plt.tight_layout()
    plt.show()

    best_single = min(a.mean() for a in arm_cates)
    gain        = best_cate.mean() - best_single

    print(f"Mean CATE — personalised routing : {best_cate.mean():+.4f}")
    for name, ca in zip(ext.arms, arm_cates):
        print(f"Mean CATE — always {name:<15}: {ca.mean():+.4f}")
    print(f"Gain from personalised routing vs. best single arm: {gain:+.4f}")

    return {
        "arm_means":   {name: ca.mean() for name, ca in zip(ext.arms, arm_cates)},
        "routed_mean": best_cate.mean(),
        "gain":        gain,
    }


# ── 8.5 Dynamic targeting ─────────────────────────────────────────────────────

def dynamic_targeting_simulation(cfg: Config) -> pd.DataFrame:
    """
    Simulate how targeting quality improves as the training sample grows.
    CATE estimation error is proxied as σ / √n (CLT scaling).

    Returns a DataFrame: n_training, profit_smart, profit_univ, personalisation_value.
    """
    ext  = cfg.extensions.dynamic_targeting
    rc   = cfg.policy.return_cost
    nc   = cfg.policy.nudge_cost
    rng  = np.random.default_rng(cfg.random_seed)
    c    = cfg.plots.colors

    rows = []
    for n_obs in ext.sample_sizes:
        noise       = 1 / np.sqrt(n_obs)
        true_cates  = rng.normal(ext.true_cate_mean, ext.true_cate_std, ext.test_set_size)
        est_cates   = true_cates + rng.normal(0, noise * ext.true_cate_std, ext.test_set_size)

        mask_smart  = est_cates < 0
        p_smart     = (-true_cates[mask_smart].clip(max=0)).sum() * rc - mask_smart.sum() * nc
        p_univ      = (-true_cates.clip(max=0)).sum() * rc - ext.test_set_size * nc
        rows.append({
            "n_training":            n_obs,
            "profit_smart":          p_smart,
            "profit_univ":           p_univ,
            "personalisation_value": p_smart - p_univ,
        })

    dyn_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=cfg.plots.figsize_wide)
    ax.plot(dyn_df["n_training"], dyn_df["profit_smart"], "o-",
            color=c["smart"],     linewidth=2.5, markersize=7, label="Smart targeting")
    ax.plot(dyn_df["n_training"], dyn_df["profit_univ"],  "s--",
            color=c["universal"], linewidth=2,   markersize=7, label="Universal nudging")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Training sample size (log scale)")
    ax.set_ylabel(f"Profit gain (€, on {ext.test_set_size:,}-customer test set)")
    ax.set_title("Dynamic Targeting: Value of Personalisation vs. Sample Size")
    ax.legend()
    plt.tight_layout()
    plt.show()

    crossover = dyn_df[dyn_df["personalisation_value"] > 0]["n_training"]
    if len(crossover):
        print(f"Smart targeting becomes profitable at ~{crossover.min():,} training observations.")
    print(dyn_df[["n_training", "profit_smart", "profit_univ", "personalisation_value"]].round(2).to_string(index=False))

    return dyn_df
