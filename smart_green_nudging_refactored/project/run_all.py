"""
run_all.py
──────────
End-to-end replication pipeline — runs the full analysis without the notebook.
Useful for reproducibility checks, CI, and batch re-runs with different configs.

Usage
-----
    python run_all.py                        # uses default config.yaml
    python run_all.py --config my_config.yaml
    python run_all.py --no-extensions        # skip Section 8 extensions
"""

from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless runs

# ── Allow imports from src/ and the existing ate/features/causal_forest modules ─
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from src.config    import load_config
from src.data      import load_data, describe_split, prepare_matrices
from src.robustness import run_all_robustness_checks
from src.policy    import compute_policy, compute_qini, compute_cate_segments
from src.reporting import build_summary_table, print_summary
from src.extensions import (
    profit_lever_analysis,
    cate_dgp_simulation,
    domain_translation_table,
    multi_arm_simulation,
    dynamic_targeting_simulation,
)

from features      import engineer_features, get_feature_cols, check_covariate_balance
from ate           import run_ate_suite
from causal_forest import fit_causal_forest, compute_rate


def main(config_path: str = "config.yaml", run_extensions: bool = True) -> None:
    print("\n" + "═" * 65)
    print("  Smart Green Nudging — Full Replication Pipeline")
    print("═" * 65 + "\n")

    # ── 0. Config ──────────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    np.random.seed(cfg.random_seed)
    T = cfg.columns.treatment
    Y = cfg.columns.outcome
    print(f"Config loaded from '{config_path}'  (seed={cfg.random_seed})\n")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("── 1. Loading data ──")
    df_train, df_test = load_data(cfg)
    describe_split(df_train, df_test, cfg)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n── 2. Feature engineering ──")
    df_train = engineer_features(df_train)
    df_test  = engineer_features(df_test)
    feature_cols = get_feature_cols(df_train, exclude=[T, Y])
    print(f"   {len(feature_cols)} features: {feature_cols}")

    balance = check_covariate_balance(df_train, T, feature_cols)
    max_smd = balance["smd"].abs().max() if "smd" in balance.columns else float("nan")
    print(f"   Max SMD (covariate balance): {max_smd:.4f}  {'✓ OK' if max_smd < 0.1 else '⚠ Check balance'}")

    X_train, T_train, Y_train, X_test, T_test, Y_test = prepare_matrices(
        df_train, df_test, feature_cols, cfg
    )

    # ── 3. ATE estimation ──────────────────────────────────────────────────────
    print("\n── 3. ATE estimation ──")
    ate_results = run_ate_suite(Y_train, T_train, X_train,
                                n_boot=cfg.ate.n_boot, seed=cfg.random_seed)
    print(ate_results.summary())
    ate_ref = ate_results.ate[1]   # OLS + controls

    # ── 4. Robustness ──────────────────────────────────────────────────────────
    print("\n── 4. Robustness checks ──")
    robustness = run_all_robustness_checks(
        df_train, Y_train, T_train, X_train, ate_ref, feature_cols, cfg
    )
    robustness.print_summary()

    # ── 5. Causal forest ───────────────────────────────────────────────────────
    print("\n── 5. Causal forest (CATE) ──")
    cf = fit_causal_forest(
        Y_train, T_train, X_train, X_test,
        feature_names=feature_cols,
        n_estimators=cfg.causal_forest.n_estimators,
        seed=cfg.random_seed,
    )
    print(cf.summary())

    df_test = df_test.copy()
    df_test["cate"]    = cf.cate
    df_test["cate_lb"] = cf.cate_lb
    df_test["cate_ub"] = cf.cate_ub
    pct_benefit = (cf.cate < 0).mean()
    print(f"   {pct_benefit:.1%} of customers have CATE < 0 (benefit from nudge)")

    rate_res = compute_rate(cf.cate, Y_test.values, T_test.values)
    print(f"   RATE = {rate_res['rate']:+.4f}")

    # ── 6. Policy ──────────────────────────────────────────────────────────────
    print("\n── 6. Targeting policy ──")
    policy = compute_policy(df_test, ate_ref, cfg)
    policy.print_summary()

    qini = compute_qini(df_test, Y_test, T_test)
    print(f"   AUQC = {qini['auqc']:.4f}")

    seg = compute_cate_segments(df_test, cfg)
    print("\n   CATE segments:")
    print(seg.to_string())

    # ── 7. Summary ─────────────────────────────────────────────────────────────
    print("\n── 7. Replication summary ──")
    summary = build_summary_table(
        ate_results, robustness, cf, rate_res, policy, qini, pct_benefit
    )
    print_summary(summary)

    # ── 8. Extensions ──────────────────────────────────────────────────────────
    if run_extensions:
        print("\n── 8. Extensions ──\n")

        print("  8.1 Profit lever sensitivity …")
        profit_lever_analysis(policy.df_pol, policy.fracs, ate_ref, cfg)

        print("\n  8.2 CATE DGP simulation …")
        cate_dgp_simulation(cfg)

        print("\n  8.3 Cross-domain translation table:")
        dt = domain_translation_table()
        print(dt.to_string(index=False))

        print("\n  8.4 Multi-arm nudge simulation …")
        multi_arm_simulation(cfg)

        print("\n  8.5 Dynamic targeting simulation …")
        dynamic_targeting_simulation(cfg)

    print("\n" + "═" * 65)
    print("  Pipeline complete.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Green Nudging replication pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--no-extensions", dest="extensions", action="store_false",
                        help="Skip Section 8 extension analyses")
    args = parser.parse_args()
    main(config_path=args.config, run_extensions=args.extensions)
