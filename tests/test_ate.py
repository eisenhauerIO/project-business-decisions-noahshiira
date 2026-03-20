"""
tests/test_ate.py
─────────────────
Unit tests for ATE estimation, policy computation, and config loading.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.policy import compute_cate_segments, compute_policy, compute_qini
from src.robustness import run_placebo_test, run_sutva_sensitivity

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    """Return a default Config (no YAML needed for unit tests)."""
    return Config()


@pytest.fixture
def small_rct(cfg):
    """Tiny synthetic RCT dataset (n=500)."""
    rng = np.random.default_rng(42)
    n = 500
    T = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(0, 1, n),
        }
    )
    # True ATE = -0.05
    Y = pd.Series((rng.random(n) < (0.35 + T * (-0.05))).astype(float))
    T = pd.Series(T.astype(float))
    return X, T, Y


@pytest.fixture
def df_test_with_cate(cfg):
    """Small df_test with synthetic CATE, T, Y columns."""
    rng = np.random.default_rng(42)
    n = 200
    cate = rng.normal(-0.05, 0.07, n)
    T = rng.integers(0, 2, n).astype(float)
    Y = (rng.random(n) < 0.35).astype(float)
    return pd.DataFrame(
        {
            "cate": cate,
            cfg.columns.treatment: T,
            cfg.columns.outcome: Y,
        }
    )


# ── Config tests ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config_builds(self, cfg):
        assert cfg.random_seed == 42
        assert cfg.policy.return_cost == 15.0
        assert cfg.policy.nudge_cost == 0.10

    def test_config_fields_accessible(self, cfg):
        assert hasattr(cfg, "causal_forest")
        assert hasattr(cfg, "extensions")
        assert hasattr(cfg.extensions, "multi_arm")

    def test_missing_yaml_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


# ── Policy tests ───────────────────────────────────────────────────────────────


class TestPolicy:
    def test_compute_policy_returns_correct_shape(self, df_test_with_cate, cfg):
        policy = compute_policy(df_test_with_cate, ate_reference=-0.026, cfg=cfg)
        assert len(policy.fracs) == cfg.policy.n_fracs
        assert len(policy.profit_smart) == cfg.policy.n_fracs
        assert len(policy.profit_univ) == cfg.policy.n_fracs

    def test_profit_at_zero_frac_is_zero(self, df_test_with_cate, cfg):
        policy = compute_policy(df_test_with_cate, ate_reference=-0.026, cfg=cfg)
        assert policy.profit_smart[0] == pytest.approx(0.0)
        assert policy.profit_univ[0] == pytest.approx(0.0)

    def test_best_frac_in_unit_interval(self, df_test_with_cate, cfg):
        policy = compute_policy(df_test_with_cate, ate_reference=-0.026, cfg=cfg)
        assert 0.0 <= policy.best_frac <= 1.0

    def test_personalisation_value_finite(self, df_test_with_cate, cfg):
        policy = compute_policy(df_test_with_cate, ate_reference=-0.026, cfg=cfg)
        assert np.isfinite(policy.personalisation_value)

    def test_df_pol_sorted_by_cate(self, df_test_with_cate, cfg):
        policy = compute_policy(df_test_with_cate, ate_reference=-0.026, cfg=cfg)
        cates = policy.df_pol["cate"].values
        assert np.all(cates[:-1] <= cates[1:]), (
            "df_pol should be sorted ascending by CATE"
        )


class TestQINI:
    def test_qini_returns_auqc(self, df_test_with_cate, cfg):
        T = df_test_with_cate[cfg.columns.treatment]
        Y = df_test_with_cate[cfg.columns.outcome]
        result = compute_qini(df_test_with_cate, Y, T)
        assert "auqc" in result
        assert "fracs" in result
        assert "qini_vals" in result
        assert np.isfinite(result["auqc"])

    def test_qini_fracs_length_matches_data(self, df_test_with_cate, cfg):
        T = df_test_with_cate[cfg.columns.treatment]
        Y = df_test_with_cate[cfg.columns.outcome]
        result = compute_qini(df_test_with_cate, Y, T)
        assert len(result["fracs"]) == len(df_test_with_cate)


class TestCATESegments:
    def test_segments_has_four_quartiles(self, df_test_with_cate, cfg):
        seg = compute_cate_segments(df_test_with_cate.copy(), cfg)
        assert len(seg) == 4

    def test_segment_return_rates_in_unit_interval(self, df_test_with_cate, cfg):
        seg = compute_cate_segments(df_test_with_cate.copy(), cfg)
        assert (seg["return_rate"] >= 0).all()
        assert (seg["return_rate"] <= 1).all()

    def test_segments_cover_all_rows(self, df_test_with_cate, cfg):
        df = df_test_with_cate.copy()
        seg = compute_cate_segments(df, cfg)
        assert seg["n"].sum() == len(df)


# ── Robustness tests ───────────────────────────────────────────────────────────


class TestRobustness:
    def test_placebo_pvalue_in_unit_interval(self, small_rct, cfg):
        X, T, Y = small_rct
        # Speed up: override n_permutations for the test
        cfg.robustness.n_permutations = 100
        result = run_placebo_test(Y, T, X, cfg)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_sutva_sensitivity_returns_correct_rows(self, small_rct, cfg):
        X, T, Y = small_rct
        df_sutva = run_sutva_sensitivity(Y, T, ate_reference=-0.026, cfg=cfg)
        assert len(df_sutva) == len(cfg.robustness.sutva_contamination_rates)
        assert "biased_ate" in df_sutva.columns
        assert "bias" in df_sutva.columns

    def test_zero_contamination_bias_is_near_zero(self, small_rct, cfg):
        X, T, Y = small_rct
        df_sutva = run_sutva_sensitivity(Y, T, ate_reference=-0.026, cfg=cfg)
        zero_row = df_sutva[df_sutva["contamination_rate"] == "0%"]
        assert len(zero_row) == 1
        assert abs(zero_row["bias"].iloc[0]) < 1e-9
