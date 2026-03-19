"""
src/data.py
───────────
Data loading, validation, and train/test preparation.

Public API
----------
    load_data(cfg)         → (df_train, df_test)
    validate_data(df, cfg) → raises AssertionError on problems
    prepare_matrices(df_train, df_test, feature_cols, cfg)
                           → (X_train, T_train, Y_train, X_test, T_test, Y_test)
    describe_split(df_train, df_test, cfg) → prints a quick summary
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config

# ── Loading ────────────────────────────────────────────────────────────────────

def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test CSVs as configured. Returns (df_train, df_test)."""
    train_path = Path(cfg.data.train)
    test_path  = Path(cfg.data.test)

    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Data file not found: '{p.resolve()}'\n"
                "Place your CSVs in the data/ directory or update config.yaml."
            )

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    validate_data(df_train, cfg, label="train")
    validate_data(df_test,  cfg, label="test")

    return df_train, df_test


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, cfg: Config, label: str = "") -> None:
    """
    Run basic sanity checks on a dataset. Raises AssertionError with a
    descriptive message if any check fails.
    """
    tag = f"[{label}] " if label else ""
    T   = cfg.columns.treatment
    Y   = cfg.columns.outcome

    assert T in df.columns, f"{tag}Missing treatment column '{T}'"
    assert Y in df.columns, f"{tag}Missing outcome column '{Y}'"

    assert df[T].isin([0, 1]).all(), \
        f"{tag}Treatment column '{T}' must be binary (0/1); found: {df[T].unique()}"
    assert df[Y].isin([0, 1]).all(), \
        f"{tag}Outcome column '{Y}' must be binary (0/1); found: {df[Y].unique()}"

    assert df[T].nunique() == 2, \
        f"{tag}Treatment column has only one value — no variation to estimate effects from."

    assert len(df) > 100, \
        f"{tag}Dataset has only {len(df)} rows — too small for reliable estimates."

    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5]
    if len(high_missing):
        import warnings
        warnings.warn(
            f"{tag}The following columns have >50% missing values: "
            f"{high_missing.index.tolist()}"
        )


# ── Matrix preparation ─────────────────────────────────────────────────────────

def prepare_matrices(
    df_train:     pd.DataFrame,
    df_test:      pd.DataFrame,
    feature_cols: list[str],
    cfg:          Config,
) -> tuple[pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract (X, T, Y) matrices from train and test sets.
    Fills NaN in features with 0.

    Returns
    -------
    X_train, T_train, Y_train, X_test, T_test, Y_test
    """
    T = cfg.columns.treatment
    Y = cfg.columns.outcome

    X_train = df_train[feature_cols].fillna(0)
    T_train = df_train[T]
    Y_train = df_train[Y]

    X_test  = df_test[feature_cols].fillna(0)
    T_test  = df_test[T]
    Y_test  = df_test[Y]

    return X_train, T_train, Y_train, X_test, T_test, Y_test


# ── Descriptive summary ────────────────────────────────────────────────────────

def describe_split(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    cfg:      Config,
) -> None:
    """Print a concise summary of the train/test split and treatment balance."""
    T = cfg.columns.treatment
    Y = cfg.columns.outcome

    print("─" * 55)
    print("  DATA SUMMARY")
    print("─" * 55)
    print(f"  Train rows : {len(df_train):,}  |  Test rows : {len(df_test):,}")

    for label, df in [("Train", df_train), ("Test", df_test)]:
        t_share    = df[T].mean()
        ret_ctrl   = df.loc[df[T] == 0, Y].mean()
        ret_treat  = df.loc[df[T] == 1, Y].mean()
        naive_ate  = ret_treat - ret_ctrl
        print(f"\n  [{label}]")
        print(f"    Treatment share   : {t_share:.3f}")
        print(f"    Return rate ctrl  : {ret_ctrl:.4f}")
        print(f"    Return rate treat : {ret_treat:.4f}")
        print(f"    Naive diff-means  : {naive_ate:+.4f}")

    print(f"{'─'*55}")
