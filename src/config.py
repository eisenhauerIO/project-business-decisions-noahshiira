"""
src/config.py
─────────────
Loads config.yaml and exposes a single typed `cfg` object.

Usage
-----
    from src.config import load_config
    cfg = load_config()          # reads config.yaml from project root
    print(cfg.policy.return_cost)
    print(cfg.random_seed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# ── Sub-configs ────────────────────────────────────────────────────────────────


@dataclass
class ColumnsConfig:
    treatment: str = "treatment"
    outcome: str = "returned"


@dataclass
class DataConfig:
    train: str = "data/train.csv"
    test: str = "data/test.csv"


@dataclass
class ATEConfig:
    n_boot: int = 500


@dataclass
class RobustnessConfig:
    n_permutations: int = 1000
    placebo_seed_offset: int = 99
    sutva_contamination_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.30]
    )


@dataclass
class CausalForestConfig:
    n_estimators: int = 2000


@dataclass
class PolicyConfig:
    return_cost: float = 15.0
    nudge_cost: float = 0.10
    n_fracs: int = 101


@dataclass
class ProfitLeverConfig:
    return_costs: List[float] = field(default_factory=lambda: [5, 10, 15, 25, 40])
    nudge_costs: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.25, 0.50])
    breakeven_sweep_max: float = 50.0


@dataclass
class CateDGPConfig:
    n_sim: int = 2000
    scenarios: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class MultiArmConfig:
    n_sim: int = 3000
    arms: List[str] = field(
        default_factory=lambda: ["Environmental", "Social Norm", "Financial"]
    )
    arm_colors: List[str] = field(
        default_factory=lambda: ["#4CAF82", "#5B8DB8", "#E07B54"]
    )


@dataclass
class DynamicTargetingConfig:
    sample_sizes: List[int] = field(
        default_factory=lambda: [200, 500, 1000, 2000, 5000, 10000, 20000]
    )
    test_set_size: int = 5000
    true_cate_mean: float = -0.05
    true_cate_std: float = 0.07


@dataclass
class ExtensionsConfig:
    profit_lever: ProfitLeverConfig = field(default_factory=ProfitLeverConfig)
    cate_dgp_simulation: CateDGPConfig = field(default_factory=CateDGPConfig)
    multi_arm: MultiArmConfig = field(default_factory=MultiArmConfig)
    dynamic_targeting: DynamicTargetingConfig = field(
        default_factory=DynamicTargetingConfig
    )


@dataclass
class PlotsConfig:
    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "smart": "#4CAF82",
            "universal": "#5B8DB8",
            "accent": "#E07B54",
            "purple": "#9B59B6",
        }
    )
    figsize_wide: List[float] = field(default_factory=lambda: [9.0, 4.0])
    figsize_square: List[float] = field(default_factory=lambda: [8.0, 5.0])


# ── Root config ────────────────────────────────────────────────────────────────


@dataclass
class Config:
    random_seed: int = 42
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ate: ATEConfig = field(default_factory=ATEConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    causal_forest: CausalForestConfig = field(default_factory=CausalForestConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────


def load_config(path: str | Path = "config.yaml") -> Config:
    """
    Load config.yaml from *path* and return a fully typed Config object.
    Falls back to dataclass defaults for any key not present in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at '{path.resolve()}'. "
            "Make sure you're running from the project root."
        )

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        random_seed=raw.get("random_seed", 42),
        columns=ColumnsConfig(**raw.get("columns", {})),
        data=DataConfig(**raw.get("data", {})),
        ate=ATEConfig(**raw.get("ate", {})),
        robustness=RobustnessConfig(**raw.get("robustness", {})),
        causal_forest=CausalForestConfig(**raw.get("causal_forest", {})),
        policy=PolicyConfig(**raw.get("policy", {})),
        extensions=_parse_extensions(raw.get("extensions", {})),
        plots=_parse_plots(raw.get("plots", {})),
    )
    return cfg


def _parse_extensions(raw: dict) -> ExtensionsConfig:
    pl = raw.get("profit_lever", {})
    dgp = raw.get("cate_dgp_simulation", {})
    ma = raw.get("multi_arm", {})
    dyn = raw.get("dynamic_targeting", {})
    return ExtensionsConfig(
        profit_lever=ProfitLeverConfig(**pl) if pl else ProfitLeverConfig(),
        cate_dgp_simulation=CateDGPConfig(**dgp) if dgp else CateDGPConfig(),
        multi_arm=MultiArmConfig(**ma) if ma else MultiArmConfig(),
        dynamic_targeting=DynamicTargetingConfig(**dyn)
        if dyn
        else DynamicTargetingConfig(),
    )


def _parse_plots(raw: dict) -> PlotsConfig:
    return PlotsConfig(
        colors=raw.get("colors", PlotsConfig().colors),
        figsize_wide=raw.get("figsize_wide", [9.0, 4.0]),
        figsize_square=raw.get("figsize_square", [8.0, 5.0]),
    )
