# Smart Green Nudging — Replication Project

Replication of **von Zahn et al. (2024), "Smart Green Nudging: Reducing Product Returns
Through Digital Footprints and Causal Machine Learning"**, *Marketing Science*.  
https://doi.org/10.1287/mksc.2022.0393

---

## Project Structure

```
project/
├── notebook.ipynb          ← Main replication notebook (clean, narrative-first)
├── config.yaml             ← ALL parameters — edit here, nowhere else
├── run_all.py              ← Headless pipeline (no notebook needed)
│
├── src/                    ← Refactored Python modules
│   ├── config.py           ← Loads config.yaml → typed Config dataclass
│   ├── data.py             ← load_data(), validate_data(), prepare_matrices()
│   ├── robustness.py       ← Permutation, bandwidth, placebo, SUTVA checks
│   ├── policy.py           ← compute_policy(), compute_qini(), cate_segments()
│   ├── extensions.py       ← Section 8 simulations (profit levers, multi-arm, etc.)
│   └── reporting.py        ← build_summary_table(), print_summary()
│
├── src/ (existing, unchanged)
│   ├── features.py         ← engineer_features(), get_feature_cols()
│   ├── ate.py              ← run_ate_suite(), permutation_test(), bandwidth_sensitivity()
│   ├── causal_forest.py    ← fit_causal_forest(), compute_rate()
│   └── plots.py            ← All plotting functions
│
├── data/
│   ├── train.csv           ← Synthetic training data
│   └── test.csv            ← Synthetic test data
│
└── tests/
    └── test_ate.py         ← Unit tests (pytest)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install econml pandas numpy matplotlib seaborn scikit-learn statsmodels pyyaml pytest
```

### 2. Open the notebook
```bash
jupyter notebook notebook.ipynb
```

### 3. Run the full pipeline headlessly
```bash
python run_all.py
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## Changing Parameters

**All magic numbers live in `config.yaml`.** You never need to edit a notebook cell
or a Python file to change an assumption.

| What you want to change | Where in config.yaml |
|-------------------------|----------------------|
| Return cost per shipment | `policy.return_cost` |
| Nudge cost per impression | `policy.nudge_cost` |
| Number of causal forest trees | `causal_forest.n_estimators` |
| Number of permutation test draws | `robustness.n_permutations` |
| Cost sweep grid for Section 8.1 | `extensions.profit_lever` |
| DGP scenarios for Section 8.2 | `extensions.cate_dgp_simulation` |
| Multi-arm simulation size | `extensions.multi_arm` |
| Dynamic targeting sample sizes | `extensions.dynamic_targeting` |

---

## Key Findings (Synthetic Replication)

| Metric | Replication | Paper |
|--------|-------------|-------|
| Naïve nudge ATE | ~−0.026 (p < 0.05) | −2.6% (p < 0.03) |
| Permutation p-value | < 0.05 | < 0.05 |
| % customers with CATE < 0 | ~50–95% | ~95% |
| Optimal targeting share | ~90–95% | ~95% |
| Personalisation gain | ~2× | ~2× |

---

## Citation

```
von Zahn, M., Bauer, K., Mihale-Wilson, C., Jagow, J., Speicher, M., & Hinz, O. (2024).
Smart Green Nudging: Reducing Product Returns Through Digital Footprints and
Causal Machine Learning. Marketing Science.
https://doi.org/10.1287/mksc.2022.0393
```
