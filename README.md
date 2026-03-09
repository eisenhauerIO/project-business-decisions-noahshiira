# Replication: "Smart Green Nudging" — von Zahn et al. (2024)

> A replication and extension of **"Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning"**  
> *Marketing Science* (2024) · DOI: [10.1287/mksc.2022.0393](https://doi.org/10.1287/mksc.2022.0393)

---

## Overview

This repository replicates the core methodology of von Zahn et al. (2024) using **simulated data** generated to match the paper's described data-generating process. The original retailer dataset is proprietary and not publicly available.

The paper asks: *do personalised green nudges — shown selectively to the customers most likely to respond — reduce e-commerce product returns, and are they more profitable than nudging everyone?*

We replicate:
- Average Treatment Effect (ATE) estimation via OLS, IPW, and AIPW (doubly-robust)
- Robustness checks: permutation/placebo test and bandwidth sensitivity sweep
- Heterogeneous Treatment Effects (CATE) via Causal Forest (Wager & Athey, 2018)
- Smart targeting policy curve and profit analysis

We also extend the analysis with ad hoc sections on **retail profit levers**, **transferring the method to other domains**, and **multi-arm / dynamic targeting simulations**.

---

## Repository Structure

```
.
├── data/
│   ├── train.csv               # Simulated training data
│   └── test.csv                # Simulated test data
├── src/
│   ├── features.py             # Feature engineering & covariate balance checks
│   ├── ate.py                  # ATE estimators: OLS, IPW, AIPW, permutation test, bandwidth sensitivity
│   ├── causal_forest.py        # Causal Forest fitting, CATE inference, BLP test, RATE
│   ├── plots.py                # All visualisations (one function per chart)
├── replication_smart_green_nudging_v2.ipynb   # Main notebook
├── environment.yml             # Conda environment
└── README.md
```

---

## `src/` Module Reference

All heavy logic is kept out of the notebook. Each cell in the notebook is a one-liner import + call — the `src/` modules contain the actual implementations.

| Module | What it does |
|---|---|
| `features.py` | `engineer_features()` builds digital footprint features (return rate quintiles, bracketing rate, log order value, device flags). `check_covariate_balance()` computes standardised mean differences (SMD) to verify randomisation quality. |
| `ate.py` | `run_ate_suite()` runs four ATE estimators in one call and returns a structured `ATEResults` object with `.summary()`. Also provides `permutation_test()` (1,000-shuffle empirical p-value) and `bandwidth_sensitivity()` (ATE stability across sample subsets). |
| `causal_forest.py` | `fit_causal_forest()` wraps `CausalForestDML` with full diagnostics: point estimates, 95% CIs, a Best Linear Predictor (BLP) calibration test for significant heterogeneity, and a per-customer confidence score. `compute_rate()` computes the RATE (Rank Average Treatment Effect) and TOC curve. |
| `plots.py` | One self-contained function per chart. Pass in data, get back a `matplotlib` Figure. Covers: descriptive overview, correlation heatmap, Love plot, ATE forest plot, permutation null, bandwidth sensitivity, CATE distribution, feature importance, policy curve, CATE segments, TOC curve. |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate replication-smart-green-nudging
```

### 3. Add your data

Place your simulation CSVs in the `data/` folder:

```
data/
├── train.csv
└── test.csv
```

The notebook expects at minimum a `treatment` column (binary 0/1) and a `returned` column (binary 0/1). Feature columns are detected automatically. See `src/features.py` for the expected column names used in feature engineering.

### 4. Run the notebook

```bash
jupyter lab replication_smart_green_nudging_v2.ipynb
```

Run all cells top to bottom. The Causal Forest step (Section 5) takes approximately 1–2 minutes.

---

## Reproducing Results

The random seed is fixed at `RANDOM_SEED = 42` throughout. All stochastic steps (bootstrap SE, permutation test, causal forest subsampling) use this seed, so results are fully reproducible given the same input data.

To run the notebook non-interactively (e.g. for CI or grading):

```bash
jupyter nbconvert --to notebook --execute replication_smart_green_nudging_v2.ipynb
```

---

## Key Results (on simulated data)

| Metric | Value |
|---|---|
| ATE — OLS + controls | see notebook output |
| ATE — AIPW (doubly-robust) | see notebook output |
| Permutation p-value | see notebook output |
| % customers with CATE < 0 | see notebook output |
| Value of personalisation over universal nudging | see notebook output |

> Results will vary with your simulation parameters. The table above populates when the notebook is executed.

---

## Limitations & Caveats

**Simulated data.** The original retailer data is proprietary. Our simulated dataset is generated to broadly match the paper's described distributions, but cannot perfectly reproduce the original findings. All numerical results should be interpreted as illustrative rather than confirmatory.

**Identifying assumption.** The causal interpretation of all estimates rests on the treatment being as-good-as randomly assigned within the experiment. Covariate balance checks (Section 2, Love plot) verify this for the simulated data; with real observational data, unconfoundedness would need to be argued more carefully.

**SUTVA.** The analysis assumes no spillover effects between customers — i.e., showing a nudge to one customer does not affect another's return behaviour. In practice, social influence and platform-level effects could violate this assumption.

**BLP calibration test.** The Best Linear Predictor test for CATE heterogeneity is implemented via a simplified proxy (Y − Ȳ regressed on T residuals). A fully rigorous implementation would use the cross-fitted nuisance residuals from the causal forest's internal cross-fitting procedure.

**Causal forest tuning.** Hyperparameters (`n_estimators=2000`, `min_samples_leaf=10`) were chosen to match the paper's GRF specification but were not re-tuned for the simulated data. Results may improve with cross-validated hyperparameter selection.

---

## Citation

If you reference this replication, please also cite the original paper:

```bibtex
@article{vonzahn2024smartgreen,
  title   = {Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning},
  author  = {von Zahn, Moritz and Feuerriegel, Stefan and Kuehl, Niklas},
  journal = {Marketing Science},
  year    = {2024},
  doi     = {10.1287/mksc.2022.0393}
}
```

---

## Dependencies

See `environment.yml` for the full specification. Key packages:

| Package | Role |
|---|---|
| `econml>=0.15` | Causal Forest (CausalForestDML) |
| `scikit-learn>=1.3` | Nuisance models (GBM, Logistic Regression) |
| `statsmodels>=0.14` | OLS, HC3 robust standard errors |
| `pandas>=2.0` | Data manipulation |
| `numpy>=2.0` | Numerical operations |
| `matplotlib>=3.9` · `seaborn>=0.13` | Visualisation |

---

## References

- von Zahn, M., Feuerriegel, S., & Kuehl, N. (2024). Smart Green Nudging. *Marketing Science*. https://doi.org/10.1287/mksc.2022.0393
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113(523), 1228–1242.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Athey, S., & Wager, S. (2021). Policy learning with observational data. *Econometrica*, 89(1), 133–161.
