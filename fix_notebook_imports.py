import json
from pathlib import Path

path = Path("notebook.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", [])
    if any("import numpy as np" in line for line in src):
        cell["source"] = [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "from ate import run_ate_suite\n",
            "from causal_forest import compute_rate, fit_causal_forest\n",
            "from features import check_covariate_balance, engineer_features, get_feature_cols\n",
            "from plots import (\n",
            "    plot_ate_forest,\n",
            "    plot_bandwidth_sensitivity,\n",
            "    plot_cate_distribution,\n",
            "    plot_cate_segments,\n",
            "    plot_correlation_heatmap,\n",
            "    plot_covariate_balance,\n",
            "    plot_descriptive_overview,\n",
            "    plot_feature_importance,\n",
            "    plot_permutation_null,\n",
            "    plot_policy_curve,\n",
            "    plot_toc_curve,\n",
            ")\n",
            "\n",
            "from src.config import load_config\n",
            "from src.data import describe_split, load_data, prepare_matrices\n",
            "from src.extensions import (\n",
            "    cate_dgp_simulation,\n",
            "    dynamic_targeting_simulation,\n",
            "    domain_translation_table,\n",
            "    multi_arm_simulation,\n",
            "    profit_lever_analysis,\n",
            ")\n",
            "from src.policy import compute_cate_segments, compute_policy, compute_qini\n",
            "from src.reporting import build_summary_table, print_summary\n",
            "from src.robustness import run_all_robustness_checks\n",
            "\n",
            "cfg = load_config('config.yaml')\n",
            "np.random.seed(cfg.random_seed)\n",
        ]
        break

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook import cell updated.")
