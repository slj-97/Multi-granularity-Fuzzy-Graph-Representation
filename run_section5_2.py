"""Section 5.2 experiments: Real-world Application — Multi-granularity Disease Subtyping.

Sections covered:
  5.2.3  Comparison with classical clustering baselines
  5.2.4  Single-granularity ablation
  5.2.5  Interpretability on Dermatology
  5.2.6  Robustness to feature perturbation

Usage:  python run_section5_2.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import MinMaxScaler

from mgfgr import (
    evaluate_representation,
    gaussian_similarity,
    granule_count,
    max_min_closure,
    mgfgr_fusion,
    run_kmeans,
    run_mgfgr_pipeline,
    select_lambdas,
    select_single_layers,
    single_granularity_vector,
)

# Silence only known-benign, environment-specific noise; keep genuine
# convergence / numerical / deprecation warnings visible in a final
# experiment script.
warnings.filterwarnings("ignore", message="Graph is not fully connected")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")

OUT = Path(__file__).resolve().parent
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# matplotlib style
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
    }
)

BLUE = "#3B6EA8"
ORANGE = "#D9892B"
GREEN = "#5A9E6F"
RED = "#C44E52"
GRAY = "#6E6E6E"
PURPLE = "#9372BD"
COLORS_5 = [BLUE, ORANGE, GREEN, RED, PURPLE]

N_RUNS = 50  # independent k-means runs for stochastic methods


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_dermatology():
    """Load Dermatology dataset from UCI repository.

    Returns X (366, 34), y (366,) 0-indexed, feature_names, class_names.
    """
    import io
    from urllib.request import Request, urlopen

    # Prefer a local copy for reproducibility / offline runs. Place the UCI
    # file (or any compatible CSV) at data/dermatology.data to skip the
    # network entirely; otherwise it is downloaded once and cached there.
    cache = OUT / "data" / "dermatology.data"
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
    else:
        url = (
            "https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/dermatology/dermatology.data"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as f:
            raw = f.read().decode("utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")

    # replace '?' with NaN
    raw = raw.replace("?", "NaN")
    data = np.genfromtxt(io.StringIO(raw), delimiter=",", dtype=np.float64)

    X = data[:, :-1]  # first 34 columns are features
    # median imputation for missing values
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        if nan_mask[:, j].any():
            X[nan_mask[:, j], j] = col_medians[j]

    y = data[:, -1].astype(int) - 1  # labels 1-6 -> 0-5

    feature_names = [f"f{i+1}" for i in range(34)]
    feature_names[33] = "age"

    class_names = [
        "psoriasis",
        "seboreic_dermatitis",
        "lichen_planus",
        "pityriasis_rosea",
        "cronic_dermatitis",
        "pityriasis_rubra_pilaris",
    ]
    return X, y, feature_names, class_names


def _cached_text(name, url):
    cache = OUT / "data" / name
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as f:
        raw = f.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw, encoding="utf-8")
    return raw


def _load_numeric_csv(name, url, label_col=0, missing="?"):
    import io

    raw = _cached_text(name, url).replace(missing, "NaN")
    data = np.genfromtxt(io.StringIO(raw), delimiter=",", dtype=np.float64)
    y_raw = data[:, label_col].astype(int)
    X = np.delete(data, label_col, axis=1)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        if nan_mask[:, j].any():
            X[nan_mask[:, j], j] = col_medians[j]
    _, y = np.unique(y_raw, return_inverse=True)
    return X, y.astype(int)


def load_lymphography():
    X, y = _load_numeric_csv(
        "lymphography.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data",
        label_col=0,
    )
    return X, y, [f"f{i+1}" for i in range(X.shape[1])], [f"class_{c}" for c in np.unique(y)]


def load_new_thyroid():
    X, y = _load_numeric_csv(
        "new-thyroid.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data",
        label_col=0,
    )
    return X, y, [f"f{i+1}" for i in range(X.shape[1])], [f"class_{c}" for c in np.unique(y)]


def load_cleveland_heart():
    X, y = _load_numeric_csv(
        "processed.cleveland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        label_col=-1,
    )
    return X, y, [f"f{i+1}" for i in range(X.shape[1])], [f"severity_{c}" for c in np.unique(y)]


def load_postoperative():
    import io

    raw = _cached_text(
        "post-operative.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data",
    ).replace("?", "NaN")
    df = pd.read_csv(io.StringIO(raw), header=None)
    y_raw = df.iloc[:, -1].astype(str)
    X_df = pd.get_dummies(df.iloc[:, :-1].astype(str), dummy_na=True)
    _, y = np.unique(y_raw, return_inverse=True)
    return X_df.to_numpy(float), y.astype(int), list(X_df.columns), list(np.unique(y_raw))


def load_primary_tumor():
    X, y = _load_numeric_csv(
        "primary-tumor.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data",
        label_col=0,
    )
    return X, y, [f"f{i+1}" for i in range(X.shape[1])], [f"class_{c}" for c in np.unique(y)]


# ---------------------------------------------------------------------------
# Fuzzy C-Means (minimal — avoids extra dependency)
# ---------------------------------------------------------------------------


def fuzzy_cmeans(X, n_clusters, m=2.0, max_iter=100, tol=1e-4, random_state=0):
    """Minimal fuzzy c-means. Returns hard cluster labels."""
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    # initialise memberships randomly
    U = rng.rand(n, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        # compute cluster centres
        um = U ** m
        centers = (um.T @ X) / um.T.sum(axis=1, keepdims=True)

        # update memberships
        dist = np.zeros((n, n_clusters))
        for k in range(n_clusters):
            dist[:, k] = np.sum((X - centers[k]) ** 2, axis=1)

        dist = np.maximum(dist, 1e-16)
        inv_dist = dist ** (-1.0 / (m - 1.0))
        U_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        if np.abs(U_new - U).max() < tol:
            break
        U = U_new

    return np.argmax(U, axis=1)


# ---------------------------------------------------------------------------
# Baseline clustering
# ---------------------------------------------------------------------------


def run_baseline_clustering(X, y_true, C, n_runs=N_RUNS):
    """Run all classical clustering baselines on raw features.

    Returns dict: method_name -> {"metrics_mean": {...}, "metrics_std": {...}}
    """
    results = {}
    n = X.shape[0]

    # -- k-means on raw features --
    mean_m, std_m = run_kmeans(X, n_clusters=C, y_true=y_true, n_runs=n_runs)
    results["k-means (raw)"] = {"metrics_mean": mean_m, "metrics_std": std_m}

    # -- spectral clustering --
    # Standard RBF spectral clustering with the median-distance bandwidth
    # heuristic: sklearn's rbf affinity is exp(-gamma * d^2), so gamma =
    # 1/(2*sigma^2) with sigma the median pairwise Euclidean distance.
    sigma = float(np.median(pdist(X, metric="euclidean")))
    if sigma < 1e-12:
        sigma = 1.0
    rbf_gamma = 1.0 / (2.0 * sigma ** 2)
    ari_list, nmi_list, fmi_list, acc_list = [], [], [], []
    for seed in range(n_runs):
        sc = SpectralClustering(
            n_clusters=C, affinity="rbf", gamma=rbf_gamma,
            random_state=seed, assign_labels="kmeans",
        )
        labels = sc.fit_predict(X)
        ari_list.append(adjusted_rand_score(y_true, labels))
        nmi_list.append(normalized_mutual_info_score(y_true, labels))
        fmi_list.append(fowlkes_mallows_score(y_true, labels))
        acc_list.append(cluster_accuracy(y_true, labels))
    results["spectral"] = {
        "metrics_mean": {
            "ARI": np.mean(ari_list), "NMI": np.mean(nmi_list),
            "FMI": np.mean(fmi_list), "ACC": np.mean(acc_list),
        },
        "metrics_std": {
            "ARI": np.std(ari_list, ddof=1), "NMI": np.std(nmi_list, ddof=1),
            "FMI": np.std(fmi_list, ddof=1), "ACC": np.std(acc_list, ddof=1),
        },
    }

    # -- hierarchical (deterministic, single value) --
    for method_name, method in [("Ward", "ward"), ("average", "average"), ("single", "single")]:
        Z = linkage(X, method=method)
        labels = fcluster(Z, t=C, criterion="maxclust") - 1
        m = {
            "ARI": adjusted_rand_score(y_true, labels),
            "NMI": normalized_mutual_info_score(y_true, labels),
            "FMI": fowlkes_mallows_score(y_true, labels),
            "ACC": cluster_accuracy(y_true, labels),
        }
        results[method_name] = {"metrics_mean": m, "metrics_std": {k: 0.0 for k in m}}

    # -- fuzzy c-means --
    ari_list, nmi_list, fmi_list, acc_list = [], [], [], []
    for seed in range(n_runs):
        labels = fuzzy_cmeans(X, n_clusters=C, random_state=seed)
        ari_list.append(adjusted_rand_score(y_true, labels))
        nmi_list.append(normalized_mutual_info_score(y_true, labels))
        fmi_list.append(fowlkes_mallows_score(y_true, labels))
        acc_list.append(cluster_accuracy(y_true, labels))
    results["FCM"] = {
        "metrics_mean": {
            "ARI": np.mean(ari_list), "NMI": np.mean(nmi_list),
            "FMI": np.mean(fmi_list), "ACC": np.mean(acc_list),
        },
        "metrics_std": {
            "ARI": np.std(ari_list, ddof=1), "NMI": np.std(nmi_list, ddof=1),
            "FMI": np.std(fmi_list, ddof=1), "ACC": np.std(acc_list, ddof=1),
        },
    }

    return results


def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


# ---------------------------------------------------------------------------
# 5.2.3  Comparison with classical clustering baselines
# ---------------------------------------------------------------------------


def section_5_2_3(datasets):
    """Generate LaTeX table comparing MG-FGR with baselines across datasets."""
    all_results = {}

    for ds_name, (X, y, _, _) in datasets.items():
        print(f"  [5.2.3] {ds_name}: running MG-FGR pipeline ...")
        pipe = run_mgfgr_pipeline(X, y)

        # MG-FGR evaluation
        mgfgr_mean, mgfgr_std = evaluate_representation(pipe["V"], y, pipe["C"])
        print(f"    MG-FGR ARI={mgfgr_mean['ARI']:.4f} +/- {mgfgr_std['ARI']:.4f}")

        # baselines
        print(f"  [5.2.3] {ds_name}: running baselines ...")
        baseline_results = run_baseline_clustering(X, y, pipe["C"])
        baseline_results["MG-FGR"] = {"metrics_mean": mgfgr_mean, "metrics_std": mgfgr_std}

        all_results[ds_name] = baseline_results

    # Write LaTeX table
    _write_comparison_table(all_results)
    return all_results


# Display names so the generated LaTeX matches the manuscript wording.
METHOD_DISPLAY = {
    "k-means (raw)": r"$k$-means",
    "spectral": "Spectral clustering",
    "Ward": "Ward linkage",
    "average": "Average linkage",
    "single": "Single linkage",
    "FCM": r"Fuzzy $c$-means",
    "MG-FGR": r"MG-FGR (ours)",
}


def _fmt_cell(mean, std, best, second):
    """Format a 'mean ± std' cell, bolding the best and underlining 2nd best."""
    cell = f"{mean:.4f} $\\pm$ {std:.4f}"
    if mean == best:
        return r"\textbf{" + cell + r"}"
    if second is not None and mean == second:
        return r"\underline{" + cell + r"}"
    return cell


def _write_comparison_table(all_results):
    """Write a drop-in LaTeX table for 5.2.3 to tables/comparison_baselines.tex.

    Layout matches the manuscript: metrics as columns, one block per dataset,
    methods as rows. Best per (dataset, metric) column is bold, 2nd underlined.
    Label is tab:cluster-comp so the manuscript can \\input it directly.
    """
    metrics = ["ARI", "NMI", "FMI", "ACC"]
    datasets_names = list(all_results.keys())
    methods = list(all_results[datasets_names[0]].keys())

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\footnotesize")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Clustering performance of MG-FGR alongside classical "
                 r"baselines (mean $\pm$ std over "
                 + str(N_RUNS) + r" runs). The table evaluates whether the fused "
                 r"representation preserves subtype information; the layer-wise "
                 r"traceability of MG-FGR is analysed in Section~\ref{sec:interpretability}. "
                 r"Best per column in \textbf{bold}, second best \underline{underlined}.}")
    lines.append(r"  \label{tab:cluster-comp}")
    lines.append(r"  \begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append(r"    \toprule")
    header = " & ".join([r"\textbf{Method}"] + [rf"\textbf{{{m}}}" for m in metrics])
    lines.append("    " + header + r" \\")

    for ds in datasets_names:
        lines.append(r"    \midrule")
        lines.append(r"    \multicolumn{" + str(len(metrics) + 1)
                     + r"}{c}{\textit{" + ds + r"}} \\")
        lines.append(r"    \midrule")

        # per-metric best / second-best within this dataset (higher is better)
        best, second = {}, {}
        for metric in metrics:
            vals = sorted(
                (all_results[ds][m]["metrics_mean"][metric] for m in methods),
                reverse=True,
            )
            best[metric] = vals[0]
            second[metric] = next((v for v in vals if v < vals[0]), None)

        for method in methods:
            row = "    " + METHOD_DISPLAY.get(method, method)
            for metric in metrics:
                mean = all_results[ds][method]["metrics_mean"][metric]
                std = all_results[ds][method]["metrics_std"][metric]
                row += " & " + _fmt_cell(mean, std, best[metric], second[metric])
            row += r" \\"
            lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    (TAB_DIR / "comparison_baselines.tex").write_text(tex, encoding="utf-8")
    print(f"  -> saved {TAB_DIR / 'comparison_baselines.tex'}")


# ---------------------------------------------------------------------------
# 5.2.4  Single-granularity ablation (Dermatology only)
# ---------------------------------------------------------------------------


def section_5_2_4(pipe, y_true, C):
    """Single-granularity ablation on Dermatology.

    Compares: Single-coarse, Single-medium, Single-fine, Equal-weight, MG-FGR.
    """
    E_tilde = pipe["E_tilde"]
    lambdas_full = pipe["lambdas"]
    single = pipe["single_layers"]
    n = E_tilde.shape[0]

    variants = {}

    # -- single-coarse --
    v_coarse = single_granularity_vector(E_tilde, single["coarse"])
    mean_m, std_m = run_kmeans(v_coarse, n_clusters=C, y_true=y_true)
    variants["Single-coarse"] = {"mean": mean_m, "std": std_m}

    # -- single-medium --
    v_medium = single_granularity_vector(E_tilde, single["medium"])
    mean_m, std_m = run_kmeans(v_medium, n_clusters=C, y_true=y_true)
    variants["Single-medium"] = {"mean": mean_m, "std": std_m}

    # -- single-fine --
    v_fine = single_granularity_vector(E_tilde, single["fine"])
    mean_m, std_m = run_kmeans(v_fine, n_clusters=C, y_true=y_true)
    variants["Single-fine"] = {"mean": mean_m, "std": std_m}

    # -- equal-weight fusion --
    V_equal = np.zeros((n, n), dtype=np.float64)
    for lam in lambdas_full:
        v_lam = single_granularity_vector(E_tilde, lam)
        V_equal += v_lam
    mean_m, std_m = run_kmeans(V_equal, n_clusters=C, y_true=y_true)
    variants["Equal-weight"] = {"mean": mean_m, "std": std_m}

    # -- MG-FGR (weighted) --
    mean_m, std_m = run_kmeans(pipe["V"], n_clusters=C, y_true=y_true)
    variants["MG-FGR"] = {"mean": mean_m, "std": std_m}

    # Print
    print("\n  [5.2.4] Single-granularity ablation:")
    for name, v in variants.items():
        print(f"    {name:20s}  ARI={v['mean']['ARI']:.4f} +/- {v['std']['ARI']:.4f}  "
              f"NMI={v['mean']['NMI']:.4f} +/- {v['std']['NMI']:.4f}")

    # LaTeX table
    _write_ablation_table(variants)

    # Bar chart
    _plot_ablation_barchart(variants)

    return variants


def _write_ablation_table(variants):
    metrics = ["ARI", "NMI", "FMI", "ACC"]
    order = ["Single-coarse", "Single-medium", "Single-fine", "Equal-weight", "MG-FGR"]
    display = {
        "Single-coarse": "Single-coarse",
        "Single-medium": "Single-medium",
        "Single-fine": "Single-fine",
        "Equal-weight": "Equal-weight fusion",
        "MG-FGR": r"\textbf{MG-FGR (ours)}",
    }

    # per-metric best / second-best across variants (higher is better)
    best, second = {}, {}
    for metric in metrics:
        vals = sorted((variants[n]["mean"][metric] for n in order), reverse=True)
        best[metric] = vals[0]
        second[metric] = next((v for v in vals if v < vals[0]), None)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\footnotesize")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Single-granularity ablation on Dermatology (mean $\pm$ std over "
                 + str(N_RUNS) + r" runs). Best per column in \textbf{bold}, second best \underline{underlined}.}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Variant} & \textbf{ARI} & \textbf{NMI} & \textbf{FMI} & \textbf{ACC} \\")
    lines.append(r"    \midrule")
    for name in order:
        v = variants[name]
        row = "    " + display[name]
        for metric in metrics:
            row += " & " + _fmt_cell(v["mean"][metric], v["std"][metric],
                                     best[metric], second[metric])
        row += r" \\"
        lines.append(row)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    (TAB_DIR / "ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> saved {TAB_DIR / 'ablation.tex'}")


def _plot_ablation_barchart(variants):
    names = ["Single-coarse", "Single-medium", "Single-fine", "Equal-weight", "MG-FGR"]
    ari_mean = [variants[n]["mean"]["ARI"] for n in names]
    ari_std = [variants[n]["std"]["ARI"] for n in names]
    nmi_mean = [variants[n]["mean"]["NMI"] for n in names]
    nmi_std = [variants[n]["std"]["NMI"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars1 = ax.bar(x - w/2, ari_mean, w, yerr=ari_std, color=BLUE, capsize=3, label="ARI")
    bars2 = ax.bar(x + w/2, nmi_mean, w, yerr=nmi_std, color=ORANGE, capsize=3, label="NMI")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Single-granularity ablation (Dermatology)", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", color="#E2E2E2", lw=0.5)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_ablation{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_ablation.pdf'}")


# ---------------------------------------------------------------------------
# 5.2.5  Interpretability on Dermatology
# ---------------------------------------------------------------------------


def section_5_2_5(pipe, y_true, class_names):
    """Four interpretability sub-figures for Dermatology."""
    E_tilde = pipe["E_tilde"]
    lambdas = pipe["lambdas"]
    V = pipe["V"]
    layer_info = pipe["layer_info"]

    _fig_layerwise_ari(E_tilde, lambdas, y_true)
    _fig_heatmaps(E_tilde, pipe["single_layers"], y_true)
    _fig_merging_paths(E_tilde, lambdas, V, y_true, class_names)
    _fig_V_heatmap(V, y_true, class_names)


def _fig_layerwise_ari(E_tilde, lambdas, y_true):
    """(a) Layer-wise ARI vs lambda."""
    ari_vals = []
    n_gr_vals = []
    for lam in lambdas:
        v = single_granularity_vector(E_tilde, lam)
        _, labels = np.unique(v, axis=0, return_inverse=True)
        labels = labels.ravel()
        ari_vals.append(adjusted_rand_score(y_true, labels))
        n_gr_vals.append(len(np.unique(labels)))

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10

    fig, ax1 = plt.subplots(figsize=(5.0, 3.0))
    color1 = BLUE
    ax1.plot(lambdas, ari_vals, "o-", color=color1, lw=1.2, ms=5, label="ARI")
    ax1.set_xlabel(r"$\lambda$", fontsize=11)
    ax1.set_ylabel("ARI", color=color1, fontsize=11)
    ax1.set_ylim(-0.15, 1.30)
    ax1.set_yticks([0.0, 0.4, 0.8, 1.2])
    ax1.tick_params(axis="both", direction="in", top=False, right=False,
                    which="both", labelsize=9)
    ax1.tick_params(axis="y", labelcolor=color1)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    ax2 = ax1.twinx()
    color2 = ORANGE
    ax2.plot(lambdas, n_gr_vals, "s--", color=color2, lw=1.2, ms=5,
             label=r"$|FGS_\lambda|$")
    ax2.set_ylabel(r"$|FGS_\lambda|$", color=color2, fontsize=11)
    ax2.tick_params(axis="both", direction="in", top=False, left=False,
                    which="both", labelsize=9)
    ax2.tick_params(axis="y", labelcolor=color2)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               loc="upper left", frameon=True, edgecolor="black", ncol=2)

    ax1.set_title("Layer-wise ARI and granule count (Dermatology)",
                  fontweight="bold", fontsize=12, pad=14)
    ax1.grid(color="#E2E2E2", lw=0.4)
    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_layerwise_ari{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_layerwise_ari.pdf'}")


def _fig_heatmaps(E_tilde, single_layers, y_true):
    """(b) Equivalence-class heatmaps at coarse / medium / fine lambdas."""
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]
    lam_names = ["coarse", "medium", "fine"]
    lambdas_sel = [single_layers[k] for k in lam_names]

    # class boundaries for separator lines
    boundaries = np.where(np.diff(y_sorted))[0]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6),
                             constrained_layout=True)
    for ax, lam, name in zip(axes, lambdas_sel, lam_names):
        v = single_granularity_vector(E_tilde, lam)
        v_sorted = v[sort_idx][:, sort_idx]
        im = ax.imshow(v_sorted, cmap=mpl.colors.ListedColormap(["#F5F5F5", BLUE]),
                       vmin=0, vmax=1, aspect="equal", interpolation="none")

        # class boundary lines
        for b in boundaries:
            ax.axhline(y=b + 0.5, color="#C44E52", lw=0.6, alpha=1.0)
            ax.axvline(x=b + 0.5, color="#C44E52", lw=0.6, alpha=1.0)

        n_gr = granule_count(E_tilde, lam)
        ax.set_title(f"{name.capitalize()}  "
                     + r"$\lambda=" + f"{lam:.3f}$  "
                     + r"$|FGS|=" + f"{n_gr}$",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # single colorbar shared across subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                        ticks=[0, 1])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Co-membership", fontsize=7)

    fig.suptitle("Equivalence-class heatmaps at three granularities (Dermatology)",
                 fontweight="bold", fontsize=9)
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_heatmaps{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_heatmaps.pdf'}")


def _fig_merging_paths(E_tilde, lambdas, V, y_true, class_names):
    """(c) Merging paths for 2--3 borderline samples (one per class)."""
    classes = np.unique(y_true)
    scores = np.full(V.shape[0], -np.inf, dtype=np.float64)
    for i in range(V.shape[0]):
        own = y_true[i]
        own_mask = y_true == own
        own_mask[i] = False
        within = V[i, own_mask].mean() if own_mask.any() else 0.0
        cross = max(V[i, y_true == c].mean() for c in classes if c != own)
        scores[i] = cross - within

    # pick the single most borderline sample from each of the 3 strongest classes
    class_top = {}
    for c in classes:
        mask = y_true == c
        masked_scores = np.where(mask, scores, -np.inf)
        class_top[c] = int(np.argmax(masked_scores))
    ranked_classes = sorted(class_top.keys(),
                            key=lambda c: scores[class_top[c]], reverse=True)
    borderline_idx = [class_top[c] for c in ranked_classes[:3]]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for ax, idx in zip(axes, borderline_idx):
        n_members = []
        for lam in lambdas:
            v = single_granularity_vector(E_tilde, lam)
            n_members.append(int(v[idx].sum()))
        ax.plot(lambdas, n_members, "o-", color=BLUE, lw=1.2, ms=5)
        ax.set_xlabel(r"$\lambda$", fontsize=10)
        ax.set_ylabel("Granule size", fontsize=10)
        ax.tick_params(axis="both", direction="in", top=False, right=False,
                       which="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        true_class = class_names[y_true[idx]]
        ax.set_title(f"Sample {idx} ({true_class})", fontsize=10, fontweight="bold")
        ax.grid(color="#E2E2E2", lw=0.4)

    fig.suptitle("Merging paths of borderline samples (Dermatology)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_merging_paths{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_merging_paths.pdf'}")


def _fig_V_heatmap(V, y_true, class_names):
    """(d) V(x) heatmap — one row per subtype representative."""
    sort_idx = np.argsort(y_true)
    # pick one representative per class: the sample whose V row has highest
    # mean value within its own class
    unique_classes = np.unique(y_true)
    representatives = []
    for c in unique_classes:
        mask = y_true == c
        # pick sample with highest mean V on within-class indices
        within_mean = V[mask][:, mask].mean(axis=1)
        best_local = np.where(mask)[0][np.argmax(within_mean)]
        representatives.append(best_local)

    V_rep = V[representatives][:, sort_idx]

    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    im = ax.imshow(V_rep, cmap="RdYlBu", aspect="auto")
    ax.set_yticks(range(len(representatives)))
    ax.set_yticklabels([class_names[y_true[r]] for r in representatives], fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel(f"All {V.shape[0]} samples (sorted by true class)", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$V(x)_j$", fontsize=7)
    ax.set_title("MG-FGR fusion vector V(x) for representative samples (Dermatology)", fontweight="bold", fontsize=9)
    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_V_heatmap{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_V_heatmap.pdf'}")


# ---------------------------------------------------------------------------
# 5.2.6  Robustness to feature perturbation
# ---------------------------------------------------------------------------


def section_5_2_6(X, y_true, pipe):
    """Feature perturbation robustness on Dermatology."""
    epsilons = [0.00, 0.05, 0.10, 0.15, 0.20]
    n_repeats = 20
    C = pipe["C"]

    s_j = np.std(X, axis=0, ddof=1)  # per-feature std
    rng = np.random.RandomState(2026)

    # methods to compare. Single-layer methods store the granularity *key*
    # ("coarse"/"medium"/"fine"): the actual lambda is re-selected on each
    # noisy draw, matching how MG-FGR re-selects its lambda set — so every
    # representation faces the perturbed data under the same selection rule.
    method_configs = [
        ("MG-FGR", "mgfgr"),
        ("Single-coarse", "coarse"),
        ("Single-medium", "medium"),
        ("Single-fine", "fine"),
        ("k-means (raw)", "raw"),
    ]

    all_curves = {name: {"mean": [], "std": []} for name, _ in method_configs}

    for eps in epsilons:
        print(f"  [5.2.6] epsilon = {eps:.2f}")
        ari_per_method = {name: [] for name, _ in method_configs}

        for rep in range(n_repeats):
            noise = rng.normal(0, eps * s_j, size=X.shape)
            X_noisy = np.clip(X + noise, 0.0, 1.0)

            # Recompute the closure / lambda set once per draw and reuse it
            # across MG-FGR and all single-layer variants (fair + cheaper).
            S_n = gaussian_similarity(X_noisy, sigma=None)
            E_n = max_min_closure(S_n)
            lam_n = select_lambdas(E_n, max_layers=15)
            single_n = select_single_layers(E_n, lam_n, C)

            for method_name, config in method_configs:
                if config == "raw":
                    rep_mat = X_noisy
                elif config in ("coarse", "medium", "fine"):
                    rep_mat = single_granularity_vector(E_n, single_n[config])
                else:  # mgfgr — full fusion on noisy data
                    rep_mat = mgfgr_fusion(E_n, lam_n)

                km = KMeans(n_clusters=C, n_init=10, random_state=2026)
                labels = km.fit_predict(rep_mat)
                ari = adjusted_rand_score(y_true, labels)
                ari_per_method[method_name].append(ari)

        for method_name, _ in method_configs:
            arr = np.array(ari_per_method[method_name])
            all_curves[method_name]["mean"].append(np.mean(arr))
            all_curves[method_name]["std"].append(np.std(arr, ddof=1))

    # Plot
    _plot_robustness(epsilons, all_curves)

    # LaTeX table
    _write_robustness_table(epsilons, all_curves)

    return all_curves


def _plot_robustness(epsilons, all_curves):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for (name, _), color in zip(all_curves.items(), COLORS_5):
        mean_arr = np.array(all_curves[name]["mean"])
        std_arr = np.array(all_curves[name]["std"])
        ax.plot(epsilons, mean_arr, "o-", color=color, lw=1.2, ms=5, label=name)
        ax.fill_between(epsilons, mean_arr - std_arr, mean_arr + std_arr,
                        color=color, alpha=0.12)

    ax.set_xlabel(r"Noise level $\epsilon$", fontsize=11)
    ax.set_ylabel("ARI", fontsize=11)
    ax.set_ylim(-0.12, 1.19)
    ax.tick_params(axis="both", direction="in", top=False, right=False,
                   which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title("Robustness to feature perturbation (Dermatology)",
                 fontweight="bold", fontsize=12, pad=14)
    ax.legend(fontsize=8, frameon=True, edgecolor="black",
              loc="upper left", ncol=3)
    ax.grid(color="#E2E2E2", lw=0.4)

    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_robustness{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_robustness.pdf'}")


def _write_robustness_table(epsilons, all_curves):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{ARI under feature perturbation (Dermatology). Mean $\pm$ std over 20 noise draws.}")
    lines.append(r"  \label{tab:robustness}")
    lines.append(r"  \begin{tabular}{l" + "c" * len(epsilons) + "}")
    lines.append(r"    \toprule")
    header = " & ".join([r"    \textbf{Method}"] + [rf"$\epsilon={e:.2f}$" for e in epsilons])
    lines.append(header + r" \\")
    lines.append(r"    \midrule")
    for name in all_curves:
        row = f"    {name}"
        for i, e in enumerate(epsilons):
            m = all_curves[name]["mean"][i]
            s = all_curves[name]["std"][i]
            row += f" & {m:.4f} $\\pm$ {s:.4f}"
        row += r" \\"
        lines.append(row)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    (TAB_DIR / "robustness.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> saved {TAB_DIR / 'robustness.tex'}")


# ---------------------------------------------------------------------------
# 5.2.7  Sensitivity to the neighbour count k
# ---------------------------------------------------------------------------


def section_k_sensitivity(X, y_true, C, ks=(3, 5, 7, 10, 15, 20, 30, 40)):
    """MG-FGR ARI/NMI as a function of the local-scaling / kNN neighbour count k.

    k (n_neighbors) replaces the global bandwidth as the method's main
    neighbourhood parameter, so its sensitivity must be reported. A Ward-linkage
    ARI reference line marks the strongest classical baseline. k=7 is the standard
    Zelnik-Manor & Perona default (not tuned to this dataset).
    """
    print("\n  [5.2.7] Sensitivity to neighbour count k:")
    ari_m, ari_s, nmi_m, nmi_s = [], [], [], []
    for k in ks:
        S = gaussian_similarity(X, n_neighbors=k, knn_sparsify=True)
        E = max_min_closure(S)
        V = mgfgr_fusion(E, select_lambdas(E))
        m, s = evaluate_representation(V, y_true, C, n_runs=N_RUNS)
        ari_m.append(m["ARI"]); ari_s.append(s["ARI"])
        nmi_m.append(m["NMI"]); nmi_s.append(s["NMI"])
        print(f"    k={k:2d}  ARI={m['ARI']:.4f} +/- {s['ARI']:.4f}  "
              f"NMI={m['NMI']:.4f} +/- {s['NMI']:.4f}")

    # Ward-linkage reference (deterministic best classical baseline)
    Zw = linkage(X, method="ward")
    ward_labels = fcluster(Zw, t=C, criterion="maxclust") - 1
    ward_ari = adjusted_rand_score(y_true, ward_labels)

    _plot_k_sensitivity(ks, ari_m, ari_s, nmi_m, nmi_s, ward_ari)
    _write_k_sensitivity_table(ks, ari_m, ari_s, nmi_m, nmi_s)
    return {"ks": list(ks), "ari_mean": ari_m, "ari_std": ari_s,
            "nmi_mean": nmi_m, "nmi_std": nmi_s, "ward_ari": ward_ari}


def _plot_k_sensitivity(ks, ari_m, ari_s, nmi_m, nmi_s, ward_ari):
    ks = np.array(ks, dtype=float)
    ari_m, ari_s = np.array(ari_m), np.array(ari_s)
    nmi_m, nmi_s = np.array(nmi_m), np.array(nmi_s)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(ks, ari_m, "o-", color=BLUE, lw=1.2, ms=5, label="ARI")
    ax.fill_between(ks, ari_m - ari_s, ari_m + ari_s, color=BLUE, alpha=0.12)
    ax.plot(ks, nmi_m, "s--", color=ORANGE, lw=1.2, ms=5, label="NMI")
    ax.fill_between(ks, nmi_m - nmi_s, nmi_m + nmi_s, color=ORANGE, alpha=0.12)
    ax.axvline(7, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.annotate(r"$k=7$", xy=(7, 0.06), xytext=(8.5, 0.10),
                fontsize=8, color=GRAY)

    ax.set_xlabel(r"$k$", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(ks)
    ax.tick_params(axis="both", direction="in", top=False, right=False,
                   which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title("Sensitivity to neighbour count (Dermatology)",
                 fontweight="bold", fontsize=12, pad=14)
    ax.legend(fontsize=9, frameon=True, edgecolor="black", loc="lower right")
    ax.grid(color="#E2E2E2", lw=0.4)

    fig.tight_layout()
    for fmt in [".pdf", ".png", ".svg"]:
        fig.savefig(FIG_DIR / f"fig_k_sensitivity{fmt}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR / 'fig_k_sensitivity.pdf'}")


def _write_k_sensitivity_table(ks, ari_m, ari_s, nmi_m, nmi_s):
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\footnotesize")
    lines.append(r"  \centering")
    lines.append(r"  \caption{MG-FGR sensitivity to the neighbour count $k$ on Dermatology "
                 r"(mean $\pm$ std over " + str(N_RUNS) + r" runs). $k=7$ is the "
                 r"Zelnik-Manor \& Perona default.}")
    lines.append(r"  \label{tab:k_sensitivity}")
    lines.append(r"  \begin{tabular}{l" + "c" * len(ks) + "}")
    lines.append(r"    \toprule")
    header = " & ".join([r"    $k$"] + [rf"\textbf{{{k}}}" if k == 7 else str(k) for k in ks])
    lines.append(header + r" \\")
    lines.append(r"    \midrule")
    row_a = "    ARI" + "".join(f" & {m:.3f}" for m in ari_m) + r" \\"
    row_n = "    NMI" + "".join(f" & {m:.3f}" for m in nmi_m) + r" \\"
    lines.append(row_a)
    lines.append(row_n)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    (TAB_DIR / "k_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> saved {TAB_DIR / 'k_sensitivity.tex'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Section 5.2: Real-world Application Experiments")
    print("=" * 60)

    # ---- Load datasets ----
    print("\n[Loading datasets]")
    datasets = {}

    print("  Loading Dermatology ...")
    X_derm, y_derm, feat_derm, class_derm = load_dermatology()
    X_derm = MinMaxScaler().fit_transform(X_derm)
    datasets["Dermatology"] = (X_derm, y_derm, feat_derm, class_derm)
    print(f"    X={X_derm.shape}, y={len(np.unique(y_derm))} classes")

    
    for ds_name, loader in extra_loaders:
        print(f"  Loading {ds_name} ...")
        X, y, feat, cls = loader()
        X = MinMaxScaler().fit_transform(X)
        datasets[ds_name] = (X, y, feat, cls)
        print(f"    X={X.shape}, y={len(np.unique(y))} classes")

    # ---- Run MG-FGR on flagship ----
    print("\n[Running MG-FGR on Dermatology (flagship)]")
    pipe_derm = run_mgfgr_pipeline(X_derm, y_derm)
    print(f"  Lambdas selected: {len(pipe_derm['lambdas'])} (max_layers=15)")
    for li in pipe_derm["layer_info"]:
        print(f"    lambda={li['lambda']:.4f}  |FGS|={li['n_granules']}")
    print(f"  Single layers: coarse={pipe_derm['single_layers']['coarse']:.4f}  "
          f"medium={pipe_derm['single_layers']['medium']:.4f}  "
          f"fine={pipe_derm['single_layers']['fine']:.4f}")

    # ---- 5.2.3 Comparison with baselines ----
    print("\n" + "=" * 40)
    print("5.2.3 Comparison with classical clustering baselines")
    print("=" * 40)
    results_5_2_3 = section_5_2_3(datasets)

    # ---- 5.2.4 Single-granularity ablation ----
    print("\n" + "=" * 40)
    print("5.2.4 Single-granularity ablation")
    print("=" * 40)
    results_5_2_4 = section_5_2_4(pipe_derm, y_derm, pipe_derm["C"])

    # ---- 5.2.5 Interpretability ----
    print("\n" + "=" * 40)
    print("5.2.5 Interpretability on Dermatology")
    print("=" * 40)
    section_5_2_5(pipe_derm, y_derm, class_derm)

    # ---- 5.2.6 Robustness ----
    print("\n" + "=" * 40)
    print("5.2.6 Robustness to feature perturbation")
    print("=" * 40)
    results_5_2_6 = section_5_2_6(X_derm, y_derm, pipe_derm)

    # ---- 5.2.7 Sensitivity to neighbour count k ----
    print("\n" + "=" * 40)
    print("5.2.7 Sensitivity to neighbour count k")
    print("=" * 40)
    results_5_2_7 = section_k_sensitivity(X_derm, y_derm, pipe_derm["C"])

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("All experiments complete.")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TAB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
