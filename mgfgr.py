"""MG-FGR: Multi-Granularity Fuzzy Graph Representation.

Core pipeline: similarity -> max-min transitive closure -> lambda-cuts -> weighted fusion.

What actually drives the output (important, and easy to misread).  Thresholding
the max-min closure at lambda yields the connected components of the graph
{S >= lambda}; sweeping lambda therefore reproduces the *single-linkage*
hierarchy.  The fusion V(x) is assembled from the lambda-cut **partitions**
(binary co-membership matrices ``v_lambda``) weighted by ``1/|FGS_lambda|``, so
V depends ONLY on (i) the *order* of the pairwise affinities and (ii) the kNN
sparsity pattern -- never on the numeric similarity values.  Consequently any
strictly monotone kernel, and any global bandwidth ``sigma``, produces the
exact same V: the Gaussian form and the bandwidth are inert.  The genuine
modelling choices are the local scaling and the mutual-kNN sparsification, both
governed by the neighbour count ``n_neighbors`` (k) -- that k, not sigma, is the
method's real hyper-parameter.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    confusion_matrix,
)
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------------


def gaussian_similarity(X, sigma=None, n_neighbors=7, knn_sparsify=True):
    """Affinity graph for the MG-FGR pipeline.

    The name is kept for backward compatibility, but note (see the module
    docstring) that the downstream lambda-cut fusion is invariant to any
    strictly monotone reweighting of the edges: only the *order* of the
    affinities and the kNN sparsity pattern reach ``V``.  The exponential form
    and the bandwidth are therefore cosmetic; the two ingredients that actually
    shape the result are produced here:

    * **Local scaling** (Zelnik-Manor & Perona, 2004), ``sigma=None`` default:
      ``S_ij = exp(-d_ij^2 / (sigma_i * sigma_j))`` with ``sigma_i`` the distance
      from ``x_i`` to its ``n_neighbors``-th nearest neighbour.  This is what
      sets the affinity *ordering* in a density-adaptive way and spreads the
      lambda-cut ladder; it replaces the global bandwidth with the single,
      interpretable neighbour count ``n_neighbors`` (k) -- the real
      hyper-parameter (it is not tuning-free, just better-conditioned).
    * **Mutual-kNN sparsification** (``knn_sparsify=True``): only edges that are
      mutual local neighbours are kept before the closure, removing weak
      long-range bridges that drive single-linkage chaining.  This *does* change
      which components form, so it is a genuine modelling choice, not cosmetic.

    Passing a float ``sigma`` selects the legacy fixed global bandwidth
    ``S_ij = exp(-d_ij^2 / (2 sigma^2))``; under it the closure reproduces plain
    single-linkage and the bandwidth value has no effect on ``V``.
    """
    n = X.shape[0]
    dists = squareform(pdist(X, metric="euclidean"))

    if sigma is None:
        k = int(min(max(1, n_neighbors), n - 1))
        # distance to the k-th nearest neighbour (column 0 is the self-distance)
        dist_sorted = np.sort(dists, axis=1)
        sigma_i = dist_sorted[:, k]
        sigma_i = np.maximum(sigma_i, 1e-12)
        denom = np.outer(sigma_i, sigma_i)
        S = np.exp(-(dists ** 2) / denom)
        if knn_sparsify:
            nn = np.argsort(dists, axis=1)[:, 1 : k + 1]
            mask = np.zeros((n, n), dtype=bool)
            rows = np.repeat(np.arange(n), k)
            mask[rows, nn.ravel()] = True
            mask = mask & mask.T
            S = np.where(mask, S, 0.0)
            np.fill_diagonal(S, 1.0)
        return S

    # legacy fixed-bandwidth path
    if sigma < 1e-12:
        sigma = 1.0
    return np.exp(-(dists ** 2) / (2 * sigma ** 2))


def pearson_similarity(X):
    """Pearson correlation similarity rescaled to [0, 1].

    S_ij = (corr(x_i, x_j) + 1) / 2
    """
    corr = np.corrcoef(X)
    return (corr + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Transitive closure
# ---------------------------------------------------------------------------


def max_min_closure(S):
    """Max-min transitive closure (Warshall-style, vectorised).

    Computes the fuzzy equivalence relation E = TC(S) such that
    E_ij = sup_k min(E_ik, E_kj), repeated to convergence.

    Complexity: O(n^3), but vectorised per iteration.
    """
    E = S.copy().astype(np.float64)
    n = E.shape[0]
    for k in range(n):
        ik = E[:, k : k + 1]   # shape (n, 1)
        kj = E[k : k + 1, :]   # shape (1, n)
        E = np.maximum(E, np.minimum(ik, kj))
    return E


# ---------------------------------------------------------------------------
# Lambda selection
# ---------------------------------------------------------------------------


def select_lambdas(E_tilde, max_layers=15):
    """Select lambdas so the granule counts follow a geometric ladder.

    Thresholding directly in lambda-space is ill-conditioned: max-min closure
    crowds the distinct merge heights into a narrow high band, so evenly-spaced
    lambdas give wildly uneven granularities (jumping 1 -> 10 -> 22 ...).  We
    instead fix a target ladder of granule counts (geometric from 2 to n) and,
    for each target, pick the lambda whose ``|FGS_lambda|`` is closest to it.
    This yields a controlled, well-separated granularity resolution.

    Returns sorted array of lambdas (ascending).
    """
    n = E_tilde.shape[0]
    upper = E_tilde[np.triu_indices_from(E_tilde, k=1)]
    distinct = np.unique(upper)
    distinct = distinct[distinct > 0.0]
    if len(distinct) == 0:
        # degenerate: all entries equal zero — shouldn't happen
        return np.array([0.5])

    # granule count is monotone non-decreasing in lambda.  We scan all distinct
    # nonzero merge heights so the implementation exactly matches the paper's
    # "closest granule count" rule rather than an approximate subsampling.
    candidates = distinct
    # Exact granule counts for all candidate thresholds.  A naive implementation
    # would rebuild the binary relation for every lambda; here we sweep the
    # off-diagonal entries once from high to low and maintain connected
    # components with union-find.  At a threshold lambda, the components of
    # {E_tilde >= lambda} are exactly the lambda-cut granules.
    tri_i, tri_j = np.triu_indices(n, k=1)
    weights = E_tilde[tri_i, tri_j]
    keep = weights > 0.0
    edge_i = tri_i[keep]
    edge_j = tri_j[keep]
    edge_w = weights[keep]
    order = np.argsort(-edge_w)
    edge_i = edge_i[order]
    edge_j = edge_j[order]
    edge_w = edge_w[order]

    parent = np.arange(n)
    size = np.ones(n, dtype=int)
    n_components = n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        nonlocal n_components
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        n_components -= 1

    count_by_weight = {}
    pos = 0
    while pos < len(edge_w):
        w = edge_w[pos]
        while pos < len(edge_w) and edge_w[pos] == w:
            union(int(edge_i[pos]), int(edge_j[pos]))
            pos += 1
        count_by_weight[w] = n_components

    counts = np.array([count_by_weight[lam] for lam in candidates])

    # target granule counts: geometric ladder 2, ..., n (deduplicated integers)
    targets = np.unique(np.round(np.geomspace(2, n, num=max_layers)).astype(int))

    chosen = []
    for t in targets:
        j = int(np.argmin(np.abs(counts - t)))
        chosen.append(candidates[j])
    return np.unique(chosen)


# ---------------------------------------------------------------------------
# Single-granularity representation
# ---------------------------------------------------------------------------


def single_granularity_vector(E_tilde, lam):
    """Binary indicator matrix at threshold lambda.

    v_lambda(x_i)[j] = 1 if E_tilde[i,j] >= lambda, else 0.

    Returns (n, n) binary matrix.
    """
    return (E_tilde >= lam).astype(np.float64)


def granule_count(E_tilde, lam):
    """Number of equivalence classes (distinct rows) at threshold lambda."""
    v = single_granularity_vector(E_tilde, lam)
    return len(np.unique(v, axis=0))


# ---------------------------------------------------------------------------
# Multi-granularity fusion
# ---------------------------------------------------------------------------


def mgfgr_fusion(E_tilde, lambdas):
    """Weighted multi-granularity fusion.

    V(x) = sum_{lambda in Lambda} omega_lambda * v_lambda(x)
    where omega_lambda = 1 / |FGS_lambda|  (coarser -> larger weight).

    ``v_lambda`` is the crisp 0/1 co-membership matrix of the partition
    ``FGS_lambda``, so V is a function of the selected *partitions* alone -- the
    graded (fuzzy) closure degrees are discarded at the cut.  A genuinely fuzzy
    variant would accumulate the soft degrees ``E_tilde`` gated by each cut
    instead of the binary indicator; that is left as a separate, to-be-validated
    design (it changes the method, not just its labelling).

    Returns (n, n) fusion matrix V.
    """
    n = E_tilde.shape[0]
    V = np.zeros((n, n), dtype=np.float64)
    for lam in lambdas:
        v_lam = single_granularity_vector(E_tilde, lam)
        n_granules = len(np.unique(v_lam, axis=0))
        if n_granules <= 1:
            continue
        omega = 1.0 / n_granules
        V += omega * v_lam
    return V


# ---------------------------------------------------------------------------
# Single-layer selection (for ablation — 5.2.4)
# ---------------------------------------------------------------------------


def select_single_layers(E_tilde, lambdas, C):
    """Select three *distinct* coarse / medium / fine single layers for ablation.

    Earlier versions keyed the layers off the class count C (coarsest with
    ``2 <= |FGS| < C``, medium closest to C).  With the granule-count ladder and
    kNN sparsification the granularities no longer straddle C -- the coarsest
    non-trivial layer can already have many more than C granules -- which made
    'coarse' and 'medium' collapse onto the same layer.  We instead pick three
    layers by their *position* along the non-trivial part of the ladder
    (fewest / middle / most granules), which is the natural reading of
    coarse/medium/fine and guarantees distinct layers whenever at least three
    non-trivial layers exist.

    Returns dict with lambda values keyed by 'coarse', 'medium', 'fine'.
    """
    n = E_tilde.shape[0]
    lambdas = sorted(lambdas)
    counts = [granule_count(E_tilde, lam) for lam in lambdas]

    # indices of non-trivial layers (more than one granule, not all singletons)
    valid = [i for i, c in enumerate(counts) if 2 <= c < n]
    if not valid:
        valid = list(range(len(lambdas)))

    coarse_i = valid[0]            # fewest granules (smallest lambda)
    fine_i = valid[-1]             # most granules (largest lambda)
    medium_i = valid[len(valid) // 2]

    # if the middle collides with an endpoint, nudge it to stay distinct
    if len(valid) >= 3:
        if medium_i in (coarse_i, fine_i):
            interior = [i for i in valid if i not in (coarse_i, fine_i)]
            medium_i = interior[len(interior) // 2]

    return {
        "coarse": lambdas[coarse_i],
        "medium": lambdas[medium_i],
        "fine": lambdas[fine_i],
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def cluster_accuracy(y_true, y_pred):
    """Clustering accuracy via Hungarian (linear assignment) matching."""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


def evaluate_clustering(y_true, y_pred):
    """Return dict with ARI, NMI, FMI, ACC."""
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "ACC": cluster_accuracy(y_true, y_pred),
    }


def run_kmeans(X, n_clusters, y_true, n_runs=50, random_seed=0):
    """Run k-means n_runs times with independent random seeds.

    Returns (mean_metrics, std_metrics) as dicts.
    """
    all_metrics = {"ARI": [], "NMI": [], "FMI": [], "ACC": []}
    for i in range(n_runs):
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=random_seed + i)
        labels = km.fit_predict(X)
        m = evaluate_clustering(y_true, labels)
        for k, v in m.items():
            all_metrics[k].append(v)
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v, ddof=1) for k, v in all_metrics.items()}
    return mean_metrics, std_metrics


def evaluate_representation(V, y_true, C, n_runs=50, random_seed=0):
    """Run k-means on representation V, return (mean, std) of metrics.

    V: (n, n) or (n, d) representation matrix.
    y_true: ground truth labels (0-indexed).
    C: number of clusters (= number of true classes).
    """
    return run_kmeans(V, n_clusters=C, y_true=y_true, n_runs=n_runs, random_seed=random_seed)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_mgfgr_pipeline(X, y_true, max_layers=15):
    """End-to-end MG-FGR pipeline.

    Returns dict with:
        S, E_tilde, lambdas, V, layer_info, single_layers
    """
    S = gaussian_similarity(X)
    E_tilde = max_min_closure(S)
    lambdas = select_lambdas(E_tilde, max_layers=max_layers)
    V = mgfgr_fusion(E_tilde, lambdas)
    C = len(np.unique(y_true))

    layer_info = []
    for lam in lambdas:
        n_gr = granule_count(E_tilde, lam)
        layer_info.append({"lambda": lam, "n_granules": n_gr})

    single_layers = select_single_layers(E_tilde, lambdas, C)

    return {
        "S": S,
        "E_tilde": E_tilde,
        "lambdas": lambdas,
        "V": V,
        "layer_info": layer_info,
        "single_layers": single_layers,
        "C": C,
        "n": X.shape[0],
    }
