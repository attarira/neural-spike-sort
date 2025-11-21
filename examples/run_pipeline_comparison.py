#!/usr/bin/env python3
"""
Compare dimensionality reduction + clustering combinations on synthetic spike data.

Workflow
--------
1. Generate SpikeInterface drifting recordings (≈20 units, ~1 minute) and extract spike features.
2. Derive intrinsic PCA dimension d* per dataset using an explained-variance rule (≥98%, cap at 20).
3. Preprocess features with PCA to d* (whitened, decorrelated representation).
4. Evaluate DR methods (PCA, ICA, UMAP, t-SNE, manifold methods, deep learning) with clustering.
5. Tune (DR, clusterer) configurations on theory-backed grids to maximize ARI.
6. Report ARI (primary), V-measure, NMI, and optional unsupervised metrics plus correlations.

Rationale
---------
All DR methods receive PCA-preprocessed input because:
  - Many manifold learners (Isomap/LLE/UMAP/etc) assume isotropic/whitened spaces
  - Raw waveforms have correlated dimensions with amplitude-dominated variance
  - PCA preprocessing decorrelates features and improves numerical conditioning

Run:
    python examples/run_pipeline_comparison.py --num-datasets 2 --output-dir ./results/pipeline_cmp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils.runner_helpers import (
    create_synthetic_recording,
    match_ground_truth,
    prepare_spike_features,
    slice_recording,
)

# Mapping from high-level DR groups requested by the user to the concrete reducer names
# exposed by `create_reducer`. These groupings power the CLI switches that limit runs to
# linear, nonlinear, or deep-learning methods only.
DR_GROUPS: Dict[str, Set[str]] = {
    "linear": {"PCA", "ICA"},
    "nonlinear": {
        "KPCA",
        "Isomap",
        "LLE",
        "ModifiedLLE",
        "LaplacianEigenmaps",
        "DiffusionMaps",
        "PHATE",
        "TriMap",
        "TSNE",
        "UMAP",
    },
    "deeplearning": {"Autoencoder", "VAE", "CEED"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DR + clustering combinations on PCA-preprocessed spike features.",
    )
    parser.add_argument("--num-datasets", type=int, default=2, help="Number of synthetic datasets/seeds.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seeds (overrides --num-datasets when provided).",
    )
    parser.add_argument("--base-seed", type=int, default=2024, help="Base seed for reproducibility.")
    parser.add_argument("--num-units", type=int, default=20, help="Units to simulate per dataset.")
    parser.add_argument(
        "--synthetic-duration",
        type=float,
        default=300.0,
        help="Total duration (seconds) of the synthetic recording before slicing.",
    )
    parser.add_argument(
        "--subset-duration",
        type=float,
        default=60.0,
        help="Recording duration (seconds) to slice from each simulation.",
    )
    parser.add_argument("--detect-threshold", type=float, default=5.0, help="Spike detection threshold (MAD).")
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.98,
        help="Cumulative explained variance target when selecting d*.",
    )
    parser.add_argument(
        "--max-pca-components",
        type=int,
        default=20,
        help="Cap on retained PCA components for d*.",
    )
    parser.add_argument(
        "--max-pca-fit",
        type=int,
        default=60,
        help="Upper bound on PCA components fitted before truncation.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs passed to ExperimentRunner.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/pipeline_comparison",
        help="Output directory for results, metadata, and analysis.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="pipeline_comparison",
        help="Base filename for serialized ExperimentRunner results (pkl/json).",
    )
    parser.add_argument(
        "--pca-train-fraction",
        type=float,
        default=0.8,
        help="Fraction of spikes used to fit PCA when estimating d* (mitigates leakage).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose ExperimentRunner progress output.",
    )
    parser.add_argument(
        "--silent-errors",
        action="store_true",
        help="Prevent ExperimentRunner from printing per-experiment failures.",
    )
    parser.add_argument(
        "--dr-groups",
        type=str,
        nargs="+",
        default=["linear", "nonlinear", "deeplearning"],
        choices=list(DR_GROUPS.keys()),
        help=(
            "Dimensionality-reduction groups to evaluate. "
            "Choose one or more of {linear, nonlinear, deeplearning}."
        ),
    )
    return parser.parse_args()


def resolve_enabled_methods(selected_groups: Sequence[str]) -> Set[str]:
    """
    Expand high-level group selections (linear / nonlinear / deeplearning) into the concrete
    reducer names that should be kept in each experiment configuration.
    """
    enabled: Set[str] = set()
    for group in selected_groups:
        enabled.update(DR_GROUPS[group])
    if not enabled:
        raise ValueError("No dimensionality-reduction methods enabled. Check --dr-groups.")
    return enabled


def filter_dimensionality_grid(
    grid: Dict[str, List[Dict[str, Any]]],
    enabled_methods: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Remove reducers that fall outside the requested DR groups while preserving the order of
    the remaining method definitions.
    """
    return {method: params for method, params in grid.items() if method in enabled_methods}


def determine_intrinsic_pca_dim(
    X: np.ndarray,
    variance_threshold: float,
    component_cap: int,
    fit_cap: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Estimate d* via cumulative explained variance using a train split to avoid leakage."""
    assert 0 < train_fraction <= 1.0, "train_fraction must be in (0, 1]"
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if train_fraction < 1.0:
        train_size = max(2, int(np.ceil(train_fraction * n_samples)))
        subset_idx = rng.choice(n_samples, size=train_size, replace=False)
        X_fit = X[subset_idx]
    else:
        X_fit = X
    n_components = min(n_features, fit_cap, X_fit.shape[0])
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X_fit)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cumulative, variance_threshold, side="left"))
    d_star = min(max(2, idx + 1), component_cap, n_components)
    X_pca = pca.transform(X)[:, :d_star]
    return {
        "pca_model": pca,
        "X_pca": X_pca,
        "d_star": d_star,
        "cumulative_variance": cumulative.tolist(),
    }


def _neighbor_candidates(n_samples: int) -> List[int]:
    if n_samples <= 3:
        return [2]
    # Use log-scale heuristics so the graph sparsity adapts to dataset size without grids.
    base = max(5, int(round(np.log(n_samples))))
    candidates = sorted({min(base, n_samples - 1), min(base * 2, n_samples - 1)})
    return [int(c) for c in candidates if c >= 2]


def _median_gamma(X: np.ndarray, rng: np.random.Generator, sample_size: int = 2048) -> float:
    if X.shape[0] <= 1:
        return 1.0
    idx = rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
    sample = X[idx]
    distances = pairwise_distances(sample, metric="euclidean")
    squared = distances**2
    positive = squared[squared > 0]
    if positive.size == 0:
        median = 1.0
    else:
        median = float(np.median(positive))
    if not np.isfinite(median) or median == 0:
        median = float(np.mean(positive)) if positive.size else 1.0
    # Convert the distance scale into an RBF gamma following the median heuristic.
    gamma = 1.0 / (2.0 * median)
    return float(max(gamma, 1e-6))


def _estimate_distance_scale(X: np.ndarray, rng: np.random.Generator, sample_size: int = 2048) -> float:
    """
    Estimate a characteristic distance scale for DBSCAN by computing the median pairwise
    distance on a random subset of the data.
    """
    if X.shape[0] <= 1:
        return 1.0
    idx = rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
    sample = X[idx]
    distances = pairwise_distances(sample, metric="euclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _component_options(d_star: int) -> List[int]:
    """Generate dimensionality options for PCA and similar methods.
    
    Uses coarse fractions of d* to probe compression vs noise trade-offs:
    - Always include d*/2 and d* (the "sweet spot" range)
    - Add d*/4 only when d* >= 12 (otherwise it's too small to be useful)
    
    This avoids fine-grained spacing (like 3/4 d*) that provides diminishing returns.
    """
    base = max(2, int(d_star))
    half = max(2, int(round(d_star / 2)))
    quarter = max(2, int(round(d_star / 4)))
    
    # Only include quarter if d* is reasonably large
    if d_star >= 12:
        return sorted({quarter, half, base})
    else:
        return sorted({half, base})


def _manifold_components(d_star: int) -> List[int]:
    """Generate dimensionality options for manifold learning methods.
    
    Uses a coarse ladder of dimensions to test different compression levels:
    - Low dimension (2 or d*/4): Tests aggressive compression
    - Medium dimension (d*/2): Balanced compression
    - Full dimension (d*): Preserves most structure
    
    When d* is small, we use fixed small values to avoid over-compression.
    When d* is large (>= 12), we add d*/4 to explore aggressive bottlenecks.
    """
    base = max(2, int(d_star))
    half = max(2, int(round(d_star / 2)))
    
    if d_star >= 12:
        # Large d*: use fractions including aggressive compression (d*/4)
        quarter = max(2, int(round(d_star / 4)))
        return sorted({quarter, half, base})
    elif d_star >= 8:
        # Medium d*: use {2, d*/2, d*} for coarse sampling
        return sorted({2, half, base})
    else:
        # Small d*: just use {2, d*} to avoid redundancy
        return sorted({2, base})


def create_dimensionality_grid(
    d_star: int,
    X_reference: np.ndarray,
    include_supervised: bool,
    rng: np.random.Generator,
    neighbor_vals: Optional[List[int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build DR hyperparameter grids tied to heuristics described in the design doc."""
    config: Dict[str, List[Dict[str, Any]]] = {}
    component_options = _component_options(d_star)
    n_samples = X_reference.shape[0]

    # --- Linear methods ----------------------------------------------------
    config["PCA"] = [
        {"n_components": comp, "svd_solver": "auto", "whiten": False}
        for comp in component_options
    ]
    config["ICA"] = [
        {"n_components": max(2, int(d_star)), "max_iter": 400, "whiten": "unit-variance", "random_state": 42}
    ]

    # --- Kernel / spectral methods -----------------------------------------
    gamma = _median_gamma(X_reference, rng=rng)
    config["KPCA"] = [
        {
            "n_components": max(2, int(d_star)),
            "kernel": "rbf",
            "gamma": gamma,
            "fit_inverse_transform": False,
        }
    ]

    # --- Graph / manifold learners -----------------------------------------
    manifold_dims = _manifold_components(d_star)
    if neighbor_vals is None:
        neighbor_vals = _neighbor_candidates(n_samples)
    for method in ("Isomap", "LLE", "ModifiedLLE", "LaplacianEigenmaps", "DiffusionMaps"):
        params: List[Dict[str, Any]] = []
        for n_nb in neighbor_vals:
            for comp in manifold_dims:
                # ModifiedLLE requires n_neighbors >= n_components
                if method == "ModifiedLLE" and n_nb < comp:
                    continue
                # Ensure n_neighbors < n_samples for all graph-based methods
                if n_nb >= n_samples:
                    continue
                # Ensure n_components < n_samples
                if comp >= n_samples:
                    continue
                    
                entry = {"n_neighbors": n_nb, "n_components": comp}
                if method == "DiffusionMaps":
                    for alpha in (0.5, 1.0):
                        params.append({**entry, "alpha": alpha})
                else:
                    params.append(entry)
        config[method] = params

    # --- Stochastic neighbor methods ---------------------------------------
    # t-SNE requires perplexity < n_samples and typically perplexity < n_samples/3
    perplexities = [p for p in (20, 30, 50) if p < (n_samples - 1) / 3]
    if not perplexities:
        fallback = max(5, min(30, (n_samples - 1) // 3))
        perplexities = [fallback]
    tsne_dims = sorted({2, max(2, min(3, d_star))})
    # Ensure n_components < n_samples for t-SNE
    tsne_dims = [d for d in tsne_dims if d < n_samples]
    if tsne_dims:
        config["TSNE"] = [
            {
                "perplexity": float(perp),
                "n_components": comp,
                "learning_rate": "auto",
                "init": "pca",
                "n_iter": 1000,
            }
            for perp in perplexities
            for comp in tsne_dims
        ]
    else:
        config["TSNE"] = []

    # --- UMAP & relatives ---------------------------------------------------
    umap_neighbors = sorted(set(neighbor_vals + [10, 20, 50]))
    umap_neighbors = [n for n in umap_neighbors if n < n_samples]
    if len(umap_neighbors) > 3:
        umap_neighbors = [umap_neighbors[0], umap_neighbors[len(umap_neighbors) // 2], umap_neighbors[-1]]
    if not umap_neighbors:
        umap_neighbors = [min(10, max(2, n_samples - 1))]
    umap_dims = sorted({2, max(2, min(3, d_star)), max(2, int(d_star))})
    # Ensure n_components < n_samples for UMAP
    umap_dims = [d for d in umap_dims if d < n_samples]
    if umap_dims:
        config["UMAP"] = [
            {
                "n_neighbors": n_nb,
                "min_dist": min_dist,
                "n_components": comp,
                "metric": "euclidean",
                "random_state": 42,
            }
            for n_nb in umap_neighbors
            for min_dist in (0.0, 0.1, 0.3)
            for comp in umap_dims
        ]
    else:
        config["UMAP"] = []

    # --- Density-preserving global methods ---------------------------------
    phate_dims = [d for d in (2, max(2, int(d_star))) if d < n_samples]
    if phate_dims:
        config["PHATE"] = [
            {
                "knn": min(max(5, n_nb), n_samples - 1),
                "decay": 40,
                "t": "auto",
                "n_components": comp,
            }
            for n_nb in neighbor_vals
            for comp in phate_dims
        ]
    else:
        config["PHATE"] = []

    trimap_dims = [d for d in (2, max(2, int(d_star))) if d < n_samples]
    if trimap_dims:
        config["TriMap"] = [
            {
                "n_dims": comp,
                "n_inliers": inliers,
                "n_outliers": outliers,
                "n_random": 5,
                "distance_metric": "euclidean",
            }
            for comp in trimap_dims
            for inliers in (10, 20)
            for outliers in (5, 10)
            if inliers < n_samples  # Ensure n_inliers < n_samples
        ]
    else:
        config["TriMap"] = []

    # --- Deep learning methods ---------------------------------------------
    encoding_dim = max(2, int(d_star))
    deep_hidden = (50, 100, 200)
    deep_epochs = (20, 50)
    config["Autoencoder"] = [
        {
            "encoding_dim": encoding_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": 1e-3,
        }
        for hidden_dim in deep_hidden
        for epochs in deep_epochs
    ]
    config["VAE"] = [
        {
            "encoding_dim": encoding_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": 1e-3,
        }
        for hidden_dim in deep_hidden
        for epochs in deep_epochs
    ]
    if include_supervised:
        config["CEED"] = [
            {
                "encoding_dim": encoding_dim,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "temperature": temp,
            }
            for hidden_dim in deep_hidden
            for epochs in deep_epochs
            for temp in (0.5, 1.0)
        ]

    return config


def create_clustering_grid(
    true_k: int,
    distance_scale: float,
    spectral_neighbors: Sequence[int],
    n_samples: int,
) -> Dict[str, List[Dict[str, Any]]]:
    min_distance = max(distance_scale, 1e-3)
    eps_multipliers = (0.6, 0.9, 1.3)
    dbscan_configs = [
        {
            "eps": float(max(min_distance * mult, 1e-3)),
            "min_samples": max(5, int(np.log(max(n_samples, 2))) + offset),
        }
        for mult, offset in zip(eps_multipliers, (0, 2, 4))
    ]
    if not spectral_neighbors:
        spectral_neighbors = [15, 20]
    spectral_values = list(dict.fromkeys(spectral_neighbors))  # preserve order, unique
    if len(spectral_values) == 1:
        spectral_values.append(spectral_values[0])

    return {
        "KMeans": [
            {"n_clusters": true_k, "n_init": 10, "max_iter": 300, "algorithm": "lloyd"},
            {"n_clusters": true_k, "n_init": 20, "max_iter": 300, "algorithm": "elkan"},
        ],
        "GMM": [
            {"n_components": true_k, "covariance_type": "full", "reg_covar": 1e-4, "max_iter": 300},
            {"n_components": true_k, "covariance_type": "diag", "reg_covar": 1e-4, "max_iter": 300},
        ],
        "DBSCAN": dbscan_configs,
        "SpectralClustering": [
            {"n_clusters": true_k, "n_neighbors": spectral_values[0], "assign_labels": "kmeans"},
            {"n_clusters": true_k, "n_neighbors": spectral_values[1], "assign_labels": "cluster_qr"},
        ],
    }


def prepare_dataset(
    dataset_idx: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate synthetic recording and prepare PCA-preprocessed features.
    
    Returns a dataset dict containing:
        - X_raw: Raw spike features (for reference/distance estimation)
        - X_pca: PCA-preprocessed features (main input to DR methods)
        - labels: Ground-truth spike labels
        - d_star: Intrinsic dimensionality (PCA components retained)
        - Metadata: Dataset name, seed, true_k, distance_scale, etc.
    """
    static_rec, _, gt_sorting = create_synthetic_recording(
        num_units=args.num_units,
        duration=args.synthetic_duration,
        sampling_frequency=30_000.0,
        seed=seed,
    )
    recording, gt_crop = slice_recording(
        static_rec,
        duration_s=args.subset_duration,
        sf=static_rec.get_sampling_frequency(),
        gt_sorting=gt_sorting,
    )
    X_raw, peak_locations, preprocess_meta = prepare_spike_features(recording, threshold=args.detect_threshold)
    labels_full = match_ground_truth(gt_crop, peak_locations, recording)
    matched_mask = labels_full >= 0  # keep only spikes that found a GT match
    if not np.any(matched_mask):
        raise RuntimeError("No spikes matched to ground truth; consider lowering detection threshold.")
    X_raw = X_raw[matched_mask]
    labels = labels_full[matched_mask]
    pca_rng = np.random.default_rng(seed + 73)
    metadata = determine_intrinsic_pca_dim(
        X_raw,
        variance_threshold=args.variance_threshold,
        component_cap=args.max_pca_components,
        fit_cap=args.max_pca_fit,
        train_fraction=args.pca_train_fraction,
        rng=pca_rng,
    )
    distance_scale = _estimate_distance_scale(X_raw, rng=np.random.default_rng(seed + 19))
    dataset_name = f"synthetic_seed{seed}_idx{dataset_idx}"
    return {
        "dataset_idx": dataset_idx,
        "seed": seed,
        "dataset_name": dataset_name,
        "X_raw": X_raw,
        "labels": labels,
        "n_spikes": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "true_k": int(len(np.unique(labels))),
        "preprocess_meta": preprocess_meta,
        "distance_scale": distance_scale,
        **metadata,
    }


def run_single_pipeline(
    runner: ExperimentRunner,
    dataset: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run experiments on PCA-preprocessed features for a single dataset.
    
    Args:
        runner: ExperimentRunner instance
        dataset: Dataset dict containing X_pca, labels, and metadata
        config: Combined DR + clustering configuration
        
    Returns:
        List of result dictionaries with metadata attached
    """
    X_preprocessed = dataset["X_pca"]
    dataset_name = dataset["dataset_name"]
    
    results = runner.run_experiments(
        X=X_preprocessed,
        config=config,
        y=dataset["labels"],
        dataset_name=dataset_name,
    )
    
    # Attach dataset metadata to each result
    for result in results:
        result["dataset_idx"] = dataset["dataset_idx"]
        result["dataset_seed"] = dataset["seed"]
        result["d_star"] = dataset["d_star"]
        result["true_k"] = dataset["true_k"]
        result["n_spikes"] = dataset["n_spikes"]
        result["n_features_raw"] = dataset["n_features"]
        result["n_features_preprocessed"] = dataset["X_pca"].shape[1]
    return results


def build_results_dataframe(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten experiment results into a DataFrame for analysis.
    
    Args:
        results: List of result dictionaries from ExperimentRunner
        
    Returns:
        DataFrame with flattened metrics and serialized parameter configurations
    """
    rows: List[Dict[str, Any]] = []
    for res in results:
        if not res.get("success"):
            continue
        evaluation = res.get("evaluation") or {}
        row = {
            "dataset_name": res.get("dataset_name"),
            "dataset_idx": res.get("dataset_idx"),
            "dataset_seed": res.get("dataset_seed"),
            "dim_reduction_method": res.get("dim_reduction_method"),
            "clustering_method": res.get("clustering_method"),
            "ari": evaluation.get("adjusted_rand_index"),
            "v_measure": evaluation.get("v_measure"),
            "nmi": evaluation.get("normalized_mutual_info"),
            "silhouette": evaluation.get("silhouette_score"),
            "davies_bouldin": evaluation.get("davies_bouldin_index"),
            "calinski_harabasz": evaluation.get("calinski_harabasz_index"),
            "dim_params": json.dumps(res.get("dim_reduction_params", {}), sort_keys=True),
            "clust_params": json.dumps(res.get("clustering_params", {}), sort_keys=True),
            "d_star": res.get("d_star"),
            "true_k": res.get("true_k"),
            "n_spikes": res.get("n_spikes"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Select the best hyperparameter configuration for each (DR, clusterer) pair.
    
    Best configurations are determined by averaging ARI across all dataset seeds,
    with V-measure and NMI as tie-breakers.
    
    Args:
        df: Results DataFrame with columns for metrics and parameters
        
    Returns:
        DataFrame with best configuration per (DR method, clustering method) pair
    """
    if df.empty or "ari" not in df:
        return pd.DataFrame()
    
    metric_df = df.dropna(subset=["ari"]).copy()
    if metric_df.empty:
        return pd.DataFrame()
    
    group_cols = [
        "dim_reduction_method",
        "clustering_method",
        "dim_params",
        "clust_params",
    ]
    
    # Average ARI (and tie-breaker metrics) over seeds to score each hyperparameter combo
    summary = (
        metric_df.groupby(group_cols, as_index=False)[["ari", "v_measure", "nmi"]]
        .mean()
        .fillna({"v_measure": -np.inf, "nmi": -np.inf})
    )

    def pick_best(group: pd.DataFrame) -> pd.Series:
        # Deterministic ranking order: ARI primary, V-measure and NMI as tie-breakers
        ordered = group.sort_values(
            by=["ari", "v_measure", "nmi"],
            ascending=[False, False, False],
        )
        return ordered.iloc[0]

    best = (
        summary.groupby(
            ["dim_reduction_method", "clustering_method"],
            group_keys=False,
        )
        .apply(pick_best)
        .reset_index(drop=True)
    )
    
    best = best.rename(
        columns={
            "ari": "best_ari",
            "v_measure": "best_v_measure",
            "nmi": "best_nmi",
        }
    )
    best["dim_reduction_params"] = best["dim_params"].apply(json.loads)
    best["clustering_params"] = best["clust_params"].apply(json.loads)
    
    # Sort by best ARI descending
    best = best.sort_values(by="best_ari", ascending=False)
    
    return best


def compute_unsupervised_correlations(df: pd.DataFrame) -> Dict[str, float]:
    correlations: Dict[str, float] = {}
    for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        if metric not in df:
            continue
        sub = df[["ari", metric]].dropna()
        if len(sub) < 2:
            continue
        # Store simple Pearson correlations to see whether unsupervised scores mirror ARI.
        correlations[metric] = float(sub["ari"].corr(sub[metric]))
    return correlations


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.base_seed + i for i in range(args.num_datasets)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
        silent_errors=args.silent_errors,
    )

    # Translate the CLI group selection into the reducer names we keep in each configuration.
    enabled_methods = resolve_enabled_methods(args.dr_groups)

    all_results: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds):
        print(f"\n=== Dataset {idx + 1}/{len(seeds)} (seed={seed}) ===")
        dataset = prepare_dataset(idx, seed, args)
        dataset["variance_threshold"] = args.variance_threshold
        dataset_summaries.append(
            {
                "dataset_name": dataset["dataset_name"],
                "seed": seed,
                "n_spikes": dataset["n_spikes"],
                "n_features": dataset["n_features"],
                "d_star": dataset["d_star"],
                "true_k": dataset["true_k"],
                "variance_threshold": args.variance_threshold,
                "synthetic_duration": args.synthetic_duration,
                "subset_duration": args.subset_duration,
            }
        )

        include_supervised = dataset["labels"] is not None  # enables CEED only when GT labels exist
        neighbor_vals = _neighbor_candidates(dataset["n_spikes"])
        
        # Build DR grid using PCA-preprocessed features as reference
        dim_grid = create_dimensionality_grid(
            d_star=dataset["d_star"],
            X_reference=dataset["X_pca"],
            include_supervised=include_supervised,
            rng=np.random.default_rng(seed + 17),
            neighbor_vals=neighbor_vals,
        )
        dim_grid = filter_dimensionality_grid(dim_grid, enabled_methods)
        if not dim_grid:
            raise ValueError(
                "No dimensionality-reduction methods left after applying --dr-groups filter."
            )

        clustering_grid = create_clustering_grid(
            dataset["true_k"],
            dataset["distance_scale"],
            neighbor_vals,
            dataset["n_spikes"],
        )

        # Run single pipeline on PCA-preprocessed features
        config = {"dimensionality_reduction": dim_grid, "clustering": clustering_grid}
        all_results.extend(run_single_pipeline(runner, dataset, config))

    runner.save_results(filename=args.results_name)

    results_df = build_results_dataframe(all_results)
    best_configs = select_best_configs(results_df)
    correlations = compute_unsupervised_correlations(results_df)

    best_path = output_dir / "best_configs.csv"
    full_results_path = output_dir / "all_results.csv"
    meta_path = output_dir / "dataset_metadata.json"
    corr_path = output_dir / "unsupervised_metric_correlations.json"

    if not best_configs.empty:
        best_configs.to_csv(best_path, index=False)
        print(f"\nSaved best configurations (per DR + clustering method) to: {best_path}")
    
    if not results_df.empty:
        results_df.to_csv(full_results_path, index=False)
        print(f"Saved full results table to: {full_results_path}")
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summaries, f, indent=2)
    print(f"Dataset metadata stored at: {meta_path}")
    
    if correlations:
        with open(corr_path, "w", encoding="utf-8") as f:
            json.dump(correlations, f, indent=2)
        print(f"Unsupervised metric correlations stored at: {corr_path}")

    print("\n=== DR + Clustering Comparison Complete ===")

    analyzer = ResultsAnalyzer(all_results)
    print("\nSummary statistics across all runs:")
    print(analyzer.get_summary_statistics())


if __name__ == "__main__":
    main()
