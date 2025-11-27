#!/usr/bin/env python3
"""
Compare dimensionality reduction + clustering combinations on synthetic spike data.

Workflow
--------
1. Generate SpikeInterface drifting recordings (≈20 units, ~1 minute) and extract spike features.
2. Derive intrinsic PCA dimension d* per dataset using an explained-variance rule (≥98%, cap at 20).
3. Preprocess features with PCA to d* for nonlinear and deep learning methods.
4. Evaluate DR methods (PCA, ICA, UMAP, t-SNE, manifold methods, deep learning) with clustering.
5. Tune (DR, clusterer) configurations on theory-backed grids to maximize ARI.
6. Report ARI (primary), V-measure, NMI, and optional unsupervised metrics plus correlations.

Rationale
---------
Linear methods (PCA, ICA) receive raw waveforms directly to avoid redundant preprocessing.
Nonlinear and deep learning methods receive PCA-preprocessed input because:
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils.runner_helpers import (
    create_synthetic_recording,
    match_ground_truth,
    prepare_spike_features,
    slice_recording,
    visualize_best_clusters,
)

# Mapping from high-level DR groups requested by the user to the concrete reducer names
# exposed by `create_reducer`. These groupings power the CLI switches that limit runs to
# linear, nonlinear, or deep-learning methods only.
DR_GROUPS: Dict[str, Set[str]] = {
    "linear": {"PCA", "ICA"},
    "nonlinear": {
        "KPCA",
        "Isomap",
        "MDS",
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
    parser.add_argument(
        "--d-star-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of d_star values to test (e.g., 3 5 8 12 20). "
            "If provided, tests each value and reports the best. "
            "If not provided, uses auto-computed d_star from variance threshold."
        ),
    )
    parser.add_argument(
        "--apply-ibl-pipeline",
        action="store_true",
        help="Apply IBL preprocessing pipeline (bandpass filter, spatial filtering, etc.)",
    )
    parser.add_argument(
        "--correct-motion",
        action="store_true",
        help="Apply motion correction (requires --apply-ibl-pipeline)",
    )
    parser.add_argument(
        "--motion-preset",
        type=str,
        default="dredge",
        choices=["dredge", "kilosort_like", "nonrigid_accurate", "rigid_fast"],
        help="Motion correction preset to use (default: dredge)",
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
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build DR hyperparameter grids tied to heuristics described in the design doc.
    
    Returns:
        Tuple of (config_dict, skipped_configs_list) where skipped_configs_list contains
        configurations that were filtered out due to constraints.
    """
    config: Dict[str, List[Dict[str, Any]]] = {}
    skipped_configs: List[Dict[str, Any]] = []
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
    
    # --- MDS (distance-preserving embedding) --------------------------------
    mds_dims = _manifold_components(d_star)
    mds_dims = [d for d in mds_dims if d < n_samples]
    if mds_dims:
        config["MDS"] = [
            {
                "n_components": comp,
                "metric": True,
                "n_init": 4,
                "max_iter": 300,
                "random_state": 42,
            }
            for comp in mds_dims
        ]
    else:
        config["MDS"] = []

    # --- Graph / manifold learners -----------------------------------------
    manifold_dims = _manifold_components(d_star)
    if neighbor_vals is None:
        neighbor_vals = _neighbor_candidates(n_samples)
    for method in ("Isomap", "LLE", "ModifiedLLE", "LaplacianEigenmaps", "DiffusionMaps"):
        params: List[Dict[str, Any]] = []
        for n_nb in neighbor_vals:
            for comp in manifold_dims:
                skip_reason = None
                
                # DiffusionMaps: cap n_components at 5 to avoid ARPACK convergence issues
                if method == "DiffusionMaps" and comp > 5:
                    skip_reason = f"DiffusionMaps n_components > 5 (requested {comp})"
                # ModifiedLLE requires n_neighbors > n_components
                elif method == "ModifiedLLE" and n_nb <= comp:
                    skip_reason = f"ModifiedLLE requires n_neighbors > n_components (n_neighbors={n_nb}, n_components={comp})"
                # Ensure n_neighbors < n_samples for all graph-based methods
                elif n_nb >= n_samples:
                    skip_reason = f"n_neighbors >= n_samples ({n_nb} >= {n_samples})"
                # Ensure n_components < n_samples
                elif comp >= n_samples:
                    skip_reason = f"n_components >= n_samples ({comp} >= {n_samples})"
                
                if skip_reason:
                    entry = {"n_neighbors": n_nb, "n_components": comp}
                    if method == "DiffusionMaps":
                        for alpha in (0.5, 1.0):
                            skipped_configs.append({
                                "method": method,
                                "params": {**entry, "alpha": alpha},
                                "skip_reason": skip_reason
                            })
                    else:
                        skipped_configs.append({
                            "method": method,
                            "params": entry,
                            "skip_reason": skip_reason
                        })
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
    umap_dims = sorted({2, max(2, int(d_star / 2)), max(2, min(3, d_star)), max(2, int(d_star))})
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
    phate_dims = [d for d in (2, max(2, int(d_star / 2)), max(2, int(d_star))) if d < n_samples]
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

    trimap_dims = [d for d in (2, max(2, int(d_star / 2)), max(2, int(d_star))) if d < n_samples]
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

    return config, skipped_configs


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


def preprocess_dataset_no_split(
    X_raw: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    d_star_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Preprocess data with PCA without train/test split.
    
    For methods that don't require validation (linear/nonlinear), we can use all data.
    
    Args:
        X_raw: Raw spike features (n_samples, n_features)
        labels: Ground truth labels
        args: Argument namespace with configuration
        seed: Random seed for reproducibility
        d_star_override: If provided, use this d_star instead of computing from variance
        
    Returns:
        data: Dict with X_pca, labels, d_star, etc.
    """
    pca_rng = np.random.default_rng(seed + 73)
    
    if d_star_override is not None:
        # Use fixed d_star value
        d_star = d_star_override
        n_components = min(X_raw.shape[1], args.max_pca_fit, X_raw.shape[0])
        pca_model = PCA(n_components=n_components, svd_solver="full")
        pca_model.fit(X_raw)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_).tolist()
    else:
        # Compute d_star from variance threshold
        pca_result = determine_intrinsic_pca_dim(
            X_raw,
            variance_threshold=args.variance_threshold,
            component_cap=args.max_pca_components,
            fit_cap=args.max_pca_fit,
            train_fraction=args.pca_train_fraction,  # Use fraction to avoid leakage in d_star estimation
            rng=pca_rng,
        )
        pca_model = pca_result['pca_model']
        d_star = pca_result['d_star']
        cumulative_variance = pca_result['cumulative_variance']
    
    # Transform all data
    X_pca = pca_model.transform(X_raw)[:, :d_star]
    
    # Compute hyperparameter statistics
    distance_scale = _estimate_distance_scale(X_pca, rng=np.random.default_rng(seed + 19))
    
    data = {
        'X_raw': X_raw,
        'X_pca': X_pca,
        'labels': labels,
        'n_spikes': len(X_raw),
        'd_star': d_star,
        'distance_scale': distance_scale,
        'pca_model': pca_model,
        'cumulative_variance': cumulative_variance,
    }
    
    variance_pct = cumulative_variance[d_star-1] if d_star <= len(cumulative_variance) else cumulative_variance[-1]
    print(f"   Using all {len(X_raw)} spikes")
    print(f"   d* = {d_star} (variance explained: {variance_pct:.1%})")
    
    return data


def split_and_preprocess_dataset(
    X_raw: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    d_star_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split data and fit PCA on training set only to avoid leakage.
    
    This function ensures proper train/test isolation:
    1. Splits data using stratified sampling
    2. Fits PCA and computes d_star on TRAINING set only
    3. Transforms both train and test using training-fit PCA
    4. Computes hyperparameter statistics on TRAINING set only
    
    Args:
        X_raw: Raw spike features (n_samples, n_features)
        labels: Ground truth labels
        args: Argument namespace with configuration
        seed: Random seed for reproducibility
        d_star_override: If provided, use this d_star instead of computing from variance
        
    Returns:
        train_data: Dict with X_train, X_pca_train, labels_train, d_star, etc.
        test_data: Dict with X_test, X_pca_test, labels_test
    """
    from sklearn.model_selection import train_test_split
    
    # Stratified train/test split (default 80/20)
    test_size = 0.2
    train_idx, test_idx = train_test_split(
        np.arange(len(X_raw)),
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    
    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    
    # Fit PCA on TRAINING set only
    pca_rng = np.random.default_rng(seed + 73)
    
    if d_star_override is not None:
        # Use fixed d_star value
        d_star = d_star_override
        n_components = min(X_train.shape[1], args.max_pca_fit, X_train.shape[0])
        pca_model = PCA(n_components=n_components, svd_solver="full")
        pca_model.fit(X_train)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_).tolist()
    else:
        # Compute d_star from variance threshold
        pca_result = determine_intrinsic_pca_dim(
            X_train,  # Only training data!
            variance_threshold=args.variance_threshold,
            component_cap=args.max_pca_components,
            fit_cap=args.max_pca_fit,
            train_fraction=1.0,  # Use all training data (already split)
            rng=pca_rng,
        )
        pca_model = pca_result['pca_model']
        d_star = pca_result['d_star']
        cumulative_variance = pca_result['cumulative_variance']
    
    # Transform train and test using training-fit PCA
    X_pca_train = pca_model.transform(X_train)[:, :d_star]
    X_pca_test = pca_model.transform(X_test)[:, :d_star]
    
    # Compute hyperparameter statistics on TRAINING set only
    distance_scale = _estimate_distance_scale(X_pca_train, rng=np.random.default_rng(seed + 19))
    
    train_data = {
        'X_raw': X_train,
        'X_pca': X_pca_train,
        'labels': y_train,
        'n_spikes': len(X_train),
        'd_star': d_star,
        'distance_scale': distance_scale,
        'pca_model': pca_model,
        'cumulative_variance': cumulative_variance,
    }
    
    test_data = {
        'X_raw': X_test,
        'X_pca': X_pca_test,
        'labels': y_test,
        'n_spikes': len(X_test),
        'indices': test_idx,  # Track which samples are in test set
    }
    
    variance_pct = cumulative_variance[d_star-1] if d_star <= len(cumulative_variance) else cumulative_variance[-1]
    print(f"   Train/Test split: {len(X_train)} train, {len(X_test)} test")
    print(f"   d* = {d_star} (variance explained: {variance_pct:.1%})")
    
    return train_data, test_data


def prepare_dataset(
    dataset_idx: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate synthetic recording and extract raw spike features.
    
    IMPORTANT: No PCA preprocessing is done here to avoid data leakage!
    PCA will be fit on the training set only after train/test split.
    
    Returns a dataset dict containing:
        - X_raw: Raw spike features (NOT preprocessed yet)
        - labels: Ground-truth spike labels
        - Metadata: Dataset name, seed, true_k, etc.
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
    X_raw, peak_locations, preprocess_meta = prepare_spike_features(
        recording, 
        threshold=args.detect_threshold,
        synthetic_pipeline=True,  # Using synthetic data preprocessing
        apply_ibl_pipeline=args.apply_ibl_pipeline,
        correct_motion=args.correct_motion,
        motion_preset=args.motion_preset,
        cache_dir=str(Path(args.output_dir) / "preprocessing_cache") if args.apply_ibl_pipeline else None,
        save_motion=args.correct_motion,  # Save motion data if correction is applied
        save_rec=False,  # Don't save intermediate recordings by default
        job_kwargs={"n_jobs": args.n_jobs} if args.n_jobs > 1 else None,
    )
    labels_full = match_ground_truth(gt_crop, peak_locations, recording)
    matched_mask = labels_full >= 0  # keep only spikes that found a GT match
    if not np.any(matched_mask):
        raise RuntimeError("No spikes matched to ground truth; consider lowering detection threshold.")
    X_raw = X_raw[matched_mask]
    labels = labels_full[matched_mask]
    
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
    }


def run_single_pipeline(
    runner: ExperimentRunner,
    dataset: Dict[str, Any],
    config: Dict[str, Any],
    enabled_methods: Set[str],
) -> List[Dict[str, Any]]:
    """Run experiments with appropriate preprocessing for each method type.
    
    Linear methods (PCA, ICA) run on raw waveforms.
    Nonlinear and deep learning methods run on PCA-preprocessed features.
    
    Args:
        runner: ExperimentRunner instance
        dataset: Dataset dict containing X_raw, X_pca, labels, and metadata
        config: Combined DR + clustering configuration
        enabled_methods: Set of enabled DR method names
        
    Returns:
        List of result dictionaries with metadata attached
    """
    dataset_name = dataset["dataset_name"]
    all_results = []
    
    # Determine which method groups are enabled
    linear_methods = enabled_methods & DR_GROUPS["linear"]
    nonlinear_methods = enabled_methods & DR_GROUPS["nonlinear"]
    deeplearning_methods = enabled_methods & DR_GROUPS["deeplearning"]
    
    # Run linear methods on RAW data (no PCA preprocessing)
    if linear_methods:
        linear_config = {
            "dimensionality_reduction": {
                k: v for k, v in config["dimensionality_reduction"].items() 
                if k in linear_methods
            },
            "clustering": config["clustering"]
        }
        if linear_config["dimensionality_reduction"]:  # Only run if there are configs
            linear_results = runner.run_experiments(
                X=dataset["X_raw"],
                config=linear_config,
                y=dataset["labels"],
                dataset_name=dataset_name,
            )
            all_results.extend(linear_results)
    
    # Run nonlinear + deep learning methods on PCA-preprocessed data
    if nonlinear_methods or deeplearning_methods:
        nonlinear_deeplearning_config = {
            "dimensionality_reduction": {
                k: v for k, v in config["dimensionality_reduction"].items() 
                if k in (nonlinear_methods | deeplearning_methods)
            },
            "clustering": config["clustering"]
        }
        if nonlinear_deeplearning_config["dimensionality_reduction"]:  # Only run if there are configs
            nonlinear_deeplearning_results = runner.run_experiments(
                X=dataset["X_pca"],
                config=nonlinear_deeplearning_config,
                y=dataset["labels"],
                dataset_name=dataset_name,
            )
            all_results.extend(nonlinear_deeplearning_results)
    
    # Attach dataset metadata to each result
    for result in all_results:
        result["dataset_idx"] = dataset["dataset_idx"]
        result["dataset_seed"] = dataset["seed"]
        result["d_star"] = dataset["d_star"]
        result["true_k"] = dataset["true_k"]
        result["n_spikes"] = dataset["n_spikes"]
        result["n_features_raw"] = dataset["n_features"]
        result["n_features_preprocessed"] = dataset["X_pca"].shape[1]
        
        # Add string versions of params for compatibility with visualization
        if "dim_reduction_params" in result:
            result["dim_params_str"] = json.dumps(result["dim_reduction_params"])
        if "clustering_params" in result:
            result["clust_params_str"] = json.dumps(result["clustering_params"])
    
    return all_results


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


def analyze_method_performance(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze performance by method pair using both peak and average metrics.
    
    Creates two views:
    1. Peak performance: Best single result for each (DR, clustering) pair
    2. Average performance: Mean performance across all configurations for each pair
    
    Args:
        df: Results DataFrame with columns for metrics and parameters
        
    Returns:
        Tuple of (peak_performance_df, average_performance_df)
    """
    if df.empty or "ari" not in df:
        return pd.DataFrame(), pd.DataFrame()
    
    metric_df = df.dropna(subset=["ari"]).copy()
    if metric_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Group by method pair (ignoring hyperparameters)
    method_cols = ["dim_reduction_method", "clustering_method"]
    
    # Peak performance: max ARI achieved by any configuration for each method pair
    peak = metric_df.groupby(method_cols, as_index=False).agg({
        "ari": ["max", "mean", "std", "count"],
        "v_measure": ["max", "mean"],
        "nmi": ["max", "mean"],
    })
    
    # Flatten column names
    peak.columns = [
        "dim_reduction_method", "clustering_method",
        "peak_ari", "mean_ari", "std_ari", "n_configs",
        "peak_v_measure", "mean_v_measure",
        "peak_nmi", "mean_nmi"
    ]
    
    # Sort by peak ARI
    peak = peak.sort_values(by="peak_ari", ascending=False).reset_index(drop=True)
    
    # Average performance: mean ARI across all configurations for each method pair
    avg = metric_df.groupby(method_cols, as_index=False).agg({
        "ari": ["mean", "std", "min", "max", "count"],
        "v_measure": ["mean", "std"],
        "nmi": ["mean", "std"],
    })
    
    # Flatten column names
    avg.columns = [
        "dim_reduction_method", "clustering_method",
        "mean_ari", "std_ari", "min_ari", "max_ari", "n_configs",
        "mean_v_measure", "std_v_measure",
        "mean_nmi", "std_nmi"
    ]
    
    # Sort by mean ARI
    avg = avg.sort_values(by="mean_ari", ascending=False).reset_index(drop=True)
    
    return peak, avg


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


def _create_single_dstar_heatmap(df: pd.DataFrame, d_star: int, output_dir: Path) -> None:
    """Create heatmap for a single d_star value."""
    df_clean = df.dropna(subset=['ari']).copy()
    
    if len(df_clean) == 0:
        return
    
    # Calculate average ARI
    avg_ari = df_clean.groupby(['dim_reduction_method', 'clustering_method']).agg({
        'ari': ['mean', 'count']
    }).reset_index()
    avg_ari.columns = ['dim_reduction_method', 'clustering_method', 'mean_ari', 'n_configs']
    
    # Create pivot table
    heatmap_data = avg_ari.pivot(
        index='dim_reduction_method',
        columns='clustering_method',
        values='mean_ari'
    )
    
    if heatmap_data.empty:
        return
    
    # Sort by performance
    row_means = heatmap_data.mean(axis=1, skipna=True).sort_values(ascending=False)
    col_means = heatmap_data.mean(axis=0, skipna=True).sort_values(ascending=False)
    heatmap_data = heatmap_data.loc[row_means.index, col_means.index]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
        center=0.3, vmin=0.0, vmax=0.6,
        cbar_kws={'label': 'Average ARI'},
        linewidths=0.5, linecolor='gray', ax=ax
    )
    ax.set_title(f'Average ARI by DR and Clustering Method\nd* = {d_star} (PCA dimensions)\n' +
                 f'Overall Mean ARI: {df_clean["ari"].mean():.4f} | n={len(df_clean)} experiments',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Clustering Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimensionality Reduction Method', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / f"heatmap_ari.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_single_dstar_boxplots(df: pd.DataFrame, d_star: int, output_dir: Path) -> None:
    """Create box plots for a single d_star value."""
    df_clean = df.dropna(subset=['ari']).copy()
    
    if len(df_clean) == 0:
        return
    
    sns.set_style("whitegrid")
    
    # 1. Box plot by DR method
    fig, ax = plt.subplots(figsize=(14, 8))
    dr_order = df_clean.groupby('dim_reduction_method')['ari'].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df_clean, x='dim_reduction_method', y='ari',
        order=dr_order, palette='Set2', ax=ax
    )
    
    means = df_clean.groupby('dim_reduction_method')['ari'].mean()
    positions = range(len(dr_order))
    ax.scatter(positions, [means[method] for method in dr_order],
               color='red', s=100, zorder=3, marker='D', label='Mean')
    
    ax.set_title(f'ARI Distribution by DR Method (d* = {d_star})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dimensionality Reduction Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('ARI', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df_clean['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = output_dir / "boxplot_by_dr_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot by clustering method
    fig, ax = plt.subplots(figsize=(10, 8))
    clust_order = df_clean.groupby('clustering_method')['ari'].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df_clean, x='clustering_method', y='ari',
        order=clust_order, palette='Set3', ax=ax
    )
    
    means = df_clean.groupby('clustering_method')['ari'].mean()
    positions = range(len(clust_order))
    ax.scatter(positions, [means[method] for method in clust_order],
               color='red', s=100, zorder=3, marker='D', label='Mean')
    
    ax.set_title(f'ARI Distribution by Clustering Method (d* = {d_star})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Clustering Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('ARI', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df_clean['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = output_dir / "boxplot_by_clustering_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top combinations
    df_clean['method_combo'] = df_clean['dim_reduction_method'] + ' + ' + df_clean['clustering_method']
    combo_medians = df_clean.groupby('method_combo')['ari'].median().sort_values(ascending=False)
    top_combos = combo_medians.head(10).index.tolist()
    
    df_top = df_clean[df_clean['method_combo'].isin(top_combos)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=df_top, x='method_combo', y='ari',
        order=top_combos, palette='husl', ax=ax
    )
    
    means = df_top.groupby('method_combo')['ari'].mean()
    positions = range(len(top_combos))
    ax.scatter(positions, [means[combo] for combo in top_combos],
               color='red', s=80, zorder=3, marker='D', label='Mean')
    
    ax.set_title(f'Top 10 Method Combinations (d* = {d_star})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Method Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('ARI', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df_clean['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = output_dir / "boxplot_top_combinations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_distance_matrix_visualization(
    X_data: np.ndarray,
    y_true: Optional[np.ndarray],
    results: List[Dict[str, Any]],
    output_dir: Path,
    d_star: Optional[int] = None,
    max_samples: int = 500
) -> None:
    """Create distance matrix visualization for the best performing method.
    
    Shows pairwise distances ordered by cluster labels to reveal block diagonal structure.
    
    Args:
        X_data: Feature matrix (n_samples, n_features) - should be PCA-reduced data
        y_true: Ground truth labels (optional, for comparison)
        results: List of experiment results
        output_dir: Directory to save visualization
        d_star: d_star value for title (optional)
        max_samples: Maximum samples to visualize (for performance)
    """
    from spike_sort.dimensionality_reduction import create_reducer
    from spike_sort.clustering import create_clustering
    
    # Find best method by mean ARI
    successful_results = [r for r in results if r.get('success') and r.get('evaluation')]
    
    if not successful_results:
        print("    No successful results for distance matrix visualization")
        return
    
    # Get method with best mean ARI
    method_aris = {}
    for result in successful_results:
        if 'adjusted_rand_index' not in result.get('evaluation', {}):
            continue
        
        key = (result['dim_reduction_method'], result['clustering_method'],
               str(result.get('dim_reduction_params', {})), str(result.get('clustering_params', {})))
        
        if key not in method_aris:
            method_aris[key] = []
        method_aris[key].append(result['evaluation']['adjusted_rand_index'])
    
    if not method_aris:
        print("    No ARI values available for distance matrix visualization")
        return
    
    # Find method with best mean ARI
    best_method = max(method_aris.items(), key=lambda x: np.mean(x[1]))
    dr_method, clust_method, dr_params_str, clust_params_str = best_method[0]
    mean_ari = np.mean(best_method[1])
    
    # Parse parameters
    import json
    try:
        dr_params = json.loads(dr_params_str.replace("'", '"'))
        clust_params = json.loads(clust_params_str.replace("'", '"'))
    except:
        dr_params = {}
        clust_params = {}
    
    print(f"    Creating distance matrix for: {dr_method} + {clust_method} (ARI: {mean_ari:.4f})")
    
    # Subsample if needed
    if len(X_data) > max_samples:
        idx = np.random.choice(len(X_data), max_samples, replace=False)
        X_subset = X_data[idx]
        y_subset = y_true[idx] if y_true is not None else None
    else:
        X_subset = X_data
        y_subset = y_true
    
    # Run the best method
    reducer = create_reducer(dr_method, **dr_params)
    if reducer is None:
        return
    
    X_reduced = reducer.fit_transform(X_subset, y_subset)
    if X_reduced is None:
        return
    
    clusterer = create_clustering(clust_method, **clust_params)
    if clusterer is None:
        return
    
    labels_pred = clusterer.fit_predict(X_reduced)
    if labels_pred is None:
        return
    
    # Compute pairwise distance matrix
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(X_reduced, metric='euclidean')
    
    # Create figure with 2 subplots (removed histogram panel)
    if y_subset is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]
    
    # 1. Distance matrix ordered by predicted clusters
    ax1 = axes[0]
    order_pred = np.argsort(labels_pred)
    dist_ordered_pred = dist_matrix[order_pred][:, order_pred]
    
    im1 = ax1.imshow(dist_ordered_pred, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Distance Matrix (Predicted Clusters)\n{dr_method} + {clust_method}\nARI: {mean_ari:.3f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index (ordered by predicted cluster)', fontsize=11)
    ax1.set_ylabel('Sample Index (ordered by predicted cluster)', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Euclidean Distance')
    
    # Add cluster boundaries
    unique_labels = np.unique(labels_pred[order_pred])
    boundaries = []
    for label in unique_labels[:-1]:
        boundary = np.where(labels_pred[order_pred] == label)[0][-1] + 0.5
        boundaries.append(boundary)
        ax1.axhline(boundary, color='red', linewidth=1, alpha=0.5)
        ax1.axvline(boundary, color='red', linewidth=1, alpha=0.5)
    
    # 2. Distance matrix ordered by ground truth (if available)
    if y_subset is not None:
        ax2 = axes[1]
        order_true = np.argsort(y_subset)
        dist_ordered_true = dist_matrix[order_true][:, order_true]
        
        im2 = ax2.imshow(dist_ordered_true, cmap='viridis', aspect='auto', interpolation='nearest')
        ax2.set_title(f'Distance Matrix (Ground Truth Clusters)\nTrue Labels',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sample Index (ordered by true cluster)', fontsize=11)
        ax2.set_ylabel('Sample Index (ordered by true cluster)', fontsize=11)
        plt.colorbar(im2, ax=ax2, label='Euclidean Distance')
        
        # Add true cluster boundaries
        unique_true = np.unique(y_subset[order_true])
        for label in unique_true[:-1]:
            boundary = np.where(y_subset[order_true] == label)[0][-1] + 0.5
            ax2.axhline(boundary, color='red', linewidth=1, alpha=0.5)
            ax2.axvline(boundary, color='red', linewidth=1, alpha=0.5)
    
    title_suffix = f" (d* = {d_star})" if d_star is not None else ""
    fig.suptitle(f'Distance Matrix Analysis - Best Method{title_suffix}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = output_dir / "distance_matrix_best_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved to: {output_path.name}")
    plt.close()


def create_performance_heatmaps(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps showing ARI performance for DR + Clustering combinations.
    
    Creates:
    1. Overall heatmap (all d_star values combined)
    2. Separate heatmaps for each d_star value
    3. Summary comparison plot across d_star values
    
    Args:
        results_df: DataFrame with columns: dim_reduction_method, clustering_method, ari, d_star, etc.
        output_dir: Directory to save visualizations
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if results_df.empty or "ari" not in results_df.columns:
        logger.warning("No valid results to create heatmaps")
        return
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    df = results_df.dropna(subset=['ari']).copy()
    
    if len(df) == 0:
        logger.warning("No valid ARI values to plot")
        return
    
    print("\n[Heatmap Generation] Creating performance heatmaps...")
    
    # 1. Create overall heatmap (all d_star values)
    print("  Creating overall heatmap (all d_star values combined)...")
    avg_ari = df.groupby(['dim_reduction_method', 'clustering_method']).agg({
        'ari': ['mean', 'std', 'count']
    }).reset_index()
    avg_ari.columns = ['dim_reduction_method', 'clustering_method', 'mean_ari', 'std_ari', 'n_configs']
    
    heatmap_data = avg_ari.pivot(
        index='dim_reduction_method',
        columns='clustering_method',
        values='mean_ari'
    )
    
    if not heatmap_data.empty:
        # Sort by performance
        row_means = heatmap_data.mean(axis=1, skipna=True).sort_values(ascending=False)
        col_means = heatmap_data.mean(axis=0, skipna=True).sort_values(ascending=False)
        heatmap_data = heatmap_data.loc[row_means.index, col_means.index]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0.3, vmin=0.0, vmax=0.6,
            cbar_kws={'label': 'Average ARI'},
            linewidths=0.5, linecolor='gray', ax=ax
        )
        ax.set_title('Average ARI by DR and Clustering Method\n(Across all d_star values and hyperparameters)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Clustering Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dimensionality Reduction Method', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = viz_dir / "ari_heatmap_overall.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved to: {output_path.name}")
        plt.close()
    
    # 2. Create per-d_star heatmaps
    if 'd_star' in df.columns:
        d_star_values = sorted(df['d_star'].dropna().unique())
        
        if len(d_star_values) > 1:
            print(f"  Creating separate heatmaps for {len(d_star_values)} d_star values: {d_star_values}")
            
            dstar_dir = viz_dir / "dstar_heatmaps"
            dstar_dir.mkdir(parents=True, exist_ok=True)
            
            summary_stats = []
            
            for d_star in d_star_values:
                df_dstar = df[df['d_star'] == d_star].copy()
                
                if len(df_dstar) == 0:
                    continue
                
                avg_ari_dstar = df_dstar.groupby(['dim_reduction_method', 'clustering_method']).agg({
                    'ari': ['mean', 'std', 'count']
                }).reset_index()
                avg_ari_dstar.columns = ['dim_reduction_method', 'clustering_method', 'mean_ari', 'std_ari', 'n_configs']
                
                heatmap_dstar = avg_ari_dstar.pivot(
                    index='dim_reduction_method',
                    columns='clustering_method',
                    values='mean_ari'
                )
                
                if heatmap_dstar.empty:
                    continue
                
                # Sort by performance
                row_means_dstar = heatmap_dstar.mean(axis=1, skipna=True).sort_values(ascending=False)
                col_means_dstar = heatmap_dstar.mean(axis=0, skipna=True).sort_values(ascending=False)
                heatmap_dstar = heatmap_dstar.loc[row_means_dstar.index, col_means_dstar.index]
                
                # Store summary
                best_combo = avg_ari_dstar.loc[avg_ari_dstar['mean_ari'].idxmax()]
                summary_stats.append({
                    'd_star': d_star,
                    'n_results': len(df_dstar),
                    'best_dr': best_combo['dim_reduction_method'],
                    'best_clustering': best_combo['clustering_method'],
                    'best_ari': best_combo['mean_ari'],
                    'overall_mean_ari': df_dstar['ari'].mean()
                })
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    heatmap_dstar, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0.3, vmin=0.0, vmax=0.6,
                    cbar_kws={'label': 'Average ARI'},
                    linewidths=0.5, linecolor='gray', ax=ax
                )
                ax.set_title(f'Average ARI by DR and Clustering Method\nd* = {int(d_star)} (PCA dimensions)\n' +
                             f'Overall Mean ARI: {df_dstar["ari"].mean():.4f} | n={len(df_dstar)} experiments',
                             fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Clustering Method', fontsize=12, fontweight='bold')
                ax.set_ylabel('Dimensionality Reduction Method', fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                output_path = dstar_dir / f"ari_heatmap_dstar{int(d_star)}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"    ✓ Created {len(d_star_values)} d_star-specific heatmaps in: {dstar_dir.name}/")
            
            # 3. Create summary comparison plot
            if len(summary_stats) > 1:
                summary_df = pd.DataFrame(summary_stats).sort_values('d_star')
                
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # Plot 1: Overall mean ARI vs d_star
                ax1 = axes[0]
                ax1.plot(summary_df['d_star'], summary_df['overall_mean_ari'],
                        marker='o', linewidth=2, markersize=10, color='steelblue')
                ax1.set_xlabel('d* (PCA Dimensions)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Overall Mean ARI', fontsize=12, fontweight='bold')
                ax1.set_title('Overall Performance vs Intrinsic Dimensionality',
                             fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Annotate best d_star
                best_idx = summary_df['overall_mean_ari'].idxmax()
                best_row = summary_df.loc[best_idx]
                ax1.annotate(f"Best: d*={int(best_row['d_star'])}\nARI={best_row['overall_mean_ari']:.4f}",
                            xy=(best_row['d_star'], best_row['overall_mean_ari']),
                            xytext=(20, 20), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                # Plot 2: Best method combinations per d_star
                ax2 = axes[1]
                bars = ax2.bar(range(len(summary_df)), summary_df['best_ari'],
                              color='lightgreen', edgecolor='darkgreen', linewidth=2)
                ax2.set_xticks(range(len(summary_df)))
                ax2.set_xticklabels([f"d*={int(d)}" for d in summary_df['d_star']])
                ax2.set_xlabel('d* (PCA Dimensions)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Best ARI', fontsize=12, fontweight='bold')
                ax2.set_title('Best Method Combination at Each d*', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add labels on bars
                for i, (bar, row) in enumerate(zip(bars, summary_df.itertuples())):
                    height = bar.get_height()
                    label = f"{row.best_dr}\n+\n{row.best_clustering}"
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}\n{label}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                
                summary_path = dstar_dir / "dstar_comparison_summary.png"
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"    ✓ Summary comparison saved to: {dstar_dir.name}/{summary_path.name}")
                plt.close()
                
                # Save stats to CSV
                summary_csv = dstar_dir / "dstar_summary_stats.csv"
                summary_df.to_csv(summary_csv, index=False)
    
    print("  ✓ Heatmap generation complete!")


def create_performance_boxplots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots showing ARI distribution for different method combinations.
    
    Creates:
    1. Box plots by dimensionality reduction method
    2. Box plots by clustering method
    3. Box plots by (DR, Clustering) combinations (top methods only)
    4. Box plots per d_star value (if multiple d_star tested)
    
    Args:
        results_df: DataFrame with columns: dim_reduction_method, clustering_method, ari, d_star, etc.
        output_dir: Directory to save visualizations
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if results_df.empty or "ari" not in results_df.columns:
        logger.warning("No valid results to create box plots")
        return
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    df = results_df.dropna(subset=['ari']).copy()
    
    if len(df) == 0:
        logger.warning("No valid ARI values to plot")
        return
    
    print("\n[Box Plot Generation] Creating performance box plots...")
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # 1. Box plots by Dimensionality Reduction method
    print("  Creating box plots by DR method...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by median ARI
    dr_order = df.groupby('dim_reduction_method')['ari'].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df, x='dim_reduction_method', y='ari',
        order=dr_order, palette='Set2', ax=ax
    )
    
    # Add mean markers
    means = df.groupby('dim_reduction_method')['ari'].mean()
    positions = range(len(dr_order))
    ax.scatter(positions, [means[method] for method in dr_order], 
               color='red', s=100, zorder=3, marker='D', label='Mean')
    
    ax.set_title('ARI Distribution by Dimensionality Reduction Method\n(Box: IQR, Line: Median, Diamond: Mean)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dimensionality Reduction Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = viz_dir / "boxplot_by_dr_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved to: {output_path.name}")
    plt.close()
    
    # 2. Box plots by Clustering method
    print("  Creating box plots by clustering method...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by median ARI
    clust_order = df.groupby('clustering_method')['ari'].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df, x='clustering_method', y='ari',
        order=clust_order, palette='Set3', ax=ax
    )
    
    # Add mean markers
    means = df.groupby('clustering_method')['ari'].mean()
    positions = range(len(clust_order))
    ax.scatter(positions, [means[method] for method in clust_order],
               color='red', s=100, zorder=3, marker='D', label='Mean')
    
    ax.set_title('ARI Distribution by Clustering Method\n(Box: IQR, Line: Median, Diamond: Mean)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Clustering Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = viz_dir / "boxplot_by_clustering_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved to: {output_path.name}")
    plt.close()
    
    # 3. Box plots for top method combinations
    print("  Creating box plots for top method combinations...")
    
    # Get top 15 combinations by median ARI
    df['method_combo'] = df['dim_reduction_method'] + ' + ' + df['clustering_method']
    combo_medians = df.groupby('method_combo')['ari'].median().sort_values(ascending=False)
    top_combos = combo_medians.head(15).index.tolist()
    
    df_top = df[df['method_combo'].isin(top_combos)].copy()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.boxplot(
        data=df_top, x='method_combo', y='ari',
        order=top_combos, palette='husl', ax=ax
    )
    
    # Add mean markers
    means = df_top.groupby('method_combo')['ari'].mean()
    positions = range(len(top_combos))
    ax.scatter(positions, [means[combo] for combo in top_combos],
               color='red', s=80, zorder=3, marker='D', label='Mean')
    
    ax.set_title('ARI Distribution for Top 15 Method Combinations\n(Ranked by Median ARI)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Method Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = viz_dir / "boxplot_top_combinations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved to: {output_path.name}")
    plt.close()
    
    # 4. Box plots per d_star (if multiple d_star values)
    if 'd_star' in df.columns:
        d_star_values = sorted(df['d_star'].dropna().unique())
        
        if len(d_star_values) > 1:
            print(f"  Creating box plots comparing d_star values...")
            
            dstar_dir = viz_dir / "dstar_heatmaps"
            dstar_dir.mkdir(parents=True, exist_ok=True)
            
            # Overall comparison across d_star
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.boxplot(
                data=df, x='d_star', y='ari',
                palette='coolwarm', ax=ax
            )
            
            # Add mean markers
            means = df.groupby('d_star')['ari'].mean()
            positions = range(len(d_star_values))
            ax.scatter(positions, [means[d] for d in d_star_values],
                       color='red', s=150, zorder=3, marker='D', label='Mean')
            
            # Add counts
            counts = df.groupby('d_star').size()
            for i, d_star in enumerate(d_star_values):
                ax.text(i, df['ari'].max() + 0.02, f'n={counts[d_star]}',
                       ha='center', fontsize=10, fontweight='bold')
            
            ax.set_title('ARI Distribution Across d* Values\n(Intrinsic Dimensionality)',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('d* (PCA Dimensions)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, df['ari'].max() + 0.08)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            output_path = dstar_dir / "boxplot_by_dstar.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved to: {dstar_dir.name}/{output_path.name}")
            plt.close()
            
            # Box plots for top DR methods at each d_star
            fig, axes = plt.subplots(1, len(d_star_values), figsize=(8*len(d_star_values), 8), sharey=True)
            
            if len(d_star_values) == 1:
                axes = [axes]
            
            for idx, d_star in enumerate(d_star_values):
                ax = axes[idx]
                df_dstar = df[df['d_star'] == d_star].copy()
                
                # Get top 10 DR methods at this d_star
                dr_medians = df_dstar.groupby('dim_reduction_method')['ari'].median().sort_values(ascending=False)
                top_dr = dr_medians.head(10).index.tolist()
                
                df_dstar_top = df_dstar[df_dstar['dim_reduction_method'].isin(top_dr)].copy()
                
                sns.boxplot(
                    data=df_dstar_top, x='dim_reduction_method', y='ari',
                    order=top_dr, palette='Set2', ax=ax
                )
                
                ax.set_title(f'd* = {int(d_star)}\n(n={len(df_dstar)} experiments)',
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('DR Method', fontsize=11, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('ARI', fontsize=11, fontweight='bold')
                else:
                    ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
            
            fig.suptitle('Top DR Methods by d* Value', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            output_path = dstar_dir / "boxplot_dr_by_dstar.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved to: {dstar_dir.name}/{output_path.name}")
            plt.close()
    
    # 5. Violin plots for better distribution visualization (bonus!)
    print("  Creating violin plots for top combinations...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.violinplot(
        data=df_top, x='method_combo', y='ari',
        order=top_combos, palette='muted', ax=ax,
        inner='box'  # Show box plot inside violin
    )
    
    ax.set_title('ARI Distribution for Top 15 Method Combinations\n(Violin plot shows full distribution)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Method Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, df['ari'].max() + 0.05)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = viz_dir / "violinplot_top_combinations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved to: {output_path.name}")
    plt.close()
    
    print("  ✓ Box plot generation complete!")


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
    
    # Check if we're using deep learning methods (which need train/test split)
    has_deeplearning = bool(enabled_methods & DR_GROUPS["deeplearning"])

    all_results: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []
    
    # Track d_star performance if testing multiple values
    d_star_performance: List[Dict[str, Any]] = []
    
    # Track last dataset's features and labels for visualization
    last_X_pca = None
    last_labels = None

    for idx, seed in enumerate(seeds):
        print(f"\n=== Dataset {idx + 1}/{len(seeds)} (seed={seed}) ===")
        
        # Step 1: Prepare raw dataset (NO PCA yet to avoid leakage)
        dataset = prepare_dataset(idx, seed, args)
        
        # Determine which d_star values to test
        if args.d_star_values:
            d_star_values = args.d_star_values
            print(f"   Testing d_star values: {d_star_values}")
        else:
            d_star_values = [None]  # None means auto-compute
        
        # Test each d_star value
        for d_star_test in d_star_values:
            if d_star_test is not None:
                print(f"\n   --- Testing d_star = {d_star_test} ---")
            
            # Step 2: Preprocess data (with or without train/test split)
            if has_deeplearning:
                # Deep learning needs train/test split to avoid overfitting
                print(f"   Splitting data for deep learning methods...")
                train_data, test_data = split_and_preprocess_dataset(
                    X_raw=dataset["X_raw"],
                    labels=dataset["labels"],
                    args=args,
                    seed=seed,
                    d_star_override=d_star_test,
                )
                use_train_only = True
            else:
                # Linear/nonlinear methods can use all data
                print(f"   Using all data (no split needed for linear/nonlinear methods)...")
                train_data = preprocess_dataset_no_split(
                    X_raw=dataset["X_raw"],
                    labels=dataset["labels"],
                    args=args,
                    seed=seed,
                    d_star_override=d_star_test,
                )
                use_train_only = False
            
            # Metadata for tracking
            if has_deeplearning:
                n_spikes_train = train_data["n_spikes"]
                n_spikes_test = dataset["n_spikes"] - train_data["n_spikes"]
            else:
                n_spikes_train = train_data["n_spikes"]
                n_spikes_test = 0
            
            dataset_summary = {
                "dataset_name": dataset["dataset_name"],
                "seed": seed,
                "n_spikes_total": dataset["n_spikes"],
                "n_spikes_train": n_spikes_train,
                "n_spikes_test": n_spikes_test,
                "n_features": dataset["n_features"],
                "d_star": train_data["d_star"],
                "true_k": dataset["true_k"],
                "variance_threshold": args.variance_threshold,
                "synthetic_duration": args.synthetic_duration,
                "subset_duration": args.subset_duration,
                "used_train_test_split": has_deeplearning,
            }
            dataset_summaries.append(dataset_summary)

            include_supervised = train_data["labels"] is not None
            
            # Step 3: Build grids using statistics from the data
            neighbor_vals = _neighbor_candidates(train_data["n_spikes"])
            
            dim_grid, skipped_dr_configs = create_dimensionality_grid(
                d_star=train_data["d_star"],
                X_reference=train_data["X_pca"],
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
                train_data["distance_scale"],
                neighbor_vals,
                train_data["n_spikes"],
            )

            # Step 4: Run experiments
            config = {"dimensionality_reduction": dim_grid, "clustering": clustering_grid}
            dataset_suffix = f"_dstar{train_data['d_star']}" if args.d_star_values else ""
            split_suffix = "_train" if use_train_only else ""
            train_dataset_info = {
                "dataset_name": dataset["dataset_name"] + split_suffix + dataset_suffix,
                "X_raw": train_data["X_raw"],
                "X_pca": train_data["X_pca"],
                "labels": train_data["labels"],
                "d_star": train_data["d_star"],
                "n_spikes": train_data["n_spikes"],
                "dataset_idx": dataset["dataset_idx"],
                "seed": dataset["seed"],
                "true_k": dataset["true_k"],
                "n_features": dataset["n_features"],
            }
            
            # Run actual experiments
            experiment_results = run_single_pipeline(runner, train_dataset_info, config, enabled_methods)
            all_results.extend(experiment_results)
            
            # Save for visualization (use last dataset processed)
            # For deep learning: combine train and test data for complete visualization
            if has_deeplearning:
                # Combine train and test data back together
                last_X_pca = np.vstack([train_data["X_pca"], test_data["X_pca"]])
                last_labels = np.concatenate([train_data["labels"], test_data["labels"]])
            else:
                # For non-deep learning, we already have all the data
                last_X_pca = train_data["X_pca"]
                last_labels = train_data["labels"]
            
            # Track performance for this d_star value
            if args.d_star_values:
                successful_results = [r for r in experiment_results if r.get("success")]
                if successful_results:
                    aris = [r["evaluation"]["adjusted_rand_index"] 
                           for r in successful_results 
                           if r.get("evaluation") and "adjusted_rand_index" in r["evaluation"]]
                    if aris:
                        mean_ari = float(np.mean(aris))
                        max_ari = float(np.max(aris))
                        d_star_performance.append({
                            "dataset_idx": idx,
                            "seed": seed,
                            "d_star": train_data["d_star"],
                            "mean_ari": mean_ari,
                            "max_ari": max_ari,
                            "n_experiments": len(aris),
                        })
            
            # Generate visualizations for this d_star immediately after experiments complete
            if experiment_results:
                try:
                    # Convert current results to DataFrame
                    current_results_df = pd.DataFrame(all_results)
                    
                    # Create heatmap and box plots for this specific d_star
                    print(f"\n   [Visualization] Generating heatmaps and box plots for d_star={train_data['d_star']}...")
                    
                    # Filter to only this d_star
                    dstar_results = current_results_df[
                        current_results_df['d_star'] == train_data['d_star']
                    ].copy()
                    
                    if not dstar_results.empty and 'ari' in dstar_results.columns:
                        # Create a specific output directory for this d_star
                        dstar_viz_dir = output_dir / "visualizations" / f"dstar_{train_data['d_star']}"
                        dstar_viz_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate heatmap for this d_star
                        _create_single_dstar_heatmap(dstar_results, train_data['d_star'], dstar_viz_dir)
                        
                        # Generate box plots for this d_star  
                        _create_single_dstar_boxplots(dstar_results, train_data['d_star'], dstar_viz_dir)
                        
                        # Generate distance matrix for best method
                        # Filter results list to this d_star
                        dstar_result_list = [r for r in all_results if r.get('d_star') == train_data['d_star']]
                        create_distance_matrix_visualization(
                            X_data=train_data['X_pca'],
                            y_true=train_data['labels'],
                            results=dstar_result_list,
                            output_dir=dstar_viz_dir,
                            d_star=train_data['d_star'],
                            max_samples=500
                        )
                        
                        print(f"   ✓ Visualizations saved to: visualizations/dstar_{train_data['d_star']}/")
                
                except Exception as e:
                    import logging
                    logging.warning(f"Incremental visualization failed for d_star={train_data['d_star']}: {e}")
            
            # Document skipped configurations as failures
            for skipped in skipped_dr_configs:
                # Only document if the method is in enabled_methods
                if skipped["method"] not in enabled_methods:
                    continue
                    
                # Create a failure result for each clustering method
                for clust_method, clust_params_list in clustering_grid.items():
                    for clust_params in clust_params_list:
                        failure_result = {
                            "experiment_idx": -1,  # Marker for skipped configs
                            "dim_reduction_method": skipped["method"],
                            "dim_reduction_params": skipped["params"],
                            "clustering_method": clust_method,
                            "clustering_params": clust_params,
                            "success": False,
                            "error": f"Configuration skipped: {skipped['skip_reason']}",
                            "dataset_name": train_dataset_info["dataset_name"],
                            "dataset_idx": dataset["dataset_idx"],
                            "dataset_seed": dataset["seed"],
                            "d_star": train_data["d_star"],
                            "true_k": dataset["true_k"],
                            "n_spikes": train_data["n_spikes"],
                            "n_features_raw": dataset["n_features"],
                            "n_features_preprocessed": train_data["X_pca"].shape[1],
                            "data_shape": train_data["X_pca"].shape,
                            "has_ground_truth": train_data["labels"] is not None,
                        }
                        all_results.append(failure_result)

    runner.save_results(filename=args.results_name)

    results_df = build_results_dataframe(all_results)
    best_configs = select_best_configs(results_df)
    peak_performance, avg_performance = analyze_method_performance(results_df)
    correlations = compute_unsupervised_correlations(results_df)

    best_path = output_dir / "best_configs.csv"
    peak_performance_path = output_dir / "peak_performance.csv"
    avg_performance_path = output_dir / "average_performance.csv"
    full_results_path = output_dir / "all_results.csv"
    meta_path = output_dir / "dataset_metadata.json"
    corr_path = output_dir / "unsupervised_metric_correlations.json"

    if not best_configs.empty:
        best_configs.to_csv(best_path, index=False)
        print(f"\nSaved best configurations (per DR + clustering method) to: {best_path}")
    
    if not peak_performance.empty:
        peak_performance.to_csv(peak_performance_path, index=False)
        print(f"Saved peak performance analysis to: {peak_performance_path}")
    
    if not avg_performance.empty:
        avg_performance.to_csv(avg_performance_path, index=False)
        print(f"Saved average performance analysis to: {avg_performance_path}")
    
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
    
    # Save and print d_star comparison if multiple values were tested
    if d_star_performance:
        d_star_path = output_dir / "d_star_comparison.json"
        with open(d_star_path, "w", encoding="utf-8") as f:
            json.dump(d_star_performance, f, indent=2)
        print(f"d_star comparison results stored at: {d_star_path}")
        
        # Analyze and print best d_star
        print("\n=== d_star Performance Comparison ===")
        d_star_df = pd.DataFrame(d_star_performance)
        
        # Group by d_star and compute average performance
        d_star_summary = d_star_df.groupby("d_star").agg({
            "mean_ari": ["mean", "std"],
            "max_ari": ["mean", "std"],
            "n_experiments": "sum"
        }).round(4)
        
        print("\nAverage performance across all datasets by d_star:")
        print(d_star_summary)
        
        # Find best d_star
        best_mean_ari = d_star_df.groupby("d_star")["mean_ari"].mean()
        best_d_star = int(best_mean_ari.idxmax())
        best_ari_value = best_mean_ari.max()
        
        print(f"\n*** BEST d_star: {best_d_star} (average ARI: {best_ari_value:.4f}) ***")
        
        # Per-dataset best d_star
        print("\nBest d_star per dataset:")
        for dataset_idx in d_star_df["dataset_idx"].unique():
            dataset_perf = d_star_df[d_star_df["dataset_idx"] == dataset_idx]
            best_for_dataset = dataset_perf.loc[dataset_perf["mean_ari"].idxmax()]
            print(f"  Dataset {dataset_idx} (seed={int(best_for_dataset['seed'])}): "
                  f"d_star={int(best_for_dataset['d_star'])} "
                  f"(mean ARI={best_for_dataset['mean_ari']:.4f}, "
                  f"max ARI={best_for_dataset['max_ari']:.4f})")

    # Print performance analysis summaries
    if not peak_performance.empty:
        print("\n=== Peak Performance (Best Single Config per Method Pair) ===")
        print("\nTop 10 method combinations by peak ARI:")
        top_peak = peak_performance.head(10)[["dim_reduction_method", "clustering_method", 
                                               "peak_ari", "mean_ari", "std_ari", "n_configs"]]
        print(top_peak.to_string(index=False))
    
    if not avg_performance.empty:
        print("\n=== Average Performance (Mean Across All Configs per Method Pair) ===")
        print("\nTop 10 method combinations by average ARI:")
        top_avg = avg_performance.head(10)[["dim_reduction_method", "clustering_method",
                                             "mean_ari", "std_ari", "min_ari", "max_ari", "n_configs"]]
        print(top_avg.to_string(index=False))
        
        # Identify most consistent methods (high mean, low std)
        if len(avg_performance) > 0:
            avg_performance["consistency_score"] = avg_performance["mean_ari"] / (avg_performance["std_ari"] + 1e-6)
            most_consistent = avg_performance.nlargest(5, "consistency_score")
            print("\n=== Most Consistent Methods (High Mean / Low Std) ===")
            consistent_display = most_consistent[["dim_reduction_method", "clustering_method",
                                                   "mean_ari", "std_ari", "consistency_score"]]
            print(consistent_display.to_string(index=False))

    print("\n=== DR + Clustering Comparison Complete ===")

    analyzer = ResultsAnalyzer(all_results)
    print("\nSummary statistics across all runs:")
    print(analyzer.get_summary_statistics())
    
    # Print mean ARI per DR method
    if not results_df.empty and "ari" in results_df.columns:
        mean_ari = results_df.groupby("dim_reduction_method")["ari"].mean().sort_values(ascending=False)
        print("\nBest mean ARI per dimensionality reduction method:")
        print(mean_ari)
    
    # Visualize best clustering configuration (by mean ARI)
    if last_X_pca is not None and len(all_results) > 0:
        try:
            if has_deeplearning:
                print(f"\n[Visualization] Using complete dataset (train + test) for visualization: {last_X_pca.shape[0]} samples")
            visualize_best_clusters(
                X=last_X_pca,
                y=last_labels if last_labels is not None else None,
                results=all_results,
                output_dir=output_dir / "visualizations",
                max_points=50000,
                plot_3d=True,
                recording=None,
                peak_locations=None,
                export_to_phy_flag=False,
                launch_phy_gui=False,
                job_kwargs=None,
                use_mean_ari=True,  # Use mean ARI across datasets for more robust selection
            )
        except Exception as e:
            import logging
            logging.warning(f"Visualization failed: {e}")
    
    # Create performance heatmaps
    if not results_df.empty:
        try:
            create_performance_heatmaps(results_df, output_dir)
        except Exception as e:
            import logging
            logging.warning(f"Heatmap generation failed: {e}")
    
    # Create performance box plots
    if not results_df.empty:
        try:
            create_performance_boxplots(results_df, output_dir)
        except Exception as e:
            import logging
            logging.warning(f"Box plot generation failed: {e}")
    
    # Create distance matrix for overall best method
    if last_X_pca is not None and len(all_results) > 0:
        try:
            print("\n[Distance Matrix] Creating visualization for overall best method...")
            create_distance_matrix_visualization(
                X_data=last_X_pca,
                y_true=last_labels,
                results=all_results,
                output_dir=output_dir / "visualizations",
                d_star=None,  # Overall, not specific to d_star
                max_samples=500
            )
        except Exception as e:
            import logging
            logging.warning(f"Distance matrix visualization failed: {e}")


if __name__ == "__main__":
    main()
