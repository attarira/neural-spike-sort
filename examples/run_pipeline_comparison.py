#!/usr/bin/env python3
"""
Enhanced spike sorting pipeline comparison with improved logging and visualizations.

Key improvements:
- Better structured console output with progress indicators
- Rich visualization suite including performance metrics, timing, and distance matrices
- 2D-only cluster plots (removed 3D)
- Block diagonal distance matrix visualization
- Performance comparison bar charts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.dimensionality_reduction import create_reducer
from spike_sort.clustering import create_clustering
from spike_sort.utils.runner_helpers import (
    create_synthetic_recording,
    match_ground_truth,
    prepare_spike_features,
    slice_recording,
)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Mapping from high-level DR groups to concrete reducer names
DR_GROUPS: Dict[str, Set[str]] = {
    "linear": {"PCA", "ICA"},
    "nonlinear": {
        "KPCA", "Isomap", "MDS", "LLE", "ModifiedLLE",
        "LaplacianEigenmaps", "DiffusionMaps", "PHATE", "TriMap", "TSNE", "UMAP",
    },
    "deeplearning": {"Autoencoder", "VAE", "CEED"},
}


def print_header(text: str, level: int = 1) -> None:
    """Print formatted section headers."""
    if level == 1:
        print(f"\n{'=' * 80}")
        print(f"  {text}")
        print(f"{'=' * 80}")
    elif level == 2:
        print(f"\n{'─' * 80}")
        print(f"  {text}")
        print(f"{'─' * 80}")
    else:
        print(f"\n► {text}")


def print_metric(label: str, value: Any, indent: int = 2) -> None:
    """Print a metric in a consistent format."""
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}• {label}: {value:.4f}")
    elif isinstance(value, int):
        print(f"{prefix}• {label}: {value:,}")
    else:
        print(f"{prefix}• {label}: {value}")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress indicator."""
    pct = 100 * current / total
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix}: [{bar}] {pct:.1f}% ({current}/{total})", end='', flush=True)
    if current == total:
        print()  # New line when complete


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
    """Expand high-level group selections into concrete reducer names."""
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
    """Remove reducers that fall outside the requested DR groups."""
    return {method: params for method, params in grid.items() if method in enabled_methods}


def determine_intrinsic_pca_dim(
    X: np.ndarray,
    variance_threshold: float,
    component_cap: int,
    fit_cap: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Estimate d* via cumulative explained variance using a train split."""
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
    gamma = 1.0 / (2.0 * median)
    return float(max(gamma, 1e-6))


def _estimate_distance_scale(X: np.ndarray, rng: np.random.Generator, sample_size: int = 2048) -> float:
    """Estimate characteristic distance scale for DBSCAN."""
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
    """Generate dimensionality options for PCA."""
    base = max(2, int(d_star))
    half = max(2, int(round(d_star / 2)))
    quarter = max(2, int(round(d_star / 4)))
    if d_star >= 12:
        return sorted({quarter, half, base})
    else:
        return sorted({half, base})


def _manifold_components(d_star: int) -> List[int]:
    """Generate dimensionality options for manifold learning."""
    base = max(2, int(d_star))
    half = max(2, int(round(d_star / 2)))
    
    if d_star >= 12:
        quarter = max(2, int(round(d_star / 4)))
        return sorted({quarter, half, base})
    elif d_star >= 8:
        return sorted({2, half, base})
    else:
        return sorted({2, base})


def create_dimensionality_grid(
    d_star: int,
    X_reference: np.ndarray,
    include_supervised: bool,
    rng: np.random.Generator,
    neighbor_vals: Optional[List[int]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build DR hyperparameter grids."""
    config: Dict[str, List[Dict[str, Any]]] = {}
    skipped_configs: List[Dict[str, Any]] = []
    component_options = _component_options(d_star)
    n_samples = X_reference.shape[0]

    # Linear methods
    config["PCA"] = [
        {"n_components": comp, "svd_solver": "auto", "whiten": False}
        for comp in component_options
    ]
    config["ICA"] = [
        {"n_components": max(2, int(d_star)), "max_iter": 400, "whiten": "unit-variance", "random_state": 42}
    ]

    # Kernel methods
    gamma = _median_gamma(X_reference, rng=rng)
    config["KPCA"] = [
        {
            "n_components": max(2, int(d_star)),
            "kernel": "rbf",
            "gamma": gamma,
            "fit_inverse_transform": False,
        }
    ]
    
    # MDS
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

    # Graph/manifold learners
    manifold_dims = _manifold_components(d_star)
    if neighbor_vals is None:
        neighbor_vals = _neighbor_candidates(n_samples)
    for method in ("Isomap", "LLE", "ModifiedLLE", "LaplacianEigenmaps", "DiffusionMaps"):
        params: List[Dict[str, Any]] = []
        for n_nb in neighbor_vals:
            for comp in manifold_dims:
                skip_reason = None
                
                if method == "DiffusionMaps" and comp > 5:
                    skip_reason = f"DiffusionMaps n_components > 5 (requested {comp})"
                elif method == "ModifiedLLE" and n_nb <= comp:
                    skip_reason = f"ModifiedLLE requires n_neighbors > n_components (n_neighbors={n_nb}, n_components={comp})"
                elif n_nb >= n_samples:
                    skip_reason = f"n_neighbors >= n_samples ({n_nb} >= {n_samples})"
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

    # t-SNE
    perplexities = [p for p in (20, 30, 50) if p < (n_samples - 1) / 3]
    if not perplexities:
        fallback = max(5, min(30, (n_samples - 1) // 3))
        perplexities = [fallback]
    tsne_dims = sorted({2, max(2, min(3, d_star))})
    tsne_dims = [d for d in tsne_dims if d < n_samples]
    if tsne_dims:
        config["TSNE"] = [
            {
                "perplexity": float(perp),
                "n_components": comp,
                "learning_rate": "auto",
                "init": "pca",
            }
            for perp in perplexities
            for comp in tsne_dims
        ]
    else:
        config["TSNE"] = []

    # UMAP
    umap_neighbors = sorted(set(neighbor_vals + [10, 20, 50]))
    umap_neighbors = [n for n in umap_neighbors if n < n_samples]
    if len(umap_neighbors) > 3:
        umap_neighbors = [umap_neighbors[0], umap_neighbors[len(umap_neighbors) // 2], umap_neighbors[-1]]
    if not umap_neighbors:
        umap_neighbors = [min(10, max(2, n_samples - 1))]
    umap_dims = sorted({2, max(2, int(d_star / 2)), max(2, min(3, d_star)), max(2, int(d_star))})
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

    # PHATE and TriMap
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
            if inliers < n_samples
        ]
    else:
        config["TriMap"] = []

    # Deep learning
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
    spectral_values = list(dict.fromkeys(spectral_neighbors))
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
    """Preprocess data with PCA without train/test split."""
    pca_rng = np.random.default_rng(seed + 73)
    
    if d_star_override is not None:
        d_star = d_star_override
        n_components = min(X_raw.shape[1], args.max_pca_fit, X_raw.shape[0])
        pca_model = PCA(n_components=n_components, svd_solver="full")
        pca_model.fit(X_raw)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_).tolist()
    else:
        pca_result = determine_intrinsic_pca_dim(
            X_raw,
            variance_threshold=args.variance_threshold,
            component_cap=args.max_pca_components,
            fit_cap=args.max_pca_fit,
            train_fraction=args.pca_train_fraction,
            rng=pca_rng,
        )
        pca_model = pca_result['pca_model']
        d_star = pca_result['d_star']
        cumulative_variance = pca_result['cumulative_variance']
    
    X_pca = pca_model.transform(X_raw)[:, :d_star]
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
    print_metric("Total spikes", len(X_raw))
    print_metric("d*", f"{d_star} (variance explained: {variance_pct:.1%})")
    
    return data


def split_and_preprocess_dataset(
    X_raw: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    d_star_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split data and fit PCA on training set only."""
    test_size = 0.2
    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = len(unique_labels) >= 2 and np.all(counts >= 2)
    stratify_param = labels if can_stratify else None
    if not can_stratify:
        print_progress(1, 1, prefix="Stratified split disabled (small classes)")
    train_idx, test_idx = train_test_split(
        np.arange(len(X_raw)),
        test_size=test_size,
        stratify=stratify_param,
        random_state=seed
    )
    
    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    
    pca_rng = np.random.default_rng(seed + 73)
    
    if d_star_override is not None:
        d_star = d_star_override
        n_components = min(X_train.shape[1], args.max_pca_fit, X_train.shape[0])
        pca_model = PCA(n_components=n_components, svd_solver="full")
        pca_model.fit(X_train)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_).tolist()
    else:
        pca_result = determine_intrinsic_pca_dim(
            X_train,
            variance_threshold=args.variance_threshold,
            component_cap=args.max_pca_components,
            fit_cap=args.max_pca_fit,
            train_fraction=1.0,
            rng=pca_rng,
        )
        pca_model = pca_result['pca_model']
        d_star = pca_result['d_star']
        cumulative_variance = pca_result['cumulative_variance']
    
    X_pca_train = pca_model.transform(X_train)[:, :d_star]
    X_pca_test = pca_model.transform(X_test)[:, :d_star]
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
        'indices': test_idx,
    }
    
    variance_pct = cumulative_variance[d_star-1] if d_star <= len(cumulative_variance) else cumulative_variance[-1]
    print_metric("Train/Test split", f"{len(X_train)} train, {len(X_test)} test")
    print_metric("d*", f"{d_star} (variance explained: {variance_pct:.1%})")
    
    return train_data, test_data


def prepare_dataset(
    dataset_idx: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate synthetic recording and extract raw spike features."""
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
        synthetic_pipeline=True,
        apply_ibl_pipeline=args.apply_ibl_pipeline,
        correct_motion=args.correct_motion,
        motion_preset=args.motion_preset,
        cache_dir=str(Path(args.output_dir) / "preprocessing_cache") if args.apply_ibl_pipeline else None,
        save_motion=args.correct_motion,
        save_rec=False,
        job_kwargs={"n_jobs": args.n_jobs} if args.n_jobs > 1 else None,
    )
    labels_full = match_ground_truth(gt_crop, peak_locations, recording)
    matched_mask = labels_full >= 0
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
    """Run experiments with appropriate preprocessing."""
    dataset_name = dataset["dataset_name"]
    all_results = []
    
    linear_methods = enabled_methods & DR_GROUPS["linear"]
    nonlinear_methods = enabled_methods & DR_GROUPS["nonlinear"]
    deeplearning_methods = enabled_methods & DR_GROUPS["deeplearning"]
    
    if linear_methods:
        linear_config = {
            "dimensionality_reduction": {
                k: v for k, v in config["dimensionality_reduction"].items() 
                if k in linear_methods
            },
            "clustering": config["clustering"]
        }
        if linear_config["dimensionality_reduction"]:
            linear_results = runner.run_experiments(
                X=dataset["X_raw"],
                config=linear_config,
                y=dataset["labels"],
                dataset_name=dataset_name,
            )
            all_results.extend(linear_results)
    
    if nonlinear_methods or deeplearning_methods:
        nonlinear_deeplearning_config = {
            "dimensionality_reduction": {
                k: v for k, v in config["dimensionality_reduction"].items() 
                if k in (nonlinear_methods | deeplearning_methods)
            },
            "clustering": config["clustering"]
        }
        if nonlinear_deeplearning_config["dimensionality_reduction"]:
            nonlinear_deeplearning_results = runner.run_experiments(
                X=dataset["X_pca"],
                config=nonlinear_deeplearning_config,
                y=dataset["labels"],
                dataset_name=dataset_name,
            )
            all_results.extend(nonlinear_deeplearning_results)
    
    for result in all_results:
        result["dataset_idx"] = dataset["dataset_idx"]
        result["dataset_seed"] = dataset["seed"]
        result["d_star"] = dataset["d_star"]
        result["true_k"] = dataset["true_k"]
        result["n_spikes"] = dataset["n_spikes"]
        result["n_features_raw"] = dataset["n_features"]
        result["n_features_preprocessed"] = dataset["X_pca"].shape[1]
        
        if "dim_reduction_params" in result:
            result["dim_params_str"] = json.dumps(result["dim_reduction_params"])
        if "clustering_params" in result:
            result["clust_params_str"] = json.dumps(result["clustering_params"])
    
    return all_results


def build_results_dataframe(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten experiment results into a DataFrame with timing and grouping keys."""
    rows: List[Dict[str, Any]] = []
    for res in results:
        if not res.get("success"):
            continue
        evaluation = res.get("evaluation") or {}
        dim_meta = res.get("dim_reduction_metadata") or {}
        clust_meta = res.get("clustering_metadata") or {}
        dim_params_str = json.dumps(res.get("dim_reduction_params", {}), sort_keys=True)
        clust_params_str = json.dumps(res.get("clustering_params", {}), sort_keys=True)
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
            "dim_params": dim_params_str,
            "clust_params": clust_params_str,
            "d_star": res.get("d_star"),
            "true_k": res.get("true_k"),
            "n_spikes": res.get("n_spikes"),
            "dim_key": f"{res.get('dim_reduction_method')}:{dim_params_str}",
            "clust_key": f"{res.get('clustering_method')}:{clust_params_str}",
            "dim_reduction_time": dim_meta.get("computation_time"),
            "dim_reduction_memory": dim_meta.get("memory_usage_mb"),
            "clustering_time": clust_meta.get("computation_time"),
            "n_clusters_pred": clust_meta.get("n_clusters"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["is_dim_key_primary"] = ~df.duplicated(subset=["dim_key"], keep="first")
    df.loc[~df["is_dim_key_primary"], ["dim_reduction_time", "dim_reduction_memory"]] = np.nan
    return df


def select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Select best hyperparameter configuration for each (DR, clusterer) pair."""
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
    
    summary = (
        metric_df.groupby(group_cols, as_index=False)[["ari", "v_measure", "nmi"]]
        .mean()
        .fillna({"v_measure": -np.inf, "nmi": -np.inf})
    )

    def pick_best(group: pd.DataFrame) -> pd.Series:
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
    best = best.sort_values(by="best_ari", ascending=False)
    
    return best


def analyze_method_performance(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze performance by method pair using peak and average metrics."""
    if df.empty or "ari" not in df:
        return pd.DataFrame(), pd.DataFrame()
    
    metric_df = df.dropna(subset=["ari"]).copy()
    if metric_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    method_cols = ["dim_reduction_method", "clustering_method"]
    
    peak = metric_df.groupby(method_cols, as_index=False).agg({
        "ari": ["max", "mean", "std", "count"],
        "v_measure": ["max", "mean"],
        "nmi": ["max", "mean"],
    })
    
    peak.columns = [
        "dim_reduction_method", "clustering_method",
        "peak_ari", "mean_ari", "std_ari", "n_configs",
        "peak_v_measure", "mean_v_measure",
        "peak_nmi", "mean_nmi"
    ]
    peak = peak.sort_values(by="peak_ari", ascending=False).reset_index(drop=True)
    
    avg = metric_df.groupby(method_cols, as_index=False).agg({
        "ari": ["mean", "std", "min", "max", "count"],
        "v_measure": ["mean", "std"],
        "nmi": ["mean", "std"],
    })
    
    avg.columns = [
        "dim_reduction_method", "clustering_method",
        "mean_ari", "std_ari", "min_ari", "max_ari", "n_configs",
        "mean_v_measure", "std_v_measure",
        "mean_nmi", "std_nmi"
    ]
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
        correlations[metric] = float(sub["ari"].corr(sub[metric]))
    return correlations


 


def create_block_diagonal_matrix(
    X: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Create block diagonal distance matrix visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sort by true labels
    true_order = np.argsort(labels_true)
    X_sorted_true = X[true_order]
    labels_sorted_true = labels_true[true_order]
    
    # Sort by predicted labels
    pred_order = np.argsort(labels_pred)
    X_sorted_pred = X[pred_order]
    labels_sorted_pred = labels_pred[pred_order]
    
    # Compute distance matrices (subsample if too large)
    max_samples = 500
    if len(X) > max_samples:
        idx_true = np.random.choice(len(X_sorted_true), max_samples, replace=False)
        idx_true = np.sort(idx_true)
        D_true = pairwise_distances(X_sorted_true[idx_true], metric='euclidean')
        labels_plot_true = labels_sorted_true[idx_true]
        
        idx_pred = np.random.choice(len(X_sorted_pred), max_samples, replace=False)
        idx_pred = np.sort(idx_pred)
        D_pred = pairwise_distances(X_sorted_pred[idx_pred], metric='euclidean')
        labels_plot_pred = labels_sorted_pred[idx_pred]
    else:
        D_true = pairwise_distances(X_sorted_true, metric='euclidean')
        labels_plot_true = labels_sorted_true
        D_pred = pairwise_distances(X_sorted_pred, metric='euclidean')
        labels_plot_pred = labels_sorted_pred
    
    # Plot ground truth
    im1 = axes[0].imshow(D_true, cmap='viridis', aspect='auto')
    axes[0].set_title('Distance Matrix (Ground Truth Order)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Spike Index')
    axes[0].set_ylabel('Spike Index')
    
    # Add cluster boundaries for ground truth
    boundaries_true = np.where(np.diff(labels_plot_true))[0] + 0.5
    for b in boundaries_true:
        axes[0].axhline(b, color='red', linewidth=1.5, alpha=0.7)
        axes[0].axvline(b, color='red', linewidth=1.5, alpha=0.7)
    
    plt.colorbar(im1, ax=axes[0], label='Euclidean Distance')
    
    # Plot predicted
    im2 = axes[1].imshow(D_pred, cmap='viridis', aspect='auto')
    axes[1].set_title('Distance Matrix (Predicted Order)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Spike Index')
    axes[1].set_ylabel('Spike Index')
    
    # Add cluster boundaries for predicted
    boundaries_pred = np.where(np.diff(labels_plot_pred))[0] + 0.5
    for b in boundaries_pred:
        axes[1].axhline(b, color='red', linewidth=1.5, alpha=0.7)
        axes[1].axvline(b, color='red', linewidth=1.5, alpha=0.7)
    
    plt.colorbar(im2, ax=axes[1], label='Euclidean Distance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_matrix_block_diagonal.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_bars(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar charts comparing performance metrics across methods."""
    
    # Get top 10 method combinations by ARI
    method_pairs = results_df.groupby(['dim_reduction_method', 'clustering_method']).agg({
        'ari': 'mean',
        'v_measure': 'mean',
        'nmi': 'mean',
        'silhouette': 'mean',
    }).reset_index()
    
    method_pairs['method_combo'] = (method_pairs['dim_reduction_method'] + 
                                     ' + ' + method_pairs['clustering_method'])
    method_pairs = method_pairs.sort_values('ari', ascending=False).head(10)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics Comparison (Top 10 Method Combinations)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [('ari', 'Adjusted Rand Index', axes[0, 0]),
               ('v_measure', 'V-Measure', axes[0, 1]),
               ('nmi', 'Normalized Mutual Information', axes[1, 0]),
               ('silhouette', 'Silhouette Score', axes[1, 1])]
    
    for metric, title, ax in metrics:
        data = method_pairs.sort_values(metric, ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(data)))
        
        bars = ax.barh(range(len(data)), data[metric], color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['method_combo'], fontsize=9)
        ax.set_xlabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, data[metric])):
            if not np.isnan(val):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_method_heatmap(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap of method performance."""
    
    # Pivot to get DR methods vs clustering methods
    pivot = results_df.pivot_table(
        values='ari',
        index='dim_reduction_method',
        columns='clustering_method',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Mean ARI'},
                ax=ax)
    ax.set_title('Mean ARI by Method Combination', fontsize=14, fontweight='bold')
    ax.set_xlabel('Clustering Method', fontsize=12)
    ax.set_ylabel('Dimensionality Reduction', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_timing_plots(timing_info: Dict[str, float], output_dir: Path) -> None:
    """Create timing analysis plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of timing
    methods = list(timing_info.keys())
    times = list(timing_info.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    axes[0].barh(methods, times, color=colors)
    axes[0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Computation Time by Method Group', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    for i, (method, time_val) in enumerate(zip(methods, times)):
        axes[0].text(time_val, i, f' {time_val:.2f}s', va='center', fontsize=9)
    
    # Pie chart of time proportion
    axes[1].pie(times, labels=methods, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Computation Time Proportion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'computation_timing.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_distribution(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create distribution plots for various metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Distribution Across All Experiments', fontsize=14, fontweight='bold')
    
    metrics = ['ari', 'v_measure', 'nmi', 'silhouette']
    titles = ['ARI Distribution', 'V-Measure Distribution', 
              'NMI Distribution', 'Silhouette Score Distribution']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        data = results_df[metric].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {data.median():.3f}')
            ax.set_xlabel(metric.upper(), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_scatter(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plots showing correlation between supervised and unsupervised metrics."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Supervised vs Unsupervised Metrics Correlation', 
                 fontsize=14, fontweight='bold')
    
    unsupervised = [('silhouette', 'Silhouette Score'),
                    ('davies_bouldin', 'Davies-Bouldin Index'),
                    ('calinski_harabasz', 'Calinski-Harabasz Index')]
    
    for ax, (metric, label) in zip(axes, unsupervised):
        data = results_df[['ari', metric]].dropna()
        
        if len(data) > 5:
            ax.scatter(data[metric], data['ari'], alpha=0.5, s=30)
            
            # Add trend line
            z = np.polyfit(data[metric], data['ari'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[metric].min(), data[metric].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = data['ari'].corr(data[metric])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('ARI', fontsize=11)
            ax.set_title(f'ARI vs {label}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    """Enhanced main function with improved logging."""
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.base_seed + i for i in range(args.num_datasets)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("SPIKE SORTING PIPELINE COMPARISON", level=1)
    print(f"\n📋 Configuration:")
    print_metric("Datasets", len(seeds))
    print_metric("Units per dataset", args.num_units)
    print_metric("Recording duration", f"{args.subset_duration}s")
    print_metric("DR groups", ", ".join(args.dr_groups))
    print_metric("Output directory", str(output_dir))
    
    runner = ExperimentRunner(
        output_dir=str(output_dir),
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
        silent_errors=args.silent_errors,
    )
    
    enabled_methods = resolve_enabled_methods(args.dr_groups)
    has_deeplearning = bool(enabled_methods & DR_GROUPS["deeplearning"])
    
    all_results: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []
    timing_info: Dict[str, List[float]] = {"linear": [], "nonlinear": [], "deeplearning": []}
    
    # Track for visualization
    last_X_pca = None
    last_labels_true = None
    last_labels_pred = None
    last_best_config = None
    
    # Process datasets
    print_header(f"Processing {len(seeds)} Dataset(s)", level=2)
    
    for idx, seed in enumerate(seeds):
        print_header(f"Dataset {idx + 1}/{len(seeds)} (seed={seed})", level=3)
        
        dataset_start = time.time()
        
        # Prepare dataset
        dataset = prepare_dataset(idx, seed, args)
        print_metric("Spikes detected", dataset["n_spikes"])
        print_metric("True clusters", dataset["true_k"])
        
        # Determine d_star values
        d_star_values = args.d_star_values if args.d_star_values else [None]
        
        for d_star_test in d_star_values:
            if d_star_test is not None:
                print(f"\n   Testing d_star = {d_star_test}")
            
            # Preprocess
            if has_deeplearning:
                print("   Using train/test split for deep learning...")
                train_data, test_data = split_and_preprocess_dataset(
                    X_raw=dataset["X_raw"],
                    labels=dataset["labels"],
                    args=args,
                    seed=seed,
                    d_star_override=d_star_test,
                )
                use_train_only = True
            else:
                train_data = preprocess_dataset_no_split(
                    X_raw=dataset["X_raw"],
                    labels=dataset["labels"],
                    args=args,
                    seed=seed,
                    d_star_override=d_star_test,
                )
                use_train_only = False
            
            # Store metadata
            dataset_summary = {
                "dataset_name": dataset["dataset_name"],
                "seed": seed,
                "n_spikes": dataset["n_spikes"],
                "n_features": dataset["n_features"],
                "d_star": train_data["d_star"],
                "true_k": dataset["true_k"],
            }
            dataset_summaries.append(dataset_summary)
            
            # Build grids
            neighbor_vals = _neighbor_candidates(train_data["n_spikes"])
            
            dim_grid, skipped_dr_configs = create_dimensionality_grid(
                d_star=train_data["d_star"],
                X_reference=train_data["X_pca"],
                include_supervised=train_data["labels"] is not None,
                rng=np.random.default_rng(seed + 17),
                neighbor_vals=neighbor_vals,
            )
            dim_grid = filter_dimensionality_grid(dim_grid, enabled_methods)
            
            clustering_grid = create_clustering_grid(
                dataset["true_k"],
                train_data["distance_scale"],
                neighbor_vals,
                train_data["n_spikes"],
            )
            
            # Run experiments with timing
            config = {"dimensionality_reduction": dim_grid, "clustering": clustering_grid}
            split_suffix = "_train" if use_train_only else ""
            train_dataset_info = {
                "dataset_name": dataset["dataset_name"] + split_suffix,
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
            
            # Track timing by method group
            linear_methods = enabled_methods & DR_GROUPS["linear"]
            nonlinear_methods = enabled_methods & DR_GROUPS["nonlinear"]
            deeplearning_methods = enabled_methods & DR_GROUPS["deeplearning"]
            
            if linear_methods:
                start_t = time.time()
                linear_config = {
                    "dimensionality_reduction": {k: v for k, v in dim_grid.items() if k in linear_methods},
                    "clustering": clustering_grid
                }
                if linear_config["dimensionality_reduction"]:
                    linear_results = runner.run_experiments(
                        X=train_dataset_info["X_raw"],
                        config=linear_config,
                        y=train_dataset_info["labels"],
                        dataset_name=train_dataset_info["dataset_name"],
                    )
                    all_results.extend(linear_results)
                    timing_info["linear"].append(time.time() - start_t)
            
            if nonlinear_methods or deeplearning_methods:
                combined_methods = nonlinear_methods | deeplearning_methods
                start_t = time.time()
                combined_config = {
                    "dimensionality_reduction": {k: v for k, v in dim_grid.items() if k in combined_methods},
                    "clustering": clustering_grid
                }
                if combined_config["dimensionality_reduction"]:
                    combined_results = runner.run_experiments(
                        X=train_dataset_info["X_pca"],
                        config=combined_config,
                        y=train_dataset_info["labels"],
                        dataset_name=train_dataset_info["dataset_name"],
                    )
                    all_results.extend(combined_results)
                    elapsed = time.time() - start_t
                    if nonlinear_methods:
                        timing_info["nonlinear"].append(elapsed)
                    if deeplearning_methods:
                        timing_info["deeplearning"].append(elapsed)
            
            # Add metadata to results
            for result in all_results[-len(all_results):]:  # Only process new results
                result["dataset_idx"] = dataset["dataset_idx"]
                result["dataset_seed"] = dataset["seed"]
                result["d_star"] = train_data["d_star"]
                result["true_k"] = dataset["true_k"]
                result["n_spikes"] = train_data["n_spikes"]
                result["n_features_raw"] = dataset["n_features"]
                result["n_features_preprocessed"] = train_data["X_pca"].shape[1]
            
            # Save for visualization
            if has_deeplearning:
                last_X_pca = np.vstack([train_data["X_pca"], test_data["X_pca"]])
                last_labels_true = np.concatenate([train_data["labels"], test_data["labels"]])
            else:
                last_X_pca = train_data["X_pca"]
                last_labels_true = train_data["labels"]
        
        dataset_time = time.time() - dataset_start
        print_metric("Dataset processing time", f"{dataset_time:.2f}s")
    
    # Save results
    print_header("Saving Results", level=2)
    runner.save_results(filename=args.results_name)
    
    # Analysis
    results_df = build_results_dataframe(all_results)
    best_configs = select_best_configs(results_df)
    peak_performance, avg_performance = analyze_method_performance(results_df)
    correlations = compute_unsupervised_correlations(results_df)
    
    # Save CSVs
    if not best_configs.empty:
        best_configs.to_csv(output_dir / "best_configs.csv", index=False)
        print_metric("Best configs saved", "best_configs.csv")
    
    if not peak_performance.empty:
        peak_performance.to_csv(output_dir / "peak_performance.csv", index=False)
        print_metric("Peak performance saved", "peak_performance.csv")
    
    if not avg_performance.empty:
        avg_performance.to_csv(output_dir / "average_performance.csv", index=False)
        print_metric("Average performance saved", "average_performance.csv")
    
    if not results_df.empty:
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        print_metric("All results saved", "all_results.csv")
    
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summaries, f, indent=2)
    print_metric("Metadata saved", "dataset_metadata.json")
    
    if correlations:
        with open(output_dir / "unsupervised_metric_correlations.json", "w", encoding="utf-8") as f:
            json.dump(correlations, f, indent=2)
        print_metric("Correlations saved", "unsupervised_metric_correlations.json")
    
    # Print summary
    print_header("Performance Summary", level=2)
    if not peak_performance.empty:
        print("\n🏆 Top 5 Method Combinations (by Peak ARI):")
        top5 = peak_performance.head(5)
        for i, row in top5.iterrows():
            print(f"\n  {i+1}. {row['dim_reduction_method']} + {row['clustering_method']}")
            print_metric("Peak ARI", row['peak_ari'], indent=6)
            print_metric("Mean ARI", row['mean_ari'], indent=6)
            print_metric("Configurations tested", int(row['n_configs']), indent=6)
    
    # Visualizations
    if not results_df.empty and last_X_pca is not None and last_labels_true is not None:
        try:
            if not best_configs.empty:
                top = best_configs.iloc[0]
                dim_method = top['dim_reduction_method']
                clust_method = top['clustering_method']
                dim_params = top['dim_reduction_params']
                clust_params = top['clustering_params']
                reducer = create_reducer(dim_method, **dim_params)
                X_emb = reducer.fit_transform(last_X_pca, last_labels_true)
                if X_emb is None:
                    raise RuntimeError("Reducer returned None embedding")
                viz_indices = getattr(reducer, 'test_indices', None)
                if viz_indices is None:
                    viz_indices = np.arange(len(last_X_pca))
                X_base = last_X_pca[viz_indices]
                y_true_viz = last_labels_true[viz_indices]
                clusterer = create_clustering(clust_method, **clust_params)
                labels_pred = clusterer.fit_predict(X_emb)
                best_cfg = {
                    'dim_reduction_method': dim_method,
                    'clustering_method': clust_method,
                    'ari': float(top.get('best_ari', np.nan)),
                    'dim_params': dim_params,
                    'clust_params': clust_params,
                }
                timing_avg = {k: np.mean(v) if v else 0 for k, v in timing_info.items() if v}
                create_enhanced_visualizations(
                    X_pca=X_base,
                    labels_true=y_true_viz,
                    labels_pred=labels_pred,
                    results_df=results_df,
                    best_config=best_cfg,
                    output_dir=output_dir / "visualizations",
                    timing_info=timing_avg if timing_avg else None,
                )
        except Exception as e:
            print(f"[Visualization] Enhanced plots failed: {e}")
    
    
    print_header("Pipeline Comparison Complete! ✓", level=1)
    print(f"\n📁 Results saved to: {output_dir}")
    viz_dir = output_dir / 'visualizations'
    if viz_dir.exists():
        print(f"📊 Visualizations saved to: {viz_dir}\n")


 


def create_enhanced_visualizations(
    X_pca: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    results_df: pd.DataFrame,
    best_config: Dict[str, Any],
    output_dir: Path,
    timing_info: Optional[Dict[str, float]] = None,
) -> None:
    """Create comprehensive visualization suite."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print_header("Generating Visualizations", level=3)
    
    # Set up color schemes
    n_true = len(np.unique(labels_true))
    n_pred = len(np.unique(labels_pred))
    colors_true = plt.cm.tab20(np.linspace(0, 1, n_true))
    colors_pred = plt.cm.tab20(np.linspace(0, 1, n_pred))
    
    # 1. 2D Cluster Comparison (Ground Truth vs Predicted)
    print("  • Creating 2D cluster comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Reduce to 2D for visualization
    if X_pca.shape[1] > 2:
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_pca)
        var_explained = pca_2d.explained_variance_ratio_.sum()
    else:
        X_2d = X_pca
        var_explained = 1.0
    
    # Ground truth
    for i, label in enumerate(np.unique(labels_true)):
        mask = labels_true == label
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors_true[i]], label=f'Unit {label}',
                       alpha=0.6, s=20, edgecolors='none')
    axes[0].set_title('Ground Truth Clusters', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({var_explained*100:.1f}% var explained)')
    axes[0].set_ylabel('PC2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Predicted
    for i, label in enumerate(np.unique(labels_pred)):
        mask = labels_pred == label
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors_pred[i]], label=f'Cluster {label}',
                       alpha=0.6, s=20, edgecolors='none')
    axes[1].set_title(f'Predicted Clusters (ARI: {best_config.get("ari", 0):.3f})', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({var_explained*100:.1f}% var explained)')
    axes[1].set_ylabel('PC2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_comparison_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Block Diagonal Distance Matrix
    print("  • Creating block diagonal distance matrix...")
    create_block_diagonal_matrix(X_pca, labels_true, labels_pred, output_dir)
    
    # 3. Performance Metrics Bar Chart
    print("  • Creating performance metrics comparison...")
    create_performance_bars(results_df, output_dir)
    
    # 4. Method Comparison Heatmap
    print("  • Creating method comparison heatmap...")
    create_method_heatmap(results_df, output_dir)
    
    # 5. Computation Time Analysis
    if timing_info:
        print("  • Creating timing analysis...")
        create_timing_plots(timing_info, output_dir)
    
    # 6. Metrics Distribution
    print("  • Creating metrics distribution plots...")
    create_metrics_distribution(results_df, output_dir)
    
    # 7. ARI vs Silhouette Scatter
    print("  • Creating ARI vs unsupervised metrics scatter...")
    create_correlation_scatter(results_df, output_dir)
    
    print(f"\n  ✓ All visualizations saved to: {output_dir}")


def create_block_diagonal_matrix(
    X: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Create block diagonal distance matrix visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sort by true labels
    true_order = np.argsort(labels_true)
    X_sorted_true = X[true_order]
    labels_sorted_true = labels_true[true_order]
    
    # Sort by predicted labels
    pred_order = np.argsort(labels_pred)
    X_sorted_pred = X[pred_order]
    labels_sorted_pred = labels_pred[pred_order]
    
    # Compute distance matrices (subsample if too large)
    max_samples = 500
    if len(X) > max_samples:
        idx_true = np.random.choice(len(X_sorted_true), max_samples, replace=False)
        idx_true = np.sort(idx_true)
        D_true = pairwise_distances(X_sorted_true[idx_true], metric='euclidean')
        labels_plot_true = labels_sorted_true[idx_true]
        
        idx_pred = np.random.choice(len(X_sorted_pred), max_samples, replace=False)
        idx_pred = np.sort(idx_pred)
        D_pred = pairwise_distances(X_sorted_pred[idx_pred], metric='euclidean')
        labels_plot_pred = labels_sorted_pred[idx_pred]
    else:
        D_true = pairwise_distances(X_sorted_true, metric='euclidean')
        labels_plot_true = labels_sorted_true
        D_pred = pairwise_distances(X_sorted_pred, metric='euclidean')
        labels_plot_pred = labels_sorted_pred
    
    # Plot ground truth
    im1 = axes[0].imshow(D_true, cmap='viridis', aspect='auto')
    axes[0].set_title('Distance Matrix (Ground Truth Order)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Spike Index')
    axes[0].set_ylabel('Spike Index')
    
    # Add cluster boundaries for ground truth
    boundaries_true = np.where(np.diff(labels_plot_true))[0] + 0.5
    for b in boundaries_true:
        axes[0].axhline(b, color='red', linewidth=1.5, alpha=0.7)
        axes[0].axvline(b, color='red', linewidth=1.5, alpha=0.7)
    
    plt.colorbar(im1, ax=axes[0], label='Euclidean Distance')
    
    # Plot predicted
    im2 = axes[1].imshow(D_pred, cmap='viridis', aspect='auto')
    axes[1].set_title('Distance Matrix (Predicted Order)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Spike Index')
    axes[1].set_ylabel('Spike Index')
    
    # Add cluster boundaries for predicted
    boundaries_pred = np.where(np.diff(labels_plot_pred))[0] + 0.5
    for b in boundaries_pred:
        axes[1].axhline(b, color='red', linewidth=1.5, alpha=0.7)
        axes[1].axvline(b, color='red', linewidth=1.5, alpha=0.7)
    
    plt.colorbar(im2, ax=axes[1], label='Euclidean Distance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_matrix_block_diagonal.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_bars(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar charts comparing performance metrics across methods."""
    
    # Get top 10 method combinations by ARI
    method_pairs = results_df.groupby(['dim_reduction_method', 'clustering_method']).agg({
        'ari': 'mean',
        'v_measure': 'mean',
        'nmi': 'mean',
        'silhouette': 'mean',
    }).reset_index()
    
    method_pairs['method_combo'] = (method_pairs['dim_reduction_method'] + 
                                     ' + ' + method_pairs['clustering_method'])
    method_pairs = method_pairs.sort_values('ari', ascending=False).head(10)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics Comparison (Top 10 Method Combinations)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [('ari', 'Adjusted Rand Index', axes[0, 0]),
               ('v_measure', 'V-Measure', axes[0, 1]),
               ('nmi', 'Normalized Mutual Information', axes[1, 0]),
               ('silhouette', 'Silhouette Score', axes[1, 1])]
    
    for metric, title, ax in metrics:
        data = method_pairs.sort_values(metric, ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(data)))
        
        bars = ax.barh(range(len(data)), data[metric], color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['method_combo'], fontsize=9)
        ax.set_xlabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, data[metric])):
            if not np.isnan(val):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_method_heatmap(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap of method performance."""
    
    # Pivot to get DR methods vs clustering methods
    pivot = results_df.pivot_table(
        values='ari',
        index='dim_reduction_method',
        columns='clustering_method',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Mean ARI'},
                ax=ax)
    ax.set_title('Mean ARI by Method Combination', fontsize=14, fontweight='bold')
    ax.set_xlabel('Clustering Method', fontsize=12)
    ax.set_ylabel('Dimensionality Reduction', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_timing_plots(timing_info: Dict[str, float], output_dir: Path) -> None:
    """Create timing analysis plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of timing
    methods = list(timing_info.keys())
    times = list(timing_info.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    axes[0].barh(methods, times, color=colors)
    axes[0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Computation Time by Method Group', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    for i, (method, time_val) in enumerate(zip(methods, times)):
        axes[0].text(time_val, i, f' {time_val:.2f}s', va='center', fontsize=9)
    
    # Pie chart of time proportion
    axes[1].pie(times, labels=methods, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Computation Time Proportion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'computation_timing.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_distribution(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create distribution plots for various metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Distribution Across All Experiments', fontsize=14, fontweight='bold')
    
    metrics = ['ari', 'v_measure', 'nmi', 'silhouette']
    titles = ['ARI Distribution', 'V-Measure Distribution', 
              'NMI Distribution', 'Silhouette Score Distribution']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        data = results_df[metric].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {data.median():.3f}')
            ax.set_xlabel(metric.upper(), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_scatter(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plots showing correlation between supervised and unsupervised metrics."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Supervised vs Unsupervised Metrics Correlation', 
                 fontsize=14, fontweight='bold')
    
    unsupervised = [('silhouette', 'Silhouette Score'),
                    ('davies_bouldin', 'Davies-Bouldin Index'),
                    ('calinski_harabasz', 'Calinski-Harabasz Index')]
    
    for ax, (metric, label) in zip(axes, unsupervised):
        data = results_df[['ari', metric]].dropna()
        
        if len(data) > 5:
            ax.scatter(data[metric], data['ari'], alpha=0.5, s=30)
            
            # Add trend line
            z = np.polyfit(data[metric], data['ari'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[metric].min(), data[metric].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = data['ari'].corr(data[metric])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('ARI', fontsize=11)
            ax.set_title(f'ARI vs {label}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()