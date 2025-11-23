import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import json
import spikeinterface.full as si
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from spike_sort import ResultsAnalyzer
from spike_sort.utils.preprocess import preprocess_recording
from spike_sort.dimensionality_reduction import create_reducer
from spike_sort.clustering import create_clustering


def setup_logging(debug: bool) -> None:
    """Configure logging for runners.

    Args:
        debug: When True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    import warnings
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
    except Exception:
        pass
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        from warnings import WarningMessage
        warnings.simplefilter("ignore", UserWarning)
    except Exception:
        pass
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("spikeinterface").setLevel(logging.INFO)
    logging.getLogger("sklearn").setLevel(logging.ERROR)
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def print_header(title: str) -> None:
    """Print a standardized header for CLI runners."""
    print("=" * 70)
    print(title)
    print("=" * 70)


def create_synthetic_recording(
    num_units:            int,
    duration:           float = 300.0,
    sampling_frequency: float = 30000.0,
    seed:                 int = 4776,
    probe_name:           str = "Neuropixels1-384"
) -> Tuple[si.BaseRecording, si.BaseRecording, si.BaseSorting]:
    """Generate synthetic recording with drift and ground truth.
    
    Args:
        num_units: Number of neurons to simulate
        duration: Recording length in seconds (default: 300)
        sampling_frequency: Sampling rate in Hz (default: 30000)
        seed: Random seed for reproducibility (default: 4776)
        probe_name: Name of probe (default: "Neuropixels1-384", 384 channels @ 30kHz)
    
    Returns:
        Tuple of (static_recording, drift_recording, ground_truth_sorting)
    """
    # Scale drift parameters based on duration to avoid assertion errors
    # Default t_start_drift is 60s, which fails for short recordings
    # Use 10% of duration as drift start, with a minimum of 0.5s
    t_start_drift = max(0.5, duration * 0.1)
    # Period should be reasonable relative to duration
    period_s = max(20.0, duration * 0.5)
    
    generate_displacement_vector_kwargs = {
        'displacement_sampling_frequency': 5.0,
        'drift_start_um': [0, 20],
        'drift_stop_um': [0, -20],
        'drift_step_um': 1,
        'motion_list': [{
            'drift_mode': 'zigzag',
            'non_rigid_gradient': None,
            't_start_drift': t_start_drift,
            't_end_drift': None,
            'period_s': period_s
        }]
    }
    
    static_rec, drift_rec, gt_sorting = si.generate_drifting_recording(
        probe_name         = probe_name,
        num_units          = num_units,
        duration           = duration,
        sampling_frequency = sampling_frequency,
        seed               = seed,
        generate_displacement_vector_kwargs = generate_displacement_vector_kwargs,
    )
    return static_rec, drift_rec, gt_sorting


def load_real_recording(recording_folder: str) -> si.BaseRecording:
    """Load a previously extracted recording from disk."""
    rec_path = Path(recording_folder)
    return si.load_extractor(rec_path)


def slice_recording(
    recording:          si.BaseRecording,
    duration_s:          Optional[float],
    sf:                  Optional[float] = None,
    gt_sorting: Optional[si.BaseSorting] = None
) -> Tuple[si.BaseRecording, Optional[si.BaseSorting]]:
    """Slice the recording (and optional ground truth) to a given duration.

    Args:
        recording: Input recording extractor
        duration_s: Desired duration in seconds (None to skip)
        sf: Sampling frequency to compute end frame (optional)
        gt_sorting: Ground-truth sorting (optional)

    Returns:
        Tuple of (sliced recording, sliced ground truth or None)
    """
    if duration_s is None:
        return recording, gt_sorting
    if sf is None:
        sf = recording.get_sampling_frequency()
    end_frame = int(duration_s * sf)
    recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
    if gt_sorting is not None:
        gt_sorting = gt_sorting.frame_slice(start_frame=0, end_frame=end_frame)
    return recording, gt_sorting


def print_recording_info(recording: si.BaseRecording, duration_s: Optional[float]) -> None:
    """Print basic recording metadata."""
    print("Recording ready:")
    print(f"  Duration: {duration_s} seconds")
    print(f"  Samples: {recording.get_num_frames():,}")
    print(f"  Channels: {recording.get_num_channels()}")
    print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")


def prepare_spike_features(recording: si.BaseRecording,
                           threshold: float,
                           apply_ibl_pipeline: bool = False,
                           synthetic_pipeline: bool = False,
                           correct_motion: bool = False,
                           motion_preset: str = "dredge",
                           cache_dir: Optional[str] = None,
                           save_motion: bool = False,
                           save_rec: bool = False,
                           job_kwargs: Optional[Dict[str, Any]] = None):
    """Run detection and waveform-based feature extraction.

    Returns:
        X: Feature matrix (n_spikes, n_features)
        peak_locations: Array of (sample_index, channel)
        meta: Preprocessing metadata dictionary
    """
    return preprocess_recording(
        recording,
        method              = "locally_exclusive",
        peak_sign           = "neg",
        detect_threshold    = threshold,
        exclude_sweep_ms    = 0.1,
        ms_before           = 0.6,
        ms_after            = 1.4,
        channels_per_spike  = 8,
        max_spikes_per_unit = 1000,
        max_spikes_global   = 5000,
        scale_features      = True,
        n_jobs              = 1,
        verbose             = False,
        apply_ibl_pipeline  = apply_ibl_pipeline,
        synthetic_pipeline  = synthetic_pipeline,
        correct_motion      = correct_motion,
        motion_preset       = motion_preset,
        cache_dir           = cache_dir,
        save_motion         = save_motion,
        save_rec            = save_rec,
        job_kwargs          = job_kwargs,
    )


def match_ground_truth(gt_sorting: si.BaseSorting,
                       peak_locations: np.ndarray,
                       recording: si.BaseRecording) -> np.ndarray:
    """Assign detected spikes to ground-truth units using a 1 ms tolerance."""
    sf = recording.get_sampling_frequency()
    tol = int(1.0 * sf / 1000)
    labels = np.full(len(peak_locations), -1, dtype=np.int32)
    units = gt_sorting.get_unit_ids()
    matched = 0
    for i, (s, c) in enumerate(peak_locations):
        best_u = -1
        best_d = 1e12
        for u in units:
            st = gt_sorting.get_unit_spike_train(unit_id=u)
            if st.size == 0:
                continue
            d = np.min(np.abs(st - s))
            if d < best_d and d <= tol:
                best_d = d
                best_u = u
        if best_u != -1:
            labels[i] = best_u
            matched += 1
    logging.debug(f"Matched spikes: {matched}/{len(peak_locations)} within {tol} samples tolerance")
    return labels


def align_true_k(config: Dict[str, Any], true_k: Optional[int]) -> Dict[str, Any]:
    """Inject ground-truth component count into fixed-k clusterers.

    Args:
        config: Combined dim/clust configuration dict
        true_k: Ground-truth number of units (None to skip)

    Returns:
        Updated configuration dict with fixed-k methods aligned
    """
    if true_k is None:
        return config
    clust = config.get("clustering", {})
    for name, params in list(clust.items()):
        if name in ("KMeans", "K-Means", "SpectralClustering", "Spectral", "Agglomerative"):
            clust[name] = [{**p, "n_clusters": true_k} for p in params] if params else [{"n_clusters": true_k}]
        elif name in ("GMM", "GaussianMixture", "DirichletProcess"):
            clust[name] = [{**p, "n_components": true_k} for p in params] if params else [{"n_components": true_k}]
        elif name == "HMM":
            clust[name] = [{**p, "n_components": true_k} for p in params] if params else [{"n_components": true_k}]
    config["clustering"] = clust
    logging.debug(f"Aligned true_k={true_k} into clustering config for fixed-k methods")
    return config


def print_spike_feature_info(X: np.ndarray) -> None:
    """Print basic information about the extracted spike features."""
    print("\nSpike features ready:")
    print(f"  Number of spikes: {X.shape[0]:,}")
    print(f"  Feature dimensionality: {X.shape[1]}")


def analyze_and_print(results: List[Dict[str, Any]]) -> None:
    """Print summary stats and top configurations by key metrics."""
    print("\n[Analysis] Summary Statistics:")
    analyzer = ResultsAnalyzer(results)
    print(analyzer.get_summary_statistics())
    successful_results = analyzer.get_successful_results()
    if successful_results:
        metrics = ["silhouette_score", "adjusted_rand_index", "davies_bouldin_index"]
        print(f"\nTop configurations by metrics:")
        for metric in metrics:
            best = analyzer.get_best_configurations(metric=metric, n_top=1)
            if best is not None and not best.empty:
                row = best.iloc[0]
                dim_params = row.get('dim_reduction_params', row.get('dim_params_str', '{}'))
                clust_params = row.get('clustering_params', row.get('clust_params_str', '{}'))
                print(f"  Best {metric}: {row[metric]:.4f}")
                print(f"    Dim. Reduction: {row['dim_reduction_method']} {dim_params}")
                print(f"    Clustering: {row['clustering_method']} {clust_params}")


def _parse_params_maybe(value: Any) -> Dict[str, Any]:
    """Parse parameters from various formats (dict, JSON string, or repr string)."""
    try:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            # Try JSON first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Try eval for dict repr strings like "{'n_components': 20}"
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    except Exception as e:
        logging.debug(f"Failed to parse params: {value}, error: {e}")
    
    return {}


def _find_best_config_by_mean_ari(results: List[Dict[str, Any]]) -> Optional[Tuple[str, str, Dict[str, Any], Dict[str, Any], float]]:
    """Find configuration with best mean ARI across all datasets.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        Tuple of (dim_method, clust_method, dim_params, clust_params, mean_ari) or None
    """
    import pandas as pd
    
    # Filter successful results with ARI
    successful = [
        r for r in results 
        if r.get('success') and r.get('evaluation') and 'adjusted_rand_index' in r.get('evaluation', {})
    ]
    
    if not successful:
        return None
    
    # Build dataframe for analysis
    rows = []
    for r in successful:
        rows.append({
            'dim_method': r.get('dim_reduction_method'),
            'clust_method': r.get('clustering_method'),
            'dim_params_str': json.dumps(r.get('dim_reduction_params', {}), sort_keys=True),
            'clust_params_str': json.dumps(r.get('clustering_params', {}), sort_keys=True),
            'ari': r['evaluation']['adjusted_rand_index'],
        })
    
    df = pd.DataFrame(rows)
    
    # Group by (method, params) and compute mean ARI
    grouped = df.groupby(['dim_method', 'clust_method', 'dim_params_str', 'clust_params_str'])['ari'].agg(['mean', 'count']).reset_index()
    
    # Find best mean ARI
    best_idx = grouped['mean'].idxmax()
    best_row = grouped.loc[best_idx]
    
    dim_method = best_row['dim_method']
    clust_method = best_row['clust_method']
    dim_params = json.loads(best_row['dim_params_str'])
    clust_params = json.loads(best_row['clust_params_str'])
    mean_ari = float(best_row['mean'])
    
    return (dim_method, clust_method, dim_params, clust_params, mean_ari)


def visualize_best_clusters(
    X: np.ndarray,
    y: Optional[np.ndarray],
    results: List[Dict[str, Any]],
    output_dir: Path,
    max_points: int = 50000,
    plot_3d: bool = True,
    recording: Optional[si.BaseRecording] = None,
    peak_locations: Optional[np.ndarray] = None,
    export_to_phy_flag: bool = False,
    launch_phy_gui: bool = False,
    job_kwargs: Optional[Dict[str, Any]] = None,
    use_mean_ari: bool = True,
) -> Optional[Path]:
    """Visualize best clustering results in 2D and optionally 3D.
    
    Args:
        X: Feature matrix
        y: Optional ground truth labels
        results: List of experiment results
        output_dir: Output directory
        max_points: Maximum points to plot (for performance)
        plot_3d: Whether to create 3D visualizations
        recording: Optional recording for Phy export
        peak_locations: Optional peak locations for Phy export
        export_to_phy_flag: Whether to export to Phy format
        launch_phy_gui: Whether to launch Phy GUI
        job_kwargs: Job parameters for Phy export
        use_mean_ari: If True, select best config by mean ARI across datasets (default)
        
    Returns:
        Path to output directory with plots
    """
    print("\n[Visualization] Creating plots...")
    
    # Determine metric to use
    metric = 'adjusted_rand_index' if (y is not None and len(y) > 0) else 'silhouette_score'
    
    # Determine best configuration
    if use_mean_ari and y is not None and len(y) > 0:
        # Use mean ARI across all datasets/runs
        best_config = _find_best_config_by_mean_ari(results)
        if best_config is None:
            print("[Visualization] No successful results to visualize")
            return None
        
        dim_method, clust_method, dim_params, clust_params, mean_ari = best_config
        print(f"  Best configuration (by mean ARI across datasets):")
        print(f"    Dim. Reduction: {dim_method} {dim_params}")
        print(f"    Clustering: {clust_method} {clust_params}")
        print(f"    Mean ARI: {mean_ari:.4f}")
    else:
        # Use single best result (original behavior)
        analyzer = ResultsAnalyzer(results)
        best = analyzer.get_best_configurations(metric=metric, n_top=1)
        
        if best is None or best.empty:
            print("[Visualization] No successful results to visualize")
            return None
        
        row = best.iloc[0]
        dim_method = row['dim_reduction_method']
        clust_method = row['clustering_method']
        
        # Parse parameters with fallback to original results
        dim_params = _parse_params_maybe(
            row.get('dim_reduction_params') or row.get('dim_params_str') or {}
        )
        clust_params = _parse_params_maybe(
            row.get('clustering_params') or row.get('clust_params_str') or {}
        )
        
        # Fallback to original results if params empty
        if not clust_params:
            for result in results:
                if (result.get('dim_reduction_method') == dim_method and 
                    result.get('clustering_method') == clust_method and
                    result.get('success', False)):
                    clust_params = result.get('clustering_params', {})
                    dim_params = result.get('dim_reduction_params', {})
                    break
        
        print(f"  Best configuration (by peak {metric}):")
        print(f"    Dim. Reduction: {dim_method} {dim_params}")
        print(f"    Clustering: {clust_method} {clust_params}")
        print(f"    Score: {row[metric]:.4f}")
    
    # Re-run best pipeline
    reducer = create_reducer(dim_method, **dim_params)
    if reducer is None:
        return None
    
    # Pass ground truth labels for supervised methods (e.g., CEED)
    X_reduced = reducer.fit_transform(X, y)
    if X_reduced is None:
        return None
    
    clusterer = create_clustering(clust_method, **clust_params)
    if clusterer is None:
        return None
    
    labels = clusterer.fit_predict(X_reduced)
    if labels is None:
        return None
    
    # Subsample if needed
    n = X_reduced.shape[0]
    y_sub = y
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        X_reduced = X_reduced[idx]
        labels = labels[idx]
        if y is not None and len(y) == n:
            y_sub = y[idx]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create 2D visualization using PCA
    print("  Creating 2D visualization...")
    plot_reducer_2d = create_reducer('PCA', n_components=2)
    X_2d = plot_reducer_2d.fit_transform(X_reduced, None)
    
    if X_2d is not None and X_2d.shape[1] >= 2:
        _create_2d_plot(
            X_2d, labels, y_sub, 
            dim_method, clust_method, 
            output_dir, metric
        )
    
    # 2. Create 3D visualization if requested
    if plot_3d and X_reduced.shape[1] >= 3:
        print("  Creating 3D visualization...")
        plot_reducer_3d = create_reducer('PCA', n_components=3)
        X_3d = plot_reducer_3d.fit_transform(X_reduced, None)
        
        if X_3d is not None and X_3d.shape[1] >= 3:
            _create_3d_plot(
                X_3d, labels, y_sub,
                dim_method, clust_method,
                output_dir
            )
    
    # 3. Create confusion matrix if ground truth available
    if y_sub is not None and len(y_sub) > 0:
        _create_confusion_matrix(
            labels, y_sub,
            dim_method, clust_method,
            output_dir
        )
    
    phy_path = None
    if export_to_phy_flag and recording is not None and peak_locations is not None:
        phy_path = export_best_clustering_to_phy(
            recording=recording,
            peak_locations=peak_locations,
            labels=labels,
            output_dir=output_dir,
            dim_method=dim_method,
            clust_method=clust_method,
            job_kwargs=job_kwargs,
        )
        if launch_phy_gui and phy_path is not None:
            launch_phy(phy_path, auto_launch=True)
    print(f"  Visualizations saved to: {output_dir}")
    return phy_path or output_dir


def _create_2d_plot(
    X_2d: np.ndarray,
    labels: np.ndarray,
    y: Optional[np.ndarray],
    dim_method: str,
    clust_method: str,
    output_dir: Path,
    metric: str
) -> None:
    """Create 2D scatter plots of clusters and ground truth."""
    unique_labels = np.unique(labels)
    
    # Plot predicted clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for lab in unique_labels:
        mask = labels == lab
        color = 'lightgray' if lab == -1 else None
        label = 'Noise' if lab == -1 else f'Cluster {lab}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  s=8, alpha=0.6, label=label, c=color, edgecolors='none')
    
    ax.set_title(f'{dim_method} + {clust_method}\n({len(unique_labels)} clusters)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if len(unique_labels) <= 20:
        ax.legend(markerscale=2, fontsize=9, frameon=True, 
                 loc='upper right', ncol=2 if len(unique_labels) > 10 else 1)
    
    plt.tight_layout()
    out_path = output_dir / f"clusters_2d_{dim_method}_{clust_method}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot ground truth if available
    if y is not None and len(y) == X_2d.shape[0]:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        unique_y = np.unique(y[y != -1])
        
        for lab in unique_y:
            mask = y == lab
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       s=8, alpha=0.6, label=f'Unit {lab}', edgecolors='none')
        
        ax2.set_title(f'Ground Truth ({len(unique_y)} units)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Component 1', fontsize=12)
        ax2.set_ylabel('Component 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        if len(unique_y) <= 20:
            ax2.legend(markerscale=2, fontsize=9, frameon=True,
                      loc='upper right', ncol=2 if len(unique_y) > 10 else 1)
        
        plt.tight_layout()
        out_gt = output_dir / f"ground_truth_2d_{dim_method}.png"
        fig2.savefig(out_gt, dpi=300, bbox_inches='tight')
        plt.close(fig2)


def _create_3d_plot(
    X_3d: np.ndarray,
    labels: np.ndarray,
    y: Optional[np.ndarray],
    dim_method: str,
    clust_method: str,
    output_dir: Path
) -> None:
    """Create 3D scatter plots of clusters and ground truth."""
    unique_labels = np.unique(labels)
    
    # Plot predicted clusters in 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for lab in unique_labels:
        mask = labels == lab
        color = 'lightgray' if lab == -1 else None
        label = 'Noise' if lab == -1 else f'Cluster {lab}'
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                  s=8, alpha=0.5, label=label, c=color, edgecolors='none')
    
    ax.set_title(f'{dim_method} + {clust_method} (3D)\n({len(unique_labels)} clusters)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=11)
    ax.set_ylabel('Component 2', fontsize=11)
    ax.set_zlabel('Component 3', fontsize=11)
    
    if len(unique_labels) <= 15:
        ax.legend(markerscale=2, fontsize=8, frameon=True, loc='upper right')
    
    plt.tight_layout()
    out_path = output_dir / f"clusters_3d_{dim_method}_{clust_method}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot ground truth in 3D if available
    if y is not None and len(y) == X_3d.shape[0]:
        fig2 = plt.figure(figsize=(12, 9))
        ax2 = fig2.add_subplot(111, projection='3d')
        unique_y = np.unique(y[y != -1])
        
        for lab in unique_y:
            mask = y == lab
            ax2.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                       s=8, alpha=0.5, label=f'Unit {lab}', edgecolors='none')
        
        ax2.set_title(f'Ground Truth (3D) - {len(unique_y)} units',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Component 1', fontsize=11)
        ax2.set_ylabel('Component 2', fontsize=11)
        ax2.set_zlabel('Component 3', fontsize=11)
        
        if len(unique_y) <= 15:
            ax2.legend(markerscale=2, fontsize=8, frameon=True, loc='upper right')
        
        plt.tight_layout()
        out_gt = output_dir / f"ground_truth_3d_{dim_method}.png"
        fig2.savefig(out_gt, dpi=300, bbox_inches='tight')
        plt.close(fig2)


def _create_confusion_matrix(
    labels: np.ndarray,
    y: np.ndarray,
    dim_method: str,
    clust_method: str,
    output_dir: Path
) -> None:
    """Create confusion matrix comparing predicted clusters to ground truth."""
    try:
        from sklearn.metrics import confusion_matrix, adjusted_rand_score
        
        # Filter to valid ground truth
        valid = y != -1
        if not np.any(valid):
            return
        
        pred = labels[valid]
        gt = y[valid]
        
        # Compute confusion matrix
        cm = confusion_matrix(gt, pred)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax.set_xlabel('Predicted Cluster', fontsize=12)
        ax.set_ylabel('Ground Truth Unit', fontsize=12)
        ax.set_title(f'Confusion Matrix: {dim_method} + {clust_method}',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Count', fontsize=11)
        
        # Add text annotations for small matrices
        if cm.shape[0] <= 20 and cm.shape[1] <= 20:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = ax.text(j, i, str(cm[i, j]),
                                 ha="center", va="center",
                                 color="black" if cm[i, j] < cm.max()/2 else "white",
                                 fontsize=8)
        
        plt.tight_layout()
        out_cm = output_dir / f"confusion_matrix_{dim_method}_{clust_method}.png"
        fig.savefig(out_cm, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Compute and save metrics
        ari = float(adjusted_rand_score(gt, pred))
        metrics = {
            "adjusted_rand_index": ari,
            "n_ground_truth_units": int(len(np.unique(gt))),
            "n_predicted_clusters": int(len(np.unique(pred))),
            "n_spikes_evaluated": int(len(gt))
        }
        
        metrics_path = output_dir / f"metrics_{dim_method}_{clust_method}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Adjusted Rand Index: {ari:.4f}")
        
    except Exception as e:
        logging.warning(f"Failed to create confusion matrix: {e}")


def save_metadata(
    output_dir: Path,
    recording: si.BaseRecording,
    meta: Dict[str, Any],
    mode: str,
    duration: Optional[float],
    filename: str
) -> Path:
    """Persist run metadata to a JSON file.

    Args:
        output_dir: Base output directory
        recording: Recording extractor
        meta: Preprocessing metadata
        mode: Type of recording (ie., 'real_no_gt' or 'synthetic_with_drift' etc.)
        duration: Subset duration in seconds
        filename: Target JSON filename

    Returns:
        Path to the saved metadata JSON file
    """
    meta_out = {
        "recording": {
            "mode": mode,
            "duration": duration,
            "sampling_frequency": recording.get_sampling_frequency(),
            "channels": recording.get_num_channels(),
        },
        "preprocessing": meta,
    }
    meta_path = Path(output_dir) / filename
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    print(f"Metadata saved to: {meta_path}")
    return meta_path


def export_best_clustering_to_phy(
    recording: si.BaseRecording,
    peak_locations: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    dim_method: str,
    clust_method: str,
    job_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    try:
        sf = recording.get_sampling_frequency()
        valid = labels >= 0
        if not np.any(valid):
            return None
        unit_ids = np.sort(np.unique(labels[valid])).tolist()
        times_list = []
        for uid in unit_ids:
            times = peak_locations[labels == uid, 0].astype(np.int64)
            times_list.append(times)
        sorting = se.NumpySorting(times_list=times_list, sampling_frequency=sf, unit_ids=unit_ids)
        phy_dir = Path(output_dir) / f"phy_{dim_method}_{clust_method}"
        phy_dir.mkdir(parents=True, exist_ok=True)
        kwargs = job_kwargs or {}
        si.export_to_phy(
            recording=recording,
            sorting=sorting,
            output_folder=str(phy_dir),
            **kwargs,
        )
        print(f"Phy export saved to: {phy_dir}")
        return phy_dir
    except Exception as e:
        logging.warning(f"Failed to export to Phy: {e}")
        return None


def launch_phy(output_dir: Path, auto_launch: bool = True) -> None:
    try:
        params = Path(output_dir) / "params.py"
        if auto_launch:
            subprocess.Popen(["phy", "template-gui", str(params)])
        else:
            print(f"Run: phy template-gui {params}")
    except Exception as e:
        logging.warning(f"Failed to launch Phy: {e}")
        print(f"Run: phy template-gui {Path(output_dir) / 'params.py'}")
