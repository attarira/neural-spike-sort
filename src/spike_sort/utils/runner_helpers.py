import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import json
import spikeinterface.full as si

from spike_sort import ResultsAnalyzer
from spike_sort.utils.preprocess import preprocess_recording


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


def print_header(title: str) -> None:
    """Print a standardized header for CLI runners."""
    print("=" * 70)
    print(title)
    print("=" * 70)


def create_synthetic_recording(num_units: int) -> Tuple[si.BaseRecording, si.BaseRecording, si.BaseSorting]:
    """Create a synthetic Neuropixels recording and ground truth sorting."""
    static_rec, drift_rec, gt_sorting = si.generate_drifting_recording(
        probe_name="Neuropixels1-384",
        num_units=num_units,
        duration=300,
        sampling_frequency=30000,
        seed=4776,
    )
    return static_rec, drift_rec, gt_sorting


def load_real_recording(recording_folder: str) -> si.BaseRecording:
    """Load a previously extracted recording from disk."""
    rec_path = Path(recording_folder)
    return si.load_extractor(rec_path)


def slice_recording(recording: si.BaseRecording,
                    duration_s: Optional[float],
                    sf: Optional[float] = None,
                    gt_sorting: Optional[si.BaseSorting] = None) -> Tuple[si.BaseRecording, Optional[si.BaseSorting]]:
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


def prepare_spike_features(recording: si.BaseRecording, threshold: float):
    """Run detection and waveform-based feature extraction.

    Returns:
        X: Feature matrix (n_spikes, n_features)
        peak_locations: Array of (sample_index, channel)
        meta: Preprocessing metadata dictionary
    """
    return preprocess_recording(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=threshold,
        exclude_sweep_ms=0.1,
        ms_before=0.6,
        ms_after=1.4,
        channels_per_spike=4,
        max_spikes_per_unit=1000,
        scale_features=True,
        n_jobs=1,
        verbose=False,
    )


def match_ground_truth(gt_sorting: si.BaseSorting,
                       peak_locations: np.ndarray,
                       recording: si.BaseRecording) -> np.ndarray:
    """Assign detected spikes to ground-truth units using a 1 ms tolerance.

    Args:
        gt_sorting: Ground-truth sorting extractor
        peak_locations: Array of (sample_index, channel)
        recording: Recording extractor (used for sampling frequency)

    Returns:
        Array of unit labels for each detected spike (-1 for unmatched)
    """
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


def save_metadata(output_dir: Path,
                  recording: si.BaseRecording,
                  meta: Dict[str, Any],
                  mode: str,
                  duration: Optional[float],
                  filename: str) -> Path:
    """Persist run metadata to a JSON file.

    Args:
        output_dir: Base output directory
        recording: Recording extractor
        meta: Preprocessing metadata
        mode: 'synthetic' or 'real'
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