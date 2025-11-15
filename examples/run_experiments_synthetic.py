"""
Improved spike sorting script with proper spike detection and feature extraction.

This script demonstrates the correct approach to spike sorting:
1. Generate synthetic recordings
2. Detect spikes (not just cluster raw traces)
3. Extract spike waveforms as features
4. Run dimensionality reduction and clustering on spike features
5. Compare results with ground truth
"""

import numpy as np
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils import save_config

# Import SpikeInterface
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spike_sort.utils.preprocess import preprocess_recording


def detect_and_extract_spikes(
    recording,
    method='locally_exclusive',
    peak_sign='neg',
    detect_threshold=5.0,
    exclude_sweep_ms=0.1,
    ms_before=0.6,
    ms_after=1.4,
    channels_per_spike=4,
    max_spikes_per_unit=1000,
    verbose=True
):
    """
    Detect spikes and extract waveform features.
    
    Parameters:
    -----------
    recording : BaseRecording
        SpikeInterface recording object
    method : str
        Peak detection method ('locally_exclusive', 'by_channel', etc.)
    peak_sign : str
        'neg', 'pos', or 'both'
    detect_threshold : float
        Detection threshold in MAD (median absolute deviation) units
    exclude_sweep_ms : float
        Exclusion period around detected peaks in milliseconds
    ms_before : float
        Time before peak to extract (milliseconds)
    ms_after : float
        Time after peak to extract (milliseconds)
    max_spikes_per_unit : int
        Maximum spikes to extract (for memory management)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    spike_features : np.ndarray
        Array of shape (n_spikes, n_features) containing spike waveforms
    peak_locations : np.ndarray
        Array of detected peak locations (sample_index, channel_index)
    metadata : dict
        Information about spike detection and extraction
    """
    
    if verbose:
        print("\n" + "="*70)
        print("Spike Detection and Feature Extraction")
        print("="*70)
    
    # Step 1: Detect peaks
    if verbose:
        print(f"\nDetecting spikes...")
        print(f"  Method: {method}")
        print(f"  Threshold: {detect_threshold} MAD")
        print(f"  Peak sign: {peak_sign}")
    
    peaks = detect_peaks(
        recording,
        method=method,
        peak_sign=peak_sign,
        detect_threshold=detect_threshold,
        exclude_sweep_ms=exclude_sweep_ms,
        n_jobs=1,
        progress_bar=verbose
    )
    
    n_peaks = len(peaks)
    if verbose:
        print(f"\nDetected {n_peaks:,} spikes")
    
    if n_peaks == 0:
        raise ValueError("No spikes detected! Try lowering the detection threshold.")
    
    # Limit number of spikes for memory management
    if n_peaks > max_spikes_per_unit * 200:  # Assume max 200 units
        if verbose:
            print(f"  Too many spikes detected. Randomly sampling {max_spikes_per_unit * 200:,} spikes...")
        indices = np.random.choice(n_peaks, max_spikes_per_unit * 200, replace=False)
        indices = np.sort(indices)
        peaks = peaks[indices]
        n_peaks = len(peaks)
    
    # Step 2: Extract waveforms around each peak
    if verbose:
        print(f"\nExtracting waveform features...")
        print(f"  Window: {ms_before}ms before, {ms_after}ms after peak")
    
    # Calculate number of samples for waveform window
    sampling_frequency = recording.get_sampling_frequency()
    nbefore = int(ms_before * sampling_frequency / 1000)
    nafter = int(ms_after * sampling_frequency / 1000)
    
    if verbose:
        print(f"  Waveform length: {nbefore + nafter} samples")
    
    # Extract waveforms
    # We'll extract from a few channels around the peak channel
    n_channels = recording.get_num_channels()
    channels_per_spike = min(channels_per_spike, n_channels)
    spike_waveforms = []
    valid_peaks = []
    n_frames = recording.get_num_frames()
    segment_span = int(0.066 * sampling_frequency)
    order = np.argsort(peaks['sample_index'])
    peaks_sorted = peaks[order]
    i = 0
    while i < n_peaks:
        start = peaks_sorted[i]['sample_index'] - nbefore
        if start < 0:
            start = 0
        end = peaks_sorted[i]['sample_index'] + nafter
        if end > n_frames:
            end = n_frames
        j = i + 1
        while j < n_peaks:
            s = peaks_sorted[j]['sample_index'] - nbefore
            e = peaks_sorted[j]['sample_index'] + nafter
            if e - start <= segment_span:
                if s < start:
                    start = s
                if e > end:
                    end = e
                j += 1
            else:
                break
        traces = recording.get_traces(
            start_frame=start,
            end_frame=end,
            channel_ids=recording.channel_ids
        )
        for k in range(i, j):
            sample_idx = peaks_sorted[k]['sample_index']
            channel_idx = peaks_sorted[k]['channel_index']
            if sample_idx < nbefore or sample_idx + nafter >= n_frames:
                continue
            offset = sample_idx - start
            amp_row = traces[offset, :]
            top_idx = np.argsort(np.abs(amp_row))[-channels_per_spike:][::-1]
            w_start = offset - nbefore
            w_end = offset + nafter
            if w_start < 0 or w_end > traces.shape[0]:
                continue
            center_col = channel_idx
            local = np.argmin(traces[w_start:w_end, center_col])
            shift = local - nbefore
            w_start_s = w_start + shift
            w_end_s = w_end + shift
            if w_start_s < 0 or w_end_s > traces.shape[0]:
                w_start_s = w_start
                w_end_s = w_end
            wf = traces[w_start_s:w_end_s, top_idx]
            v = wf.reshape(-1)
            m = v.mean()
            sdev = v.std() + 1e-8
            v = (v - m) / sdev
            min_amp = float(np.min(wf))
            energy = float(np.sum(wf * wf))
            v = np.concatenate([v, np.array([min_amp, energy], dtype=v.dtype)])
            spike_waveforms.append(v)
            valid_peaks.append(peaks_sorted[k])
        if verbose and j % 10000 == 0:
            print(f"  Processed {j:,} / {n_peaks:,} spikes...")
        i = j
    
    spike_features = np.array(spike_waveforms, dtype=np.float32)
    peak_locations = np.array([(p['sample_index'], p['channel_index']) for p in valid_peaks])
    
    if verbose:
        print(f"\nExtracted features from {len(spike_features):,} spikes")
        print(f"  Feature shape: {spike_features.shape}")
        print(f"  Feature dimensionality: {spike_features.shape[1]}")
        print(f"  Memory size: {spike_features.nbytes / 1e6:.2f} MB")
    
    metadata = {
        'n_spikes': len(spike_features),
        'n_features': spike_features.shape[1],
        'waveform_length': nbefore + nafter,
        'channels_per_spike': channels_per_spike,
        'ms_before': ms_before,
        'ms_after': ms_after,
        'detect_threshold': detect_threshold,
        'detection_method': method,
        'peak_sign': peak_sign,
        'sampling_frequency': sampling_frequency
    }
    
    return spike_features, peak_locations, metadata


def create_ground_truth_labels(gt_sorting, peak_locations, recording):
    """
    Create ground truth labels for detected spikes by matching to true spike times.
    
    Parameters:
    -----------
    gt_sorting : BaseSorting
        Ground truth sorting object
    peak_locations : np.ndarray
        Detected peak locations (sample_index, channel_index)
    recording : BaseRecording
        Recording object
        
    Returns:
    --------
    labels : np.ndarray
        Ground truth unit labels for each detected spike (-1 for unmatched)
    match_quality : dict
        Statistics about matching quality
    """
    
    print("\nMatching detected spikes to ground truth...")
    
    sampling_frequency = recording.get_sampling_frequency()
    tolerance_ms = 1.0  # Match spikes within 1ms
    tolerance_samples = int(tolerance_ms * sampling_frequency / 1000)
    
    labels = np.full(len(peak_locations), -1, dtype=np.int32)
    
    # Get all ground truth spike trains
    unit_ids = gt_sorting.get_unit_ids()
    
    matched_count = 0
    for spike_idx, (sample_idx, channel_idx) in enumerate(peak_locations):
        # Find the closest ground truth spike within tolerance
        min_distance = float('inf')
        best_unit = -1
        
        for unit_id in unit_ids:
            spike_train = gt_sorting.get_unit_spike_train(unit_id)
            
            # Find closest spike in this unit's spike train
            distances = np.abs(spike_train - sample_idx)
            min_dist = np.min(distances) if len(distances) > 0 else float('inf')
            
            if min_dist < min_distance and min_dist <= tolerance_samples:
                min_distance = min_dist
                best_unit = unit_id
        
        if best_unit != -1:
            labels[spike_idx] = best_unit
            matched_count += 1
        
        if (spike_idx + 1) % 5000 == 0:
            print(f"  Matched {spike_idx+1:,} / {len(peak_locations):,} spikes...")
    
    match_rate = matched_count / len(peak_locations) * 100
    
    print(f"\nMatching complete:")
    print(f"  Matched spikes: {matched_count:,} / {len(peak_locations):,} ({match_rate:.1f}%)")
    print(f"  Unmatched spikes: {len(peak_locations) - matched_count:,}")
    
    # Count spikes per unit
    unique_units, counts = np.unique(labels[labels != -1], return_counts=True)
    print(f"  Units detected: {len(unique_units)} / {len(unit_ids)}")
    
    match_quality = {
        'matched_count': matched_count,
        'total_detected': len(peak_locations),
        'match_rate': match_rate,
        'units_detected': len(unique_units),
        'total_units': len(unit_ids),
        'tolerance_ms': tolerance_ms
    }
    
    return labels, match_quality


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Run spike sorting with proper spike detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python run_experiments_synthetic_improved.py
  
  # Run with custom duration and threshold
  python run_experiments_synthetic_improved.py --subset-duration 30 --threshold 4.0
  
  # Run single experiment
  python run_experiments_synthetic_improved.py --single --pca-components 30
        """
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single experiment (PCA + K-means) instead of all combinations"
    )
    
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components for single experiment (default: 50)"
    )
    
    parser.add_argument(
        "--k-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: uses true k from synthetic data)"
    )
    
    parser.add_argument(
        "--subset-duration",
        type=float,
        default=10.0,
        help="Duration of recording subset in seconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--num-units",
        type=int,
        default=200,
        help="Number of units/neurons to simulate (default: 200)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Spike detection threshold in MAD units (default: 5.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=4776,
        help="Random seed for synthetic data generation (default: 4776)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IMPROVED Spike Sorting with Spike Detection")
    if args.single:
        print("MODE: Single Experiment (PCA + K-means)")
    else:
        print("MODE: Full Experiment Suite")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Generate synthetic recording
    # ========================================================================
    print("\n[Step 1] Generating synthetic recording...")
    
    static_rec, drift_rec, gt_sorting = si.generate_drifting_recording(
        probe_name="Neuropixels1-384",
        num_units=args.num_units,
        duration=300,
        sampling_frequency=30000,
        seed=args.seed,
    )
    
    # Use static recording (no drift)
    recording = static_rec
    
    # Extract subset
    if args.subset_duration is not None:
        end_frame = int(args.subset_duration * 30000)
        recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
        # Also slice ground truth
        gt_sorting = gt_sorting.frame_slice(start_frame=0, end_frame=end_frame)
    
    print(f"\nRecording ready:")
    print(f"  Duration: {args.subset_duration} seconds")
    print(f"  Samples: {recording.get_num_frames():,}")
    print(f"  Channels: {recording.get_num_channels()}")
    print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
    
    num_units_true = gt_sorting.get_num_units()
    print(f"\nGround truth: {num_units_true} units")
    
    # ========================================================================
    # Step 2: Detect spikes and extract features
    # ========================================================================
    print("\n[Step 2] Detecting spikes and extracting features...")
    
    X, peak_locations, spike_metadata = preprocess_recording(
        recording,
        method='locally_exclusive',
        peak_sign='neg',
        detect_threshold=args.threshold,
        exclude_sweep_ms=0.1,
        ms_before=0.6,
        ms_after=1.4,
        channels_per_spike=4,
        max_spikes_per_unit=1000,
        scale_features=True,
        n_jobs=1,
        verbose=True
    )
    
    print(f"\nSpike features ready:")
    print(f"  Number of spikes: {X.shape[0]:,}")
    print(f"  Feature dimensionality: {X.shape[1]}")
    print(f"  Expected: ~{num_units_true * args.subset_duration * 2:.0f} spikes (assuming ~2 Hz firing rate)")
    
    # ========================================================================
    # Step 3: Match detected spikes to ground truth
    # ========================================================================
    print("\n[Step 3] Creating ground truth labels...")
    
    y, match_quality = create_ground_truth_labels(gt_sorting, peak_locations, recording)
    
    # ========================================================================
    # Step 4: Configure experiments
    # ========================================================================
    print("\n[Step 4] Setting up experiment configuration...")
    
    true_k = num_units_true
    k_to_use = args.k_clusters if args.k_clusters is not None else true_k
    
    if args.single:
        config = {
            "dimensionality_reduction": {
                "PCA": [{"n_components": args.pca_components, "whiten": True, "svd_solver": "randomized"}],
                "UMAP": [{"n_neighbors": 15, "min_dist": 0.1, "n_components": 20, "metric": "cosine"}]
            },
            "clustering": {
                "KMeans": [{"n_clusters": k_to_use, "algorithm": "elkan", "n_init": 5, "max_iter": 100}],
                "GMM": [{"n_components": k_to_use, "covariance_type": "full", "reg_covar": 1e-4}],
                "DBSCAN": [{"eps": 1.0, "min_samples": 5}],
                "SpectralClustering": [{"n_clusters": k_to_use}]
            }
        }
        print(f"\nSingle experiment configuration:")
        print(f"  Dimensionality reduction methods: {list(config['dimensionality_reduction'].keys())}")
        print(f"  Clustering methods: {list(config['clustering'].keys())}")
    else:
        config = {
            "dimensionality_reduction": {
                "PCA": [
                    {"n_components": 20},
                    {"n_components": 50},
                    {"n_components": 100},
                ],
                "UMAP": [
                    {"n_neighbors": 15, "min_dist": 0.1, "n_components": 20},
                    {"n_neighbors": 30, "min_dist": 0.1, "n_components": 20},
                ],
                "ICA": [
                    {"n_components": 20},
                    {"n_components": 50},
                ]
            },
            "clustering": {
                "KMeans": [{"n_clusters": true_k}],
                "GMM": [{"n_components": true_k}],
                "HDBSCAN": [{"min_cluster_size": 50}],
            }
        }
        print(f"\nFull experiment suite:")
        print(f"  Testing {len(config['dimensionality_reduction'])} dim. reduction methods")
        print(f"  Testing {len(config['clustering'])} clustering methods")
        print(f"  Using k={true_k} for supervised methods")
    
    # Save configuration
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_filename = (
        "./results/experiment_config_improved_single.yaml"
        if args.single
        else "./results/experiment_config_improved_full.yaml"
    )
    save_config(config, config_filename)
    
    # ========================================================================
    # Step 5: Run experiments
    # ========================================================================
    print("\n[Step 5] Running experiments on spike features...")
    
    runner = ExperimentRunner(
        output_dir="./results",
        n_jobs=1,
        verbose=True
    )
    
    dataset_name = (
        f"synthetic_improved_single_{num_units_true}units"
        if args.single
        else f"synthetic_improved_full_{num_units_true}units"
    )
    
    results = runner.run_experiments(
        X=X,
        config=config,
        y=y,  # Ground truth labels!
        dataset_name=dataset_name
    )
    
    # ========================================================================
    # Step 6: Save results and metadata
    # ========================================================================
    print("\n[Step 6] Saving results...")
    
    runner.save_results()
    
    # Save comprehensive metadata
    metadata = {
        'recording': {
            'probe': 'Neuropixels1-384',
            'num_units': args.num_units,
            'duration': args.subset_duration,
            'sampling_frequency': 30000,
            'seed': args.seed
        },
        'spike_detection': spike_metadata,
        'ground_truth_matching': match_quality,
        'experiment': {
            'mode': 'single' if args.single else 'full',
            'true_k': true_k,
            'k_used': k_to_use
        }
    }
    
    metadata_path = Path("./results/improved_recording_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")
    
    # ========================================================================
    # Step 7: Analyze results
    # ========================================================================
    print("\n[Step 7] Analyzing results...")
    
    analyzer = ResultsAnalyzer(results)
    
    print("\nSummary Statistics:")
    print(analyzer.get_summary_statistics())
    
    if args.single:
        successful_results = analyzer.get_successful_results()
        if successful_results:
            r = successful_results[0]
            evaluation = r.get('evaluation', {})
            
            print(f"\n{'='*70}")
            print("Single Experiment Results")
            print(f"{'='*70}")
            print(f"\nConfiguration:")
            print(f"  Dimensionality Reduction: {r['dim_reduction_method']} {r['dim_reduction_params']}")
            print(f"  Clustering: {r['clustering_method']} {r.get('clustering_params', {})}")
            
            print(f"\nClustering Quality Metrics:")
            for metric in ['silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_index']:
                if metric in evaluation and evaluation[metric] is not None:
                    print(f"  {metric}: {evaluation[metric]:.4f}")
            
            print(f"\nGround Truth Comparison Metrics:")
            for metric in ['adjusted_rand_index', 'adjusted_mutual_info', 'v_measure', 'homogeneity', 'completeness']:
                if metric in evaluation and evaluation[metric] is not None:
                    print(f"  {metric}: {evaluation[metric]:.4f}")
            
            print(f"\nCluster Statistics:")
            print(f"  Number of clusters: {evaluation.get('n_clusters', 'N/A')}")
            print(f"  Noise points: {evaluation.get('n_noise_points', 0)} ({evaluation.get('noise_ratio', 0)*100:.1f}%)")
            print(f"  Mean cluster size: {evaluation.get('mean_cluster_size', 'N/A'):.1f}")
    else:
        print("\nTop 5 Configurations (by Adjusted Rand Index):")
        print(analyzer.get_best_configurations(metric="adjusted_rand_index", n_top=5))
        
        print("\nTop 5 Configurations (by Silhouette Score):")
        print(analyzer.get_best_configurations(metric="silhouette_score", n_top=5))
    
    # ========================================================================
    # Step 8: Generate visualizations
    # ========================================================================
    print("\n[Step 8] Generating visualizations...")
    
    if not args.single:
        analyzer.create_performance_heatmap(
            metric="adjusted_rand_index",
            save_path="./results/heatmap_ari_improved.png"
        )
        
        analyzer.create_method_comparison_barplot(
            metric="adjusted_rand_index",
            method_type="dim_reduction",
            save_path="./results/barplot_dimreduction_improved.png"
        )
        
        analyzer.create_method_comparison_barplot(
            metric="adjusted_rand_index",
            method_type="clustering",
            save_path="./results/barplot_clustering_improved.png"
        )
    
    # ========================================================================
    # Step 9: Generate report
    # ========================================================================
    print("\n[Step 9] Generating analysis report...")
    
    if not args.single:
        metrics_to_analyze = [
            'adjusted_rand_index',
            'adjusted_mutual_info',
            'silhouette_score',
            'v_measure'
        ]
        
        analyzer.generate_full_report(
            output_dir="./results/full_report_improved",
            metrics=metrics_to_analyze
        )
        
        analyzer.export_summary_table(
            output_path="./results/summary_table_improved.csv",
            format="csv",
            metric="adjusted_rand_index"
        )
    
    print("\n" + "=" * 70)
    print("Improved experiment completed successfully!")
    print("=" * 70)
    print(f"\nKey improvements over raw trace clustering:")
    print(f"  ✓ Spike detection: {X.shape[0]:,} spikes detected")
    print(f"  ✓ Feature extraction: {X.shape[1]} features per spike")
    print(f"  ✓ Ground truth labels: {match_quality['match_rate']:.1f}% spikes matched")
    print(f"  ✓ Expected better clustering performance!")
    print("\nResults saved to: ./results/")
    print("=" * 70)


if __name__ == "__main__":
    main()