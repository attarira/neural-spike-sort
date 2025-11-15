"""
Example script for running spike sorting experiments with synthetic recordings
from SpikeInterface.

This script demonstrates how to:
1. Generate synthetic recordings using SpikeInterface
2. Convert recordings to numpy arrays
3. Define experiment configuration
4. Run experiments on recording traces
5. Analyze results
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils import create_default_config, save_config

# Import SpikeInterface
import spikeinterface.full as si


def recording_to_array(recording, start_frame=None, end_frame=None, save_path=None, verbose=True):
    """
    Convert a SpikeInterface recording to a numpy array.
    
    Parameters:
    -----------
    recording : BaseRecording
        A SpikeInterface recording object (e.g., from si.generate_drifting_recording())
    start_frame : int, optional
        Start frame index (inclusive). If None, starts from beginning.
    end_frame : int, optional
        End frame index (exclusive). If None, goes to end.
    save_path : str or Path, optional
        If provided, save the numpy array to this path (e.g., 'data/synthetic_recording.npy').
        Will create parent directories if they don't exist.
    verbose : bool, default=True
        If True, print information about the extracted array.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (num_samples, num_channels) where:
        - Each row = one time point (all channels at that moment)
        - Each column = one channel's trace over all time
    """
    from pathlib import Path
    
    # Extract traces from the recording
    if start_frame is not None or end_frame is not None:
        traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    else:
        traces = recording.get_traces()
    
    # Print information if requested
    if verbose:
        print(f"Recording shape: {traces.shape}")
        print(f"Data type: {traces.dtype}")
        print(f"Memory size: {traces.nbytes / 1e9:.2f} GB")
        print(f"\nNumber of samples: {traces.shape[0]:,}")
        print(f"Number of channels: {traces.shape[1]}")
    
    # Save to disk if path provided
    if save_path is not None:
        save_path = Path(save_path)
        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure .npy extension
        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')
        
        np.save(save_path, traces)
        if verbose:
            print(f"\nSaved to: {save_path.absolute()}")
            print(f"File size: {save_path.stat().st_size / 1e9:.2f} GB")
    
    return traces


def generate_synthetic_recording(
    probe_name="Neuropixels1-384",
    num_units=200,
    duration=300,
    sampling_frequency=30000,
    seed=4776,
    use_drift=True,
    subset_duration=None
):
    """
    Generate synthetic recording using SpikeInterface.
    
    Args:
        probe_name: Name of the probe type (default: Neuropixels1-384)
        num_units: Number of neurons/units to simulate
        duration: Length of recording in seconds
        sampling_frequency: Sampling rate in Hz (default: 30000)
        seed: Random seed for reproducibility
        use_drift: If True, use drift_rec; if False, use static_rec
        subset_duration: If provided, extract only this many seconds from the start
        
    Returns:
        X: Data matrix (num_samples, num_channels) - recording traces
        gt_sorting: Ground truth sorting object (contains spike times and unit IDs)
        metadata: Dictionary with recording metadata
    """
    print("Generating synthetic recording...")
    print(f"  Probe: {probe_name}")
    print(f"  Units: {num_units}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {sampling_frequency} Hz")
    print(f"  Using drift: {use_drift}")
    
    # Generate synthetic recording with drift
    static_rec, drift_rec, gt_sorting = si.generate_drifting_recording(
        probe_name=probe_name,
        num_units=num_units,
        duration=duration,
        sampling_frequency=sampling_frequency,
        seed=seed
    )
    
    # Choose which recording to use
    recording = drift_rec if use_drift else static_rec
    
    # Extract subset if requested (to save memory)
    if subset_duration is not None:
        end_frame = int(subset_duration * sampling_frequency)
        print(f"\nExtracting subset: first {subset_duration} seconds ({end_frame:,} samples)")
        X = recording_to_array(recording, start_frame=0, end_frame=end_frame, verbose=True)
    else:
        X = recording_to_array(recording, verbose=True)
    
    # Create metadata
    metadata = {
        'probe_name': probe_name,
        'num_units': num_units,
        'duration': duration,
        'sampling_frequency': sampling_frequency,
        'use_drift': use_drift,
        'seed': seed,
        'data_shape': X.shape,
        'subset_duration': subset_duration
    }
    
    print(f"\nGenerated recording with shape: {X.shape}")
    print(f"  Samples (time points): {X.shape[0]:,}")
    print(f"  Channels (features): {X.shape[1]}")
    
    return X, gt_sorting, metadata


def main():
    """Main execution function."""
    
    print("="*70)
    print("Spike Sorting Experiment Runner - Synthetic Recordings")
    print("="*70)
    
    # ========================================================================
    # Step 1: Generate synthetic recording data
    # ========================================================================
    print("\n[Step 1] Generating synthetic recording...")
    
    # Generate synthetic recording
    # Using small subset to limit memory usage
    # 1 second = 30,000 samples × 384 channels ≈ 23 MB
    X, gt_sorting, metadata = generate_synthetic_recording(
        probe_name="Neuropixels1-384",
        num_units=200,
        duration=300,
        sampling_frequency=30000,
        seed=4776,
        use_drift=True,  # Set to False for static recording
        subset_duration=1  # Extract first 1 second for faster testing
    )
    
    # Note: X has shape (num_samples, num_channels)
    # Each row is a time point, each column is a channel
    # For clustering/dimensionality reduction, we treat each time point as a sample
    # with channels as features
    
    print(f"\nData ready:")
    print(f"  Shape: {X.shape}")
    print(f"  Samples (time points): {X.shape[0]:,}")
    print(f"  Features (channels): {X.shape[1]}")
    
    # Get ground truth number of units/clusters
    num_units_true = gt_sorting.get_num_units()
    print(f"\nGround truth sorting: {gt_sorting}")
    print(f"  Number of units (true clusters): {num_units_true}")
    
    # Ground truth: gt_sorting contains spike times and unit IDs
    # For trace-based clustering, we don't have per-sample labels
    # But we know the true number of clusters!
    y = None  # No ground truth labels for time points (would need spike detection)
    
    print(f"  Note: Ground truth is in spike times, not per-sample labels")
    print(f"  However, we know the true number of clusters: {num_units_true}")
    
    # ========================================================================
    # Step 2: Create or load experiment configuration
    # ========================================================================
    print("\n[Step 2] Setting up experiment configuration...")
    print(f"Using ONLY the true number of clusters: k={num_units_true}")
    
    # Use only the true k value for clustering
    true_k = num_units_true
    
    # Simplified configuration - only test true k value
    config = {
        'dimensionality_reduction': {
            'PCA': [
                {'n_components': 10},
                {'n_components': 20},
                {'n_components': 50},
                {'n_components': min(100, true_k)}
            ],
            'UMAP': [
                {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 10},
                {'n_neighbors': 30, 'min_dist': 0.1, 'n_components': 10},
                {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 20}
            ],
            'ICA': [
                {'n_components': 10},
                {'n_components': 20}
            ]
        },
        'clustering': {
            'KMeans': [
                {'n_clusters': true_k}  # Only true k value
            ],
            'GMM': [
                {'n_components': true_k}  # Only true k value
            ],
            'SpectralClustering': [
                {'n_clusters': true_k} if true_k <= 200 else None  # Only true k if reasonable
            ]
        }
    }
    
    # Remove None entries (in case true_k > 200 for SpectralClustering)
    if None in config['clustering']['SpectralClustering']:
        config['clustering']['SpectralClustering'] = [
            x for x in config['clustering']['SpectralClustering'] if x is not None
        ]
    
    print(f"\nConfiguration summary:")
    print(f"  Dimensionality reduction methods: {list(config['dimensionality_reduction'].keys())}")
    print(f"  Clustering methods: {list(config['clustering'].keys())}")
    print(f"  Using k = {true_k} (true number of clusters) for all clustering methods")
    
    # Create results directory if it doesn't exist
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reference
    save_config(config, './results/experiment_config_real.yaml')
    
    # ========================================================================
    # Step 3: Initialize experiment runner
    # ========================================================================
    print("\n[Step 3] Initializing experiment runner...")
    
    runner = ExperimentRunner(
        output_dir='./results',
        n_jobs=1,  # Change to -1 for parallel execution (be careful with memory)
        verbose=True
    )
    
    # ========================================================================
    # Step 4: Run experiments
    # ========================================================================
    print("\n[Step 4] Running experiments...")
    print("Note: This may take a while depending on data size and configuration")
    
    results = runner.run_experiments(
        X=X,
        config=config,
        y=y,  # None - no ground truth labels for time points (would need spike detection)
        dataset_name=f'synthetic_recording_{num_units_true}units'
    )
    
    # ========================================================================
    # Step 5: Save results
    # ========================================================================
    print("\n[Step 5] Saving results...")
    
    runner.save_results()
    
    # Save metadata
    metadata['true_num_units'] = num_units_true
    metadata_path = Path('./results/recording_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Recording metadata saved to: {metadata_path}")
    print(f"  True number of clusters: {num_units_true}")
    
    # ========================================================================
    # Step 6: Analyze results
    # ========================================================================
    print("\n[Step 6] Analyzing results...")
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results)
    
    # Get summary statistics
    print("\nSummary Statistics:")
    print(analyzer.get_summary_statistics())
    
    # Get best configurations
    print("\nTop 5 Configurations (by Silhouette Score):")
    print(analyzer.get_best_configurations(metric='silhouette_score', n_top=5))
    
    # All results use the true k value
    print(f"\n{'='*70}")
    print(f"Results using true number of clusters (k={num_units_true}):")
    print(f"{'='*70}")
    successful_results = analyzer.get_successful_results()
    if successful_results:
        for r in sorted(successful_results, 
                       key=lambda x: x.get('evaluation', {}).get('silhouette_score', -999), 
                       reverse=True)[:10]:
            dim_method = r['dim_reduction_method']
            dim_params = r['dim_reduction_params']
            clust_method = r['clustering_method']
            score = r.get('evaluation', {}).get('silhouette_score', 'N/A')
            print(f"\n  {dim_method} {dim_params} → {clust_method}:")
            print(f"    Silhouette Score: {score:.4f}")
    else:
        print("  No successful results found")
    
    # Note: No per-sample ground truth labels available since we're clustering time points
    
    # ========================================================================
    # Step 7: Generate visualizations
    # ========================================================================
    print("\n[Step 7] Generating visualizations...")
    
    # Create performance heatmap
    analyzer.create_performance_heatmap(
        metric='silhouette_score',
        save_path='./results/heatmap_silhouette_real.png'
    )
    
    # Create method comparison plots
    analyzer.create_method_comparison_barplot(
        metric='silhouette_score',
        method_type='dim_reduction',
        save_path='./results/barplot_dimreduction_real.png'
    )
    
    analyzer.create_method_comparison_barplot(
        metric='silhouette_score',
        method_type='clustering',
        save_path='./results/barplot_clustering_real.png'
    )
    
    # Create runtime comparison
    analyzer.create_runtime_comparison(
        save_path='./results/runtime_comparison_real.png'
    )
    
    # ========================================================================
    # Step 8: Generate full report
    # ========================================================================
    print("\n[Step 8] Generating full analysis report...")
    
    metrics_to_analyze = ['silhouette_score', 'davies_bouldin_index', 
                         'calinski_harabasz_index']
    
    # Note: No ground truth metrics available
    analyzer.generate_full_report(
        output_dir='./results/full_report_real',
        metrics=metrics_to_analyze
    )
    
    # ========================================================================
    # Step 9: Export summary tables
    # ========================================================================
    print("\n[Step 9] Exporting summary tables...")
    
    analyzer.export_summary_table(
        output_path='./results/summary_table_real.csv',
        format='csv',
        metric='silhouette_score'
    )
    
    analyzer.export_summary_table(
        output_path='./results/summary_table_real.tex',
        format='latex',
        metric='silhouette_score'
    )
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("Results saved to: ./results/")
    print("="*70)
    print("\nNote: This script uses full recording traces (time × channels).")
    print("For spike-based analysis, you may want to:")
    print("  1. Detect spikes from the recording")
    print("  2. Extract spike features (waveforms)")
    print("  3. Run clustering on spike features rather than full traces")
    print("="*70)


if __name__ == '__main__':
    main()

