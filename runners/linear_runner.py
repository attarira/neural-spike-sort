import argparse
import json
from pathlib import Path
import sys
import logging
from typing import Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spike_sort import ExperimentRunner
from spike_sort import ResultsAnalyzer
import spikeinterface.full as si
from spike_sort.utils.runner_helpers import (
    setup_logging,
    print_header,
    create_synthetic_recording,
    load_real_recording,
    slice_recording,
    print_recording_info,
    prepare_spike_features,
    match_ground_truth,
    align_true_k,
    print_spike_feature_info,
    analyze_and_print,
    save_metadata,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the linear runner."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--num-units", type=int, default=200)
    p.add_argument("--subset-duration", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--recording-folder", type=str, default=None)
    p.add_argument("--dim-config", type=str, default="configs/dimensionality_reduction/linear.yaml")
    p.add_argument("--clust-config", type=str, default="configs/clustering/common.yaml")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    """Entrypoint for running the linear pipeline on synthetic or real data."""
    
    # Initialization: Parse Arguments + Setup Logging
    # ==========================================================================================
    # Parse arguments and initialize logging for visibility
    args = parse_args()
    setup_logging(args.debug)
    print_header("Linear Runner") 
    # ==========================================================================================

    if args.mode == "synthetic":
        # Step 1: Create Synthetic Recording
        # ==========================================================================================
        # Create a synthetic recording based on Neuropixels1-384 probe (384 channels @ 30kHz)
        # with a default recording length of 300 seconds (5 minutes).
        # Comes with static and drift versions along with ground truth (actual # units)
        # TODO: May want to specify recording length (in seconds) in cmd args 
        static_rec, drift_rec, gt_sorting = create_synthetic_recording(
            num_units = args.num_units,
            duration  = 300 # seconds
        )
        # ==========================================================================================
        
        
        # Step 2: Slice Synthetic Recording to Desired Range
        # ==========================================================================================
        # Slice dataset to subset duration for faster iteration when desired
        recording, gt_sorting = slice_recording(
            recording  = static_rec,
            duration_s = args.subset_duration,
            gt_sorting = gt_sorting
        )
        
        # Some statistics
        print_recording_info(recording, args.subset_duration)
        print(f"\nGround truth: {gt_sorting.get_num_units()} units")
        # ==========================================================================================
        
        # Step 3: Preprocess Recording to Obtain: Feature Matrix, Labels, Metadata
        # ==========================================================================================
        # Preprocess recording to obtain:
        # - Feature matrix (each row is one detected spike)
        # - Labels (which channel each spike occured)
        # - Metadata (spike count, feature count, etc.)
        X, peak_locations, meta = prepare_spike_features(recording, args.threshold)
        
        # For each spike, checks if it matches (approximately) to a/the ground truth spike
        y = match_ground_truth(
            gt_sorting     = gt_sorting,
            peak_locations = peak_locations,
            recording      = recording
        )
        
        # Derive dataset name including unit count for traceability
        dataset_name = f"synthetic_linear_{gt_sorting.get_num_units()}units"
        # ==========================================================================================    
    else:
        # Real mode: requires path to a saved extractor; no ground truth available
        if args.recording_folder is None:
            raise ValueError("recording-folder is required for real mode")
        
        # Step 1: Load Real Recording
        # ==========================================================================================
        recording = load_real_recording(recording_folder = args.recording_folder)
        # ==========================================================================================
        
        # Step 2: Slice Recording to Desire Range
        # ==========================================================================================
        recording, _ = slice_recording(
            recording  = recording,
            duration_s = args.subset_duration,
            gt_sorting = None                   # Obviously!
        )
        
        # Some statistics
        print_recording_info(recording, args.subset_duration)
        # ==========================================================================================
        
        # Step 3: Preprocess Recording to Obtain: Feature Matrix, Labels, Metadata
        # ==========================================================================================
        # Preprocess recording to obtain:
        # - Feature matrix (each row is one detected spike)
        # - Labels (which channel each spike occured)
        # - Metadata (spike count, feature count, etc.)
        X, peak_locations, meta = prepare_spike_features(recording, args.threshold)
        
        # As as we have no ground truth...
        y = None
        
        dataset_name = "real_linear"
        # ==========================================================================================
  
    # Step A: Prepare Feature Matrix (Numerical Stability)
    # ==========================================================================================
    # Inspect feature matrix and cast to float64 for numerical stability
    print_spike_feature_info(X)
    X = X.astype(np.float64, copy=False)
    # ==========================================================================================

    # Step B: Initialize Runner and Configs
    # ==========================================================================================
    # Initialize ExperimentRunner and load/merge configs
    runner = ExperimentRunner(
        output_dir    = args.output_dir,
        n_jobs        = args.n_jobs,
        verbose       = True,
        silent_errors = True
    )
    config = runner.load_and_merge(args.dim_config, args.clust_config)
    # ==========================================================================================

    # Step C: Select k (for clustering) Based on Ground Truth (Synthetic Data Only)
    # ==========================================================================================
    # Align fixed-k clusterers to synthetic ground truth (if available)
    true_k = None
    if args.mode == "synthetic":
        # Use units from dataset name for consistency with previous runs
        true_k = int(dataset_name.split("_")[-1].replace("units", ""))
    config = align_true_k(config, true_k)
    # ==========================================================================================

    # Step D: Execute Runner + Save Results
    # ==========================================================================================
    # Execute configured experiments and persist outputs
    results = runner.run_experiments(
        X            = X,
        config       = config,
        y            = y,
        dataset_name = dataset_name
    )
    runner.save_results()
    # ==========================================================================================

    # Step E: Save Metadata and Summarize Results
    # ==========================================================================================
    # Save metadata for reproducibility and summarize results
    save_metadata(
        output_dir = Path(args.output_dir),
        recording  = recording,
        meta       = meta,
        mode       = args.mode,
        duration   = args.subset_duration,
        filename   = "linear_runner_metadata.json"
    )
    analyze_and_print(results)
    # ==========================================================================================

if __name__ == "__main__":
    main()
