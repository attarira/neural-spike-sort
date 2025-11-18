import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spike_sort import ExperimentRunner
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--num-units", type=int, default=200)
    p.add_argument("--subset-duration", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--recording-folder", type=str, default=None)
    p.add_argument("--dim-config", type=str, default="configs/dimensionality_reduction/nonlinear.yaml")
    p.add_argument("--clust-config", type=str, default="configs/clustering/common.yaml")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug)
    print_header("Nonlinear Runner")

    if args.mode == "synthetic":
        static_rec, drift_rec, gt_sorting = create_synthetic_recording(args.num_units)
        recording, gt_sorting = slice_recording(static_rec, args.subset_duration, 30000, gt_sorting)
        print("Recording ready:")
        print(f"  Duration: {args.subset_duration} seconds")
        print(f"  Samples: {recording.get_num_frames():,}")
        print(f"  Channels: {recording.get_num_channels()}")
        print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
        print(f"\nGround truth: {gt_sorting.get_num_units()} units")
        X, peak_locations, meta = prepare_spike_features(recording, args.threshold)
        y = match_ground_truth(gt_sorting, peak_locations, recording)
        dataset_name = f"synthetic_nonlinear_{gt_sorting.get_num_units()}units"
    else:
        if args.recording_folder is None:
            raise ValueError("recording-folder is required for real mode")
        recording = load_real_recording(args.recording_folder)
        recording, _ = slice_recording(recording, args.subset_duration)
        print("Recording ready:")
        print(f"  Duration: {args.subset_duration} seconds")
        print(f"  Samples: {recording.get_num_frames():,}")
        print(f"  Channels: {recording.get_num_channels()}")
        print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
        X, peak_locations, meta = prepare_spike_features(recording, args.threshold)
        y = None
        dataset_name = "real_nonlinear"

    print_spike_feature_info(X)
    X = X.astype(np.float64, copy=False)

    runner = ExperimentRunner(output_dir=args.output_dir, n_jobs=args.n_jobs, verbose=True, silent_errors=True)
    config = runner.load_and_merge(args.dim_config, args.clust_config)
    true_k = None
    if args.mode == "synthetic":
        true_k = int(dataset_name.split("_")[-1].replace("units", ""))
    config = align_true_k(config, true_k)
    results = runner.run_experiments(X=X, config=config, y=y, dataset_name=dataset_name)
    runner.save_results()

    save_metadata(Path(args.output_dir), recording, meta, args.mode, args.subset_duration, "nonlinear_runner_metadata.json")

    analyze_and_print(results)


if __name__ == "__main__":
    main()