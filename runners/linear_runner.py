import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spike_sort import ExperimentRunner
from spike_sort import ResultsAnalyzer
from spike_sort.utils.preprocess import preprocess_recording
import spikeinterface.full as si

def match_ground_truth(gt_sorting, peak_locations, recording):
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
    return labels

def main():
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
    args = p.parse_args()

    print("=" * 70)
    print("Linear Runner")
    print("=" * 70)

    if args.mode == "synthetic":
        static_rec, drift_rec, gt_sorting = si.generate_drifting_recording(
            probe_name="Neuropixels1-384",
            num_units=args.num_units,
            duration=300,
            sampling_frequency=30000,
            seed=4776,
        )
        recording = static_rec
        if args.subset_duration is not None:
            end_frame = int(args.subset_duration * 30000)
            recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
            gt_sorting = gt_sorting.frame_slice(start_frame=0, end_frame=end_frame)
        print("Recording ready:")
        print(f"  Duration: {args.subset_duration} seconds")
        print(f"  Samples: {recording.get_num_frames():,}")
        print(f"  Channels: {recording.get_num_channels()}")
        print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
        print(f"\nGround truth: {gt_sorting.get_num_units()} units")
        X, peak_locations, meta = preprocess_recording(
            recording,
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=args.threshold,
            exclude_sweep_ms=0.1,
            ms_before=0.6,
            ms_after=1.4,
            channels_per_spike=4,
            max_spikes_per_unit=1000,
            scale_features=True,
            n_jobs=1,
            verbose=False,
        )
        y = match_ground_truth(gt_sorting, peak_locations, recording)
        dataset_name = f"synthetic_linear_{gt_sorting.get_num_units()}units"
    else:
        if args.recording_folder is None:
            raise ValueError("recording-folder is required for real mode")
        rec_path = Path(args.recording_folder)
        recording = si.load_extractor(rec_path)
        if args.subset_duration is not None:
            end_frame = int(args.subset_duration * recording.get_sampling_frequency())
            recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
        print("Recording ready:")
        print(f"  Duration: {args.subset_duration} seconds")
        print(f"  Samples: {recording.get_num_frames():,}")
        print(f"  Channels: {recording.get_num_channels()}")
        print(f"  Sampling rate: {recording.get_sampling_frequency()} Hz")
        X, peak_locations, meta = preprocess_recording(
            recording,
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=args.threshold,
            exclude_sweep_ms=0.1,
            ms_before=0.6,
            ms_after=1.4,
            channels_per_spike=4,
            max_spikes_per_unit=1000,
            scale_features=True,
            n_jobs=1,
            verbose=False,
        )
        y = None
        dataset_name = "real_linear"

    print("\nSpike features ready:")
    print(f"  Number of spikes: {X.shape[0]:,}")
    print(f"  Feature dimensionality: {X.shape[1]}")
    X = X.astype(np.float64, copy=False)

    runner = ExperimentRunner(output_dir=args.output_dir, n_jobs=args.n_jobs, verbose=True, silent_errors=True)
    config = runner.load_and_merge(args.dim_config, args.clust_config)
    true_k = None
    if args.mode == "synthetic":
        true_k = int(dataset_name.split("_")[-1].replace("units", ""))
    if true_k is not None:
        clust = config.get("clustering", {})
        for name, params in list(clust.items()):
            if name in ("KMeans", "K-Means", "SpectralClustering", "Spectral", "Agglomerative"):
                clust[name] = [{**p, "n_clusters": true_k} for p in params] if params else [{"n_clusters": true_k}]
            elif name in ("GMM", "GaussianMixture", "DirichletProcess"):
                clust[name] = [{**p, "n_components": true_k} for p in params] if params else [{"n_components": true_k}]
            elif name == "HMM":
                clust[name] = [{**p, "n_components": true_k} for p in params] if params else [{"n_components": true_k}]
        config["clustering"] = clust
    results = runner.run_experiments(X=X, config=config, y=y, dataset_name=dataset_name)
    runner.save_results()

    meta_out = {
        "recording": {
            "mode": args.mode,
            "duration": args.subset_duration,
            "sampling_frequency": recording.get_sampling_frequency(),
            "channels": recording.get_num_channels(),
        },
        "preprocessing": meta,
    }
    meta_path = Path(args.output_dir) / "linear_runner_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    print(f"Metadata saved to: {meta_path}")

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
                print(f"  Best {metric}: {row[metric]:.4f}")
                print(f"    Dim. Reduction: {row['dim_reduction_method']} {row['dim_reduction_params']}")
                print(f"    Clustering: {row['clustering_method']} {row['clustering_params']}")

if __name__ == "__main__":
    main()
