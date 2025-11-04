"""
Example script for running spike sorting experiments.

This script demonstrates how to:
1. Load preprocessed data
2. Define experiment configuration
3. Run experiments
4. Analyze results
"""

import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils import create_default_config, save_config


def generate_synthetic_data(n_samples=1000, n_features=50, n_clusters=5, random_state=42):
    """
    Generate synthetic spike data for testing.
    
    Args:
        n_samples: Number of spike samples
        n_features: Number of features per spike
        n_clusters: Number of true clusters
        random_state: Random seed
        
    Returns:
        X: Data matrix (n_samples × n_features)
        y: Ground truth labels
    """
    np.random.seed(random_state)
    
    X = []
    y = []
    
    for cluster_id in range(n_clusters):
        # Generate cluster center
        center = np.random.randn(n_features) * 5
        
        # Generate samples around center
        n_cluster_samples = n_samples // n_clusters
        cluster_samples = center + np.random.randn(n_cluster_samples, n_features)
        
        X.append(cluster_samples)
        y.extend([cluster_id] * n_cluster_samples)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def main():
    """Main execution function."""
    
    print("="*70)
    print("Spike Sorting Experiment Runner")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load or generate data
    # ========================================================================
    print("\n[Step 1] Loading data...")
    
    # Option A: Load your preprocessed data
    # X = np.load('path/to/your/preprocessed_spikes.npy')
    # y = np.load('path/to/your/ground_truth_labels.npy')  # Optional
    
    # Option B: Generate synthetic data for testing
    X, y = generate_synthetic_data(n_samples=1000, n_features=50, n_clusters=5)
    print(f"Data shape: {X.shape}")
    print(f"Number of true clusters: {len(np.unique(y))}")
    
    # ========================================================================
    # Step 2: Create or load experiment configuration
    # ========================================================================
    print("\n[Step 2] Setting up experiment configuration...")
    
    # Option A: Use default configuration
    config = create_default_config()
    
    # Option B: Load from YAML file
    # runner = ExperimentRunner(output_dir='./results', n_jobs=4)
    # config = runner.load_config('config_example.yaml')
    
    # Option C: Define custom configuration
    # config = {
    #     'dimensionality_reduction': {
    #         'PCA': [{'n_components': 10}, {'n_components': 20}],
    #         'UMAP': [{'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 10}]
    #     },
    #     'clustering': {
    #         'KMeans': [{'n_clusters': k} for k in range(2, 11)],
    #         'GMM': [{'n_components': k} for k in range(2, 11)]
    #     }
    # }
    
    # Save configuration for reference
    save_config(config, './results/experiment_config.yaml')
    
    # ========================================================================
    # Step 3: Initialize experiment runner
    # ========================================================================
    print("\n[Step 3] Initializing experiment runner...")
    
    # Create runner with parallel processing
    # Set n_jobs=-1 to use all available cores
    # Set n_jobs=1 for sequential execution (easier for debugging)
    runner = ExperimentRunner(
        output_dir='./results',
        n_jobs=1,  # Change to -1 for parallel execution
        verbose=True
    )
    
    # ========================================================================
    # Step 4: Run experiments
    # ========================================================================
    print("\n[Step 4] Running experiments...")
    
    results = runner.run_experiments(
        X=X,
        config=config,
        y=y,  # Pass ground truth if available, otherwise None
        dataset_name='synthetic_spikes'
    )
    
    # ========================================================================
    # Step 5: Save results
    # ========================================================================
    print("\n[Step 5] Saving results...")
    
    runner.save_results()
    
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
    
    if y is not None:
        print("\nTop 5 Configurations (by Adjusted Rand Index):")
        print(analyzer.get_best_configurations(metric='adjusted_rand_index', n_top=5))
    
    # ========================================================================
    # Step 7: Generate visualizations
    # ========================================================================
    print("\n[Step 7] Generating visualizations...")
    
    # Create performance heatmap
    analyzer.create_performance_heatmap(
        metric='silhouette_score',
        save_path='./results/heatmap_silhouette.png'
    )
    
    # Create method comparison plots
    analyzer.create_method_comparison_barplot(
        metric='silhouette_score',
        method_type='dim_reduction',
        save_path='./results/barplot_dimreduction.png'
    )
    
    analyzer.create_method_comparison_barplot(
        metric='silhouette_score',
        method_type='clustering',
        save_path='./results/barplot_clustering.png'
    )
    
    # Create runtime comparison
    analyzer.create_runtime_comparison(
        save_path='./results/runtime_comparison.png'
    )
    
    # ========================================================================
    # Step 8: Generate full report
    # ========================================================================
    print("\n[Step 8] Generating full analysis report...")
    
    metrics_to_analyze = ['silhouette_score', 'davies_bouldin_index', 
                         'calinski_harabasz_index']
    
    if y is not None:
        metrics_to_analyze.extend(['adjusted_rand_index', 'normalized_mutual_info', 
                                   'v_measure', 'accuracy'])
    
    analyzer.generate_full_report(
        output_dir='./results/full_report',
        metrics=metrics_to_analyze
    )
    
    # ========================================================================
    # Step 9: Export summary tables
    # ========================================================================
    print("\n[Step 9] Exporting summary tables...")
    
    analyzer.export_summary_table(
        output_path='./results/summary_table.csv',
        format='csv',
        metric='silhouette_score'
    )
    
    analyzer.export_summary_table(
        output_path='./results/summary_table.tex',
        format='latex',
        metric='silhouette_score'
    )
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("Results saved to: ./results/")
    print("="*70)


if __name__ == '__main__':
    main()
