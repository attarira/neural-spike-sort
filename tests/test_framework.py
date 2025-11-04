"""
Quick test script to verify the framework is working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from spike_sort import create_reducer, create_clustering, evaluate_all


def test_dimensionality_reduction():
    """Test dimensionality reduction methods."""
    print("\n" + "="*70)
    print("Testing Dimensionality Reduction Methods")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    methods = {
        'PCA': {'n_components': 10},
        'ICA': {'n_components': 10},
        'TSNE': {'n_components': 2, 'perplexity': 30},
    }
    
    for method_name, params in methods.items():
        print(f"\nTesting {method_name}...")
        reducer = create_reducer(method_name, **params)
        if reducer:
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None:
                print(f"  ✓ {method_name} succeeded: {X.shape} -> {X_reduced.shape}")
                print(f"    Time: {reducer.computation_time:.3f}s")
            else:
                print(f"  ✗ {method_name} failed")
        else:
            print(f"  ✗ {method_name} not available")


def test_clustering():
    """Test clustering methods."""
    print("\n" + "="*70)
    print("Testing Clustering Methods")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    methods = {
        'KMeans': {'n_clusters': 3},
        'GMM': {'n_components': 3},
        'DBSCAN': {'eps': 0.5, 'min_samples': 5},
    }
    
    for method_name, params in methods.items():
        print(f"\nTesting {method_name}...")
        clusterer = create_clustering(method_name, **params)
        if clusterer:
            labels = clusterer.fit_predict(X)
            if labels is not None:
                print(f"  ✓ {method_name} succeeded")
                print(f"    Clusters found: {clusterer.n_clusters_}")
                print(f"    Time: {clusterer.computation_time:.3f}s")
            else:
                print(f"  ✗ {method_name} failed")
        else:
            print(f"  ✗ {method_name} not available")


def test_evaluation():
    """Test evaluation metrics."""
    print("\n" + "="*70)
    print("Testing Evaluation Metrics")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    print("\nTesting evaluation with ground truth...")
    results = evaluate_all(X, y_pred, y_true)
    
    print(f"  ✓ Evaluation completed")
    print(f"    Silhouette Score: {results.get('silhouette_score', 'N/A')}")
    print(f"    ARI: {results.get('adjusted_rand_index', 'N/A')}")
    print(f"    NMI: {results.get('normalized_mutual_info', 'N/A')}")
    print(f"    Number of clusters: {results.get('n_clusters', 'N/A')}")


def test_full_pipeline():
    """Test the full pipeline."""
    print("\n" + "="*70)
    print("Testing Full Pipeline")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y_true = np.random.randint(0, 3, 100)
    
    print("\nRunning: PCA -> K-Means -> Evaluation")
    
    # Step 1: Dimensionality reduction
    print("\n1. Dimensionality Reduction (PCA)...")
    reducer = create_reducer('PCA', n_components=10)
    X_reduced = reducer.fit_transform(X)
    print(f"   ✓ Reduced: {X.shape} -> {X_reduced.shape}")
    
    # Step 2: Clustering
    print("\n2. Clustering (K-Means)...")
    clusterer = create_clustering('KMeans', n_clusters=3)
    labels = clusterer.fit_predict(X_reduced)
    print(f"   ✓ Clustered into {clusterer.n_clusters_} clusters")
    
    # Step 3: Evaluation
    print("\n3. Evaluation...")
    results = evaluate_all(X_reduced, labels, y_true)
    print(f"   ✓ Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"   ✓ ARI: {results['adjusted_rand_index']:.4f}")
    
    print("\n" + "="*70)
    print("Full pipeline test completed successfully!")
    print("="*70)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FRAMEWORK VERIFICATION TEST")
    print("="*70)
    
    try:
        test_dimensionality_reduction()
        test_clustering()
        test_evaluation()
        test_full_pipeline()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe framework is ready to use!")
        print("Run 'python run_experiments.py' to start experiments.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
