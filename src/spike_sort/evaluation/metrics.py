"""
Evaluation Module for Spike Sorting Analysis

Provides metrics for evaluating clustering quality with and without ground truth.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score
)
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy using Hungarian algorithm for label matching.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    try:
        # Remove noise points (labeled as -1 in DBSCAN)
        mask = y_pred >= 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        if len(y_true_filtered) == 0:
            return 0.0
        
        # Create confusion matrix
        unique_true = np.unique(y_true_filtered)
        unique_pred = np.unique(y_pred_filtered)
        
        confusion_matrix = np.zeros((len(unique_true), len(unique_pred)))
        for i, true_label in enumerate(unique_true):
            for j, pred_label in enumerate(unique_pred):
                confusion_matrix[i, j] = np.sum((y_true_filtered == true_label) & 
                                                (y_pred_filtered == pred_label))
        
        # Use Hungarian algorithm to find best label matching
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # Compute accuracy
        correct = confusion_matrix[row_ind, col_ind].sum()
        accuracy = correct / len(y_true_filtered)
        
        return accuracy
    except Exception as e:
        print(f"Error computing accuracy: {str(e)}")
        return 0.0


def evaluate_with_ground_truth(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering results against ground truth labels.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    try:
        # Adjusted Rand Index
        results['adjusted_rand_index'] = adjusted_rand_score(y_true, y_pred)
    except Exception as e:
        print(f"Error computing ARI: {str(e)}")
        results['adjusted_rand_index'] = None
    
    try:
        # Normalized Mutual Information
        results['normalized_mutual_info'] = normalized_mutual_info_score(y_true, y_pred)
    except Exception as e:
        print(f"Error computing NMI: {str(e)}")
        results['normalized_mutual_info'] = None
    
    try:
        # V-measure
        results['v_measure'] = v_measure_score(y_true, y_pred)
    except Exception as e:
        print(f"Error computing V-measure: {str(e)}")
        results['v_measure'] = None
    
    try:
        # Accuracy (with Hungarian matching)
        results['accuracy'] = compute_accuracy(y_true, y_pred)
    except Exception as e:
        print(f"Error computing accuracy: {str(e)}")
        results['accuracy'] = None
    
    return results


def evaluate_clustering_quality(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality without ground truth.
    
    Args:
        X: Data matrix (n_samples × n_features)
        labels: Cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Filter out noise points (labeled as -1)
    mask = labels >= 0
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    # Check if we have enough clusters and samples
    n_clusters = len(np.unique(labels_filtered))
    n_samples = len(labels_filtered)
    
    if n_clusters < 2 or n_samples < 2:
        results['silhouette_score'] = None
        results['davies_bouldin_index'] = None
        results['calinski_harabasz_index'] = None
        return results
    
    try:
        # Silhouette Score (higher is better, range [-1, 1])
        results['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"Error computing Silhouette score: {str(e)}")
        results['silhouette_score'] = None
    
    try:
        # Davies-Bouldin Index (lower is better, range [0, inf))
        results['davies_bouldin_index'] = davies_bouldin_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"Error computing Davies-Bouldin index: {str(e)}")
        results['davies_bouldin_index'] = None
    
    try:
        # Calinski-Harabasz Index (higher is better, range [0, inf))
        results['calinski_harabasz_index'] = calinski_harabasz_score(X_filtered, labels_filtered)
    except Exception as e:
        print(f"Error computing Calinski-Harabasz index: {str(e)}")
        results['calinski_harabasz_index'] = None
    
    return results


def evaluate_all(X: np.ndarray, 
                 y_pred: np.ndarray, 
                 y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.
    
    Args:
        X: Data matrix (n_samples × n_features)
        y_pred: Predicted cluster labels
        y_true: Optional ground truth labels
        
    Returns:
        Dictionary containing all metrics
    """
    results = {}
    
    # Clustering quality metrics (no ground truth needed)
    quality_metrics = evaluate_clustering_quality(X, y_pred)
    results.update(quality_metrics)
    
    # Ground truth comparison metrics (if available)
    if y_true is not None:
        gt_metrics = evaluate_with_ground_truth(y_true, y_pred)
        results.update(gt_metrics)
    
    # Add cluster statistics
    unique_labels = np.unique(y_pred)
    n_noise = np.sum(y_pred == -1) if -1 in unique_labels else 0
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    results['n_clusters'] = n_clusters
    results['n_noise_points'] = n_noise
    results['noise_ratio'] = n_noise / len(y_pred) if len(y_pred) > 0 else 0.0
    
    # Cluster size statistics
    cluster_sizes = []
    for label in unique_labels:
        if label >= 0:  # Exclude noise
            cluster_sizes.append(np.sum(y_pred == label))
    
    if cluster_sizes:
        results['mean_cluster_size'] = np.mean(cluster_sizes)
        results['std_cluster_size'] = np.std(cluster_sizes)
        results['min_cluster_size'] = np.min(cluster_sizes)
        results['max_cluster_size'] = np.max(cluster_sizes)
    else:
        results['mean_cluster_size'] = None
        results['std_cluster_size'] = None
        results['min_cluster_size'] = None
        results['max_cluster_size'] = None
    
    return results


def compare_methods(results_list: list, method_names: list) -> Dict[str, Any]:
    """
    Compare multiple clustering methods.
    
    Args:
        results_list: List of result dictionaries from evaluate_all
        method_names: List of method names corresponding to results
        
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'methods': method_names,
        'metrics': {}
    }
    
    # Get all metric names
    all_metrics = set()
    for results in results_list:
        all_metrics.update(results.keys())
    
    # Compute statistics for each metric
    for metric in all_metrics:
        values = []
        for results in results_list:
            val = results.get(metric)
            if val is not None and not np.isnan(val):
                values.append(val)
        
        if values:
            comparison['metrics'][metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return comparison


def rank_methods(results_list: list, 
                 method_names: list, 
                 primary_metric: str = 'silhouette_score',
                 higher_is_better: bool = True) -> list:
    """
    Rank methods by a primary metric.
    
    Args:
        results_list: List of result dictionaries
        method_names: List of method names
        primary_metric: Metric to use for ranking
        higher_is_better: Whether higher values are better
        
    Returns:
        List of (method_name, metric_value) tuples, sorted by rank
    """
    rankings = []
    
    for method_name, results in zip(method_names, results_list):
        metric_value = results.get(primary_metric)
        if metric_value is not None and not np.isnan(metric_value):
            rankings.append((method_name, metric_value))
    
    # Sort by metric value
    rankings.sort(key=lambda x: x[1], reverse=higher_is_better)
    
    return rankings


def print_evaluation_summary(results: Dict[str, Any], method_name: str = "Method"):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Dictionary of evaluation metrics
        method_name: Name of the method being evaluated
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {method_name}")
    print(f"{'='*60}")
    
    # Cluster statistics
    print("\nCluster Statistics:")
    print(f"  Number of clusters: {results.get('n_clusters', 'N/A')}")
    print(f"  Noise points: {results.get('n_noise_points', 'N/A')} "
          f"({results.get('noise_ratio', 0)*100:.1f}%)")
    
    if results.get('mean_cluster_size') is not None:
        print(f"  Mean cluster size: {results.get('mean_cluster_size', 'N/A'):.1f} "
              f"± {results.get('std_cluster_size', 0):.1f}")
        print(f"  Cluster size range: [{results.get('min_cluster_size', 'N/A')}, "
              f"{results.get('max_cluster_size', 'N/A')}]")
    
    # Quality metrics
    print("\nClustering Quality Metrics:")
    if results.get('silhouette_score') is not None:
        print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
    if results.get('davies_bouldin_index') is not None:
        print(f"  Davies-Bouldin Index: {results['davies_bouldin_index']:.4f}")
    if results.get('calinski_harabasz_index') is not None:
        print(f"  Calinski-Harabasz Index: {results['calinski_harabasz_index']:.4f}")
    
    # Ground truth metrics
    if results.get('adjusted_rand_index') is not None:
        print("\nGround Truth Comparison:")
        print(f"  Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {results.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"  V-measure: {results.get('v_measure', 'N/A'):.4f}")
        print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
    
    print(f"{'='*60}\n")
