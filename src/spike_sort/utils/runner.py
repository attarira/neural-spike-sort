"""
Experiment Runner Module for Spike Sorting Analysis

Orchestrates experiments across multiple dimensionality reduction and clustering methods.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import yaml

from spike_sort.dimensionality_reduction import create_reducer
from spike_sort.clustering import create_clustering
from spike_sort.evaluation import evaluate_all
import hashlib
import json as _json


class ExperimentRunner:
    """
    Main class for running comparative analysis experiments.
    """
    
    def __init__(self, 
                 output_dir: str = './results',
                 n_jobs: int = 1,
                 verbose: bool = True,
                 silent_errors: bool = True):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to save results
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.silent_errors = silent_errors
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        return config

    def load_and_merge(self, dim_config_path: str, clust_config_path: str) -> Dict[str, Any]:
        dim = self.load_config(dim_config_path)
        clust = self.load_config(clust_config_path)
        return self.merge_configs(dim, clust)
    
    def _run_single_experiment(self,
                               X: np.ndarray,
                               y: Optional[np.ndarray],
                               dim_reduction_method: str,
                               dim_reduction_params: Dict[str, Any],
                               clustering_method: str,
                               clustering_params: Dict[str, Any],
                               experiment_idx: int) -> Dict[str, Any]:
        """
        Run a single experiment (one combination of methods and parameters).
        
        Args:
            X: Input data
            y: Optional ground truth labels
            dim_reduction_method: Name of dimensionality reduction method
            dim_reduction_params: Parameters for dimensionality reduction
            clustering_method: Name of clustering method
            clustering_params: Parameters for clustering
            experiment_idx: Index of this experiment
            
        Returns:
            Dictionary with experiment results
        """
        result = {
            'experiment_idx': experiment_idx,
            'dim_reduction_method': dim_reduction_method,
            'dim_reduction_params': dim_reduction_params,
            'clustering_method': clustering_method,
            'clustering_params': clustering_params,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Step 1: Dimensionality Reduction
            reducer = create_reducer(dim_reduction_method, **dim_reduction_params)
            if reducer is None:
                result['error'] = f"Failed to create reducer: {dim_reduction_method}"
                return result
            
            X_reduced = reducer.fit_transform(X, y)
            if X_reduced is None:
                result['error'] = "Dimensionality reduction failed"
                return result
            
            result['dim_reduction_metadata'] = reducer.get_metadata()
            result['reduced_shape'] = X_reduced.shape
            
            # Step 2: Clustering
            clusterer = create_clustering(clustering_method, **clustering_params)
            if clusterer is None:
                result['error'] = f"Failed to create clusterer: {clustering_method}"
                return result
            
            labels = clusterer.fit_predict(X_reduced)
            if labels is None:
                result['error'] = "Clustering failed"
                return result
            
            result['clustering_metadata'] = clusterer.get_metadata()
            
            # Step 3: Evaluation
            evaluation_results = evaluate_all(X_reduced, labels, y)
            result['evaluation'] = evaluation_results
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            if self.verbose and not self.silent_errors:
                print(f"Experiment {experiment_idx} failed: {str(e)}")
        
        return result
    
    def run_experiments(self,
                       X: np.ndarray,
                       config: Dict[str, Any],
                       y: Optional[np.ndarray] = None,
                       dataset_name: str = "dataset") -> List[Dict[str, Any]]:
        """
        Run all experiments defined in the configuration.
        
        Args:
            X: Input data matrix (n_samples × n_features)
            config: Configuration dictionary with method specifications
            y: Optional ground truth labels
            dataset_name: Name of the dataset for logging
            
        Returns:
            List of result dictionaries
        """
        dim_combos = []
        for dim_method, dim_params_list in config.get('dimensionality_reduction', {}).items():
            for dim_params in dim_params_list:
                # Skip invalid CCA params where n_components > 1
                if dim_method.lower() == 'cca' and dim_params.get('n_components', 1) > 1:
                    continue
                dim_combos.append((dim_method, dim_params))
        embeddings_cache: Dict[str, Dict[str, Any]] = {}
        for dim_method, dim_params in dim_combos:
            key = f"{dim_method}:{_json.dumps(dim_params, sort_keys=True)}"
            if key in embeddings_cache:
                continue
            reducer = create_reducer(dim_method, **dim_params)
            if reducer is None:
                continue
            X_reduced = reducer.fit_transform(X, y)
            if X_reduced is None:
                continue
            embeddings_cache[key] = {
                'X_reduced': X_reduced,
                'metadata': reducer.get_metadata(),
                'reduced_shape': X_reduced.shape,
                'dim_method': dim_method,
                'dim_params': dim_params
            }
        experiments = []
        experiment_idx = 0
        for dim_method, dim_params in dim_combos:
            key = f"{dim_method}:{_json.dumps(dim_params, sort_keys=True)}"
            if key not in embeddings_cache:
                continue
            for clust_method, clust_params_list in config.get('clustering', {}).items():
                for clust_params in clust_params_list:
                    experiments.append({
                        'idx': experiment_idx,
                        'cache_key': key,
                        'clust_method': clust_method,
                        'clust_params': clust_params
                    })
                    experiment_idx += 1
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Running {len(experiments)} experiments on dataset: {dataset_name}")
            print(f"Data shape: {X.shape}")
            print(f"Experiment ID: {self.experiment_id}")
            print(f"{'='*70}\n")
        
        def run_with_cache(exp):
            cache = embeddings_cache.get(exp['cache_key'], None)
            base = {
                'experiment_idx': exp['idx'],
                'dim_reduction_method': cache['dim_method'] if cache else None,
                'dim_reduction_params': cache['dim_params'] if cache else None,
                'clustering_method': exp['clust_method'],
                'clustering_params': exp['clust_params'],
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            if cache is None:
                base['error'] = 'Embedding not available'
                return base
            try:
                X_reduced = cache['X_reduced']
                params = dict(exp['clust_params'])
                method = exp['clust_method']
                n_samples = X_reduced.shape[0]
                n_features = X_reduced.shape[1]
                if method in ('HMM',):
                    if n_features > 64 or params.get('n_components', 1) > max(50, n_samples // 10):
                        base['error'] = 'HMM skipped due to high dimensionality/too many components'
                        return base
                if method in ('GMM', 'GaussianMixture', 'DirichletProcess', 'DPM'):
                    nc = params.get('n_components', 1)
                    if params.get('covariance_type', 'full') == 'full':
                        if (nc * n_features) > (n_samples * 10):
                            params['covariance_type'] = 'diag'
                            params['reg_covar'] = max(params.get('reg_covar', 1e-4), 1e-3)
                clusterer = create_clustering(method, **params)
                if clusterer is None:
                    base['error'] = f"Failed to create clusterer: {method}"
                    return base
                labels = clusterer.fit_predict(X_reduced)
                if labels is None:
                    base['error'] = 'Clustering failed'
                    return base
                base['dim_reduction_metadata'] = cache['metadata']
                base['reduced_shape'] = cache['reduced_shape']
                base['clustering_metadata'] = clusterer.get_metadata()
                # Avoid metric errors for degenerate labelings
                n_unique = len(np.unique(labels))
                if 2 <= n_unique <= (n_samples - 1):
                    evaluation_results = evaluate_all(X_reduced, labels, y)
                else:
                    evaluation_results = {}
                base['evaluation'] = evaluation_results
                base['success'] = True
            except Exception as e:
                base['error'] = str(e)
                if self.verbose:
                    print(f"Experiment {exp['idx']} failed: {str(e)}")
            return base

        if self.n_jobs == 1:
            results = []
            for exp in tqdm(experiments, desc="Running experiments", disable=not self.verbose):
                result = run_with_cache(exp)
                results.append(result)
                self._save_intermediate_result(result, dataset_name)
        else:
            if self.verbose:
                print(f"Running experiments in parallel with {self.n_jobs} jobs...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(run_with_cache)(exp) for exp in tqdm(experiments, desc="Running experiments", disable=not self.verbose)
            )
            for result in results:
                self._save_intermediate_result(result, dataset_name)
        
        # Add dataset metadata
        for result in results:
            result['dataset_name'] = dataset_name
            result['data_shape'] = X.shape
            result['has_ground_truth'] = y is not None
        
        self.results.extend(results)
        
        # Print summary
        if self.verbose:
            self._print_summary(results)
        
        return results

    def merge_configs(self, dim_config: Dict[str, Any], clust_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            'dimensionality_reduction': dim_config.get('dimensionality_reduction', {}),
            'clustering': clust_config.get('clustering', {})
        }
        return merged
    
    def _save_intermediate_result(self, result: Dict[str, Any], dataset_name: str):
        """Save a single result to avoid losing data on failure."""
        intermediate_dir = self.output_dir / 'intermediate' / self.experiment_id
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_exp_{result['experiment_idx']:04d}.json"
        filepath = intermediate_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of experiment results."""
        n_total = len(results)
        n_success = sum(1 for r in results if r['success'])
        n_failed = n_total - n_success
        
        print(f"\n{'='*70}")
        print(f"Experiment Summary")
        print(f"{'='*70}")
        print(f"Total experiments: {n_total}")
        print(f"Successful: {n_success} ({n_success/n_total*100:.1f}%)")
        print(f"Failed: {n_failed} ({n_failed/n_total*100:.1f}%)")
        
        if n_success > 0:
            # Find best performing experiments
            successful_results = [r for r in results if r['success']]
            
            # Sort by different metrics
            metrics_to_check = [
                ('silhouette_score', True),
                ('adjusted_rand_index', True),
                ('davies_bouldin_index', False)
            ]
            
            print(f"\n{'='*70}")
            print("Top Performing Configurations:")
            print(f"{'='*70}")
            
            for metric_name, higher_is_better in metrics_to_check:
                valid_results = [
                    r for r in successful_results 
                    if r.get('evaluation', {}).get(metric_name) is not None
                ]
                
                if valid_results:
                    sorted_results = sorted(
                        valid_results,
                        key=lambda x: x['evaluation'][metric_name],
                        reverse=higher_is_better
                    )
                    
                    best = sorted_results[0]
                    print(f"\nBest {metric_name}: {best['evaluation'][metric_name]:.4f}")
                    print(f"  Dim. Reduction: {best['dim_reduction_method']} {best['dim_reduction_params']}")
                    print(f"  Clustering: {best['clustering_method']} {best['clustering_params']}")
        
        print(f"{'='*70}\n")
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save all results to file.
        
        Args:
            filename: Optional custom filename (without extension)
        """
        if filename is None:
            filename = f"results_{self.experiment_id}"
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as pickle (preserves numpy arrays)
        pickle_path = self.output_dir / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        if self.verbose:
            print(f"\nResults saved to:")
            print(f"  JSON: {json_path}")
            print(f"  Pickle: {pickle_path}")
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load results from file.
        
        Args:
            filepath: Path to results file (.json or .pkl)
            
        Returns:
            List of result dictionaries
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                results = json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.results = results
        return results
    
    def get_successful_results(self) -> List[Dict[str, Any]]:
        """Get only successful experiment results."""
        return [r for r in self.results if r['success']]
    
    def filter_results(self, 
                      dim_reduction_method: Optional[str] = None,
                      clustering_method: Optional[str] = None,
                      dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter results by method or dataset.
        
        Args:
            dim_reduction_method: Filter by dimensionality reduction method
            clustering_method: Filter by clustering method
            dataset_name: Filter by dataset name
            
        Returns:
            Filtered list of results
        """
        filtered = self.results
        
        if dim_reduction_method:
            filtered = [r for r in filtered if r['dim_reduction_method'] == dim_reduction_method]
        
        if clustering_method:
            filtered = [r for r in filtered if r['clustering_method'] == clustering_method]
        
        if dataset_name:
            filtered = [r for r in filtered if r.get('dataset_name') == dataset_name]
        
        return filtered


def create_default_config() -> Dict[str, Any]:
    """
    Create a default experiment configuration.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        'dimensionality_reduction': {
            'PCA': [
                {'n_components': 10},
                {'n_components': 20},
                {'n_components': 30}
            ],
            'UMAP': [
                {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 10},
                {'n_neighbors': 30, 'min_dist': 0.1, 'n_components': 10}
            ],
            'TSNE': [
                {'perplexity': 30, 'n_components': 2},
                {'perplexity': 50, 'n_components': 2}
            ],
            'ICA': [
                {'n_components': 10},
                {'n_components': 20}
            ]
        },
        'clustering': {
            'KMeans': [
                {'n_clusters': k} for k in range(2, 11)
            ],
            'GMM': [
                {'n_components': k} for k in range(2, 11)
            ],
            'DBSCAN': [
                {'eps': 0.5, 'min_samples': 5},
                {'eps': 1.0, 'min_samples': 5},
                {'eps': 0.5, 'min_samples': 10}
            ],
            'SpectralClustering': [
                {'n_clusters': k} for k in range(2, 6)
            ]
        }
    }
    
    return config


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    filepath = Path(filepath)
    
    if filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    print(f"Configuration saved to: {filepath}")
