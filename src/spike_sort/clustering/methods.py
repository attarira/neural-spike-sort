"""
Clustering Module for Spike Sorting Analysis

Provides a unified interface for various clustering techniques.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import warnings

warnings.filterwarnings('ignore')


class ClusteringBase(ABC):
    """Base class for all clustering methods."""
    
    def __init__(self, **kwargs):
        """
        Initialize the clustering method.
        
        Args:
            **kwargs: Method-specific hyperparameters
        """
        self.params = kwargs
        self.model = None
        self.computation_time = None
        self.labels_ = None
        self.n_clusters_ = None
        
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data matrix (n_samples × n_features)
            
        Returns:
            Cluster labels or None if failed
        """
        pass
    
    def _track_performance(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Track computation time of a function.
        
        Args:
            func: Function to track
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (result, time_seconds)
        """
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            return result, elapsed_time
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {str(e)}")
            return None, time.time() - start_time
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the clustering."""
        return {
            'method': self.__class__.__name__,
            'params': self.params,
            'computation_time': self.computation_time,
            'n_clusters': self.n_clusters_
        }


class KMeansClustering(ClusteringBase):
    """K-Means Clustering with GPU acceleration support."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            use_gpu = self.params.pop('use_gpu', False)
            
            def _fit():
                # Try GPU-accelerated version first if requested
                if use_gpu:
                    try:
                        from cuml.cluster import KMeans as cuKMeans
                        import cupy as cp
                        
                        # Convert to cupy array
                        X_gpu = cp.asarray(X, dtype=cp.float32)
                        
                        # Map parameters
                        gpu_params = self.params.copy()
                        gpu_params['random_state'] = 42
                        self.model = cuKMeans(**gpu_params)
                        labels_gpu = self.model.fit_predict(X_gpu)
                        
                        # Convert back to numpy
                        return cp.asnumpy(labels_gpu)
                    except ImportError:
                        print("Warning: cuML not available. Falling back to CPU K-Means.")
                    except Exception as e:
                        print(f"GPU K-Means failed ({str(e)}), falling back to CPU")
                
                # CPU version
                self.model = KMeans(**self.params, random_state=42)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"K-Means failed: {str(e)}")
            return None


class GMMClustering(ClusteringBase):
    """Gaussian Mixture Model Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = GaussianMixture(**self.params, random_state=42)
                self.model.fit(X)
                labels = self.model.predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"GMM failed: {str(e)}")
            return None


class DBSCANClustering(ClusteringBase):
    """DBSCAN Clustering with GPU acceleration support."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            use_gpu = self.params.pop('use_gpu', False)
            
            def _fit():
                # Try GPU-accelerated version first if requested
                if use_gpu:
                    try:
                        from cuml.cluster import DBSCAN as cuDBSCAN
                        import cupy as cp
                        
                        # Convert to cupy array
                        X_gpu = cp.asarray(X, dtype=cp.float32)
                        
                        # Map parameters
                        gpu_params = self.params.copy()
                        self.model = cuDBSCAN(**gpu_params)
                        labels_gpu = self.model.fit_predict(X_gpu)
                        
                        # Convert back to numpy
                        return cp.asnumpy(labels_gpu)
                    except ImportError:
                        print("Warning: cuML not available. Falling back to CPU DBSCAN.")
                    except Exception as e:
                        print(f"GPU DBSCAN failed ({str(e)}), falling back to CPU")
                
                # CPU version
                self.model = DBSCAN(**self.params)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                # Count clusters (excluding noise points labeled as -1)
                unique_labels = np.unique(self.labels_)
                self.n_clusters_ = len(unique_labels[unique_labels >= 0])
            return self.labels_
        except Exception as e:
            print(f"DBSCAN failed: {str(e)}")
            return None


class SpectralClusteringMethod(ClusteringBase):
    """Spectral Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = SpectralClustering(**self.params, random_state=42)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"Spectral Clustering failed: {str(e)}")
            return None


class DirichletProcessMixture(ClusteringBase):
    """Dirichlet Process Gaussian Mixture (Bayesian nonparametric clustering)."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            def _fit():
                # Use BayesianGaussianMixture with Dirichlet process prior
                params = self.params.copy()
                params['weight_concentration_prior_type'] = 'dirichlet_process'
                self.model = BayesianGaussianMixture(**params, random_state=42)
                self.model.fit(X)
                labels = self.model.predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"Dirichlet Process Mixture failed: {str(e)}")
            return None


class HMMClustering(ClusteringBase):
    """Hidden Markov Model Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            from hmmlearn import hmm
            
            n_components = self.params.get('n_components', 3)
            n_iter = self.params.get('n_iter', 100)
            covariance_type = self.params.get('covariance_type', 'full')
            
            def _fit():
                self.model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=42
                )
                
                # Reshape if needed (HMM expects sequences)
                if X.ndim == 2:
                    # Treat each sample as a single time step
                    self.model.fit(X)
                    labels = self.model.predict(X)
                else:
                    self.model.fit(X)
                    labels = self.model.predict(X)
                
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except ImportError:
            print("hmmlearn not installed. Install with: pip install hmmlearn")
            return None
        except Exception as e:
            print(f"HMM failed: {str(e)}")
            return None


class AgglomerativeClustering(ClusteringBase):
    """Agglomerative Hierarchical Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            from sklearn.cluster import AgglomerativeClustering as SklearnAgglo
            
            def _fit():
                self.model = SklearnAgglo(**self.params)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"Agglomerative Clustering failed: {str(e)}")
            return None


class OPTICSClustering(ClusteringBase):
    """OPTICS Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            from sklearn.cluster import OPTICS
            
            def _fit():
                self.model = OPTICS(**self.params)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                # Count clusters (excluding noise points labeled as -1)
                unique_labels = np.unique(self.labels_)
                self.n_clusters_ = len(unique_labels[unique_labels >= 0])
            return self.labels_
        except Exception as e:
            print(f"OPTICS failed: {str(e)}")
            return None


class MeanShiftClustering(ClusteringBase):
    """Mean Shift Clustering."""
    
    def fit_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            from sklearn.cluster import MeanShift
            
            def _fit():
                self.model = MeanShift(**self.params)
                labels = self.model.fit_predict(X)
                return labels
            
            self.labels_, self.computation_time = self._track_performance(_fit)
            if self.labels_ is not None:
                self.n_clusters_ = len(np.unique(self.labels_))
            return self.labels_
        except Exception as e:
            print(f"Mean Shift failed: {str(e)}")
            return None


# Factory function to create clustering objects
def create_clustering(method_name: str, **params) -> Optional[ClusteringBase]:
    """
    Factory function to create clustering objects.
    
    Args:
        method_name: Name of the method (e.g., 'KMeans', 'GMM')
        **params: Method-specific parameters
        
    Returns:
        ClusteringBase instance or None if method not found
    """
    clusterers = {
        'KMeans': KMeansClustering,
        'K-Means': KMeansClustering,
        'GMM': GMMClustering,
        'GaussianMixture': GMMClustering,
        'DBSCAN': DBSCANClustering,
        'SpectralClustering': SpectralClusteringMethod,
        'Spectral': SpectralClusteringMethod,
        'DirichletProcess': DirichletProcessMixture,
        'DPM': DirichletProcessMixture,
        'HMM': HMMClustering,
        'Agglomerative': AgglomerativeClustering,
        'OPTICS': OPTICSClustering,
        'MeanShift': MeanShiftClustering,
    }
    
    clusterer_class = clusterers.get(method_name)
    if clusterer_class is None:
        print(f"Unknown clustering method: {method_name}")
        return None
    
    return clusterer_class(**params)
