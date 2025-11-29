"""
Dimensionality Reduction Module for Spike Sorting Analysis

Provides a unified interface for various dimensionality reduction techniques
including linear and nonlinear methods.
"""

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.cross_decomposition import CCA
import warnings

warnings.filterwarnings('ignore')


class DimensionalityReductionBase(ABC):
    """Base class for all dimensionality reduction methods."""
    
    def __init__(self, **kwargs):
        """
        Initialize the dimensionality reduction method.
        
        Args:
            **kwargs: Method-specific hyperparameters
        """
        self.params = kwargs
        self.model = None
        self.computation_time = None
        self.memory_usage = None
        
    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Fit the model and transform the data.
        
        Args:
            X: Input data matrix (n_samples × n_features)
            y: Optional labels for supervised methods
            
        Returns:
            Reduced embedding or None if failed
        """
        pass
    
    def _track_performance(self, func, *args, **kwargs) -> Tuple[Any, float, float]:
        """
        Track computation time and memory usage of a function.
        
        Args:
            func: Function to track
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (result, time_seconds, memory_mb)
        """
        tracemalloc.start()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            return result, elapsed_time, memory_mb
        except Exception as e:
            tracemalloc.stop()
            print(f"Error in {self.__class__.__name__}: {str(e)}")
            return None, time.time() - start_time, 0.0
    
    def _validate_output(self, result: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Validate that the output doesn't contain NaN or Inf values.
        
        Args:
            result: The embedding result to validate
            
        Returns:
            The result if valid, None otherwise
        """
        if result is None:
            return None
        
        if not isinstance(result, np.ndarray):
            return result
        
        if np.any(np.isnan(result)):
            print(f"Error in {self.__class__.__name__}: Output contains NaN values")
            return None
        
        if np.any(np.isinf(result)):
            print(f"Error in {self.__class__.__name__}: Output contains Inf values")
            return None
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the reduction."""
        return {
            'method': self.__class__.__name__,
            'params': self.params,
            'computation_time': self.computation_time,
            'memory_usage_mb': self.memory_usage
        }


class PCAReducer(DimensionalityReductionBase):
    """Principal Component Analysis."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = PCA(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"PCA failed: {str(e)}")
            return None


class ICAReducer(DimensionalityReductionBase):
    """Independent Component Analysis."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = FastICA(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"ICA failed: {str(e)}")
            return None


# ============================================================================
# NOT APPLICABLE: The following methods are not suitable for spike waveform data
# CCA requires two views/modalities of data
# LDA is fully supervised and requires ground truth labels
# ============================================================================

# class CCAReducer(DimensionalityReductionBase):
#     """Canonical Correlation Analysis - NOT APPLICABLE (requires two data views)."""
#     
#     def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         try:
#             if y is None:
#                 # Split X into two views for unsupervised CCA
#                 mid = X.shape[1] // 2
#                 X1, X2 = X[:, :mid], X[:, mid:]
#             else:
#                 X1, X2 = X, y.reshape(-1, 1) if y.ndim == 1 else y
#             
#             def _fit():
#                 self.model = CCA(**self.params)
#                 X_c, _ = self.model.fit_transform(X1, X2)
#                 return X_c
#             
#             result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#             return result
#         except Exception as e:
#             print(f"CCA failed: {str(e)}")
#             return None


# class LDAReducer(DimensionalityReductionBase):
#     """Linear Discriminant Analysis (supervised) - NOT APPLICABLE (requires ground truth labels)."""
#     
#     def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         try:
#             if y is None:
#                 print("LDA requires labels (y). Skipping.")
#                 return None
#             
#             def _fit():
#                 self.model = LinearDiscriminantAnalysis(**self.params)
#                 return self.model.fit_transform(X, y)
#             
#             result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#             return result
#         except Exception as e:
#             print(f"LDA failed: {str(e)}")
#             return None


class TSNEReducer(DimensionalityReductionBase):
    """t-Distributed Stochastic Neighbor Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = TSNE(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"t-SNE failed: {str(e)}")
            return None


class UMAPReducer(DimensionalityReductionBase):
    """Uniform Manifold Approximation and Projection."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            import umap
            
            def _fit():
                self.model = umap.UMAP(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            return None
        except Exception as e:
            print(f"UMAP failed: {str(e)}")
            return None


class IsomapReducer(DimensionalityReductionBase):
    """Isomap Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = Isomap(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"Isomap failed: {str(e)}")
            return None


class LaplacianEigenmapsReducer(DimensionalityReductionBase):
    """Laplacian Eigenmaps (Spectral Embedding)."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = SpectralEmbedding(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"Laplacian Eigenmaps failed: {str(e)}")
            return None


class MDSReducer(DimensionalityReductionBase):
    """Multidimensional Scaling (MDS)."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = MDS(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"MDS failed: {str(e)}")
            return None


class LLEReducer(DimensionalityReductionBase):
    """Locally Linear Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = LocallyLinearEmbedding(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"LLE failed: {str(e)}")
            return None


class ModifiedLLEReducer(DimensionalityReductionBase):
    """Modified Locally Linear Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                params = self.params.copy()
                params['method'] = 'modified'
                params["eigen_solver"] = "dense"
                self.model = LocallyLinearEmbedding(**params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"Modified LLE failed: {str(e)}")
            return None


class KPCAReducer(DimensionalityReductionBase):
    """Kernel Principal Component Analysis."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            from sklearn.decomposition import KernelPCA
            
            def _fit():
                self.model = KernelPCA(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"KPCA failed: {str(e)}")
            return None

class DiffusionMapsReducer(DimensionalityReductionBase):
    """
    Numerically stable Diffusion Maps implementation.

    Design principles:
    - NEVER use ARPACK (unstable for graph Laplacians on spike manifolds)
    - Prefer pydiffmap when available with data-scaled epsilon
    - Fallback to SpectralEmbedding with:
        * affinity = 'nearest_neighbors'
        * solver = dense (small N) or LOBPCG (large N)
    - Explicitly regularize X to break graph degeneracy
    - Hard-cap n_components <= 5
    """

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                n_samples = X.shape[0]
                if n_samples < 3:
                    print(f"Warning: Too few samples ({n_samples}) for DiffusionMaps")
                    return None

                # ---- Hard numerical caps ----
                n_components = min(self.params.get("n_components", 10), 5, n_samples - 2)
                n_neighbors = min(self.params.get("n_neighbors", 10), n_samples - 1)
                alpha = self.params.get("alpha", 0.5)

                # ---- Explicit degeneracy breaking (critical) ----
                X_reg = X + 1e-6 * np.random.standard_normal(X.shape)

                # ==========================================================
                # 1. Preferred Path: pydiffmap (Stable when epsilon scaled)
                # ==========================================================
                try:
                    from pydiffmap import diffusion_map as dm

                    # Data-scaled epsilon (NOT global heuristic)
                    from sklearn.metrics import pairwise_distances
                    sample_idx = np.random.choice(
                        n_samples, size=min(2048, n_samples), replace=False
                    )
                    D = pairwise_distances(X_reg[sample_idx], metric="euclidean")
                    eps = float(np.median(D[D > 0]) ** 2)
                    eps = max(eps, 1e-6)

                    mydmap = dm.DiffusionMap.from_sklearn(
                        n_evecs=n_components,
                        epsilon=eps,
                        alpha=alpha,
                    )

                    self.model = mydmap
                    return mydmap.fit_transform(X_reg)

                except ImportError:
                    pass  # Fall back to spectral formulation

                # ==========================================================
                # 2. Stable Fallback: SpectralEmbedding (No ARPACK)
                # ==========================================================
                eigen_solver = "dense" if n_samples <= 1500 else "lobpcg"

                params = {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "affinity": "nearest_neighbors",  # CRITICAL
                    "eigen_solver": eigen_solver,
                    "n_jobs": 1,
                }

                self.model = SpectralEmbedding(**params)
                return self.model.fit_transform(X_reg)

            # ---- Performance tracking wrapper ----
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)

        except Exception as e:
            print(f"Diffusion Maps failed (stable implementation): {str(e)}")
            return None


class PHATEReducer(DimensionalityReductionBase):
    """PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                try:
                    import phate
                    self.model = phate.PHATE(**self.params)
                    return self.model.fit_transform(X)
                except ImportError:
                    print("Warning: phate library not available. Install with: pip install phate")
                    print("Falling back to t-SNE as approximation")
                    params = {
                        'n_components': self.params.get('n_components', 2),
                        'perplexity': self.params.get('knn', 5),
                    }
                    self.model = TSNE(**params)
                    return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"PHATE failed: {str(e)}")
            return None


class TriMapReducer(DimensionalityReductionBase):
    """TriMap dimensionality reduction."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                try:
                    import trimap
                    n_components = self.params.get('n_components', 2)
                    n_inliers = self.params.get('n_inliers', 10)
                    n_outliers = self.params.get('n_outliers', 5)
                    n_random = self.params.get('n_random', 5)
                    
                    self.model = trimap.TRIMAP(
                        n_dims=n_components,
                        n_inliers=n_inliers,
                        n_outliers=n_outliers,
                        n_random=n_random
                    )
                    return self.model.fit_transform(X)
                except ImportError:
                    print("Warning: trimap library not available. Install with: pip install trimap")
                    print("Falling back to UMAP as approximation")
                    try:
                        import umap
                        params = {
                            'n_components': self.params.get('n_components', 2),
                            'n_neighbors': self.params.get('n_inliers', 10),
                        }
                        self.model = umap.UMAP(**params)
                        return self.model.fit_transform(X)
                    except ImportError:
                        print("Warning: umap library also not available. Using t-SNE")
                        params = {
                            'n_components': self.params.get('n_components', 2),
                            'perplexity': min(30, X.shape[0] // 4),
                        }
                        self.model = TSNE(**params)
                        return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"TriMap failed: {str(e)}")
            return None


class AutoencoderReducer(DimensionalityReductionBase):
    """Autoencoder-based dimensionality reduction with train/test split to avoid data leakage."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            from sklearn.model_selection import train_test_split
            
            # Default parameters
            encoding_dim = self.params.get('encoding_dim', 10)
            hidden_dim = self.params.get('hidden_dim', 50)
            epochs = self.params.get('epochs', 50)
            batch_size = self.params.get('batch_size', 32)
            learning_rate = self.params.get('learning_rate', 0.001)
            test_size = self.params.get('test_size', 0.2)  # 80/20 train/test split
            random_state = self.params.get('random_state', 42)
            
            class Autoencoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, encoding_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, encoding_dim)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim)
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
                
                def encode(self, x):
                    return self.encoder(x)
            
            def _fit():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Train/test split to avoid data leakage
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=test_size,
                    random_state=random_state
                )
                X_train = X[train_idx]
                X_test = X[test_idx]
                
                # Prepare training data
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                train_dataset = TensorDataset(X_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = Autoencoder(X.shape[1], hidden_dim, encoding_dim).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train on training set only
                self.model.train()
                for epoch in range(epochs):
                    for batch in train_loader:
                        batch_X = batch[0]
                        optimizer.zero_grad()
                        output = self.model(batch_X)
                        loss = criterion(output, batch_X)
                        loss.backward()
                        optimizer.step()
                
                # Encode ALL data points (train + test) for visualization
                # Training was done only on train set, so no data leakage
                self.model.eval()
                X_all_tensor = torch.FloatTensor(X).to(device)
                with torch.no_grad():
                    encoded_all = self.model.encode(X_all_tensor)
                    return encoded_all.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            return self._validate_output(result)
            
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"Autoencoder failed: {str(e)}")
            return None


class VAEReducer(DimensionalityReductionBase):
    """Variational Autoencoder-based dimensionality reduction with train/test split."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            from sklearn.model_selection import train_test_split
            
            # Default parameters
            encoding_dim = self.params.get('encoding_dim', 10)
            hidden_dim = self.params.get('hidden_dim', 50)
            epochs = self.params.get('epochs', 50)
            batch_size = self.params.get('batch_size', 32)
            learning_rate = self.params.get('learning_rate', 0.001)
            test_size = self.params.get('test_size', 0.2)  # 80/20 train/test split
            random_state = self.params.get('random_state', 42)
            
            class VAE(nn.Module):
                def __init__(self, input_dim, hidden_dim, encoding_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU()
                    )
                    self.fc_mu = nn.Linear(hidden_dim, encoding_dim)
                    self.fc_logvar = nn.Linear(hidden_dim, encoding_dim)
                    
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim)
                    )
                
                def encode(self, x):
                    h = self.encoder(x)
                    return self.fc_mu(h), self.fc_logvar(h)
                
                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                
                def decode(self, z):
                    return self.decoder(z)
                
                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar
            
            def vae_loss(recon_x, x, mu, logvar):
                recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return recon_loss + kld
            
            def _fit():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Train/test split to avoid data leakage
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=test_size,
                    random_state=random_state
                )
                X_train = X[train_idx]
                X_test = X[test_idx]
                
                # Prepare training data
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                train_dataset = TensorDataset(X_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = VAE(X.shape[1], hidden_dim, encoding_dim).to(device)
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train on training set only
                self.model.train()
                for epoch in range(epochs):
                    for batch in train_loader:
                        batch_X = batch[0]
                        optimizer.zero_grad()
                        recon_batch, mu, logvar = self.model(batch_X)
                        loss = vae_loss(recon_batch, batch_X, mu, logvar)
                        loss.backward()
                        optimizer.step()
                
                # Encode ALL data points (train + test) for visualization
                # Training was done only on train set, so no data leakage
                self.model.eval()
                X_all_tensor = torch.FloatTensor(X).to(device)
                with torch.no_grad():
                    mu_all, _ = self.model.encode(X_all_tensor)
                    return mu_all.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            return self._validate_output(result)
            
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"VAE failed: {str(e)}")
            return None


class ContrastiveAutoEncoderReducer(DimensionalityReductionBase):
    """
    Contrastive Auto-Encoder - SUPERVISED METHOD.
    Uses supervised contrastive learning with proper positive/negative pairs.
    
    **Requires ground truth labels (y) to function.**
    
    Positive pairs: spikes from the same neuron (same y value)
    Negative pairs: spikes from different neurons (different y values)
    """
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            # Default parameters
            encoding_dim = self.params.get('encoding_dim', 10)
            hidden_dim = self.params.get('hidden_dim', 50)
            epochs = self.params.get('epochs', 50)
            batch_size = self.params.get('batch_size', 32)
            learning_rate = self.params.get('learning_rate', 0.001)
            temperature = self.params.get('temperature', 0.5)
            
            class ContrastiveEncoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, encoding_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, encoding_dim)
                    )
                
                def forward(self, x):
                    return self.encoder(x)
            
            def supervised_contrastive_loss(z, labels, temperature):
                """
                Supervised contrastive loss.
                Positive pairs: samples with the same label (same neuron)
                Negative pairs: samples with different labels (different neurons)
                """
                # Normalize embeddings
                z = nn.functional.normalize(z, dim=1)
                
                # Compute similarity matrix: (batch_size, batch_size)
                similarity_matrix = torch.matmul(z, z.T) / temperature
                
                batch_size = z.shape[0]
                
                # Create mask for positive pairs (same label, excluding self)
                labels = labels.contiguous().view(-1, 1)
                mask_positive = torch.eq(labels, labels.T).float().to(z.device)
                
                # Remove diagonal (self-similarity)
                mask_positive = mask_positive - torch.eye(batch_size).to(z.device)
                
                # Compute log probabilities
                # Subtract max for numerical stability
                logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
                logits = similarity_matrix - logits_max.detach()
                
                # Compute log-sum-exp of all negatives (for denominator)
                exp_logits = torch.exp(logits)
                
                # Mask out positive pairs and self from denominator
                mask_negative = 1 - mask_positive - torch.eye(batch_size).to(z.device)
                
                # For each anchor, compute log[ sum(exp(pos)) / sum(exp(neg)) ]
                # Sum over positive pairs in numerator
                log_prob_positive = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
                
                # Apply positive mask and average
                num_positives_per_row = mask_positive.sum(dim=1)
                
                # Only compute loss for samples that have at least one positive pair
                valid_samples = num_positives_per_row > 0
                
                if valid_samples.sum() == 0:
                    # If no valid positive pairs in batch, return zero loss
                    return torch.tensor(0.0).to(z.device)
                
                # Compute mean of log-likelihood over positive pairs
                loss = -(mask_positive * log_prob_positive).sum(dim=1) / (num_positives_per_row + 1e-8)
                loss = loss[valid_samples].mean()
                
                return loss
            
            def _fit():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Check if labels are provided - Contrastive Auto-Encoder requires labels for supervised contrastive learning
                if y is None:
                    print("Contrastive Auto-Encoder requires labels (y) for supervised contrastive learning. Skipping.")
                    return None
                
                # STRATIFIED train/test split to maintain class balance
                # This avoids data leakage while ensuring representative class distribution
                from sklearn.model_selection import train_test_split
                test_size = self.params.get('test_size', 0.2)
                random_state = self.params.get('random_state', 42)
                
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=test_size,
                    stratify=y,  # STRATIFIED: maintain class proportions
                    random_state=random_state
                )
                X_train, y_train = X[train_idx], y[train_idx]
                X_test = X[test_idx]
                
                # Prepare training data with labels
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = ContrastiveEncoder(X.shape[1], hidden_dim, encoding_dim).to(device)
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train on training set only
                self.model.train()
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    n_batches = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        z = self.model(batch_X)
                        loss = supervised_contrastive_loss(z, batch_y, temperature)
                        
                        if loss > 0:  # Only backprop if loss is non-zero
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        
                        n_batches += 1
                    
                    # Optional: print progress every 10 epochs
                    if epoch % 10 == 0 and n_batches > 0:
                        avg_loss = epoch_loss / n_batches
                        # Uncomment for debugging:
                        # print(f"Contrastive Auto-Encoder Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
                
                # Encode ALL data points (train + test) for visualization
                # Training was done only on train set, so no data leakage
                self.model.eval()
                X_all_tensor = torch.FloatTensor(X).to(device)
                with torch.no_grad():
                    encoded_all = self.model(X_all_tensor)
                    return encoded_all.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            return self._validate_output(result)
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"Contrastive Auto-Encoder failed: {str(e)}")
            return None


# Factory function to create reducers
def create_reducer(method_name: str, **params) -> Optional[DimensionalityReductionBase]:
    """
    Factory function to create dimensionality reduction objects.
    
    Args:
        method_name: Name of the method (e.g., 'PCA', 'UMAP')
        **params: Method-specific parameters
        
    Returns:
        DimensionalityReductionBase instance or None if method not found
    """
    reducers = {
        # Linear methods
        'PCA': PCAReducer,
        'ICA': ICAReducer,
        # 'CCA': CCAReducer,  # NOT APPLICABLE: requires two data views
        # 'LDA': LDAReducer,  # NOT APPLICABLE: requires ground truth labels
        
        # Nonlinear methods
        'KPCA': KPCAReducer,
        'TSNE': TSNEReducer,
        't-SNE': TSNEReducer,
        'UMAP': UMAPReducer,
        'Isomap': IsomapReducer,
        'MDS': MDSReducer,
        'LaplacianEigenmaps': LaplacianEigenmapsReducer,
        'LLE': LLEReducer,
        'ModifiedLLE': ModifiedLLEReducer,
        'DiffusionMaps': DiffusionMapsReducer,
        'PHATE': PHATEReducer,
        'TriMap': TriMapReducer,
        
        # Deep learning methods
        'Autoencoder': AutoencoderReducer,
        'VAE': VAEReducer,
        'ContrastiveAutoEncoder': ContrastiveAutoEncoderReducer,
        
        # NOT APPLICABLE methods (commented out)
        # 'GPFA': GPFAReducer,  # NOT APPLICABLE: designed for temporal population dynamics
        # 'SliceTCA': SliceTCAReducer,  # NOT APPLICABLE: designed for tensor-structured data
        # 'LFADS': LFADSReducer,  # NOT APPLICABLE: designed for temporal sequential data
    }
    
    reducer_class = reducers.get(method_name)
    if reducer_class is None:
        print(f"Unknown dimensionality reduction method: {method_name}")
        return None
    
    return reducer_class(**params)
