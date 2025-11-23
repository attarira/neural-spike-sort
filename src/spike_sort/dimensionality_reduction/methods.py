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
    """Diffusion Maps using pydiffmap library or custom implementation."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                # Try using pydiffmap if available, otherwise use a custom implementation
                try:
                    from pydiffmap import diffusion_map as dm
                    # Cap n_components at 5 to avoid convergence issues
                    n_components = min(self.params.get('n_components', 10), 5)
                    epsilon = self.params.get('epsilon', 'bgh')
                    alpha = self.params.get('alpha', 0.5)
                    
                    mydmap = dm.DiffusionMap.from_sklearn(
                        n_evecs=n_components,
                        epsilon=epsilon,
                        alpha=alpha
                    )
                    self.model = mydmap
                    return mydmap.fit_transform(X)
                except ImportError:
                    # Fallback: use Laplacian Eigenmaps as approximation with robust parameters
                    n_samples = X.shape[0]
                    n_components = min(self.params.get('n_components', 10), n_samples - 2, 5)  # Cap at 5
                    n_neighbors = min(self.params.get('n_neighbors', 10), n_samples - 1)
                    
                    # Ensure n_components is valid
                    if n_components < 2:
                        print(f"Warning: Too few samples ({n_samples}) for DiffusionMaps, need at least 3")
                        return None
                    
                    # Use dense solver for N <= 1000 to avoid ARPACK convergence issues
                    eigen_solver = 'dense' if n_samples <= 1000 else 'arpack'
                    
                    params = {
                        'n_components': n_components,
                        'n_neighbors': n_neighbors,
                        'affinity': 'rbf',
                        'eigen_solver': eigen_solver,
                        'n_jobs': 1,
                    }
                    
                    # Add ARPACK-specific parameters for larger datasets
                    if eigen_solver == 'arpack':
                        params['eigen_tol'] = 1e-4
                        params['max_iter'] = 2000
                    
                    self.model = SpectralEmbedding(**params)
                    return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return self._validate_output(result)
        except Exception as e:
            print(f"Diffusion Maps failed: {str(e)}")
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
                
                # Encode TEST set only (never seen during training)
                self.model.eval()
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    encoded = self.model.encode(X_test_tensor)
                    return encoded.cpu().numpy(), test_idx
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            # Store test indices in metadata for downstream alignment
            encoded_data, test_indices = result
            self.test_indices = test_indices
            return self._validate_output(encoded_data)
            
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
                
                # Encode TEST set only (never seen during training)
                self.model.eval()
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    mu, _ = self.model.encode(X_test_tensor)
                    return mu.cpu().numpy(), test_idx
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            # Store test indices in metadata for downstream alignment
            encoded_data, test_indices = result
            self.test_indices = test_indices
            return self._validate_output(encoded_data)
            
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"VAE failed: {str(e)}")
            return None


class CEEDReducer(DimensionalityReductionBase):
    """
    Contrastive Encoder for Event Detection (CEED) - SUPERVISED METHOD.
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
                
                # Check if labels are provided - CEED requires labels for supervised contrastive learning
                if y is None:
                    print("CEED requires labels (y) for supervised contrastive learning. Skipping.")
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
                        # print(f"CEED Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
                
                # Encode TEST set only (never seen during training)
                self.model.eval()
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    encoded = self.model(X_test_tensor)
                    return encoded.cpu().numpy(), test_idx
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            if result is None:
                return None
            
            # Store test indices in metadata for downstream alignment
            encoded_data, test_indices = result
            self.test_indices = test_indices
            return self._validate_output(encoded_data)
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"CEED failed: {str(e)}")
            return None


# ============================================================================
# NOT APPLICABLE: GPFA is designed for temporal neural population dynamics,
# not individual spike waveforms
# ============================================================================

# class GPFAReducer(DimensionalityReductionBase):
#     """
#     Gaussian Process Factor Analysis (GPFA) - NOT APPLICABLE.
#     Extracts smooth, low-dimensional neural trajectories from population activity.
#     This method is designed for time-series data, not spike waveforms.
#     Reference: https://elephant.readthedocs.io/en/latest/tutorials/gpfa.html
#     """
#     
#     def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         try:
#             from elephant.gpfa import GPFA
#             import quantities as pq
#             from neo import SpikeTrain
#             import neo
#             
#             # Default parameters
#             x_dim = self.params.get('x_dim', 10)  # Latent dimensionality
#             bin_size = self.params.get('bin_size', 20)  # ms
#             
#             def _fit():
#                 # GPFA expects data as list of trials, each trial is (n_neurons, n_timebins)
#                 # For spike sorting, we treat each sample as a "trial"
#                 n_samples, n_features = X.shape
#                 
#                 # Reshape data: treat each sample as a separate trial
#                 # GPFA works with spike trains, so we need to convert our features
#                 trials = []
#                 for i in range(n_samples):
#                     # Treat each feature as a neuron's activity over time
#                     trial_data = X[i:i+1, :].T  # (n_features, 1)
#                     trials.append(trial_data)
#                 
#                 # Initialize GPFA
#                 gpfa = GPFA(x_dim=x_dim, bin_size=bin_size*pq.ms)
#                 
#                 # Fit and transform
#                 # Note: This is a simplified approach - GPFA typically works with temporal data
#                 # For spike waveforms, we're treating features as pseudo-temporal bins
#                 try:
#                     trajectories = gpfa.fit_transform(trials)
#                     
#                     # Extract latent states
#                     latent_states = []
#                     for traj in trajectories:
#                         # Get the mean latent state for this trial
#                         latent_states.append(traj.mean(axis=1))
#                     
#                     result = np.array(latent_states)
#                     return result
#                     
#                 except Exception as e:
#                     # Fallback: Use Factor Analysis if GPFA fails
#                     print(f"GPFA fitting failed, using Factor Analysis fallback: {str(e)}")
#                     from sklearn.decomposition import FactorAnalysis
#                     fa = FactorAnalysis(n_components=x_dim, random_state=42)
#                     return fa.fit_transform(X)
#             
#             result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#             return result
#             
#         except ImportError:
#             print("Elephant not installed. Install with: pip install elephant quantities neo")
#             print("Using Factor Analysis as fallback...")
#             try:
#                 from sklearn.decomposition import FactorAnalysis
#                 x_dim = self.params.get('x_dim', 10)
#                 
#                 def _fit():
#                     fa = FactorAnalysis(n_components=x_dim, random_state=42)
#                     return fa.fit_transform(X)
#                 
#                 result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#                 return result
#             except Exception as e:
#                 print(f"Fallback also failed: {str(e)}")
#                 return None
#                 
#         except Exception as e:
#             print(f"GPFA failed: {str(e)}")
#             return None


# ============================================================================
# NOT APPLICABLE: Slice TCA is designed for tensor-structured neural population
# data across conditions/trials, not individual spike waveforms
# ============================================================================

# class SliceTCAReducer(DimensionalityReductionBase):
#     """
#     Slice Tensor Component Analysis (Slice TCA) - NOT APPLICABLE.
#     Identifies low-dimensional structure in neural population activity.
#     This method is designed for tensor-structured data, not spike waveforms.
#     Reference: https://www.nature.com/articles/s41593-024-01626-2
#     
#     Simplified implementation using Tucker decomposition.
#     """
#     
#     def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         try:
#             # Default parameters
#             n_components = self.params.get('n_components', 10)
#             n_slices = self.params.get('n_slices', 5)
#             
#             def _fit():
#                 try:
#                     # Try using tensorly for proper tensor decomposition
#                     import tensorly as tl
#                     from tensorly.decomposition import tucker
#                     
#                     n_samples, n_features = X.shape
#                     
#                     # Reshape data into tensor: (samples, features/slices, slices)
#                     # This is a simplified approach - actual Slice TCA is more complex
#                     slice_size = n_features // n_slices
#                     if slice_size * n_slices < n_features:
#                         # Pad to make it divisible
#                         pad_size = slice_size * n_slices - n_features
#                         X_padded = np.pad(X, ((0, 0), (0, abs(pad_size))), mode='constant')
#                     else:
#                         X_padded = X[:, :slice_size * n_slices]
#                     
#                     # Reshape into 3D tensor
#                     X_tensor = X_padded.reshape(n_samples, slice_size, n_slices)
#                     
#                     # Apply Tucker decomposition
#                     core, factors = tucker(X_tensor, rank=[n_components, slice_size, n_slices])
#                     
#                     # Use the first factor (across samples) as the reduced representation
#                     result = factors[0]
#                     
#                     return result
#                     
#                 except ImportError:
#                     # Fallback: Use NMF (Non-negative Matrix Factorization)
#                     print("Tensorly not installed. Using NMF as fallback for Slice TCA.")
#                     from sklearn.decomposition import NMF
#                     
#                     # Ensure non-negative data for NMF
#                     X_nonneg = X - X.min() + 1e-10
#                     
#                     nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
#                     result = nmf.fit_transform(X_nonneg)
#                     
#                     return result
#             
#             result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#             return result
#             
#         except Exception as e:
#             print(f"Slice TCA failed: {str(e)}")
#             return None


# ============================================================================
# NOT APPLICABLE: LFADS is designed for temporal neural population dynamics,
# not individual spike waveforms
# ============================================================================

# class LFADSReducer(DimensionalityReductionBase):
#     """
#     Latent Factor Analysis via Dynamical Systems (LFADS) - NOT APPLICABLE.
#     Uses recurrent neural networks to infer latent dynamics from neural data.
#     This method is designed for sequential/temporal data, not spike waveforms.
#     Reference: https://www.nature.com/articles/s41592-018-0109-9
#     
#     Simplified implementation using LSTM autoencoder.
#     """
#     
#     def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         try:
#             import torch
#             import torch.nn as nn
#             import torch.optim as optim
#             from torch.utils.data import TensorDataset, DataLoader
#             
#             # Default parameters
#             latent_dim = self.params.get('latent_dim', 10)
#             hidden_dim = self.params.get('hidden_dim', 64)
#             num_layers = self.params.get('num_layers', 2)
#             epochs = self.params.get('epochs', 50)
#             batch_size = self.params.get('batch_size', 32)
#             learning_rate = self.params.get('learning_rate', 0.001)
#             sequence_length = self.params.get('sequence_length', 10)
#             
#             class LFADSEncoder(nn.Module):
#                 def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
#                     super().__init__()
#                     self.hidden_dim = hidden_dim
#                     self.num_layers = num_layers
#                     
#                     # LSTM encoder
#                     self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
#                                        batch_first=True, dropout=0.2 if num_layers > 1 else 0)
#                     
#                     # Map to latent space
#                     self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#                     self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
#                     
#                     # LSTM decoder
#                     self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, num_layers,
#                                                 batch_first=True, dropout=0.2 if num_layers > 1 else 0)
#                     self.decoder_fc = nn.Linear(hidden_dim, input_dim)
#                 
#                 def encode(self, x):
#                     # x: (batch, seq_len, input_dim)
#                     _, (h_n, _) = self.lstm(x)
#                     # Use last hidden state
#                     h = h_n[-1]  # (batch, hidden_dim)
#                     mu = self.fc_mu(h)
#                     logvar = self.fc_logvar(h)
#                     return mu, logvar
#                 
#                 def reparameterize(self, mu, logvar):
#                     std = torch.exp(0.5 * logvar)
#                     eps = torch.randn_like(std)
#                     return mu + eps * std
#                 
#                 def decode(self, z, seq_len):
#                     # z: (batch, latent_dim)
#                     # Repeat z for each time step
#                     z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
#                     h, _ = self.decoder_lstm(z_seq)
#                     output = self.decoder_fc(h)
#                     return output
#                 
#                 def forward(self, x):
#                     mu, logvar = self.encode(x)
#                     z = self.reparameterize(mu, logvar)
#                     recon = self.decode(z, x.size(1))
#                     return recon, mu, logvar
#             
#             def lfads_loss(recon_x, x, mu, logvar):
#                 # Reconstruction loss
#                 recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#                 # KL divergence
#                 kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#                 return recon_loss + kld
#             
#             def _fit():
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 
#                 n_samples, n_features = X.shape
#                 
#                 # Reshape data into sequences
#                 # Pad if necessary
#                 if n_features % sequence_length != 0:
#                     pad_size = sequence_length - (n_features % sequence_length)
#                     X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='edge')
#                 else:
#                     X_padded = X
#                 
#                 # Reshape: (n_samples, sequence_length, features_per_step)
#                 features_per_step = X_padded.shape[1] // sequence_length
#                 X_seq = X_padded.reshape(n_samples, sequence_length, features_per_step)
#                 
#                 # Prepare data
#                 X_tensor = torch.FloatTensor(X_seq).to(device)
#                 dataset = TensorDataset(X_tensor)
#                 dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#                 
#                 # Create model
#                 self.model = LFADSEncoder(features_per_step, hidden_dim, latent_dim, num_layers).to(device)
#                 optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#                 
#                 # Train
#                 self.model.train()
#                 for epoch in range(epochs):
#                     for batch in dataloader:
#                         batch_X = batch[0]
#                         optimizer.zero_grad()
#                         recon, mu, logvar = self.model(batch_X)
#                         loss = lfads_loss(recon, batch_X, mu, logvar)
#                         loss.backward()
#                         optimizer.step()
#                 
#                 # Extract latent representations
#                 self.model.eval()
#                 with torch.no_grad():
#                     mu, _ = self.model.encode(X_tensor)
#                     return mu.cpu().numpy()
#             
#             result, self.computation_time, self.memory_usage = self._track_performance(_fit)
#             return result
#             
#         except ImportError:
#             print("PyTorch not installed. Install with: pip install torch")
#             return None
#         except Exception as e:
#             print(f"LFADS failed: {str(e)}")
#             return None


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
        'CEED': CEEDReducer,
        
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
