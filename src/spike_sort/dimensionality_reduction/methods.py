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
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
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
            return result
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
            return result
        except Exception as e:
            print(f"ICA failed: {str(e)}")
            return None


class CCAReducer(DimensionalityReductionBase):
    """Canonical Correlation Analysis."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            if y is None:
                # Split X into two views for unsupervised CCA
                mid = X.shape[1] // 2
                X1, X2 = X[:, :mid], X[:, mid:]
            else:
                X1, X2 = X, y.reshape(-1, 1) if y.ndim == 1 else y
            
            def _fit():
                self.model = CCA(**self.params)
                X_c, _ = self.model.fit_transform(X1, X2)
                return X_c
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except Exception as e:
            print(f"CCA failed: {str(e)}")
            return None


class LDAReducer(DimensionalityReductionBase):
    """Linear Discriminant Analysis (supervised)."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            if y is None:
                print("LDA requires labels (y). Skipping.")
                return None
            
            def _fit():
                self.model = LinearDiscriminantAnalysis(**self.params)
                return self.model.fit_transform(X, y)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except Exception as e:
            print(f"LDA failed: {str(e)}")
            return None


class TSNEReducer(DimensionalityReductionBase):
    """t-Distributed Stochastic Neighbor Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = TSNE(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
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
            return result
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
            return result
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
            return result
        except Exception as e:
            print(f"Laplacian Eigenmaps failed: {str(e)}")
            return None


class LLEReducer(DimensionalityReductionBase):
    """Locally Linear Embedding."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            def _fit():
                self.model = LocallyLinearEmbedding(**self.params)
                return self.model.fit_transform(X)
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except Exception as e:
            print(f"LLE failed: {str(e)}")
            return None


class AutoencoderReducer(DimensionalityReductionBase):
    """Autoencoder-based dimensionality reduction."""
    
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
                
                # Prepare data
                X_tensor = torch.FloatTensor(X).to(device)
                dataset = TensorDataset(X_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = Autoencoder(X.shape[1], hidden_dim, encoding_dim).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train
                self.model.train()
                for epoch in range(epochs):
                    for batch in dataloader:
                        batch_X = batch[0]
                        optimizer.zero_grad()
                        output = self.model(batch_X)
                        loss = criterion(output, batch_X)
                        loss.backward()
                        optimizer.step()
                
                # Encode
                self.model.eval()
                with torch.no_grad():
                    encoded = self.model.encode(X_tensor)
                    return encoded.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"Autoencoder failed: {str(e)}")
            return None


class VAEReducer(DimensionalityReductionBase):
    """Variational Autoencoder-based dimensionality reduction."""
    
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
                
                # Prepare data
                X_tensor = torch.FloatTensor(X).to(device)
                dataset = TensorDataset(X_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = VAE(X.shape[1], hidden_dim, encoding_dim).to(device)
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train
                self.model.train()
                for epoch in range(epochs):
                    for batch in dataloader:
                        batch_X = batch[0]
                        optimizer.zero_grad()
                        recon_batch, mu, logvar = self.model(batch_X)
                        loss = vae_loss(recon_batch, batch_X, mu, logvar)
                        loss.backward()
                        optimizer.step()
                
                # Encode
                self.model.eval()
                with torch.no_grad():
                    mu, _ = self.model.encode(X_tensor)
                    return mu.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"VAE failed: {str(e)}")
            return None


class CEEDReducer(DimensionalityReductionBase):
    """
    Contrastive Encoder for Event Detection (CEED).
    Simplified implementation using contrastive learning principles.
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
            
            def contrastive_loss(z, temperature):
                # Simplified contrastive loss
                z = nn.functional.normalize(z, dim=1)
                similarity_matrix = torch.matmul(z, z.T) / temperature
                batch_size = z.shape[0]
                
                # Create positive pairs (adjacent samples)
                labels = torch.arange(batch_size).to(z.device)
                loss = nn.functional.cross_entropy(similarity_matrix, labels)
                return loss
            
            def _fit():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Prepare data
                X_tensor = torch.FloatTensor(X).to(device)
                dataset = TensorDataset(X_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Create model
                self.model = ContrastiveEncoder(X.shape[1], hidden_dim, encoding_dim).to(device)
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                
                # Train
                self.model.train()
                for epoch in range(epochs):
                    for batch in dataloader:
                        batch_X = batch[0]
                        optimizer.zero_grad()
                        z = self.model(batch_X)
                        loss = contrastive_loss(z, temperature)
                        loss.backward()
                        optimizer.step()
                
                # Encode
                self.model.eval()
                with torch.no_grad():
                    encoded = self.model(X_tensor)
                    return encoded.cpu().numpy()
            
            result, self.computation_time, self.memory_usage = self._track_performance(_fit)
            return result
        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return None
        except Exception as e:
            print(f"CEED failed: {str(e)}")
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
        'PCA': PCAReducer,
        'ICA': ICAReducer,
        'CCA': CCAReducer,
        'LDA': LDAReducer,
        'TSNE': TSNEReducer,
        't-SNE': TSNEReducer,
        'UMAP': UMAPReducer,
        'Isomap': IsomapReducer,
        'LaplacianEigenmaps': LaplacianEigenmapsReducer,
        'LLE': LLEReducer,
        'Autoencoder': AutoencoderReducer,
        'VAE': VAEReducer,
        'CEED': CEEDReducer,
    }
    
    reducer_class = reducers.get(method_name)
    if reducer_class is None:
        print(f"Unknown dimensionality reduction method: {method_name}")
        return None
    
    return reducer_class(**params)
