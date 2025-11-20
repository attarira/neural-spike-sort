# Neural Spike Sorting Analysis Framework

A modular Python framework for comparative analysis of dimensionality reduction and clustering techniques for spike sorting.

**Group Project for COMS 4774: Unsupervised Learning**  
Members: Muhammad, Ryan, Rayaan

## 🎯 Features

- **Unified Interface**: Clean, extensible base classes for all methods
- **Comprehensive Methods**: 
  - **10 Dimensionality Reduction Methods**: Linear (PCA, ICA), Manifold (t-SNE, UMAP, Isomap, Laplacian Eigenmaps, LLE), Deep Learning (Autoencoder, VAE, CEED)
  - **9 Clustering Methods**: K-Means, GMM, DBSCAN, Spectral, Dirichlet Process, HMM, Agglomerative, OPTICS, Mean Shift
- **Robust Evaluation**: Multiple metrics with and without ground truth
- **Parallel Execution**: Support for multiprocessing to speed up experiments
- **Comprehensive Analysis**: Automated visualization and reporting tools
- **Flexible Configuration**: YAML/JSON-based experiment definitions

## 📁 Project Structure

```
neural-spike-sort/
├── src/spike_sort/              # Main package
│   ├── dimensionality_reduction/
│   │   └── methods.py           # 13 dimensionality reduction methods
│   ├── clustering/
│   │   └── methods.py           # 9 clustering methods
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── runner.py            # Experiment orchestration
│       └── analyzer.py          # Results analysis & visualization
├── configs/
│   └── config_example.yaml      # Example configuration
├── examples/
│   └── run_experiments.py       # Example usage script
├── tests/
│   └── test_framework.py        # Test suite
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neural-spike-sort
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
# Option 1: Install in development mode (recommended)
pip install -e .

# Option 2: Install with all optional dependencies
pip install -e ".[full]"

# Option 3: Just install requirements
pip install -r requirements.txt
```

### Test Installation

```bash
python tests/test_framework.py
```

### Basic Usage

```python
import numpy as np
from spike_sort import ExperimentRunner, ResultsAnalyzer
from spike_sort.utils import create_default_config

# Load your preprocessed spike data
X = np.load('your_spike_data.npy')  # Shape: (n_samples, n_features)
y = np.load('your_labels.npy')      # Optional ground truth

# Run experiments
runner = ExperimentRunner(output_dir='./results', n_jobs=-1)
config = create_default_config()
results = runner.run_experiments(X, config, y, dataset_name='my_data')
runner.save_results()

# Analyze results
analyzer = ResultsAnalyzer(results)
analyzer.generate_full_report(output_dir='./results/report')
```

### Run Example

```bash
python examples/run_experiments.py
```

## 📊 Configuration

Define experiments using YAML or JSON:

```yaml
dimensionality_reduction:
  PCA:
    - n_components: 10
    - n_components: 20
  UMAP:
    - n_neighbors: 15
      min_dist: 0.1
      n_components: 10

clustering:
  KMeans:
    - n_clusters: 3
    - n_clusters: 5
  GMM:
    - n_components: 3
    - n_components: 5
  DBSCAN:
    - eps: 0.5
      min_samples: 5
```

## 🔬 Implemented Methods

### Dimensionality Reduction (10 Methods)

#### Linear Methods
1. **PCA** (Principal Component Analysis) - Identifies orthogonal directions of maximum variance in spike waveforms, useful for capturing the main shape variations
2. **ICA** (Independent Component Analysis) - Separates spike waveforms into statistically independent components, effective for unmixing overlapping neural signals

#### Nonlinear Manifold Methods
3. **t-SNE** (t-Distributed Stochastic Neighbor Embedding) - Preserves local neighborhood structure of spike waveforms, excellent for visualization of cluster separation
4. **UMAP** (Uniform Manifold Approximation and Projection) - Preserves both local and global structure of spike data, faster than t-SNE with better scalability
5. **Isomap** (Isometric Mapping) - Preserves geodesic distances in spike feature space, captures nonlinear manifold structure
6. **Laplacian Eigenmaps** - Graph-based method that preserves local relationships between similar spike waveforms
7. **LLE** (Locally Linear Embedding) - Assumes spike waveforms lie on a locally linear manifold, preserves local geometric properties

#### Deep Learning Methods
8. **Autoencoder** - Neural network that learns compressed representations of spike waveforms through reconstruction
9. **VAE** (Variational Autoencoder) - Probabilistic autoencoder that learns a smooth latent space of spike features with uncertainty quantification
10. **CEED** (Contrastive Encoder for Event Detection) - **Supervised method** that uses contrastive learning to learn discriminative spike representations (requires ground truth labels)

#### Methods Not Applicable to Spike Waveforms (Commented Out)
- **CCA** (Canonical Correlation Analysis) - Requires two data views/modalities
- **LDA** (Linear Discriminant Analysis) - Requires ground truth labels (supervised)
- **GPFA** (Gaussian Process Factor Analysis) - Designed for temporal neural population dynamics
- **Slice TCA** (Slice Tensor Component Analysis) - Designed for tensor-structured data across conditions
- **LFADS** (Latent Factor Analysis via Dynamical Systems) - Designed for temporal sequential data

### Clustering Methods (9 Methods)

1. **K-Means** - Partitions spikes into K clusters by minimizing within-cluster variance; fast and effective when cluster shapes are spherical
2. **GMM** (Gaussian Mixture Model) - Probabilistic clustering assuming spikes from each neuron follow a Gaussian distribution; provides soft cluster assignments
3. **DBSCAN** (Density-Based Spatial Clustering) - Identifies clusters of arbitrary shape based on spike density; automatically detects outliers/noise
4. **Spectral Clustering** - Uses graph-based similarity to find clusters; effective for non-convex cluster shapes in spike feature space
5. **Dirichlet Process Mixture** - Bayesian nonparametric method that automatically determines the number of neurons from the data
6. **HMM** (Hidden Markov Model) - Models temporal dependencies in spike sequences; useful when spike patterns change over time
7. **Agglomerative Clustering** - Hierarchical clustering that builds a dendrogram of spike similarities; allows exploring clusters at multiple scales
8. **OPTICS** (Ordering Points To Identify Clustering Structure) - Density-based method that handles varying density clusters better than DBSCAN
9. **Mean Shift** - Non-parametric clustering that finds modes in spike feature density; automatically determines number of clusters

## 📈 Evaluation Metrics

### With Ground Truth
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- V-measure
- Accuracy (with Hungarian matching)

### Without Ground Truth
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

### Additional Statistics
- Number of clusters detected
- Cluster size distribution
- Noise point ratio
- Computation time and memory usage

## 📊 Visualization Tools

The framework provides comprehensive visualization capabilities:

- **Performance Heatmaps**: Compare method combinations
- **Bar Charts**: Compare individual methods
- **Scatter Plots**: Explore metric relationships
- **Runtime Analysis**: Compare computational efficiency
- **Hyperparameter Analysis**: Understand parameter effects

## 🔧 Advanced Usage

### Parallel Execution

```python
# Use all CPU cores
runner = ExperimentRunner(output_dir='./results', n_jobs=-1)

# Use specific number of cores
runner = ExperimentRunner(output_dir='./results', n_jobs=4)
```

### Load and Analyze Previous Results

```python
from spike_sort import ResultsAnalyzer

analyzer = ResultsAnalyzer()
analyzer.load_results('./results/results_20240101_120000.pkl')

# Get best configurations
best = analyzer.get_best_configurations(metric='silhouette_score', n_top=10)

# Create visualizations
analyzer.create_performance_heatmap(metric='silhouette_score', save_path='./heatmap.png')
analyzer.export_summary_table(output_path='./summary.csv', format='csv')
```

## 🛠️ Error Handling

The framework includes robust error handling:
- Methods that fail return `None` and log errors
- Intermediate results are saved to avoid data loss
- Failed experiments don't stop the entire run
- Detailed error messages for debugging

## 📝 Output Files

After running experiments, you'll find:

```
results/
├── results_YYYYMMDD_HHMMSS.json    # Results in JSON format
├── results_YYYYMMDD_HHMMSS.pkl     # Results in pickle format
├── experiment_config.yaml           # Configuration used
├── intermediate/                    # Individual experiment results
├── full_report/                     # Complete analysis report
│   ├── summary_statistics.csv
│   ├── best_configs_*.csv
│   ├── heatmap_*.png
│   ├── barplot_*.png
│   ├── runtime_comparison.png
│   └── summary_table_*.csv
└── *.png                           # Individual visualizations
```

## 🔍 Troubleshooting

### Import Errors
If you encounter import errors for optional dependencies:
- **UMAP**: `pip install umap-learn`
- **PyTorch** (for Autoencoder, VAE, CEED, LFADS): `pip install torch`
- **HMMlearn**: `pip install hmmlearn`
- **GPFA**: `pip install elephant quantities neo`
- **Slice TCA**: `pip install tensorly`

### Memory Issues
For large datasets:
- Use sequential execution (`n_jobs=1`)
- Reduce the number of experiments
- Use dimensionality reduction methods with lower target dimensions

### Slow Execution
- Enable parallel processing (`n_jobs=-1`)
- Use faster methods (PCA, K-Means) for initial exploration
- Reduce the number of hyperparameter combinations

## 📚 Project Information

**Course**: COMS 4774: Unsupervised Learning  
**Team**: Muhammad, Ryan, Rayaan  
**Year**: 2024

This framework was developed for comparative analysis of spike sorting algorithms, combining multiple dimensionality reduction and clustering techniques with comprehensive evaluation metrics.