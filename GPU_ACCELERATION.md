# GPU Acceleration Guide

This document explains how to enable and use GPU acceleration for dimensionality reduction and clustering methods in the spike sorting pipeline.

## Overview

GPU acceleration can dramatically speed up many dimensionality reduction and clustering algorithms, especially with large datasets. The codebase now supports GPU-accelerated versions of several methods using NVIDIA's RAPIDS cuML library.

## Supported Methods

### Dimensionality Reduction
- **PCA** - Principal Component Analysis
- **UMAP** - Uniform Manifold Approximation and Projection  
- **t-SNE** - t-Distributed Stochastic Neighbor Embedding

### Clustering
- **K-Means** - K-Means clustering
- **DBSCAN** - Density-Based Spatial Clustering

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.x or 12.x)

### Option 1: Conda (Recommended)
```bash
# Create a new conda environment
conda create -n spike-sort python=3.10

# Activate the environment
conda activate spike-sort

# Install RAPIDS cuML (automatically includes cupy)
conda install -c rapidsai -c nvidia -c conda-forge cuml

# Install other requirements
pip install -r requirements.txt
```

### Option 2: Pip (Manual cupy installation)
```bash
# Install cupy for your CUDA version
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# Then install cuML (check RAPIDS docs for pip installation)
# Note: cuML via pip may have limited support; conda is recommended
```

## Usage

### Command Line

To enable GPU acceleration when running the pipeline, simply add the `--use-gpu` flag:

```bash
python examples/run_pipeline_comparison.py \
  --num-datasets 1 \
  --num-units 20 \
  --dr-groups nonlinear \
  --use-gpu \
  --n-jobs 1
```

**Important Notes:**
- When using GPU acceleration, set `--n-jobs 1` to avoid conflicts between GPU and CPU parallelization
- GPU methods automatically fall back to CPU implementations if cuML is not available
- First run may be slower due to GPU initialization and JIT compilation

### Python API

You can also enable GPU acceleration programmatically when creating reducers or clustering methods:

```python
from spike_sort.dimensionality_reduction import create_reducer
from spike_sort.clustering import create_clustering

# Enable GPU for dimensionality reduction
reducer = create_reducer('UMAP', 
                        n_components=10,
                        n_neighbors=15,
                        use_gpu=True)

# Enable GPU for clustering
clusterer = create_clustering('KMeans',
                             n_clusters=20,
                             use_gpu=True)
```

## Performance Considerations

### When to Use GPU Acceleration

✅ **Good Use Cases:**
- Large datasets (>10,000 samples)
- Multiple hyperparameter configurations
- High-dimensional data
- Repeated experiments

❌ **When CPU May Be Faster:**
- Small datasets (<1,000 samples)
- Single runs
- Low-dimensional data
- Systems with limited GPU memory

### Memory Management

GPU memory is typically more limited than CPU RAM. If you encounter out-of-memory errors:

1. **Reduce batch size** for deep learning methods
2. **Process fewer datasets** simultaneously
3. **Use PCA preprocessing** to reduce dimensionality before GPU methods
4. **Monitor GPU memory** usage:
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv -l 1
   ```

### Benchmarking

To compare CPU vs GPU performance:

```bash
# CPU only
time python examples/run_pipeline_comparison.py \
  --num-datasets 1 --dr-groups nonlinear

# GPU accelerated
time python examples/run_pipeline_comparison.py \
  --num-datasets 1 --dr-groups nonlinear --use-gpu --n-jobs 1
```

## Troubleshooting

### cuML not found
```
Warning: cuML not available. Falling back to CPU version.
```
**Solution:** Install cuML using conda as shown in the Installation section.

### CUDA version mismatch
```
Error: CUDA version mismatch
```
**Solution:** Ensure your CUDA toolkit version matches the cupy version you installed.

### Out of memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce the number of samples using `--subset-duration`
- Use PCA to reduce dimensionality first
- Close other GPU applications
- Use a GPU with more memory

### Slower than CPU
If GPU is slower than CPU:
- Your dataset may be too small for GPU overhead to be worthwhile
- Check that you've set `--n-jobs 1` to avoid CPU/GPU conflicts  
- Verify GPU is actually being used (check `nvidia-smi` during execution)

## Implementation Details

### Fallback Mechanism

All GPU-accelerated methods include automatic fallback to CPU:

1. Check if `use_gpu=True` is specified
2. Try to import cuML and cupy
3. If available, run on GPU and convert results back to numpy
4. If not available or if GPU execution fails, fall back to CPU implementation
5. Print informative warnings about the fallback

This ensures the pipeline works seamlessly regardless of whether GPU support is available.

### Data Transfer

Data is automatically transferred between CPU and GPU:
- Input data (numpy arrays) → GPU (cupy arrays)
- GPU computation
- Output data (cupy arrays) → CPU (numpy arrays)

The transfer overhead is typically negligible compared to computation time for large datasets.

## References

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [cupy Documentation](https://docs.cupy.dev/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Questions?

If you encounter issues with GPU acceleration, please check:
1. That your GPU and drivers support CUDA
2. That cuML and cupy are properly installed
3. The error messages in the fallback warnings

For additional help, refer to the RAPIDS community forums or GitHub issues.

