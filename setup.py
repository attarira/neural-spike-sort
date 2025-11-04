"""Setup script for spike_sort package."""

from setuptools import setup, find_packages

setup(
    name="spike_sort",
    version="1.0.0",
    description="Framework for comparative analysis of spike sorting algorithms",
    author="Muhammad, Ryan, Rayaan",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "full": [
            "umap-learn>=0.5.0",
            "torch>=1.10.0",
            "hmmlearn>=0.2.7",
        ],
    },
)
