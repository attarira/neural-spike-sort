"""
Neural Spike Sorting Analysis Framework

A modular framework for comparative analysis of dimensionality reduction
and clustering techniques for spike sorting.
"""

__version__ = "1.0.0"

from .dimensionality_reduction.methods import create_reducer
from .clustering.methods import create_clustering
from .evaluation.metrics import evaluate_all
from .utils.runner import ExperimentRunner
from .utils.analyzer import ResultsAnalyzer

__all__ = [
    'create_reducer',
    'create_clustering',
    'evaluate_all',
    'ExperimentRunner',
    'ResultsAnalyzer',
]
