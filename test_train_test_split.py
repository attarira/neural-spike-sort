#!/usr/bin/env python3
"""
Quick test to verify train/test split implementation for deep learning methods.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from spike_sort.dimensionality_reduction import create_reducer
from spike_sort import ExperimentRunner

def test_deep_learning_splits():
    """Test that deep learning methods properly implement train/test splits."""
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_classes = 5
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    print("=" * 70)
    print("Testing Train/Test Split Implementation")
    print("=" * 70)
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print()
    
    # Test each deep learning method
    methods = {
        'Autoencoder': {
            'encoding_dim': 10,
            'hidden_dim': 30,
            'epochs': 5,  # Reduced for quick testing
            'batch_size': 64,
            'learning_rate': 0.001,
            'test_size': 0.2,
            'random_state': 42
        },
        'VAE': {
            'encoding_dim': 10,
            'hidden_dim': 30,
            'epochs': 5,  # Reduced for quick testing
            'batch_size': 64,
            'learning_rate': 0.001,
            'test_size': 0.2,
            'random_state': 42
        },
        'CEED': {
            'encoding_dim': 10,
            'hidden_dim': 30,
            'epochs': 5,  # Reduced for quick testing
            'batch_size': 64,
            'learning_rate': 0.001,
            'temperature': 0.5,
            'test_size': 0.2,
            'random_state': 42
        }
    }
    
    for method_name, params in methods.items():
        print(f"\n{'='*70}")
        print(f"Testing {method_name}")
        print(f"{'='*70}")
        
        try:
            # Create reducer
            reducer = create_reducer(method_name, **params)
            if reducer is None:
                print(f"❌ Failed to create {method_name} reducer")
                continue
            
            # Fit and transform
            if method_name == 'CEED':
                # CEED requires labels
                X_reduced = reducer.fit_transform(X, y)
            else:
                # Unsupervised methods
                X_reduced = reducer.fit_transform(X)
            
            if X_reduced is None:
                print(f"❌ {method_name} fit_transform returned None")
                continue
            
            # Check for test_indices attribute
            if hasattr(reducer, 'test_indices'):
                test_indices = reducer.test_indices
                print(f"✅ Train/test split implemented")
                print(f"   Original data size: {n_samples}")
                print(f"   Test set size: {len(test_indices)}")
                print(f"   Train set size: {n_samples - len(test_indices)}")
                print(f"   Reduced data shape: {X_reduced.shape}")
                print(f"   Test proportion: {len(test_indices) / n_samples:.2%}")
                
                # Verify reduced data matches test set size
                expected_test_size = int(n_samples * params['test_size'])
                actual_test_size = len(test_indices)
                if abs(actual_test_size - expected_test_size) <= 1:  # Allow ±1 for rounding
                    print(f"   ✅ Test set size matches expected ({expected_test_size})")
                else:
                    print(f"   ⚠️  Test set size mismatch: expected ~{expected_test_size}, got {actual_test_size}")
                
                if X_reduced.shape[0] == len(test_indices):
                    print(f"   ✅ Reduced data size matches test indices")
                else:
                    print(f"   ❌ Reduced data size ({X_reduced.shape[0]}) != test indices ({len(test_indices)})")
                
                # For CEED, verify stratification
                if method_name == 'CEED' and y is not None:
                    y_test = y[test_indices]
                    test_class_dist = np.bincount(y_test, minlength=n_classes) / len(y_test)
                    full_class_dist = np.bincount(y, minlength=n_classes) / len(y)
                    print(f"   Class distribution (full): {full_class_dist}")
                    print(f"   Class distribution (test): {test_class_dist}")
                    max_diff = np.max(np.abs(test_class_dist - full_class_dist))
                    if max_diff < 0.05:  # Allow 5% difference
                        print(f"   ✅ Stratification successful (max diff: {max_diff:.3f})")
                    else:
                        print(f"   ⚠️  Stratification may be imperfect (max diff: {max_diff:.3f})")
            else:
                print(f"❌ No test_indices attribute found (data leakage possible!)")
            
        except ImportError as e:
            print(f"⚠️  {method_name} skipped: {e}")
        except Exception as e:
            print(f"❌ {method_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

def test_experiment_runner_integration():
    """Test that ExperimentRunner properly handles train/test splits."""
    
    print("\n" + "=" * 70)
    print("Testing ExperimentRunner Integration")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {n_classes}\n")
    
    # Create minimal config
    config = {
        'dimensionality_reduction': {
            'CEED': [{
                'encoding_dim': 5,
                'hidden_dim': 20,
                'epochs': 3,
                'batch_size': 64,
                'learning_rate': 0.001,
                'temperature': 0.5,
                'test_size': 0.2,
                'random_state': 42
            }]
        },
        'clustering': {
            'KMeans': [{'n_clusters': n_classes, 'n_init': 10, 'max_iter': 100}]
        }
    }
    
    try:
        # Create runner
        runner = ExperimentRunner(
            output_dir='./test_results_train_test',
            n_jobs=1,
            verbose=True,
            silent_errors=False
        )
        
        # Run experiments
        results = runner.run_experiments(
            X=X,
            config=config,
            y=y,
            dataset_name='test_train_test_split'
        )
        
        # Check results
        if results:
            for result in results:
                if result['success']:
                    print(f"\n✅ Experiment successful:")
                    print(f"   DR method: {result['dim_reduction_method']}")
                    print(f"   Clustering: {result['clustering_method']}")
                    print(f"   Used train/test split: {result.get('used_train_test_split', False)}")
                    if result.get('used_train_test_split'):
                        print(f"   Train size: {result.get('train_size')}")
                        print(f"   Test size: {result.get('test_size')}")
                    eval_results = result.get('evaluation', {})
                    ari = eval_results.get('adjusted_rand_index')
                    if ari is not None:
                        print(f"   ARI: {ari:.4f}")
                else:
                    print(f"\n❌ Experiment failed: {result.get('error')}")
        else:
            print("⚠️  No results returned")
            
    except ImportError as e:
        print(f"⚠️  Test skipped: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Integration Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_deep_learning_splits()
    test_experiment_runner_integration()

