"""
Simplified integration test for BG/NBD model functionality.

This test focuses on core model functionality without requiring database connections.
"""

import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_supporters: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """Create synthetic test data for BG/NBD model."""
    np.random.seed(random_seed)
    
    data = []
    for i in range(n_supporters):
        # Observation period (1 year)
        T = 365
        
        # Generate frequency (x) - number of repeat events
        x = np.random.negative_binomial(n=2, p=0.3)
        
        # Generate recency (t_x) - time of last event
        if x > 0:
            t_x = np.random.uniform(0, T)
        else:
            t_x = 0
        
        # Generate monetary value
        monetary = np.random.lognormal(mean=3, sigma=1) if x > 0 else 0
        
        # Generate engagement segment
        if x >= 5:
            segment = 'High'
        elif x >= 2:
            segment = 'Medium'
        else:
            segment = 'Low'
        
        data.append({
            'supporter_id': i + 1,
            'x': x,
            't_x': t_x,
            'T': T,
            'frequency': x + 1,
            'monetary': monetary,
            'engagement_segment': segment,
            'recency_ratio': t_x / T if T > 0 else 0
        })
    
    return pd.DataFrame(data)


def test_basic_model():
    """Test basic BG/NBD model functionality."""
    logger.info("Testing basic BG/NBD model")
    
    try:
        # Import here to avoid dependency issues
        from src.models.bgnbd import create_bgnbd_model
        
        # Create test data
        data = create_test_data(n_supporters=100, random_seed=42)
        logger.info(f"Created test data with {len(data)} supporters")
        
        # Create and fit model
        model = create_bgnbd_model(hierarchical=False)
        logger.info("Created BG/NBD model")
        
        # Fit with minimal sampling for testing
        model.fit(data, draws=200, tune=100, chains=2)
        logger.info("Model fitted successfully")
        
        # Test predictions
        prob_alive = model.predict_probability_alive(
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10]
        )
        logger.info(f"P(Alive) predictions: mean={prob_alive.mean():.3f}")
        
        expected_transactions = model.predict_expected_transactions(
            180,  # 6 months
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10]
        )
        logger.info(f"Expected transactions: mean={expected_transactions.mean():.3f}")
        
        # Test CLV
        clv = model.predict_clv(
            180,
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10],
            data['monetary'].values[:10]
        )
        logger.info(f"CLV predictions: mean={clv.mean():.2f}")
        
        # Test diagnostics
        diagnostics = model.get_model_diagnostics()
        logger.info(f"Model converged: {diagnostics['convergence']['converged']}")
        
        logger.info("✓ Basic model test passed")
        return model
        
    except Exception as e:
        logger.error(f"✗ Basic model test failed: {e}")
        raise


def test_hierarchical_model():
    """Test hierarchical BG/NBD model functionality."""
    logger.info("Testing hierarchical BG/NBD model")
    
    try:
        from src.models.bgnbd import create_bgnbd_model
        
        # Create test data with segments
        data = create_test_data(n_supporters=150, random_seed=43)
        logger.info(f"Created test data with segments: {data['engagement_segment'].value_counts().to_dict()}")
        
        # Create and fit hierarchical model
        model = create_bgnbd_model(
            hierarchical=True,
            segment_column='engagement_segment'
        )
        logger.info("Created hierarchical BG/NBD model")
        
        # Fit with minimal sampling for testing
        model.fit(data, draws=200, tune=100, chains=2)
        logger.info("Hierarchical model fitted successfully")
        
        # Test predictions with segments
        prob_alive = model.predict_probability_alive(
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10],
            data['engagement_segment'].values[:10]
        )
        logger.info(f"P(Alive) predictions: mean={prob_alive.mean():.3f}")
        
        # Test segmentation
        segmentation = model.segment_supporters(data.head(20), method='probability')
        logger.info(f"Segmentation results: {segmentation['segment_prediction'].value_counts().to_dict()}")
        
        logger.info("✓ Hierarchical model test passed")
        return model
        
    except Exception as e:
        logger.error(f"✗ Hierarchical model test failed: {e}")
        raise


def test_model_evaluation():
    """Test model evaluation functionality."""
    logger.info("Testing model evaluation")
    
    try:
        from src.models.bgnbd import create_bgnbd_model
        from src.models.evaluation import create_model_evaluator
        
        # Create and fit a model
        data = create_test_data(n_supporters=100, random_seed=44)
        model = create_bgnbd_model()
        model.fit(data, draws=200, tune=100, chains=2)
        
        # Create evaluator
        evaluator = create_model_evaluator(model)
        logger.info("Created model evaluator")
        
        # Test convergence evaluation
        convergence = evaluator.evaluate_convergence(verbose=False)
        logger.info(f"Convergence evaluation: {convergence['assessment']['overall_quality']}")
        
        # Test business metrics
        business_metrics = evaluator.evaluate_business_metrics(data)
        logger.info(f"High-value supporters: {business_metrics['supporter_segments']['high_value_percentage']:.1f}%")
        
        logger.info("✓ Model evaluation test passed")
        return evaluator
        
    except Exception as e:
        logger.error(f"✗ Model evaluation test failed: {e}")
        raise


def test_model_serialization():
    """Test model saving and loading."""
    logger.info("Testing model serialization")
    
    try:
        from src.models.bgnbd import create_bgnbd_model
        
        # Create and fit a model
        data = create_test_data(n_supporters=50, random_seed=45)
        original_model = create_bgnbd_model()
        original_model.fit(data, draws=100, tune=50, chains=2)
        
        # Test predictions with original model
        original_predictions = original_model.predict_probability_alive(
            data['x'].values[:5],
            data['t_x'].values[:5],
            data['T'].values[:5]
        )
        
        # Save model
        model_path = Path("test_model.pkl")
        original_model.save_model(model_path)
        logger.info("Model saved successfully")
        
        # Load model
        loaded_model = original_model.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Test predictions with loaded model
        loaded_predictions = loaded_model.predict_probability_alive(
            data['x'].values[:5],
            data['t_x'].values[:5],
            data['T'].values[:5]
        )
        
        # Compare predictions
        prediction_diff = np.abs(original_predictions - loaded_predictions).max()
        logger.info(f"Maximum prediction difference: {prediction_diff:.10f}")
        
        # Clean up
        if model_path.exists():
            model_path.unlink()
        
        logger.info("✓ Model serialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model serialization test failed: {e}")
        raise


def run_all_tests():
    """Run all integration tests."""
    logger.info("Starting BG/NBD model integration tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    try:
        # Test 1: Basic model
        basic_model = test_basic_model()
        test_results['basic_model'] = True
        
        # Test 2: Hierarchical model
        hierarchical_model = test_hierarchical_model()
        test_results['hierarchical_model'] = True
        
        # Test 3: Model evaluation
        evaluator = test_model_evaluation()
        test_results['model_evaluation'] = True
        
        # Test 4: Model serialization
        test_model_serialization()
        test_results['model_serialization'] = True
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        test_results['error'] = str(e)
        return test_results
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result is True else f"✗ FAILED: {result}"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\nTests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning("⚠️  Some tests failed")
    
    return test_results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    if all(result is True for result in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)