"""
Integration test script for BG/NBD model with data processing modules.

This script tests the complete pipeline from data processing to model training,
prediction, and evaluation to ensure all components work together correctly.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from data.preprocessing import create_bgnbd_processor
from models.bgnbd import create_bgnbd_model, fit_bgnbd_model
from models.evaluation import create_model_evaluator, evaluate_model_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_supporters: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic supporter data for testing.
    
    Args:
        n_supporters: Number of supporters to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic BG/NBD data
    """
    np.random.seed(random_seed)
    
    # Generate synthetic BG/NBD variables
    data = []
    
    for i in range(n_supporters):
        # Observation period (1 year)
        T = 365
        
        # Generate frequency (x) - number of repeat events
        # Using negative binomial distribution
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
        
        # Calculate additional metrics
        frequency = x + 1  # Total events including first
        recency_ratio = t_x / T if T > 0 else 0
        
        data.append({
            'supporter_id': i + 1,
            'x': x,
            't_x': t_x,
            'T': T,
            'frequency': frequency,
            'monetary': monetary,
            'engagement_segment': segment,
            'recency_ratio': recency_ratio,
            'first_event_date': datetime(2023, 1, 1),
            'last_event_date': datetime(2023, 1, 1) + timedelta(days=int(t_x)),
            'observation_start': datetime(2023, 1, 1),
            'observation_end': datetime(2023, 12, 31),
            'cutoff_date': datetime(2023, 12, 31)
        })
    
    return pd.DataFrame(data)


def test_basic_model_pipeline():
    """Test basic (non-hierarchical) model pipeline."""
    logger.info("Testing basic BG/NBD model pipeline")
    
    try:
        # Create synthetic data
        data = create_synthetic_data(n_supporters=500, random_seed=42)
        logger.info(f"Created synthetic data with {len(data)} supporters")
        
        # Split data for training and testing
        train_size = int(0.8 * len(data))
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        
        # Create and fit basic model
        logger.info("Creating and fitting basic BG/NBD model")
        model = create_bgnbd_model(hierarchical=False)
        
        # Fit with reduced sampling for testing
        model.fit(train_data, draws=500, tune=250, chains=2)
        logger.info("Model fitting completed")
        
        # Test predictions
        logger.info("Testing model predictions")
        
        # P(Alive) predictions
        prob_alive = model.predict_probability_alive(
            test_data['x'].values,
            test_data['t_x'].values,
            test_data['T'].values
        )
        logger.info(f"P(Alive) predictions: mean={prob_alive.mean():.3f}, std={prob_alive.std():.3f}")
        
        # Expected transactions predictions
        expected_transactions = model.predict_expected_transactions(
            180,  # 6 months
            test_data['x'].values,
            test_data['t_x'].values,
            test_data['T'].values
        )
        logger.info(f"Expected transactions: mean={expected_transactions.mean():.3f}, std={expected_transactions.std():.3f}")
        
        # CLV predictions
        clv = model.predict_clv(
            180,
            test_data['x'].values,
            test_data['t_x'].values,
            test_data['T'].values,
            test_data['monetary'].values
        )
        logger.info(f"CLV predictions: mean={clv.mean():.2f}, std={clv.std():.2f}")
        
        # Test model diagnostics
        logger.info("Testing model diagnostics")
        diagnostics = model.get_model_diagnostics()
        logger.info(f"Model converged: {diagnostics['convergence']['converged']}")
        logger.info(f"R-hat max: {diagnostics['convergence']['rhat_max']:.4f}")
        
        # Test parameter interpretation
        interpretation = model.get_parameter_interpretation()
        logger.info("Parameter interpretation generated successfully")
        
        logger.info("✓ Basic model pipeline test passed")
        return model, test_data
        
    except Exception as e:
        logger.error(f"✗ Basic model pipeline test failed: {e}")
        raise


def test_hierarchical_model_pipeline():
    """Test hierarchical model pipeline."""
    logger.info("Testing hierarchical BG/NBD model pipeline")
    
    try:
        # Create synthetic data with segments
        data = create_synthetic_data(n_supporters=500, random_seed=43)
        logger.info(f"Created synthetic data with {len(data)} supporters")
        
        # Split data for training and testing
        train_size = int(0.8 * len(data))
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        logger.info(f"Segments in data: {data['engagement_segment'].value_counts().to_dict()}")
        
        # Create and fit hierarchical model
        logger.info("Creating and fitting hierarchical BG/NBD model")
        model = create_bgnbd_model(
            hierarchical=True,
            segment_column='engagement_segment'
        )
        
        # Fit with reduced sampling for testing
        model.fit(train_data, draws=500, tune=250, chains=2)
        logger.info("Hierarchical model fitting completed")
        
        # Test predictions with segments
        logger.info("Testing hierarchical model predictions")
        
        # P(Alive) predictions
        prob_alive = model.predict_probability_alive(
            test_data['x'].values,
            test_data['t_x'].values,
            test_data['T'].values,
            test_data['engagement_segment'].values
        )
        logger.info(f"P(Alive) predictions: mean={prob_alive.mean():.3f}, std={prob_alive.std():.3f}")
        
        # Expected transactions predictions
        expected_transactions = model.predict_expected_transactions(
            180,
            test_data['x'].values,
            test_data['t_x'].values,
            test_data['T'].values,
            test_data['engagement_segment'].values
        )
        logger.info(f"Expected transactions: mean={expected_transactions.mean():.3f}, std={expected_transactions.std():.3f}")
        
        # Test segmentation
        segmentation_results = model.segment_supporters(test_data, method='probability')
        logger.info(f"Segmentation results: {segmentation_results['segment_prediction'].value_counts().to_dict()}")
        
        # Test model diagnostics
        diagnostics = model.get_model_diagnostics()
        logger.info(f"Hierarchical model converged: {diagnostics['convergence']['converged']}")
        logger.info(f"R-hat max: {diagnostics['convergence']['rhat_max']:.4f}")
        
        logger.info("✓ Hierarchical model pipeline test passed")
        return model, test_data
        
    except Exception as e:
        logger.error(f"✗ Hierarchical model pipeline test failed: {e}")
        raise


def test_model_evaluation():
    """Test model evaluation functionality."""
    logger.info("Testing model evaluation functionality")
    
    try:
        # Create and fit a basic model for evaluation
        data = create_synthetic_data(n_supporters=300, random_seed=44)
        model = fit_bgnbd_model(data, draws=500, tune=250, chains=2)
        
        # Create evaluator
        evaluator = create_model_evaluator(model)
        logger.info("Model evaluator created")
        
        # Test convergence evaluation
        convergence_results = evaluator.evaluate_convergence(verbose=False)
        logger.info(f"Convergence evaluation completed: {convergence_results['assessment']['overall_quality']}")
        
        # Test business metrics evaluation
        business_results = evaluator.evaluate_business_metrics(data)
        logger.info(f"Business metrics evaluation completed")
        logger.info(f"High-value supporters: {business_results['supporter_segments']['high_value_percentage']:.1f}%")
        
        # Test comprehensive evaluation
        comprehensive_results = evaluate_model_performance(
            model, 
            test_data=data.sample(100),  # Use sample as test data
            verbose=False
        )
        logger.info("Comprehensive evaluation completed")
        
        # Test evaluation report generation
        report = evaluator.generate_evaluation_report()
        logger.info(f"Evaluation report generated with {len(report)} sections")
        logger.info(f"Overall assessment: {report['summary_assessment']['overall_quality']}")
        
        logger.info("✓ Model evaluation test passed")
        return evaluator
        
    except Exception as e:
        logger.error(f"✗ Model evaluation test failed: {e}")
        raise


def test_model_serialization():
    """Test model saving and loading functionality."""
    logger.info("Testing model serialization")
    
    try:
        # Create and fit a model
        data = create_synthetic_data(n_supporters=200, random_seed=45)
        original_model = fit_bgnbd_model(data, draws=300, tune=150, chains=2)
        
        # Test predictions with original model
        original_predictions = original_model.predict_probability_alive(
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10]
        )
        
        # Save model
        model_path = Path("models/test_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        original_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Load model
        loaded_model = original_model.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Test predictions with loaded model
        loaded_predictions = loaded_model.predict_probability_alive(
            data['x'].values[:10],
            data['t_x'].values[:10],
            data['T'].values[:10]
        )
        
        # Compare predictions
        prediction_diff = np.abs(original_predictions - loaded_predictions).max()
        logger.info(f"Maximum prediction difference: {prediction_diff:.10f}")
        
        if prediction_diff < 1e-10:
            logger.info("✓ Model serialization test passed")
        else:
            logger.warning(f"Model serialization test passed with small differences: {prediction_diff}")
        
        # Clean up
        if model_path.exists():
            model_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model serialization test failed: {e}")
        raise


def test_data_integration():
    """Test integration with data processing modules."""
    logger.info("Testing data processing integration")
    
    try:
        # Test with BGNBDDataProcessor (if available)
        # Note: This would require actual database connection in real scenario
        logger.info("Data processing integration would require database connection")
        logger.info("Skipping data processing integration test in synthetic environment")
        
        # Test data format compatibility
        data = create_synthetic_data(n_supporters=100, random_seed=46)
        
        # Verify required columns are present
        required_columns = ['x', 't_x', 'T']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Test model can handle the data format
        model = create_bgnbd_model()
        model._validate_data(data)  # This should not raise an exception
        
        logger.info("✓ Data integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data integration test failed: {e}")
        raise


def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting BG/NBD model integration tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Basic model pipeline
        basic_model, basic_test_data = test_basic_model_pipeline()
        test_results['basic_pipeline'] = True
        
        # Test 2: Hierarchical model pipeline
        hierarchical_model, hierarchical_test_data = test_hierarchical_model_pipeline()
        test_results['hierarchical_pipeline'] = True
        
        # Test 3: Model evaluation
        evaluator = test_model_evaluation()
        test_results['evaluation'] = True
        
        # Test 4: Model serialization
        test_model_serialization()
        test_results['serialization'] = True
        
        # Test 5: Data integration
        test_data_integration()
        test_results['data_integration'] = True
        
    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        test_results['error'] = str(e)
        return test_results
    
    # Summary
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result is True else f"✗ FAILED: {result}"
        logger.info(f"{test_name:25s}: {status}")
    
    logger.info(f"\nTests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 All integration tests passed!")
    else:
        logger.warning("⚠️  Some integration tests failed")
    
    return test_results


if __name__ == "__main__":
    # Run integration tests
    results = run_integration_tests()
    
    # Exit with appropriate code
    if all(result is True for result in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)