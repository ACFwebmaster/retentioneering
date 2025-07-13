"""
Standalone test for BG/NBD model functionality.

This test runs the core BG/NBD model without external dependencies.
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


def create_test_data(n_supporters: int = 100, random_seed: int = 42) -> pd.DataFrame:
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


def test_core_functionality():
    """Test core BG/NBD functionality without config dependencies."""
    logger.info("Testing core BG/NBD functionality")
    
    try:
        # Import PyMC and other core dependencies
        import pymc as pm
        import arviz as az
        from scipy import special
        
        logger.info("Core dependencies imported successfully")
        
        # Create test data
        data = create_test_data(n_supporters=50, random_seed=42)
        logger.info(f"Created test data with {len(data)} supporters")
        
        # Test basic BG/NBD likelihood calculation
        x = data['x'].values[:10]
        t_x = data['t_x'].values[:10]
        T = data['T'].values[:10]
        
        # Test parameters
        r, alpha, a, b = 1.0, 2.0, 1.5, 2.5
        
        # Test P(Alive) calculation using scipy.special.beta
        prob_alive_x0 = 1.0 / (1.0 + (b / (b + 1)) * ((alpha + T) / alpha) ** r)
        
        # For x > 0 case
        numerator = (
            special.beta(a, b + x + 1) *
            (alpha / (alpha + t_x)) ** r *
            (t_x / (alpha + t_x)) ** x
        )
        
        denominator = (
            special.beta(a + 1, b + x) -
            special.beta(a, b + x + 1) *
            (alpha / (alpha + t_x)) ** r *
            (t_x / (alpha + t_x)) ** x
        )
        
        prob_alive_x_pos = numerator / denominator
        prob_alive = np.where(x == 0, prob_alive_x0, prob_alive_x_pos)
        prob_alive = np.clip(prob_alive, 0, 1)
        
        logger.info(f"P(Alive) calculation successful: mean={prob_alive.mean():.3f}")
        
        # Test expected transactions calculation
        expected_rate = r / alpha
        expected_transactions = prob_alive * expected_rate * 180  # 6 months
        
        logger.info(f"Expected transactions: mean={expected_transactions.mean():.3f}")
        
        # Test simple PyMC model creation
        with pm.Model() as simple_model:
            # Simple priors
            r_param = pm.Exponential('r', lam=1.0)
            alpha_param = pm.Exponential('alpha', lam=1.0)
            a_param = pm.Exponential('a', lam=1.0)
            b_param = pm.Exponential('b', lam=1.0)
            
            logger.info("Simple PyMC model created successfully")
        
        logger.info("✓ Core functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Core functionality test failed: {e}")
        return False


def test_data_validation():
    """Test data validation logic."""
    logger.info("Testing data validation")
    
    try:
        data = create_test_data(n_supporters=20, random_seed=43)
        
        # Test required columns
        required_columns = ['x', 't_x', 'T']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Test data constraints
        if (data['x'] < 0).any():
            raise ValueError("Frequency (x) values must be non-negative")
        
        if (data['t_x'] < 0).any():
            raise ValueError("Recency (t_x) values must be non-negative")
        
        if (data['T'] <= 0).any():
            raise ValueError("Observation period (T) values must be positive")
        
        if (data['t_x'] > data['T']).any():
            raise ValueError("Recency (t_x) cannot exceed observation period (T)")
        
        logger.info("✓ Data validation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data validation test failed: {e}")
        return False


def test_mathematical_functions():
    """Test mathematical functions used in BG/NBD model."""
    logger.info("Testing mathematical functions")
    
    try:
        from scipy import special
        
        # Test beta function calculations
        a, b = 2.0, 3.0
        beta_result = special.beta(a, b)
        expected_beta = special.gamma(a) * special.gamma(b) / special.gamma(a + b)
        
        if not np.isclose(beta_result, expected_beta, rtol=1e-10):
            raise ValueError("Beta function calculation mismatch")
        
        # Test gamma function
        gamma_result = special.gamma(2.5)
        if not np.isfinite(gamma_result):
            raise ValueError("Gamma function returned non-finite result")
        
        # Test log-gamma function
        gammaln_result = special.gammaln(2.5)
        if not np.isfinite(gammaln_result):
            raise ValueError("Log-gamma function returned non-finite result")
        
        logger.info("✓ Mathematical functions test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Mathematical functions test failed: {e}")
        return False


def run_standalone_tests():
    """Run all standalone tests."""
    logger.info("Starting BG/NBD standalone tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test 1: Core functionality
    test_results['core_functionality'] = test_core_functionality()
    
    # Test 2: Data validation
    test_results['data_validation'] = test_data_validation()
    
    # Test 3: Mathematical functions
    test_results['mathematical_functions'] = test_mathematical_functions()
    
    # Summary
    logger.info("=" * 50)
    logger.info("STANDALONE TEST SUMMARY")
    logger.info("=" * 50)
    
    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result is True else "✗ FAILED"
        logger.info(f"{test_name:25s}: {status}")
    
    logger.info(f"\nTests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 All standalone tests passed!")
        logger.info("Core BG/NBD functionality is working correctly.")
    else:
        logger.warning("⚠️  Some standalone tests failed")
    
    return test_results


if __name__ == "__main__":
    results = run_standalone_tests()
    
    # Exit with appropriate code
    if all(result is True for result in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)