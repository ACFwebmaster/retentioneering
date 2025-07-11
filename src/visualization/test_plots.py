"""
Simple test script for the BG/NBD visualization module.

This script performs basic functionality tests to ensure the visualization
module works correctly with the existing BG/NBD model infrastructure.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Import visualization components
from .plots import BGNBDPlotter, create_plotter, VisualizationError

logger = logging.getLogger(__name__)


def create_minimal_test_data(n_supporters: int = 50) -> pd.DataFrame:
    """Create minimal test data for visualization testing."""
    np.random.seed(42)
    
    data = []
    for i in range(n_supporters):
        T = 365
        x = np.random.poisson(2)  # Simple Poisson for frequency
        t_x = np.random.uniform(0, T) if x > 0 else 0
        
        data.append({
            'supporter_id': i + 1,
            'x': x,
            't_x': t_x,
            'T': T,
            'frequency': x + 1,
            'monetary': np.random.exponential(50) if x > 0 else 0,
            'recency_ratio': t_x / T,
            'engagement_segment': np.random.choice(['High', 'Medium', 'Low', 'Inactive']),
            'first_event_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
            'total_weighted_value': (x + 1) * 1.5,
            'total_donation_amount': np.random.exponential(50) if x > 0 else 0,
            'engagement_score': np.random.uniform(0, 1),
            'avg_donation': np.random.exponential(50) if x > 0 else 0,
            'event_type_count': np.random.randint(1, 4),
            'event_diversity_index': np.random.uniform(0, 2),
            'observation_start': datetime.now() - timedelta(days=365),
            'observation_end': datetime.now(),
            'cutoff_date': datetime.now()
        })
    
    return pd.DataFrame(data)


class MockBGNBDModel:
    """Mock BG/NBD model for testing visualization without full model fitting."""
    
    def __init__(self, hierarchical=False, segment_column=None):
        self.hierarchical = hierarchical
        self.segment_column = segment_column
        self.trace = None  # No MCMC trace for mock model
        self.params = {
            'r': 1.5,
            'alpha': 2.0,
            'a': 0.8,
            'b': 1.2
        }
        self.training_data = None
        self.model_metadata = {
            'created_at': datetime.now(),
            'trained_at': datetime.now(),
            'model_version': '1.0.0',
            'hierarchical': hierarchical,
            'n_supporters': 50
        }
    
    def predict_probability_alive(self, x, t_x, T, segment=None):
        """Mock P(Alive) prediction."""
        # Simple mock calculation
        prob = np.random.beta(2, 2, size=len(x))
        return np.clip(prob, 0.1, 0.9)
    
    def predict_expected_transactions(self, t, x, t_x, T, segment=None):
        """Mock expected transactions prediction."""
        # Simple mock calculation
        expected = np.random.exponential(1.5, size=len(x))
        return np.maximum(expected, 0.1)
    
    def predict_clv(self, t, x, t_x, T, monetary_value, segment=None, discount_rate=0.1):
        """Mock CLV prediction."""
        # Simple mock calculation
        clv = monetary_value * np.random.uniform(0.5, 2.0, size=len(x))
        return np.maximum(clv, 0)
    
    def get_model_diagnostics(self):
        """Mock model diagnostics."""
        return {
            'convergence': {
                'rhat_max': 1.005,
                'ess_bulk_min': 800,
                'converged': True
            }
        }


def test_plotter_initialization():
    """Test BGNBDPlotter initialization."""
    print("Testing plotter initialization...")
    
    try:
        # Test default initialization
        plotter = BGNBDPlotter()
        assert plotter is not None
        assert hasattr(plotter, 'colors')
        assert hasattr(plotter, 'segment_colors')
        print("✓ Default initialization successful")
        
        # Test custom initialization
        plotter = create_plotter(
            style='default',
            color_palette='Set1',
            figure_size=(10, 6),
            dpi=100,
            font_scale=0.8
        )
        assert plotter.figure_size == (10, 6)
        assert plotter.dpi == 100
        print("✓ Custom initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Plotter initialization failed: {e}")
        return False


def test_data_quality_plots():
    """Test data quality visualization functions."""
    print("Testing data quality plots...")
    
    try:
        plotter = create_plotter(dpi=72)  # Low DPI for faster testing
        data = create_minimal_test_data(30)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data quality report
            fig = plotter.plot_data_quality_report(
                data,
                save_path=Path(temp_dir) / "quality_report.png",
                show_plot=False
            )
            assert fig is not None
            assert os.path.exists(Path(temp_dir) / "quality_report.png")
            print("✓ Data quality report created successfully")
            
            # Test preprocessing impact (using same data as before/after)
            fig = plotter.plot_preprocessing_impact(
                data, data,  # Using same data for simplicity
                save_path=Path(temp_dir) / "preprocessing_impact.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ Preprocessing impact plot created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Data quality plots failed: {e}")
        return False


def test_business_intelligence_plots():
    """Test business intelligence visualization functions."""
    print("Testing business intelligence plots...")
    
    try:
        plotter = create_plotter(dpi=72)
        data = create_minimal_test_data(40)
        model = MockBGNBDModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test supporter segments
            fig = plotter.plot_supporter_segments(
                data,
                save_path=Path(temp_dir) / "segments.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ Supporter segments plot created successfully")
            
            # Test campaign targeting
            fig = plotter.plot_campaign_targeting(
                model, data,
                budget_constraint=5000,
                save_path=Path(temp_dir) / "targeting.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ Campaign targeting plot created successfully")
            
            # Test engagement trends
            fig = plotter.plot_engagement_trends(
                data,
                save_path=Path(temp_dir) / "trends.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ Engagement trends plot created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Business intelligence plots failed: {e}")
        return False


def test_prediction_plots():
    """Test prediction visualization functions."""
    print("Testing prediction plots...")
    
    try:
        plotter = create_plotter(dpi=72)
        data = create_minimal_test_data(35)
        model = MockBGNBDModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test P(Alive) plots
            fig = plotter.plot_probability_alive(
                model, data,
                save_path=Path(temp_dir) / "prob_alive.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ P(Alive) plot created successfully")
            
            # Test expected transactions
            fig = plotter.plot_expected_transactions(
                model, data,
                save_path=Path(temp_dir) / "expected_trans.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ Expected transactions plot created successfully")
            
            # Test CLV analysis
            fig = plotter.plot_clv_analysis(
                model, data,
                save_path=Path(temp_dir) / "clv_analysis.png",
                show_plot=False
            )
            assert fig is not None
            print("✓ CLV analysis plot created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction plots failed: {e}")
        return False


def test_dashboard_creation():
    """Test comprehensive dashboard creation."""
    print("Testing dashboard creation...")
    
    try:
        plotter = create_plotter(dpi=72)
        data = create_minimal_test_data(25)
        model = MockBGNBDModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test dashboard creation
            fig = plotter.create_dashboard(
                model, data,
                save_path=Path(temp_dir) / "dashboard.png",
                show_plot=False
            )
            assert fig is not None
            assert os.path.exists(Path(temp_dir) / "dashboard.png")
            print("✓ Dashboard created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard creation failed: {e}")
        return False


def test_error_handling():
    """Test error handling in visualization functions."""
    print("Testing error handling...")
    
    try:
        plotter = create_plotter()
        
        # Test with invalid data
        try:
            empty_data = pd.DataFrame()
            plotter.plot_data_quality_report(empty_data, show_plot=False)
            print("✗ Should have raised error for empty data")
            return False
        except (VisualizationError, Exception):
            print("✓ Correctly handled empty data error")
        
        # Test with missing columns
        try:
            invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
            plotter.plot_supporter_segments(invalid_data, show_plot=False)
            print("✗ Should have raised error for missing columns")
            return False
        except (VisualizationError, Exception):
            print("✓ Correctly handled missing column error")
        
        # Test with unfitted model
        try:
            model = MockBGNBDModel()
            model.params = None  # Simulate unfitted model
            data = create_minimal_test_data(10)
            plotter.plot_probability_alive(model, data, show_plot=False)
            print("✗ Should have raised error for unfitted model")
            return False
        except (VisualizationError, Exception):
            print("✓ Correctly handled unfitted model error")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all visualization tests."""
    print("BG/NBD VISUALIZATION MODULE TESTS")
    print("="*50)
    
    tests = [
        test_plotter_initialization,
        test_data_quality_plots,
        test_business_intelligence_plots,
        test_prediction_plots,
        test_dashboard_creation,
        test_error_handling
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Visualization module is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)