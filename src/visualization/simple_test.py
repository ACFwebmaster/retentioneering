"""
Simple standalone test for the visualization module.

This test verifies the basic structure and imports of the visualization module
without requiring external dependencies like SQLAlchemy or PyMC.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that the visualization module can be imported."""
    print("Testing visualization module imports...")
    
    try:
        # Test basic imports without triggering config dependencies
        from src.visualization.plots import VisualizationError
        print("✓ VisualizationError imported successfully")
        
        # Test that the main class exists
        import src.visualization.plots as plots_module
        assert hasattr(plots_module, 'BGNBDPlotter')
        print("✓ BGNBDPlotter class found")
        
        # Test that key methods exist
        plotter_class = plots_module.BGNBDPlotter
        expected_methods = [
            'plot_trace_diagnostics',
            'plot_convergence_summary', 
            'plot_probability_alive',
            'plot_expected_transactions',
            'plot_clv_analysis',
            'plot_supporter_segments',
            'plot_campaign_targeting',
            'plot_engagement_trends',
            'plot_data_quality_report',
            'plot_preprocessing_impact',
            'create_dashboard',
            'generate_all_plots'
        ]
        
        for method_name in expected_methods:
            assert hasattr(plotter_class, method_name), f"Missing method: {method_name}"
        
        print(f"✓ All {len(expected_methods)} expected methods found")
        
        # Test factory functions exist
        assert hasattr(plots_module, 'create_plotter')
        assert hasattr(plots_module, 'plot_model_diagnostics')
        assert hasattr(plots_module, 'plot_predictions')
        assert hasattr(plots_module, 'create_comprehensive_report')
        print("✓ Factory functions found")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Assertion error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    viz_dir = Path(__file__).parent
    
    required_files = [
        'plots.py',
        '__init__.py',
        'example_usage.py',
        'test_plots.py',
        'simple_test.py'
    ]
    
    for filename in required_files:
        file_path = viz_dir / filename
        assert file_path.exists(), f"Missing file: {filename}"
    
    print(f"✓ All {len(required_files)} required files found")
    
    # Check that plots.py has reasonable size (should be substantial)
    plots_file = viz_dir / 'plots.py'
    file_size = plots_file.stat().st_size
    assert file_size > 50000, f"plots.py seems too small: {file_size} bytes"
    print(f"✓ plots.py has substantial content: {file_size:,} bytes")
    
    return True


def test_code_structure():
    """Test the structure of the plots.py file."""
    print("Testing code structure...")
    
    plots_file = Path(__file__).parent / 'plots.py'
    content = plots_file.read_text()
    
    # Check for key classes and functions
    required_elements = [
        'class VisualizationError',
        'class BGNBDPlotter',
        'def plot_trace_diagnostics',
        'def plot_convergence_summary',
        'def plot_probability_alive',
        'def plot_expected_transactions',
        'def plot_clv_analysis',
        'def plot_supporter_segments',
        'def plot_campaign_targeting',
        'def plot_engagement_trends',
        'def plot_data_quality_report',
        'def plot_preprocessing_impact',
        'def create_dashboard',
        'def generate_all_plots',
        'def create_plotter',
        'def create_comprehensive_report'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"✗ Missing elements: {missing_elements}")
        return False
    
    print(f"✓ All {len(required_elements)} required code elements found")
    
    # Check for proper imports
    required_imports = [
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'import pandas as pd',
        'import numpy as np',
        'import arviz as az'
    ]
    
    for import_stmt in required_imports:
        if import_stmt not in content:
            print(f"✗ Missing import: {import_stmt}")
            return False
    
    print("✓ All required imports found")
    
    return True


def test_documentation():
    """Test that functions have proper documentation."""
    print("Testing documentation...")
    
    plots_file = Path(__file__).parent / 'plots.py'
    content = plots_file.read_text()
    
    # Count docstrings
    docstring_count = content.count('"""')
    assert docstring_count >= 30, f"Expected at least 30 docstrings, found {docstring_count // 2}"
    print(f"✓ Found {docstring_count // 2} docstrings")
    
    # Check for key documentation elements
    doc_elements = [
        'Args:',
        'Returns:',
        'matplotlib Figure object',
        'DataFrame with supporter data',
        'Fitted BGNBDModel instance'
    ]
    
    for element in doc_elements:
        assert element in content, f"Missing documentation element: {element}"
    
    print("✓ Documentation elements found")
    
    return True


def run_simple_tests():
    """Run all simple tests."""
    print("VISUALIZATION MODULE SIMPLE TESTS")
    print("="*40)
    print("Testing basic structure and imports without dependencies")
    print("="*40)
    
    tests = [
        test_file_structure,
        test_code_structure,
        test_documentation,
        test_imports
    ]
    
    results = []
    for test_func in tests:
        try:
            print(f"\n{test_func.__name__.replace('_', ' ').title()}:")
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*40)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*40)
    
    if passed == total:
        print("🎉 ALL SIMPLE TESTS PASSED!")
        print("The visualization module structure is correct.")
        print("\nNote: Full functionality tests require installing dependencies:")
        print("  pip install matplotlib seaborn arviz pandas numpy pymc sqlalchemy")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)