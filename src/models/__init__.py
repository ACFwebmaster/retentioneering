"""
BG/NBD modeling module for the non-profit engagement model.

This module implements the Beta-Geometric/Negative Binomial Distribution
model using PyMC for Bayesian inference to predict supporter engagement.

Main Components:
- BGNBDModel: Core BG/NBD model implementation
- BGNBDModelEvaluator: Comprehensive model evaluation toolkit
- Factory functions for easy model creation
- Utility functions for model comparison and benchmarking

Example Usage:
    from src.models import create_bgnbd_model, create_model_evaluator
    
    # Create and fit a basic model
    model = create_bgnbd_model(hierarchical=False)
    model.fit(data, draws=2000, tune=1000, chains=4)
    
    # Generate predictions
    prob_alive = model.predict_probability_alive(x, t_x, T)
    expected_transactions = model.predict_expected_transactions(180, x, t_x, T)
    
    # Evaluate model performance
    evaluator = create_model_evaluator(model)
    convergence = evaluator.evaluate_convergence()
    business_metrics = evaluator.evaluate_business_metrics(data)
"""

# Core model classes
from .bgnbd import BGNBDModel, BGNBDModelError

# Model evaluation classes
from .evaluation import BGNBDModelEvaluator, ModelEvaluationError

# Factory functions
from .bgnbd import create_bgnbd_model, fit_bgnbd_model
from .evaluation import (
    create_model_evaluator,
    evaluate_model_performance,
    compare_models,
    benchmark_model_performance
)

# Version information
__version__ = "1.0.0"
__author__ = "BG/NBD Model Implementation Team"

# Public API
__all__ = [
    # Core classes
    'BGNBDModel',
    'BGNBDModelEvaluator',
    
    # Exceptions
    'BGNBDModelError',
    'ModelEvaluationError',
    
    # Factory functions
    'create_bgnbd_model',
    'fit_bgnbd_model',
    'create_model_evaluator',
    
    # Evaluation functions
    'evaluate_model_performance',
    'compare_models',
    'benchmark_model_performance',
    
    # Version info
    '__version__',
    '__author__'
]

# Module-level documentation
def get_model_info():
    """
    Get information about the BG/NBD model implementation.
    
    Returns:
        Dictionary with model information and capabilities
    """
    return {
        'name': 'BG/NBD Model for Non-Profit Engagement Prediction',
        'version': __version__,
        'description': 'Bayesian implementation of Beta-Geometric/Negative Binomial Distribution model',
        'capabilities': [
            'Supporter engagement probability prediction (P(Alive))',
            'Expected future transaction forecasting',
            'Customer Lifetime Value (CLV) estimation',
            'Hierarchical modeling for segment-specific insights',
            'Comprehensive model evaluation and diagnostics',
            'Model serialization and deployment support'
        ],
        'use_cases': [
            'Supporter segmentation and targeting',
            'Campaign optimization and resource allocation',
            'Churn prediction and retention strategies',
            'Donation forecasting and planning',
            'ROI optimization for outreach activities'
        ],
        'technical_features': [
            'PyMC-based Bayesian inference',
            'MCMC sampling with NUTS sampler',
            'Convergence diagnostics (R-hat, ESS)',
            'Uncertainty quantification',
            'Cross-validation support',
            'Business-focused evaluation metrics'
        ]
    }


def print_model_info():
    """Print formatted model information."""
    info = get_model_info()
    
    print(f"\n{info['name']}")
    print("=" * len(info['name']))
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    
    print(f"\n📊 Capabilities:")
    for capability in info['capabilities']:
        print(f"  • {capability}")
    
    print(f"\n🎯 Use Cases:")
    for use_case in info['use_cases']:
        print(f"  • {use_case}")
    
    print(f"\n⚙️  Technical Features:")
    for feature in info['technical_features']:
        print(f"  • {feature}")
    
    print(f"\n🚀 Quick Start:")
    print("  from src.models import create_bgnbd_model")
    print("  model = create_bgnbd_model()")
    print("  model.fit(data)")
    print("  predictions = model.predict_probability_alive(x, t_x, T)")


# Module initialization message
import logging
logger = logging.getLogger(__name__)
logger.debug(f"BG/NBD models module initialized (version {__version__})")