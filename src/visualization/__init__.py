"""
Visualization module for the non-profit engagement model.

This module provides plotting and visualization capabilities for
BG/NBD model results, engagement predictions, and data analysis.
"""

from .plots import (
    BGNBDPlotter,
    VisualizationError,
    create_plotter,
    plot_model_diagnostics,
    plot_predictions,
    create_comprehensive_report
)

__all__ = [
    'BGNBDPlotter',
    'VisualizationError',
    'create_plotter',
    'plot_model_diagnostics',
    'plot_predictions',
    'create_comprehensive_report'
]