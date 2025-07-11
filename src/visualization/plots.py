"""
Comprehensive visualization module for BG/NBD engagement model.

This module provides publication-ready plots for model diagnostics, predictions,
business insights, and data quality assessment. All plots support both interactive
and batch generation modes with customizable styling.
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy import stats

from ..config import get_config
from ..models.bgnbd import BGNBDModel
from ..models.evaluation import BGNBDModelEvaluator

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class BGNBDPlotter:
    """
    Comprehensive plotting toolkit for BG/NBD models.
    
    This class provides methods for creating publication-ready visualizations
    including model diagnostics, predictions, business insights, and data quality plots.
    """
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8',
        color_palette: str = 'husl',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        font_scale: float = 1.0
    ):
        """
        Initialize the BG/NBD plotter.
        
        Args:
            style: Matplotlib style to use
            color_palette: Seaborn color palette
            figure_size: Default figure size
            dpi: Resolution for saved figures
            font_scale: Font scaling factor
        """
        self.config = get_config()
        
        # Plotting configuration
        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.font_scale = font_scale
        
        # Set up plotting style
        self._setup_plotting_style()
        
        # Color schemes for different plot types
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'info': '#264653',
            'light': '#E9C46A',
            'dark': '#2A9D8F'
        }
        
        # Business segment colors
        self.segment_colors = {
            'Champions': '#2E86AB',
            'Loyal_Supporters': '#A23B72',
            'Potential_Loyalists': '#F18F01',
            'At_Risk': '#F4A261',
            'Lost': '#C73E1D',
            'High': '#2E86AB',
            'Medium': '#F18F01',
            'Low': '#F4A261',
            'Inactive': '#C73E1D'
        }
        
        logger.info("BGNBDPlotter initialized")
    
    def _setup_plotting_style(self) -> None:
        """Set up matplotlib and seaborn plotting style."""
        try:
            plt.style.use(self.style)
        except OSError:
            logger.warning(f"Style '{self.style}' not found, using default")
            plt.style.use('default')
        
        # Set seaborn style
        sns.set_palette(self.color_palette)
        sns.set_context("notebook", font_scale=self.font_scale)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.size'] = 10 * self.font_scale
        plt.rcParams['axes.titlesize'] = 12 * self.font_scale
        plt.rcParams['axes.labelsize'] = 10 * self.font_scale
        plt.rcParams['xtick.labelsize'] = 9 * self.font_scale
        plt.rcParams['ytick.labelsize'] = 9 * self.font_scale
        plt.rcParams['legend.fontsize'] = 9 * self.font_scale
    
    # ==================== MODEL DIAGNOSTIC PLOTS ====================
    
    def plot_trace_diagnostics(
        self,
        model: BGNBDModel,
        parameters: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create MCMC trace and posterior distribution plots.
        
        Args:
            model: Fitted BGNBDModel instance
            parameters: List of parameters to plot (if None, plots all)
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.trace is None:
            raise VisualizationError("Model must be fitted before plotting diagnostics")
        
        logger.info("Creating trace diagnostic plots")
        
        try:
            # Use ArviZ for trace plots
            if parameters:
                trace_subset = model.trace.sel(var_names=parameters)
            else:
                trace_subset = model.trace
            
            # Create trace plot
            axes = az.plot_trace(
                trace_subset,
                figsize=(15, 8),
                compact=True,
                combined=True
            )
            
            fig = axes.ravel()[0].figure
            fig.suptitle('MCMC Trace Diagnostics', fontsize=16, y=0.98)
            
            # Add convergence annotations
            self._add_convergence_annotations(fig, model)
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trace diagnostics: {e}")
            raise VisualizationError(f"Failed to create trace diagnostics: {e}")
    
    def plot_convergence_summary(
        self,
        model: BGNBDModel,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create convergence summary plots (R-hat and ESS).
        
        Args:
            model: Fitted BGNBDModel instance
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.trace is None:
            raise VisualizationError("Model must be fitted before plotting convergence")
        
        logger.info("Creating convergence summary plots")
        
        try:
            # Get convergence diagnostics
            rhat = az.rhat(model.trace)
            ess = az.ess(model.trace)
            
            # Extract ESS values from the dataset
            ess_bulk_values = []
            ess_tail_values = []
            
            for var_name in ess.data_vars:
                var_ess = ess[var_name]
                if hasattr(var_ess, 'values'):
                    ess_bulk_values.extend(var_ess.values.flatten())
                    ess_tail_values.extend(var_ess.values.flatten())  # Using same values for both
                else:
                    ess_bulk_values.append(float(var_ess))
                    ess_tail_values.append(float(var_ess))
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Convergence Diagnostics Summary', fontsize=16)
            
            # R-hat plot
            rhat_values = [float(rhat[var].values) for var in rhat.data_vars]
            param_names = list(rhat.data_vars)
            
            axes[0, 0].bar(range(len(param_names)), rhat_values, color=self.colors['primary'])
            axes[0, 0].axhline(y=1.01, color='red', linestyle='--', label='Threshold (1.01)')
            axes[0, 0].set_xticks(range(len(param_names)))
            axes[0, 0].set_xticklabels(param_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel('R-hat')
            axes[0, 0].set_title('R-hat Convergence Diagnostic')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # ESS Bulk plot
            if ess_bulk_values:
                min_ess_bulk = min(ess_bulk_values)
                axes[0, 1].bar(range(len(param_names)), [min_ess_bulk] * len(param_names), color=self.colors['secondary'])
            else:
                axes[0, 1].bar(range(len(param_names)), [1000] * len(param_names), color=self.colors['secondary'])
                min_ess_bulk = 1000
            
            axes[0, 1].axhline(y=400, color='red', linestyle='--', label='Threshold (400)')
            axes[0, 1].set_xticks(range(len(param_names)))
            axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right')
            axes[0, 1].set_ylabel('ESS Bulk')
            axes[0, 1].set_title('Effective Sample Size (Bulk)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # ESS Tail plot
            if ess_tail_values:
                min_ess_tail = min(ess_tail_values)
                axes[1, 0].bar(range(len(param_names)), [min_ess_tail] * len(param_names), color=self.colors['accent'])
            else:
                axes[1, 0].bar(range(len(param_names)), [1000] * len(param_names), color=self.colors['accent'])
                min_ess_tail = 1000
            
            axes[1, 0].axhline(y=400, color='red', linestyle='--', label='Threshold (400)')
            axes[1, 0].set_xticks(range(len(param_names)))
            axes[1, 0].set_xticklabels(param_names, rotation=45, ha='right')
            axes[1, 0].set_ylabel('ESS Tail')
            axes[1, 0].set_title('Effective Sample Size (Tail)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Summary statistics
            convergence_summary = {
                'Max R-hat': max(rhat_values),
                'Min ESS Bulk': min_ess_bulk,
                'Min ESS Tail': min_ess_tail,
                'Converged': max(rhat_values) < 1.01,
                'Adequate ESS': min_ess_bulk > 400 and min_ess_tail > 400
            }
            
            # Text summary
            summary_text = []
            for key, value in convergence_summary.items():
                if isinstance(value, bool):
                    summary_text.append(f"{key}: {'✓' if value else '✗'}")
                elif isinstance(value, float):
                    summary_text.append(f"{key}: {value:.3f}")
                else:
                    summary_text.append(f"{key}: {value}")
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(summary_text), 
                           transform=axes[1, 1].transAxes, fontsize=12,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_title('Convergence Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating convergence summary: {e}")
            raise VisualizationError(f"Failed to create convergence summary: {e}")
    
    def plot_model_comparison(
        self,
        models: List[BGNBDModel],
        model_names: List[str],
        metrics: List[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Compare multiple BG/NBD models.
        
        Args:
            models: List of fitted BGNBDModel instances
            model_names: Names for the models
            metrics: List of metrics to compare
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if len(models) != len(model_names):
            raise VisualizationError("Number of models must match number of model names")
        
        if metrics is None:
            metrics = ['rhat_max', 'ess_bulk_min', 'n_supporters']
        
        logger.info(f"Creating model comparison plot for {len(models)} models")
        
        try:
            # Collect model statistics
            comparison_data = []
            
            for model, name in zip(models, model_names):
                if model.trace is None:
                    logger.warning(f"Model '{name}' not fitted, skipping")
                    continue
                
                # Get diagnostics
                rhat = az.rhat(model.trace)
                ess = az.ess(model.trace)
                
                # Extract values properly
                rhat_values = []
                ess_values = []
                
                for var_name in rhat.data_vars:
                    var_rhat = rhat[var_name]
                    if hasattr(var_rhat, 'values'):
                        rhat_values.extend(var_rhat.values.flatten())
                    else:
                        rhat_values.append(float(var_rhat))
                
                for var_name in ess.data_vars:
                    var_ess = ess[var_name]
                    if hasattr(var_ess, 'values'):
                        ess_values.extend(var_ess.values.flatten())
                    else:
                        ess_values.append(float(var_ess))
                
                model_stats = {
                    'model_name': name,
                    'rhat_max': max(rhat_values) if rhat_values else 1.0,
                    'ess_bulk_min': min(ess_values) if ess_values else 1000.0,
                    'n_supporters': len(model.training_data) if model.training_data is not None else 0,
                    'hierarchical': model.hierarchical,
                    'n_parameters': len(model.params) if model.params else 0
                }
                
                comparison_data.append(model_stats)
            
            if not comparison_data:
                raise VisualizationError("No fitted models found for comparison")
            
            df = pd.DataFrame(comparison_data)
            
            # Create comparison plots
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            
            fig.suptitle('Model Comparison', fontsize=16)
            
            for i, metric in enumerate(metrics):
                if metric not in df.columns:
                    logger.warning(f"Metric '{metric}' not available, skipping")
                    continue
                
                # Bar plot for each metric
                bars = axes[i].bar(df['model_name'], df[metric], 
                                  color=sns.color_palette(self.color_palette, len(df)))
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[metric]):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom')
                
                # Add threshold lines for convergence metrics
                if metric == 'rhat_max':
                    axes[i].axhline(y=1.01, color='red', linestyle='--', alpha=0.7, label='Threshold')
                    axes[i].legend()
                elif 'ess' in metric:
                    axes[i].axhline(y=400, color='red', linestyle='--', alpha=0.7, label='Threshold')
                    axes[i].legend()
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model comparison: {e}")
            raise VisualizationError(f"Failed to create model comparison: {e}")
    
    # ==================== PREDICTION VISUALIZATIONS ====================
    
    def plot_probability_alive(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        bins: int = 50,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot P(Alive) probability distributions.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data
            bins: Number of histogram bins
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.params is None:
            raise VisualizationError("Model must be fitted before plotting predictions")
        
        logger.info("Creating P(Alive) distribution plots")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical:
                segment_col = data[model.segment_column].values
            
            # Calculate P(Alive) probabilities
            prob_alive = model.predict_probability_alive(
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('P(Alive) Probability Analysis', fontsize=16)
            
            # Overall distribution
            axes[0, 0].hist(prob_alive, bins=bins, alpha=0.7, color=self.colors['primary'], 
                           density=True, edgecolor='black', linewidth=0.5)
            axes[0, 0].axvline(prob_alive.mean(), color='red', linestyle='--', 
                              label=f'Mean: {prob_alive.mean():.3f}')
            axes[0, 0].axvline(np.median(prob_alive), color='orange', linestyle='--', 
                              label=f'Median: {np.median(prob_alive):.3f}')
            axes[0, 0].set_xlabel('P(Alive)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Overall P(Alive) Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # P(Alive) by frequency
            freq_bins = pd.cut(data['frequency'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            for i, (freq_bin, group_data) in enumerate(data.groupby(freq_bins)):
                if len(group_data) == 0:
                    continue
                
                group_segment_col = None
                if model.hierarchical:
                    group_segment_col = group_data[model.segment_column].values
                
                group_prob_alive = model.predict_probability_alive(
                    group_data['x'].values,
                    group_data['t_x'].values,
                    group_data['T'].values,
                    group_segment_col
                )
                
                axes[0, 1].hist(group_prob_alive, bins=20, alpha=0.6, 
                               label=f'{freq_bin} (n={len(group_data)})', density=True)
            
            axes[0, 1].set_xlabel('P(Alive)')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('P(Alive) by Frequency Groups')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # P(Alive) vs Recency
            axes[1, 0].scatter(data['recency_ratio'], prob_alive, alpha=0.6, 
                              color=self.colors['secondary'], s=20)
            axes[1, 0].set_xlabel('Recency Ratio (t_x / T)')
            axes[1, 0].set_ylabel('P(Alive)')
            axes[1, 0].set_title('P(Alive) vs Recency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data['recency_ratio'], prob_alive, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(data['recency_ratio'], p(data['recency_ratio']), 
                           "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
            axes[1, 0].legend()
            
            # Summary statistics
            stats_text = [
                f"Total Supporters: {len(data):,}",
                f"Mean P(Alive): {prob_alive.mean():.3f}",
                f"Std P(Alive): {prob_alive.std():.3f}",
                f"Active (P>0.5): {(prob_alive > 0.5).sum():,} ({(prob_alive > 0.5).mean()*100:.1f}%)",
                f"High Value (P>0.7): {(prob_alive > 0.7).sum():,} ({(prob_alive > 0.7).mean()*100:.1f}%)",
                f"At Risk (P<0.3): {(prob_alive < 0.3).sum():,} ({(prob_alive < 0.3).mean()*100:.1f}%)"
            ]
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text), 
                           transform=axes[1, 1].transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('P(Alive) Summary Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating P(Alive) plots: {e}")
            raise VisualizationError(f"Failed to create P(Alive) plots: {e}")
    
    def plot_expected_transactions(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        prediction_period: int = 180,
        uncertainty: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot expected transactions with uncertainty bands.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data
            prediction_period: Prediction period in days
            uncertainty: Whether to show uncertainty bands
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.params is None:
            raise VisualizationError("Model must be fitted before plotting predictions")
        
        logger.info(f"Creating expected transactions plots for {prediction_period} days")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical:
                segment_col = data[model.segment_column].values
            
            # Calculate expected transactions
            expected_transactions = model.predict_expected_transactions(
                prediction_period,
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Expected Transactions ({prediction_period} days)', fontsize=16)
            
            # Distribution of expected transactions
            axes[0, 0].hist(expected_transactions, bins=50, alpha=0.7, 
                           color=self.colors['primary'], density=True, 
                           edgecolor='black', linewidth=0.5)
            axes[0, 0].axvline(expected_transactions.mean(), color='red', linestyle='--', 
                              label=f'Mean: {expected_transactions.mean():.2f}')
            axes[0, 0].axvline(np.median(expected_transactions), color='orange', linestyle='--', 
                              label=f'Median: {np.median(expected_transactions):.2f}')
            axes[0, 0].set_xlabel('Expected Transactions')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distribution of Expected Transactions')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Expected transactions vs historical frequency
            axes[0, 1].scatter(data['frequency'], expected_transactions, alpha=0.6, 
                              color=self.colors['secondary'], s=20)
            axes[0, 1].set_xlabel('Historical Frequency')
            axes[0, 1].set_ylabel('Expected Transactions')
            axes[0, 1].set_title('Expected vs Historical Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data['frequency'], expected_transactions, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(data['frequency'], p(data['frequency']), 
                           "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
            axes[0, 1].legend()
            
            # Expected transactions by engagement segment
            if 'engagement_segment' in data.columns:
                segment_means = []
                segment_names = []
                segment_stds = []
                
                for segment in data['engagement_segment'].unique():
                    if pd.isna(segment):
                        continue
                    
                    segment_mask = data['engagement_segment'] == segment
                    segment_expected = expected_transactions[segment_mask]
                    
                    if len(segment_expected) > 0:
                        segment_names.append(segment)
                        segment_means.append(segment_expected.mean())
                        segment_stds.append(segment_expected.std())
                
                if segment_names:
                    bars = axes[1, 0].bar(segment_names, segment_means, 
                                         color=[self.segment_colors.get(seg, self.colors['primary']) 
                                               for seg in segment_names],
                                         alpha=0.7)
                    
                    if uncertainty:
                        axes[1, 0].errorbar(segment_names, segment_means, yerr=segment_stds, 
                                           fmt='none', color='black', capsize=5)
                    
                    axes[1, 0].set_xlabel('Engagement Segment')
                    axes[1, 0].set_ylabel('Expected Transactions')
                    axes[1, 0].set_title('Expected Transactions by Segment')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, mean_val in zip(bars, segment_means):
                        height = bar.get_height()
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                        f'{mean_val:.2f}', ha='center', va='bottom')
            
            # Summary statistics
            stats_text = [
                f"Prediction Period: {prediction_period} days",
                f"Total Expected: {expected_transactions.sum():.0f}",
                f"Mean per Supporter: {expected_transactions.mean():.2f}",
                f"Median per Supporter: {np.median(expected_transactions):.2f}",
                f"Std Deviation: {expected_transactions.std():.2f}",
                f"High Activity (>2): {(expected_transactions > 2).sum():,} ({(expected_transactions > 2).mean()*100:.1f}%)",
                f"Low Activity (<0.5): {(expected_transactions < 0.5).sum():,} ({(expected_transactions < 0.5).mean()*100:.1f}%)"
            ]
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text), 
                           transform=axes[1, 1].transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            axes[1, 1].set_title('Expected Transactions Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating expected transactions plots: {e}")
            raise VisualizationError(f"Failed to create expected transactions plots: {e}")
    
    def plot_clv_analysis(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        prediction_period: int = 365,
        discount_rate: float = 0.1,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot Customer Lifetime Value (CLV) analysis.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data (must include 'monetary' column)
            prediction_period: Prediction period in days
            discount_rate: Annual discount rate for NPV calculation
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.params is None:
            raise VisualizationError("Model must be fitted before plotting CLV")
        
        if 'monetary' not in data.columns:
            raise VisualizationError("Data must contain 'monetary' column for CLV analysis")
        
        logger.info(f"Creating CLV analysis plots for {prediction_period} days")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical:
                segment_col = data[model.segment_column].values
            
            # Calculate CLV
            predicted_clv = model.predict_clv(
                prediction_period,
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                data['monetary'].values,
                segment_col,
                discount_rate
            )
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Customer Lifetime Value Analysis ({prediction_period} days)', fontsize=16)
            
            # CLV distribution
            axes[0, 0].hist(predicted_clv, bins=50, alpha=0.7,
                           color=self.colors['primary'], density=True,
                           edgecolor='black', linewidth=0.5)
            axes[0, 0].axvline(predicted_clv.mean(), color='red', linestyle='--',
                              label=f'Mean: ${predicted_clv.mean():.2f}')
            axes[0, 0].axvline(np.median(predicted_clv), color='orange', linestyle='--',
                              label=f'Median: ${np.median(predicted_clv):.2f}')
            axes[0, 0].set_xlabel('Predicted CLV ($)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('CLV Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # CLV vs Historical Monetary Value
            axes[0, 1].scatter(data['monetary'], predicted_clv, alpha=0.6,
                              color=self.colors['secondary'], s=20)
            axes[0, 1].set_xlabel('Historical Monetary Value ($)')
            axes[0, 1].set_ylabel('Predicted CLV ($)')
            axes[0, 1].set_title('CLV vs Historical Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add trend line
            if len(data['monetary']) > 1:
                z = np.polyfit(data['monetary'], predicted_clv, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(data['monetary'], p(data['monetary']),
                               "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
                axes[0, 1].legend()
            
            # CLV by engagement segment
            if 'engagement_segment' in data.columns:
                segment_clv = []
                segment_names = []
                
                for segment in data['engagement_segment'].unique():
                    if pd.isna(segment):
                        continue
                    
                    segment_mask = data['engagement_segment'] == segment
                    segment_clv_values = predicted_clv[segment_mask]
                    
                    if len(segment_clv_values) > 0:
                        segment_names.append(segment)
                        segment_clv.append(segment_clv_values.mean())
                
                if segment_names:
                    bars = axes[1, 0].bar(segment_names, segment_clv,
                                         color=[self.segment_colors.get(seg, self.colors['primary'])
                                               for seg in segment_names],
                                         alpha=0.7)
                    
                    axes[1, 0].set_xlabel('Engagement Segment')
                    axes[1, 0].set_ylabel('Average CLV ($)')
                    axes[1, 0].set_title('CLV by Engagement Segment')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, clv_val in zip(bars, segment_clv):
                        height = bar.get_height()
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                        f'${clv_val:.0f}', ha='center', va='bottom')
            
            # CLV summary statistics
            total_clv = predicted_clv.sum()
            high_value_threshold = np.percentile(predicted_clv, 80)
            high_value_supporters = predicted_clv >= high_value_threshold
            
            stats_text = [
                f"Prediction Period: {prediction_period} days",
                f"Discount Rate: {discount_rate*100:.1f}%",
                f"Total CLV: ${total_clv:,.0f}",
                f"Mean CLV: ${predicted_clv.mean():.2f}",
                f"Median CLV: ${np.median(predicted_clv):.2f}",
                f"Top 20% Threshold: ${high_value_threshold:.2f}",
                f"High Value Count: {high_value_supporters.sum():,}",
                f"High Value CLV: ${predicted_clv[high_value_supporters].sum():,.0f}"
            ]
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text),
                           transform=axes[1, 1].transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 1].set_title('CLV Summary Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating CLV analysis plots: {e}")
            raise VisualizationError(f"Failed to create CLV analysis plots: {e}")
    
    # ==================== BUSINESS INTELLIGENCE CHARTS ====================
    
    def plot_supporter_segments(
        self,
        data: pd.DataFrame,
        segment_column: str = 'engagement_segment',
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot supporter segmentation analysis.
        
        Args:
            data: DataFrame with supporter data
            segment_column: Column containing segment information
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if segment_column not in data.columns:
            raise VisualizationError(f"Column '{segment_column}' not found in data")
        
        logger.info("Creating supporter segmentation plots")
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Supporter Segmentation Analysis', fontsize=16)
            
            # Segment distribution pie chart
            segment_counts = data[segment_column].value_counts()
            colors = [self.segment_colors.get(seg, self.colors['primary']) for seg in segment_counts.index]
            
            wedges, texts, autotexts = axes[0, 0].pie(segment_counts.values,
                                                     labels=segment_counts.index,
                                                     colors=colors,
                                                     autopct='%1.1f%%',
                                                     startangle=90)
            axes[0, 0].set_title('Segment Distribution')
            
            # Segment counts bar chart
            bars = axes[0, 1].bar(segment_counts.index, segment_counts.values,
                                 color=colors, alpha=0.7)
            axes[0, 1].set_xlabel('Segment')
            axes[0, 1].set_ylabel('Number of Supporters')
            axes[0, 1].set_title('Supporter Count by Segment')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, segment_counts.values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                f'{count:,}', ha='center', va='bottom')
            
            # Segment characteristics (if frequency and monetary data available)
            if 'frequency' in data.columns:
                segment_freq_means = data.groupby(segment_column)['frequency'].mean()
                bars = axes[1, 0].bar(segment_freq_means.index, segment_freq_means.values,
                                     color=[self.segment_colors.get(seg, self.colors['primary'])
                                           for seg in segment_freq_means.index],
                                     alpha=0.7)
                axes[1, 0].set_xlabel('Segment')
                axes[1, 0].set_ylabel('Average Frequency')
                axes[1, 0].set_title('Average Frequency by Segment')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, freq in zip(bars, segment_freq_means.values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                    f'{freq:.1f}', ha='center', va='bottom')
            
            # Monetary value by segment (if available)
            if 'monetary' in data.columns:
                segment_monetary_means = data.groupby(segment_column)['monetary'].mean()
                bars = axes[1, 1].bar(segment_monetary_means.index, segment_monetary_means.values,
                                     color=[self.segment_colors.get(seg, self.colors['primary'])
                                           for seg in segment_monetary_means.index],
                                     alpha=0.7)
                axes[1, 1].set_xlabel('Segment')
                axes[1, 1].set_ylabel('Average Monetary Value ($)')
                axes[1, 1].set_title('Average Monetary Value by Segment')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, monetary in zip(bars, segment_monetary_means.values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                    f'${monetary:.0f}', ha='center', va='bottom')
            else:
                # Show segment summary table if no monetary data
                segment_summary = data.groupby(segment_column).agg({
                    'frequency': ['count', 'mean'],
                    'recency_ratio': 'mean'
                }).round(2)
                
                # Create table
                table_data = []
                for segment in segment_summary.index:
                    row = [
                        segment,
                        f"{segment_summary.loc[segment, ('frequency', 'count')]:,}",
                        f"{segment_summary.loc[segment, ('frequency', 'mean')]:.1f}",
                        f"{segment_summary.loc[segment, ('recency_ratio', 'mean')]:.2f}"
                    ]
                    table_data.append(row)
                
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                table = axes[1, 1].table(cellText=table_data,
                                        colLabels=['Segment', 'Count', 'Avg Freq', 'Avg Recency'],
                                        cellLoc='center',
                                        loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                axes[1, 1].set_title('Segment Summary Statistics')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating supporter segmentation plots: {e}")
            raise VisualizationError(f"Failed to create supporter segmentation plots: {e}")
    
    def plot_campaign_targeting(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        prediction_period: int = 180,
        budget_constraint: Optional[float] = None,
        cost_per_contact: float = 5.0,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot campaign targeting optimization analysis.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data
            prediction_period: Prediction period in days
            budget_constraint: Optional budget constraint
            cost_per_contact: Cost per supporter contact
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if model.params is None:
            raise VisualizationError("Model must be fitted before plotting campaign targeting")
        
        logger.info("Creating campaign targeting optimization plots")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical:
                segment_col = data[model.segment_column].values
            
            # Calculate predictions
            prob_alive = model.predict_probability_alive(
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            expected_transactions = model.predict_expected_transactions(
                prediction_period,
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            # Calculate ROI metrics
            if 'monetary' in data.columns:
                expected_value = expected_transactions * data['monetary']
                roi = (expected_value - cost_per_contact) / cost_per_contact
            else:
                # Use average transaction value if monetary not available
                avg_transaction_value = 50.0  # Default assumption
                expected_value = expected_transactions * avg_transaction_value
                roi = (expected_value - cost_per_contact) / cost_per_contact
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Campaign Targeting Optimization', fontsize=16)
            
            # ROI distribution
            axes[0, 0].hist(roi, bins=50, alpha=0.7, color=self.colors['primary'], 
                           density=True, edgecolor='black', linewidth=0.5)
            axes[0, 0].axvline(0, color='red', linestyle='--', label='Break-even')
            axes[0, 0].axvline(roi.mean(), color='orange', linestyle='--', 
                              label=f'Mean ROI: {roi.mean():.2f}')
            axes[0, 0].set_xlabel('ROI')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('ROI Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # P(Alive) vs ROI scatter
            scatter = axes[0, 1].scatter(prob_alive, roi, alpha=0.6, 
                                        c=expected_transactions, cmap='viridis', s=20)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[0, 1].set_xlabel('P(Alive)')
            axes[0, 1].set_ylabel('ROI')
            axes[0, 1].set_title('P(Alive) vs ROI')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[0, 1])
            cbar.set_label('Expected Transactions')
            
            # Targeting recommendations by threshold
            thresholds = np.arange(0.1, 1.0, 0.1)
            target_counts = []
            total_roi = []
            
            for threshold in thresholds:
                targeted = data[prob_alive >= threshold]
                target_counts.append(len(targeted))
                total_roi.append(roi[prob_alive >= threshold].sum())
            
            # Plot targeting efficiency
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(thresholds, target_counts, 'b-o', label='Targeted Count', linewidth=2)
            line2 = ax2.plot(thresholds, total_roi, 'r-s', label='Total ROI', linewidth=2)
            
            ax1.set_xlabel('P(Alive) Threshold')
            ax1.set_ylabel('Number of Supporters Targeted', color='b')
            ax2.set_ylabel('Total ROI', color='r')
            ax1.set_title('Targeting Efficiency by Threshold')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            # Budget optimization or general recommendations
            if budget_constraint:
                max_contacts = int(budget_constraint / cost_per_contact)
                
                # Sort by ROI and select top performers within budget
                data_with_roi = data.copy()
                data_with_roi['roi'] = roi
                data_with_roi['prob_alive'] = prob_alive
                data_with_roi['expected_value'] = expected_value
                
                sorted_data = data_with_roi.sort_values('roi', ascending=False)
                optimal_targets = sorted_data.head(max_contacts)
                
                budget_stats = [
                    f"Budget: ${budget_constraint:,.0f}",
                    f"Cost per Contact: ${cost_per_contact:.2f}",
                    f"Max Contacts: {max_contacts:,}",
                    f"Optimal Targets: {len(optimal_targets):,}",
                    f"Expected ROI: {optimal_targets['roi'].sum():.2f}",
                    f"Expected Revenue: ${optimal_targets['expected_value'].sum():,.0f}",
                    f"Min P(Alive): {optimal_targets['prob_alive'].min():.3f}",
                    f"Avg P(Alive): {optimal_targets['prob_alive'].mean():.3f}"
                ]
                
                axes[1, 1].text(0.1, 0.9, '\n'.join(budget_stats), 
                               transform=axes[1, 1].transAxes, fontsize=11,
                               verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                axes[1, 1].set_title('Budget Optimization Results')
                axes[1, 1].axis('off')
            else:
                # General targeting recommendations
                positive_roi_supporters = (roi > 0).sum()
                high_prob_supporters = (prob_alive > 0.7).sum()
                
                general_stats = [
                    f"Total Supporters: {len(data):,}",
                    f"Positive ROI: {positive_roi_supporters:,} ({positive_roi_supporters/len(data)*100:.1f}%)",
                    f"High P(Alive) (>0.7): {high_prob_supporters:,} ({high_prob_supporters/len(data)*100:.1f}%)",
                    f"Mean ROI: {roi.mean():.2f}",
                    f"Median ROI: {np.median(roi):.2f}",
                    f"Best ROI: {roi.max():.2f}",
                    f"Worst ROI: {roi.min():.2f}",
                    "",
                    "Recommendation:",
                    f"Target {positive_roi_supporters:,} supporters",
                    f"with positive ROI"
                ]
                
                axes[1, 1].text(0.1, 0.9, '\n'.join(general_stats), 
                               transform=axes[1, 1].transAxes, fontsize=11,
                               verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                axes[1, 1].set_title('Targeting Recommendations')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating campaign targeting plots: {e}")
            raise VisualizationError(f"Failed to create campaign targeting plots: {e}")
    
    def plot_engagement_trends(
        self,
        data: pd.DataFrame,
        date_column: str = 'first_event_date',
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot engagement trends over time.
        
        Args:
            data: DataFrame with supporter data
            date_column: Column containing date information
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if date_column not in data.columns:
            raise VisualizationError(f"Column '{date_column}' not found in data")
        
        logger.info("Creating engagement trends plots")
        
        try:
            # Prepare data
            data_with_dates = data.copy()
            data_with_dates[date_column] = pd.to_datetime(data_with_dates[date_column])
            data_with_dates['year_month'] = data_with_dates[date_column].dt.to_period('M')
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Engagement Trends Analysis', fontsize=16)
            
            # New supporters over time
            new_supporters_by_month = data_with_dates.groupby('year_month').size()
            
            axes[0, 0].plot(new_supporters_by_month.index.astype(str), 
                           new_supporters_by_month.values, 
                           marker='o', linewidth=2, color=self.colors['primary'])
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('New Supporters')
            axes[0, 0].set_title('New Supporter Acquisition Trend')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line
            x_numeric = range(len(new_supporters_by_month))
            if len(x_numeric) > 1:
                z = np.polyfit(x_numeric, new_supporters_by_month.values, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(new_supporters_by_month.index.astype(str), p(x_numeric), 
                               "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.1f}/month)')
                axes[0, 0].legend()
            
            # Average frequency by month
            if 'frequency' in data.columns:
                freq_by_month = data_with_dates.groupby('year_month')['frequency'].mean()
                
                axes[0, 1].plot(freq_by_month.index.astype(str), 
                               freq_by_month.values, 
                               marker='s', linewidth=2, color=self.colors['secondary'])
                axes[0, 1].set_xlabel('Month')
                axes[0, 1].set_ylabel('Average Frequency')
                axes[0, 1].set_title('Average Engagement Frequency Trend')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # Engagement segments over time
            if 'engagement_segment' in data.columns:
                segment_trends = data_with_dates.groupby(['year_month', 'engagement_segment']).size().unstack(fill_value=0)
                
                # Stacked area plot
                axes[1, 0].stackplot(segment_trends.index.astype(str), 
                                    *[segment_trends[col] for col in segment_trends.columns],
                                    labels=segment_trends.columns,
                                    colors=[self.segment_colors.get(seg, self.colors['primary']) 
                                           for seg in segment_trends.columns],
                                    alpha=0.7)
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Number of Supporters')
                axes[1, 0].set_title('Engagement Segments Over Time')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
                axes[1, 0].grid(True, alpha=0.3)
            
            # Monthly statistics summary
            monthly_stats = data_with_dates.groupby('year_month').agg({
                'frequency': ['count', 'mean', 'std'],
                'monetary': ['sum', 'mean'] if 'monetary' in data.columns else ['count']
            }).round(2)
            
            # Show recent trends
            recent_months = monthly_stats.tail(6)
            stats_text = ["Recent 6 Months Summary:"]
            
            for month in recent_months.index:
                month_str = str(month)
                count = recent_months.loc[month, ('frequency', 'count')]
                avg_freq = recent_months.loc[month, ('frequency', 'mean')]
                stats_text.append(f"{month_str}: {count:,.0f} supporters, {avg_freq:.1f} avg freq")
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text), 
                           transform=axes[1, 1].transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Recent Trends Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating engagement trends plots: {e}")
            raise VisualizationError(f"Failed to create engagement trends plots: {e}")
    
    # ==================== DATA QUALITY VISUALIZATIONS ====================
    
    def plot_data_quality_report(
        self,
        data: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive data quality assessment plots.
        
        Args:
            data: DataFrame with supporter data
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating data quality report")
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Data Quality Assessment Report', fontsize=16)
            
            # Missing data analysis
            missing_data = data.isnull().sum()
            missing_pct = (missing_data / len(data)) * 100
            
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing_Count': missing_data.values,
                    'Missing_Pct': missing_pct.values
                }).sort_values('Missing_Pct', ascending=True)
                
                bars = axes[0, 0].barh(missing_df['Column'], missing_df['Missing_Pct'], 
                                      color=self.colors['warning'], alpha=0.7)
                axes[0, 0].set_xlabel('Missing Data (%)')
                axes[0, 0].set_title('Missing Data by Column')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add percentage labels
                for bar, pct in zip(bars, missing_df['Missing_Pct']):
                    width = bar.get_width()
                    if width > 0:
                        axes[0, 0].text(width, bar.get_y() + bar.get_height()/2,
                                       f'{pct:.1f}%', ha='left', va='center')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Missing Data Found!', 
                               transform=axes[0, 0].transAxes, ha='center', va='center',
                               fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
                axes[0, 0].set_title('Missing Data Analysis')
                axes[0, 0].axis('off')
            
            # Frequency distribution
            if 'frequency' in data.columns:
                axes[0, 1].hist(data['frequency'], bins=30, alpha=0.7, 
                               color=self.colors['primary'], edgecolor='black', linewidth=0.5)
                axes[0, 1].set_xlabel('Frequency')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Frequency Distribution')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add statistics
                freq_stats = f"Mean: {data['frequency'].mean():.1f}\nMedian: {data['frequency'].median():.1f}\nStd: {data['frequency'].std():.1f}"
                axes[0, 1].text(0.7, 0.8, freq_stats, transform=axes[0, 1].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Recency distribution
            if 'recency_ratio' in data.columns:
                axes[0, 2].hist(data['recency_ratio'], bins=30, alpha=0.7, 
                               color=self.colors['secondary'], edgecolor='black', linewidth=0.5)
                axes[0, 2].set_xlabel('Recency Ratio')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].set_title('Recency Distribution')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Add statistics
                rec_stats = f"Mean: {data['recency_ratio'].mean():.2f}\nMedian: {data['recency_ratio'].median():.2f}\nStd: {data['recency_ratio'].std():.2f}"
                axes[0, 2].text(0.7, 0.8, rec_stats, transform=axes[0, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Outlier detection for frequency
            if 'frequency' in data.columns:
                Q1 = data['frequency'].quantile(0.25)
                Q3 = data['frequency'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data['frequency'] < lower_bound) | (data['frequency'] > upper_bound)]
                
                # Box plot
                box_data = [data['frequency']]
                axes[1, 0].boxplot(box_data, labels=['Frequency'])
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title(f'Frequency Outliers (n={len(outliers)})')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add outlier statistics
                outlier_text = f"Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)\nLower bound: {lower_bound:.1f}\nUpper bound: {upper_bound:.1f}"
                axes[1, 0].text(0.02, 0.98, outlier_text, transform=axes[1, 0].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Monetary distribution (if available)
            if 'monetary' in data.columns:
                # Filter out zero values for better visualization
                monetary_nonzero = data[data['monetary'] > 0]['monetary']
                
                if len(monetary_nonzero) > 0:
                    axes[1, 1].hist(monetary_nonzero, bins=30, alpha=0.7, 
                                   color=self.colors['accent'], edgecolor='black', linewidth=0.5)
                    axes[1, 1].set_xlabel('Monetary Value ($)')
                    axes[1, 1].set_ylabel('Count')
                    axes[1, 1].set_title('Monetary Distribution (Non-zero)')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Add statistics
                    mon_stats = f"Donors: {len(monetary_nonzero)} ({len(monetary_nonzero)/len(data)*100:.1f}%)\nMean: ${monetary_nonzero.mean():.0f}\nMedian: ${monetary_nonzero.median():.0f}"
                    axes[1, 1].text(0.6, 0.8, mon_stats, transform=axes[1, 1].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Monetary Data Available', 
                                   transform=axes[1, 1].transAxes, ha='center', va='center',
                                   fontsize=12)
                    axes[1, 1].set_title('Monetary Distribution')
                    axes[1, 1].axis('off')
            
            # Data quality summary
            quality_metrics = {
                'Total Records': len(data),
                'Complete Records': len(data.dropna()),
                'Completeness Rate': f"{(len(data.dropna())/len(data)*100):.1f}%",
                'Duplicate Records': data.duplicated().sum(),
                'Unique Supporters': data['supporter_id'].nunique() if 'supporter_id' in data.columns else 'N/A'
            }
            
            if 'frequency' in data.columns:
                quality_metrics.update({
                    'Zero Frequency': (data['frequency'] == 0).sum(),
                    'High Frequency (>10)': (data['frequency'] > 10).sum()
                })
            
            if 'monetary' in data.columns:
                quality_metrics.update({
                    'Donors': (data['monetary'] > 0).sum(),
                    'High Value (>$500)': (data['monetary'] > 500).sum()
                })
            
            # Format quality summary
            quality_text = []
            for key, value in quality_metrics.items():
                quality_text.append(f"{key}: {value}")
            
            axes[1, 2].text(0.1, 0.9, '\n'.join(quality_text), 
                           transform=axes[1, 2].transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 2].set_title('Data Quality Summary')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating data quality report: {e}")
            raise VisualizationError(f"Failed to create data quality report: {e}")
    
    def plot_preprocessing_impact(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot the impact of data preprocessing transformations.
        
        Args:
            raw_data: Original raw data before preprocessing
            processed_data: Data after preprocessing
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating preprocessing impact plots")
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Data Preprocessing Impact Analysis', fontsize=16)
            
            # Record count comparison
            counts = [len(raw_data), len(processed_data)]
            labels = ['Raw Data', 'Processed Data']
            colors = [self.colors['warning'], self.colors['primary']]
            
            bars = axes[0, 0].bar(labels, counts, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Number of Records')
            axes[0, 0].set_title('Record Count: Before vs After')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{count:,}', ha='center', va='bottom')
            
            # Add percentage change
            pct_change = ((len(processed_data) - len(raw_data)) / len(raw_data)) * 100
            axes[0, 0].text(0.5, 0.8, f'Change: {pct_change:+.1f}%', 
                           transform=axes[0, 0].transAxes, ha='center',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Frequency distribution comparison (if available)
            if 'frequency' in raw_data.columns and 'frequency' in processed_data.columns:
                axes[0, 1].hist(raw_data['frequency'], bins=30, alpha=0.5, 
                               color=self.colors['warning'], label='Raw Data', density=True)
                axes[0, 1].hist(processed_data['frequency'], bins=30, alpha=0.5, 
                               color=self.colors['primary'], label='Processed Data', density=True)
                axes[0, 1].set_xlabel('Frequency')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].set_title('Frequency Distribution Comparison')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Data completeness comparison
            raw_completeness = (1 - raw_data.isnull().sum() / len(raw_data)) * 100
            processed_completeness = (1 - processed_data.isnull().sum() / len(processed_data)) * 100
            
            # Get common columns
            common_cols = list(set(raw_data.columns) & set(processed_data.columns))[:10]  # Limit to 10 for readability
            
            if common_cols:
                x = np.arange(len(common_cols))
                width = 0.35
                
                raw_comp_vals = [raw_completeness[col] for col in common_cols]
                proc_comp_vals = [processed_completeness[col] for col in common_cols]
                
                axes[1, 0].bar(x - width/2, raw_comp_vals, width, 
                              label='Raw Data', color=self.colors['warning'], alpha=0.7)
                axes[1, 0].bar(x + width/2, proc_comp_vals, width, 
                              label='Processed Data', color=self.colors['primary'], alpha=0.7)
                
                axes[1, 0].set_xlabel('Columns')
                axes[1, 0].set_ylabel('Completeness (%)')
                axes[1, 0].set_title('Data Completeness Comparison')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(common_cols, rotation=45, ha='right')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Processing summary statistics
            processing_stats = [
                f"Raw Records: {len(raw_data):,}",
                f"Processed Records: {len(processed_data):,}",
                f"Records Removed: {len(raw_data) - len(processed_data):,}",
                f"Removal Rate: {((len(raw_data) - len(processed_data)) / len(raw_data) * 100):.1f}%",
                "",
                "Quality Improvements:"
            ]
            
            # Add specific quality improvements
            if 'frequency' in processed_data.columns:
                min_freq = processed_data['frequency'].min()
                processing_stats.append(f"Min Frequency: {min_freq}")
            
            if 'monetary' in processed_data.columns:
                donors_pct = (processed_data['monetary'] > 0).mean() * 100
                processing_stats.append(f"Donors: {donors_pct:.1f}%")
            
            # Add data validation results
            if all(col in processed_data.columns for col in ['x', 't_x', 'T']):
                valid_bgnbd = (
                    (processed_data['x'] >= 0) & 
                    (processed_data['t_x'] >= 0) & 
                    (processed_data['T'] > 0) &
                    (processed_data['t_x'] <= processed_data['T'])
                ).sum()
                processing_stats.append(f"Valid BG/NBD Records: {valid_bgnbd:,}")
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(processing_stats),
                           transform=axes[1, 1].transAxes, fontsize=11,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 1].set_title('Processing Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating preprocessing impact plots: {e}")
            raise VisualizationError(f"Failed to create preprocessing impact plots: {e}")
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _add_convergence_annotations(self, fig: plt.Figure, model: BGNBDModel) -> None:
        """Add convergence annotations to trace plots."""
        try:
            # Get convergence diagnostics
            rhat = az.rhat(model.trace)
            ess = az.ess(model.trace)
            
            # Extract values properly
            rhat_values = []
            ess_values = []
            
            for var_name in rhat.data_vars:
                var_rhat = rhat[var_name]
                if hasattr(var_rhat, 'values'):
                    rhat_values.extend(var_rhat.values.flatten())
                else:
                    rhat_values.append(float(var_rhat))
            
            for var_name in ess.data_vars:
                var_ess = ess[var_name]
                if hasattr(var_ess, 'values'):
                    ess_values.extend(var_ess.values.flatten())
                else:
                    ess_values.append(float(var_ess))
            
            # Add text annotation
            max_rhat = max(rhat_values) if rhat_values else 1.0
            min_ess = min(ess_values) if ess_values else 1000.0
            convergence_text = f"Max R-hat: {max_rhat:.3f}\nMin ESS: {min_ess:.0f}"
            fig.text(0.02, 0.98, convergence_text, transform=fig.transFigure,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            logger.warning(f"Could not add convergence annotations: {e}")
    
    def _save_figure(self, fig: plt.Figure, save_path: Union[str, Path]) -> None:
        """Save figure to file with proper formatting."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI and tight bounding box
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            logger.info(f"Figure saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
            raise VisualizationError(f"Failed to save figure: {e}")
    
    def create_dashboard(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data
            save_path: Path to save the dashboard
            show_plot: Whether to display the dashboard
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating comprehensive dashboard")
        
        try:
            # Create large figure with subplots
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle('BG/NBD Model Dashboard', fontsize=20, y=0.98)
            
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical:
                segment_col = data[model.segment_column].values
            
            # Calculate predictions
            prob_alive = model.predict_probability_alive(
                data['x'].values, data['t_x'].values, data['T'].values, segment_col
            )
            expected_transactions = model.predict_expected_transactions(
                180, data['x'].values, data['t_x'].values, data['T'].values, segment_col
            )
            
            # 1. P(Alive) Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(prob_alive, bins=30, alpha=0.7, color=self.colors['primary'])
            ax1.set_title('P(Alive) Distribution')
            ax1.set_xlabel('P(Alive)')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            
            # 2. Expected Transactions Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(expected_transactions, bins=30, alpha=0.7, color=self.colors['secondary'])
            ax2.set_title('Expected Transactions (180 days)')
            ax2.set_xlabel('Expected Transactions')
            ax2.set_ylabel('Count')
            ax2.grid(True, alpha=0.3)
            
            # 3. P(Alive) vs Frequency
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.scatter(data['frequency'], prob_alive, alpha=0.6, s=20, color=self.colors['accent'])
            ax3.set_title('P(Alive) vs Frequency')
            ax3.set_xlabel('Historical Frequency')
            ax3.set_ylabel('P(Alive)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Convergence Summary (if available)
            ax4 = fig.add_subplot(gs[0, 3])
            if model.trace is not None:
                rhat = az.rhat(model.trace)
                rhat_values = [float(rhat[var].values) for var in rhat.data_vars]
                param_names = list(rhat.data_vars)
                
                ax4.bar(range(len(param_names)), rhat_values, color=self.colors['warning'])
                ax4.axhline(y=1.01, color='red', linestyle='--', label='Threshold')
                ax4.set_title('R-hat Convergence')
                ax4.set_ylabel('R-hat')
                ax4.set_xticks(range(len(param_names)))
                ax4.set_xticklabels(param_names, rotation=45, ha='right')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Trace Available', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Model Convergence')
                ax4.axis('off')
            
            # 5. Segment Distribution
            ax5 = fig.add_subplot(gs[1, :2])
            if 'engagement_segment' in data.columns:
                segment_counts = data['engagement_segment'].value_counts()
                colors = [self.segment_colors.get(seg, self.colors['primary']) for seg in segment_counts.index]
                
                wedges, texts, autotexts = ax5.pie(segment_counts.values, labels=segment_counts.index,
                                                  colors=colors, autopct='%1.1f%%', startangle=90)
                ax5.set_title('Supporter Segments')
            else:
                ax5.text(0.5, 0.5, 'No Segment Data Available', ha='center', va='center',
                        transform=ax5.transAxes, fontsize=12)
                ax5.set_title('Supporter Segments')
                ax5.axis('off')
            
            # 6. CLV Analysis (if monetary data available)
            ax6 = fig.add_subplot(gs[1, 2:])
            if 'monetary' in data.columns:
                predicted_clv = model.predict_clv(
                    365, data['x'].values, data['t_x'].values, data['T'].values,
                    data['monetary'].values, segment_col
                )
                ax6.hist(predicted_clv, bins=30, alpha=0.7, color=self.colors['success'])
                ax6.set_title('Predicted CLV Distribution (1 year)')
                ax6.set_xlabel('CLV ($)')
                ax6.set_ylabel('Count')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No Monetary Data Available', ha='center', va='center',
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('CLV Analysis')
                ax6.axis('off')
            
            # 7. Data Quality Summary
            ax7 = fig.add_subplot(gs[2, 0])
            quality_stats = [
                f"Total Supporters: {len(data):,}",
                f"Active (P>0.5): {(prob_alive > 0.5).sum():,}",
                f"High Value (P>0.7): {(prob_alive > 0.7).sum():,}",
                f"Mean P(Alive): {prob_alive.mean():.3f}",
                f"Mean Expected Trans: {expected_transactions.mean():.2f}"
            ]
            
            ax7.text(0.1, 0.9, '\n'.join(quality_stats), transform=ax7.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax7.set_title('Key Metrics')
            ax7.axis('off')
            
            # 8. Frequency Distribution
            ax8 = fig.add_subplot(gs[2, 1])
            if 'frequency' in data.columns:
                ax8.hist(data['frequency'], bins=20, alpha=0.7, color=self.colors['info'])
                ax8.set_title('Frequency Distribution')
                ax8.set_xlabel('Frequency')
                ax8.set_ylabel('Count')
                ax8.grid(True, alpha=0.3)
            
            # 9. Recency Analysis
            ax9 = fig.add_subplot(gs[2, 2])
            if 'recency_ratio' in data.columns:
                ax9.hist(data['recency_ratio'], bins=20, alpha=0.7, color=self.colors['light'])
                ax9.set_title('Recency Distribution')
                ax9.set_xlabel('Recency Ratio')
                ax9.set_ylabel('Count')
                ax9.grid(True, alpha=0.3)
            
            # 10. Model Parameters (if available)
            ax10 = fig.add_subplot(gs[2, 3])
            if model.params:
                param_text = []
                if model.hierarchical:
                    param_text.append("Hierarchical Model")
                    if 'mu_r' in model.params:
                        param_text.extend([
                            f"μ_r: {model.params['mu_r']:.3f}",
                            f"μ_α: {model.params['mu_alpha']:.3f}",
                            f"μ_a: {model.params['mu_a']:.3f}",
                            f"μ_b: {model.params['mu_b']:.3f}"
                        ])
                else:
                    param_text.append("Basic Model")
                    param_text.extend([
                        f"r: {model.params['r']:.3f}",
                        f"α: {model.params['alpha']:.3f}",
                        f"a: {model.params['a']:.3f}",
                        f"b: {model.params['b']:.3f}"
                    ])
                
                ax10.text(0.1, 0.9, '\n'.join(param_text), transform=ax10.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax10.set_title('Model Parameters')
            ax10.axis('off')
            
            # 11-16. Additional plots for bottom row
            # Expected vs Historical scatter
            ax11 = fig.add_subplot(gs[3, 0])
            ax11.scatter(data['frequency'], expected_transactions, alpha=0.6, s=15, color=self.colors['dark'])
            ax11.set_title('Expected vs Historical')
            ax11.set_xlabel('Historical Frequency')
            ax11.set_ylabel('Expected Transactions')
            ax11.grid(True, alpha=0.3)
            
            # P(Alive) vs Recency
            ax12 = fig.add_subplot(gs[3, 1])
            if 'recency_ratio' in data.columns:
                ax12.scatter(data['recency_ratio'], prob_alive, alpha=0.6, s=15, color=self.colors['accent'])
                ax12.set_title('P(Alive) vs Recency')
                ax12.set_xlabel('Recency Ratio')
                ax12.set_ylabel('P(Alive)')
                ax12.grid(True, alpha=0.3)
            
            # Monetary analysis (if available)
            ax13 = fig.add_subplot(gs[3, 2])
            if 'monetary' in data.columns:
                monetary_nonzero = data[data['monetary'] > 0]['monetary']
                if len(monetary_nonzero) > 0:
                    ax13.hist(monetary_nonzero, bins=20, alpha=0.7, color=self.colors['success'])
                    ax13.set_title('Monetary Distribution')
                    ax13.set_xlabel('Amount ($)')
                    ax13.set_ylabel('Count')
                    ax13.grid(True, alpha=0.3)
            
            # Summary insights
            ax14 = fig.add_subplot(gs[3, 3])
            insights = [
                "Key Insights:",
                f"• {(prob_alive > 0.7).mean()*100:.1f}% high-value supporters",
                f"• {(prob_alive < 0.3).mean()*100:.1f}% at-risk supporters",
                f"• Avg expected: {expected_transactions.mean():.1f} transactions",
                f"• Model: {'Hierarchical' if model.hierarchical else 'Basic'}"
            ]
            
            if 'monetary' in data.columns:
                total_clv = model.predict_clv(
                    365, data['x'].values, data['t_x'].values, data['T'].values,
                    data['monetary'].values, segment_col
                ).sum()
                insights.append(f"• Total CLV: ${total_clv:,.0f}")
            
            ax14.text(0.1, 0.9, '\n'.join(insights), transform=ax14.transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax14.set_title('Summary Insights')
            ax14.axis('off')
            
            if save_path:
                self._save_figure(fig, save_path)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise VisualizationError(f"Failed to create dashboard: {e}")
    
    def generate_all_plots(
        self,
        model: BGNBDModel,
        data: pd.DataFrame,
        output_dir: Union[str, Path],
        prediction_period: int = 180
    ) -> Dict[str, Path]:
        """
        Generate all available plots and save them to the output directory.
        
        Args:
            model: Fitted BGNBDModel instance
            data: DataFrame with supporter data
            output_dir: Directory to save all plots
            prediction_period: Prediction period for forecasting plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating all plots to {output_dir}")
        
        saved_plots = {}
        
        try:
            # Model diagnostic plots
            if model.trace is not None:
                # Trace diagnostics
                trace_path = output_dir / "trace_diagnostics.png"
                self.plot_trace_diagnostics(model, save_path=trace_path, show_plot=False)
                saved_plots['trace_diagnostics'] = trace_path
                
                # Convergence summary
                convergence_path = output_dir / "convergence_summary.png"
                self.plot_convergence_summary(model, save_path=convergence_path, show_plot=False)
                saved_plots['convergence_summary'] = convergence_path
            
            # Prediction plots
            prob_alive_path = output_dir / "probability_alive.png"
            self.plot_probability_alive(model, data, save_path=prob_alive_path, show_plot=False)
            saved_plots['probability_alive'] = prob_alive_path
            
            expected_trans_path = output_dir / "expected_transactions.png"
            self.plot_expected_transactions(model, data, prediction_period,
                                          save_path=expected_trans_path, show_plot=False)
            saved_plots['expected_transactions'] = expected_trans_path
            
            # CLV analysis (if monetary data available)
            if 'monetary' in data.columns:
                clv_path = output_dir / "clv_analysis.png"
                self.plot_clv_analysis(model, data, save_path=clv_path, show_plot=False)
                saved_plots['clv_analysis'] = clv_path
            
            # Business intelligence plots
            if 'engagement_segment' in data.columns:
                segments_path = output_dir / "supporter_segments.png"
                self.plot_supporter_segments(data, save_path=segments_path, show_plot=False)
                saved_plots['supporter_segments'] = segments_path
            
            # Campaign targeting
            targeting_path = output_dir / "campaign_targeting.png"
            self.plot_campaign_targeting(model, data, prediction_period,
                                       save_path=targeting_path, show_plot=False)
            saved_plots['campaign_targeting'] = targeting_path
            
            # Engagement trends (if date data available)
            if 'first_event_date' in data.columns:
                trends_path = output_dir / "engagement_trends.png"
                self.plot_engagement_trends(data, save_path=trends_path, show_plot=False)
                saved_plots['engagement_trends'] = trends_path
            
            # Data quality report
            quality_path = output_dir / "data_quality_report.png"
            self.plot_data_quality_report(data, save_path=quality_path, show_plot=False)
            saved_plots['data_quality_report'] = quality_path
            
            # Comprehensive dashboard
            dashboard_path = output_dir / "dashboard.png"
            self.create_dashboard(model, data, save_path=dashboard_path, show_plot=False)
            saved_plots['dashboard'] = dashboard_path
            
            logger.info(f"Generated {len(saved_plots)} plots successfully")
            return saved_plots
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise VisualizationError(f"Failed to generate all plots: {e}")


# ==================== FACTORY FUNCTIONS ====================

def create_plotter(
    style: str = 'seaborn-v0_8',
    color_palette: str = 'husl',
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    font_scale: float = 1.0
) -> BGNBDPlotter:
    """
    Factory function to create a BGNBDPlotter instance.
    
    Args:
        style: Matplotlib style to use
        color_palette: Seaborn color palette
        figure_size: Default figure size
        dpi: Resolution for saved figures
        font_scale: Font scaling factor
        
    Returns:
        BGNBDPlotter instance
    """
    return BGNBDPlotter(
        style=style,
        color_palette=color_palette,
        figure_size=figure_size,
        dpi=dpi,
        font_scale=font_scale
    )


def plot_model_diagnostics(
    model: BGNBDModel,
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = True
) -> Dict[str, plt.Figure]:
    """
    Convenience function to create all model diagnostic plots.
    
    Args:
        model: Fitted BGNBDModel instance
        output_dir: Optional directory to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary of plot names to Figure objects
    """
    plotter = create_plotter()
    figures = {}
    
    if model.trace is not None:
        # Trace diagnostics
        save_path = None
        if output_dir:
            save_path = Path(output_dir) / "trace_diagnostics.png"
        
        figures['trace_diagnostics'] = plotter.plot_trace_diagnostics(
            model, save_path=save_path, show_plot=show_plots
        )
        
        # Convergence summary
        save_path = None
        if output_dir:
            save_path = Path(output_dir) / "convergence_summary.png"
        
        figures['convergence_summary'] = plotter.plot_convergence_summary(
            model, save_path=save_path, show_plot=show_plots
        )
    
    return figures


def plot_predictions(
    model: BGNBDModel,
    data: pd.DataFrame,
    prediction_period: int = 180,
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = True
) -> Dict[str, plt.Figure]:
    """
    Convenience function to create all prediction plots.
    
    Args:
        model: Fitted BGNBDModel instance
        data: DataFrame with supporter data
        prediction_period: Prediction period in days
        output_dir: Optional directory to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary of plot names to Figure objects
    """
    plotter = create_plotter()
    figures = {}
    
    # P(Alive) distribution
    save_path = None
    if output_dir:
        save_path = Path(output_dir) / "probability_alive.png"
    
    figures['probability_alive'] = plotter.plot_probability_alive(
        model, data, save_path=save_path, show_plot=show_plots
    )
    
    # Expected transactions
    save_path = None
    if output_dir:
        save_path = Path(output_dir) / "expected_transactions.png"
    
    figures['expected_transactions'] = plotter.plot_expected_transactions(
        model, data, prediction_period, save_path=save_path, show_plot=show_plots
    )
    
    # CLV analysis (if monetary data available)
    if 'monetary' in data.columns:
        save_path = None
        if output_dir:
            save_path = Path(output_dir) / "clv_analysis.png"
        
        figures['clv_analysis'] = plotter.plot_clv_analysis(
            model, data, save_path=save_path, show_plot=show_plots
        )
    
    return figures


def create_comprehensive_report(
    model: BGNBDModel,
    data: pd.DataFrame,
    output_dir: Union[str, Path],
    prediction_period: int = 180
) -> Dict[str, Any]:
    """
    Create a comprehensive visualization report with all plots and summary.
    
    Args:
        model: Fitted BGNBDModel instance
        data: DataFrame with supporter data
        output_dir: Directory to save the report
        prediction_period: Prediction period for forecasting
        
    Returns:
        Dictionary with report summary and file paths
    """
    plotter = create_plotter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating comprehensive report in {output_dir}")
    
    # Generate all plots
    saved_plots = plotter.generate_all_plots(model, data, output_dir, prediction_period)
    
    # Create report summary
    report_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Hierarchical' if model.hierarchical else 'Basic',
        'n_supporters': len(data),
        'prediction_period_days': prediction_period,
        'plots_generated': list(saved_plots.keys()),
        'output_directory': str(output_dir),
        'file_paths': {name: str(path) for name, path in saved_plots.items()}
    }
    
    # Add model diagnostics if available
    if model.trace is not None:
        diagnostics = model.get_model_diagnostics()
        report_summary['model_diagnostics'] = {
            'converged': diagnostics['convergence']['converged'],
            'rhat_max': diagnostics['convergence']['rhat_max'],
            'ess_bulk_min': diagnostics['convergence']['ess_bulk_min']
        }
    
    # Add prediction summaries
    segment_col = None
    if model.hierarchical:
        segment_col = data[model.segment_column].values
    
    prob_alive = model.predict_probability_alive(
        data['x'].values, data['t_x'].values, data['T'].values, segment_col
    )
    expected_transactions = model.predict_expected_transactions(
        prediction_period, data['x'].values, data['t_x'].values, data['T'].values, segment_col
    )
    
    report_summary['prediction_summary'] = {
        'mean_prob_alive': float(prob_alive.mean()),
        'active_supporters_pct': float((prob_alive > 0.5).mean() * 100),
        'high_value_supporters_pct': float((prob_alive > 0.7).mean() * 100),
        'mean_expected_transactions': float(expected_transactions.mean()),
        'total_expected_transactions': float(expected_transactions.sum())
    }
    
    # Save report summary
    import json
    summary_path = output_dir / "report_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(report_summary, f, indent=2, default=str)
    
    report_summary['summary_file'] = str(summary_path)
    
    logger.info("Comprehensive report created successfully")
    return report_summary