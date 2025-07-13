"""
Model evaluation module for BG/NBD models.

This module provides comprehensive evaluation metrics, diagnostics, and validation
tools for BG/NBD models including convergence diagnostics, predictive accuracy,
calibration assessment, and business metrics.
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_score

from ..config import get_config
from .bgnbd import BGNBDModel

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ModelEvaluationError(Exception):
    """Custom exception for model evaluation errors."""
    pass


class BGNBDModelEvaluator:
    """
    Comprehensive evaluation toolkit for BG/NBD models.
    
    This class provides methods for evaluating model convergence, predictive accuracy,
    calibration, and business performance metrics.
    """
    
    def __init__(self, model: BGNBDModel):
        """
        Initialize the model evaluator.
        
        Args:
            model: Fitted BGNBDModel instance
        """
        if model.trace is None:
            raise ModelEvaluationError("Model must be fitted before evaluation")
        
        self.model = model
        self.config = get_config()
        
        # Evaluation results storage
        self.convergence_diagnostics = {}
        self.predictive_metrics = {}
        self.calibration_metrics = {}
        self.business_metrics = {}
        
        logger.info("Initialized BG/NBD model evaluator")
    
    def evaluate_convergence(self, verbose: bool = True) -> Dict:
        """
        Evaluate MCMC convergence diagnostics.
        
        Args:
            verbose: Whether to print detailed diagnostics
            
        Returns:
            Dictionary with convergence diagnostics
        """
        logger.info("Evaluating MCMC convergence")
        
        try:
            # R-hat statistic (should be < 1.01)
            rhat = az.rhat(self.model.trace)
            rhat_max = float(rhat.max().values)
            rhat_mean = float(rhat.mean().values)
            
            # Effective sample size
            ess_bulk = az.ess(self.model.trace, kind='bulk')
            ess_tail = az.ess(self.model.trace, kind='tail')
            ess_bulk_min = float(ess_bulk.min().values)
            ess_tail_min = float(ess_tail.min().values)
            
            # Monte Carlo standard error
            mcse_mean = az.mcse(self.model.trace, kind='mean')
            mcse_sd = az.mcse(self.model.trace, kind='sd')
            
            # Convergence assessment
            converged = rhat_max < 1.01
            adequate_ess = ess_bulk_min > 400 and ess_tail_min > 400
            
            self.convergence_diagnostics = {
                'rhat': {
                    'max': rhat_max,
                    'mean': rhat_mean,
                    'by_parameter': {var: float(rhat[var].values) for var in rhat.data_vars}
                },
                'ess': {
                    'bulk_min': ess_bulk_min,
                    'tail_min': ess_tail_min,
                    'bulk_by_parameter': {var: float(ess_bulk[var].values) for var in ess_bulk.data_vars},
                    'tail_by_parameter': {var: float(ess_tail[var].values) for var in ess_tail.data_vars}
                },
                'mcse': {
                    'mean_max': float(mcse_mean.max().values),
                    'sd_max': float(mcse_sd.max().values)
                },
                'assessment': {
                    'converged': converged,
                    'adequate_ess': adequate_ess,
                    'overall_quality': converged and adequate_ess
                },
                'recommendations': self._get_convergence_recommendations(
                    rhat_max, ess_bulk_min, ess_tail_min
                )
            }
            
            if verbose:
                self._print_convergence_summary()
            
            return self.convergence_diagnostics
            
        except Exception as e:
            logger.error(f"Error evaluating convergence: {e}")
            raise ModelEvaluationError(f"Convergence evaluation failed: {e}")
    
    def _get_convergence_recommendations(
        self, 
        rhat_max: float, 
        ess_bulk_min: float, 
        ess_tail_min: float
    ) -> List[str]:
        """Get recommendations for improving convergence."""
        recommendations = []
        
        if rhat_max > 1.01:
            recommendations.append(
                f"R-hat too high ({rhat_max:.3f}). Consider increasing tune/draws or checking model specification."
            )
        
        if ess_bulk_min < 400:
            recommendations.append(
                f"Bulk ESS too low ({ess_bulk_min:.0f}). Consider increasing draws or improving sampler efficiency."
            )
        
        if ess_tail_min < 400:
            recommendations.append(
                f"Tail ESS too low ({ess_tail_min:.0f}). Consider increasing draws or checking for heavy tails."
            )
        
        if not recommendations:
            recommendations.append("Convergence diagnostics look good!")
        
        return recommendations
    
    def _print_convergence_summary(self) -> None:
        """Print convergence diagnostics summary."""
        diag = self.convergence_diagnostics
        
        print("\n" + "="*60)
        print("MCMC CONVERGENCE DIAGNOSTICS")
        print("="*60)
        
        print(f"R-hat (max): {diag['rhat']['max']:.4f} {'✓' if diag['rhat']['max'] < 1.01 else '✗'}")
        print(f"R-hat (mean): {diag['rhat']['mean']:.4f}")
        
        print(f"\nEffective Sample Size:")
        print(f"  Bulk (min): {diag['ess']['bulk_min']:.0f} {'✓' if diag['ess']['bulk_min'] > 400 else '✗'}")
        print(f"  Tail (min): {diag['ess']['tail_min']:.0f} {'✓' if diag['ess']['tail_min'] > 400 else '✗'}")
        
        print(f"\nOverall Assessment: {'GOOD' if diag['assessment']['overall_quality'] else 'NEEDS IMPROVEMENT'}")
        
        print(f"\nRecommendations:")
        for rec in diag['recommendations']:
            print(f"  • {rec}")
    
    def evaluate_predictive_accuracy(
        self, 
        test_data: pd.DataFrame,
        prediction_period: int = 180,
        metrics: List[str] = None
    ) -> Dict:
        """
        Evaluate predictive accuracy on test data.
        
        Args:
            test_data: Test dataset with actual outcomes
            prediction_period: Prediction period in days
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with predictive accuracy metrics
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'accuracy', 'auc']
        
        logger.info(f"Evaluating predictive accuracy on {len(test_data)} test cases")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if self.model.hierarchical:
                segment_col = test_data[self.model.segment_column].values
            
            # Generate predictions
            prob_alive = self.model.predict_probability_alive(
                test_data['x'].values,
                test_data['t_x'].values,
                test_data['T'].values,
                segment_col
            )
            
            expected_transactions = self.model.predict_expected_transactions(
                prediction_period,
                test_data['x'].values,
                test_data['t_x'].values,
                test_data['T'].values,
                segment_col
            )
            
            # Prepare actual outcomes (assuming they exist in test_data)
            actual_outcomes = {}
            
            if 'actual_alive' in test_data.columns:
                actual_outcomes['alive'] = test_data['actual_alive'].values
            
            if 'actual_transactions' in test_data.columns:
                actual_outcomes['transactions'] = test_data['actual_transactions'].values
            
            # Calculate metrics
            self.predictive_metrics = {
                'predictions': {
                    'prob_alive': prob_alive,
                    'expected_transactions': expected_transactions
                },
                'actual': actual_outcomes,
                'metrics': {}
            }
            
            # P(Alive) metrics
            if 'alive' in actual_outcomes:
                alive_metrics = self._calculate_classification_metrics(
                    actual_outcomes['alive'], prob_alive, metrics
                )
                self.predictive_metrics['metrics']['prob_alive'] = alive_metrics
            
            # Transaction prediction metrics
            if 'transactions' in actual_outcomes:
                trans_metrics = self._calculate_regression_metrics(
                    actual_outcomes['transactions'], expected_transactions, metrics
                )
                self.predictive_metrics['metrics']['expected_transactions'] = trans_metrics
            
            return self.predictive_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictive accuracy: {e}")
            raise ModelEvaluationError(f"Predictive accuracy evaluation failed: {e}")
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        metrics: List[str]
    ) -> Dict:
        """Calculate classification metrics for binary outcomes."""
        results = {}
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        y_pred = (y_prob >= 0.5).astype(int)
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'auc' in metrics and len(np.unique(y_true)) > 1:
            results['auc'] = roc_auc_score(y_true, y_prob)
        
        # Precision, recall, F1
        if any(m in metrics for m in ['precision', 'recall', 'f1']):
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            if 'precision' in metrics:
                results['precision'] = precision
            if 'recall' in metrics:
                results['recall'] = recall
            if 'f1' in metrics:
                results['f1'] = f1
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return results
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: List[str]
    ) -> Dict:
        """Calculate regression metrics for continuous outcomes."""
        results = {}
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mape' in metrics:
            # Mean Absolute Percentage Error (handle division by zero)
            mask = y_true != 0
            if mask.sum() > 0:
                results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                results['mape'] = np.inf
        
        # R-squared
        if 'r2' in metrics:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            results['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Correlation
        if 'correlation' in metrics:
            results['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return results
    
    def evaluate_calibration(
        self, 
        test_data: pd.DataFrame, 
        n_bins: int = 10
    ) -> Dict:
        """
        Evaluate probability calibration for P(Alive) predictions.
        
        Args:
            test_data: Test dataset with actual outcomes
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        logger.info("Evaluating probability calibration")
        
        if 'actual_alive' not in test_data.columns:
            raise ModelEvaluationError("Test data must contain 'actual_alive' column for calibration evaluation")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if self.model.hierarchical:
                segment_col = test_data[self.model.segment_column].values
            
            # Generate P(Alive) predictions
            prob_alive = self.model.predict_probability_alive(
                test_data['x'].values,
                test_data['t_x'].values,
                test_data['T'].values,
                segment_col
            )
            
            actual_alive = test_data['actual_alive'].values
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_alive, prob_alive, n_bins=n_bins, strategy='uniform'
            )
            
            # Brier score (lower is better)
            brier_score = np.mean((prob_alive - actual_alive) ** 2)
            
            # Reliability (calibration error)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (prob_alive > bin_lower) & (prob_alive <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = actual_alive[in_bin].mean()
                    avg_confidence_in_bin = prob_alive[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            self.calibration_metrics = {
                'brier_score': brier_score,
                'calibration_error': calibration_error,
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                },
                'reliability_diagram_data': {
                    'bin_boundaries': bin_boundaries.tolist(),
                    'bin_accuracies': fraction_of_positives.tolist(),
                    'bin_confidences': mean_predicted_value.tolist()
                }
            }
            
            return self.calibration_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating calibration: {e}")
            raise ModelEvaluationError(f"Calibration evaluation failed: {e}")
    
    def evaluate_business_metrics(
        self, 
        data: pd.DataFrame,
        prediction_period: int = 180,
        high_value_threshold: float = 0.7,
        clv_threshold: Optional[float] = None
    ) -> Dict:
        """
        Evaluate business-relevant metrics.
        
        Args:
            data: Dataset for evaluation
            prediction_period: Prediction period in days
            high_value_threshold: Threshold for high-value supporter identification
            clv_threshold: Threshold for CLV-based segmentation
            
        Returns:
            Dictionary with business metrics
        """
        logger.info("Evaluating business metrics")
        
        try:
            # Get segment info if hierarchical
            segment_col = None
            if self.model.hierarchical:
                segment_col = data[self.model.segment_column].values
            
            # Generate predictions
            prob_alive = self.model.predict_probability_alive(
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            expected_transactions = self.model.predict_expected_transactions(
                prediction_period,
                data['x'].values,
                data['t_x'].values,
                data['T'].values,
                segment_col
            )
            
            # High-value supporter identification
            high_value_supporters = prob_alive >= high_value_threshold
            
            # CLV analysis (if monetary data available)
            clv_metrics = {}
            if 'monetary' in data.columns:
                predicted_clv = self.model.predict_clv(
                    prediction_period,
                    data['x'].values,
                    data['t_x'].values,
                    data['T'].values,
                    data['monetary'].values,
                    segment_col
                )
                
                clv_metrics = {
                    'total_predicted_clv': float(predicted_clv.sum()),
                    'mean_predicted_clv': float(predicted_clv.mean()),
                    'median_predicted_clv': float(np.median(predicted_clv)),
                    'clv_distribution': {
                        'q25': float(np.percentile(predicted_clv, 25)),
                        'q75': float(np.percentile(predicted_clv, 75)),
                        'q90': float(np.percentile(predicted_clv, 90)),
                        'q95': float(np.percentile(predicted_clv, 95))
                    }
                }
                
                if clv_threshold:
                    high_clv_supporters = predicted_clv >= clv_threshold
                    clv_metrics['high_clv_supporters'] = {
                        'count': int(high_clv_supporters.sum()),
                        'percentage': float(high_clv_supporters.mean() * 100),
                        'total_clv': float(predicted_clv[high_clv_supporters].sum())
                    }
            
            # Engagement segmentation
            engagement_segments = self._create_engagement_segments(
                prob_alive, expected_transactions
            )
            
            self.business_metrics = {
                'supporter_segments': {
                    'high_value_count': int(high_value_supporters.sum()),
                    'high_value_percentage': float(high_value_supporters.mean() * 100),
                    'engagement_distribution': {
                        segment: int(count) for segment, count in 
                        pd.Series(engagement_segments).value_counts().items()
                    }
                },
                'predicted_engagement': {
                    'total_expected_transactions': float(expected_transactions.sum()),
                    'mean_expected_transactions': float(expected_transactions.mean()),
                    'active_supporters_percentage': float((prob_alive >= 0.5).mean() * 100)
                },
                'clv_analysis': clv_metrics,
                'model_insights': self._generate_business_insights(
                    prob_alive, expected_transactions, data
                )
            }
            
            return self.business_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating business metrics: {e}")
            raise ModelEvaluationError(f"Business metrics evaluation failed: {e}")
    
    def _create_engagement_segments(
        self, 
        prob_alive: np.ndarray, 
        expected_transactions: np.ndarray
    ) -> List[str]:
        """Create engagement segments based on predictions."""
        segments = []
        
        for p_alive, exp_trans in zip(prob_alive, expected_transactions):
            if p_alive >= 0.8 and exp_trans >= np.percentile(expected_transactions, 75):
                segments.append('Champions')
            elif p_alive >= 0.6 and exp_trans >= np.percentile(expected_transactions, 50):
                segments.append('Loyal_Supporters')
            elif p_alive >= 0.5:
                segments.append('Potential_Loyalists')
            elif p_alive >= 0.3:
                segments.append('At_Risk')
            else:
                segments.append('Lost')
        
        return segments
    
    def _generate_business_insights(
        self, 
        prob_alive: np.ndarray, 
        expected_transactions: np.ndarray, 
        data: pd.DataFrame
    ) -> Dict:
        """Generate business insights from model predictions."""
        insights = {}
        
        # Churn risk analysis
        at_risk_threshold = 0.3
        at_risk_supporters = prob_alive < at_risk_threshold
        
        insights['churn_risk'] = {
            'at_risk_count': int(at_risk_supporters.sum()),
            'at_risk_percentage': float(at_risk_supporters.mean() * 100),
            'avg_prob_alive_at_risk': float(prob_alive[at_risk_supporters].mean()) if at_risk_supporters.sum() > 0 else 0
        }
        
        # Engagement opportunity analysis
        low_engagement = expected_transactions < np.percentile(expected_transactions, 25)
        high_potential = (prob_alive >= 0.6) & low_engagement
        
        insights['engagement_opportunities'] = {
            'high_potential_count': int(high_potential.sum()),
            'high_potential_percentage': float(high_potential.mean() * 100)
        }
        
        # Frequency vs. recency analysis
        if 'frequency' in data.columns:
            high_freq = data['frequency'] >= data['frequency'].quantile(0.75)
            recent_activity = data['recency_ratio'] >= 0.5
            
            insights['behavioral_patterns'] = {
                'high_freq_recent': int((high_freq & recent_activity).sum()),
                'high_freq_not_recent': int((high_freq & ~recent_activity).sum()),
                'low_freq_recent': int((~high_freq & recent_activity).sum()),
                'low_freq_not_recent': int((~high_freq & ~recent_activity).sum())
            }
        
        return insights
    
    def cross_validate_model(
        self, 
        data: pd.DataFrame, 
        cv_folds: int = 5,
        prediction_period: int = 180,
        random_state: int = 42
    ) -> Dict:
        """
        Perform cross-validation evaluation.
        
        Args:
            data: Full dataset for cross-validation
            cv_folds: Number of cross-validation folds
            prediction_period: Prediction period for evaluation
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        try:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_results = {
                'fold_results': [],
                'summary_metrics': {}
            }
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
                logger.info(f"Processing fold {fold + 1}/{cv_folds}")
                
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Create and fit model for this fold
                from .bgnbd import create_bgnbd_model
                fold_model = create_bgnbd_model(
                    hierarchical=self.model.hierarchical,
                    segment_column=self.model.segment_column,
                    random_seed=random_state + fold
                )
                
                # Fit with reduced sampling for speed
                fold_model.fit(train_data, draws=1000, tune=500, chains=2)
                
                # Evaluate on test set
                fold_evaluator = BGNBDModelEvaluator(fold_model)
                
                # Get predictions
                segment_col = None
                if fold_model.hierarchical:
                    segment_col = test_data[fold_model.segment_column].values
                
                prob_alive = fold_model.predict_probability_alive(
                    test_data['x'].values,
                    test_data['t_x'].values,
                    test_data['T'].values,
                    segment_col
                )
                
                expected_transactions = fold_model.predict_expected_transactions(
                    prediction_period,
                    test_data['x'].values,
                    test_data['t_x'].values,
                    test_data['T'].values,
                    segment_col
                )
                
                # Calculate fold metrics
                fold_metrics = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'prob_alive_mean': float(prob_alive.mean()),
                    'prob_alive_std': float(prob_alive.std()),
                    'expected_transactions_mean': float(expected_transactions.mean()),
                    'expected_transactions_std': float(expected_transactions.std())
                }
                
                # Add convergence diagnostics
                convergence = fold_evaluator.evaluate_convergence(verbose=False)
                fold_metrics['convergence'] = {
                    'rhat_max': convergence['rhat']['max'],
                    'ess_bulk_min': convergence['ess']['bulk_min'],
                    'converged': convergence['assessment']['converged']
                }
                
                cv_results['fold_results'].append(fold_metrics)
            
            # Calculate summary statistics across folds
            fold_df = pd.DataFrame(cv_results['fold_results'])
            
            cv_results['summary_metrics'] = {
                'prob_alive': {
                    'mean': float(fold_df['prob_alive_mean'].mean()),
                    'std': float(fold_df['prob_alive_mean'].std()),
                    'cv': float(fold_df['prob_alive_mean'].std() / fold_df['prob_alive_mean'].mean())
                },
                'expected_transactions': {
                    'mean': float(fold_df['expected_transactions_mean'].mean()),
                    'std': float(fold_df['expected_transactions_mean'].std()),
                    'cv': float(fold_df['expected_transactions_mean'].std() / fold_df['expected_transactions_mean'].mean())
                },
                'convergence': {
                    'avg_rhat_max': float(fold_df['rhat_max'].mean()),
                    'avg_ess_bulk_min': float(fold_df['ess_bulk_min'].mean()),
                    'convergence_rate': float(fold_df['converged'].mean())
                }
            }
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise ModelEvaluationError(f"Cross-validation failed: {e}")
    
    def generate_evaluation_report(
        self, 
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Dictionary with complete evaluation results
        """
        logger.info("Generating comprehensive evaluation report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_metadata': self.model.model_metadata,
            'convergence_diagnostics': self.convergence_diagnostics,
            'predictive_metrics': self.predictive_metrics,
            'calibration_metrics': self.calibration_metrics,
            'business_metrics': self.business_metrics,
            'parameter_interpretation': self.model.get_parameter_interpretation()
        }
        
        # Add summary assessment
        report['summary_assessment'] = self._create_summary_assessment()
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _create_summary_assessment(self) -> Dict:
        """Create overall model assessment summary."""
        assessment = {
            'overall_quality': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Convergence assessment
        if self.convergence_diagnostics:
            if self.convergence_diagnostics['assessment']['overall_quality']:
                assessment['strengths'].append("Excellent MCMC convergence")
            else:
                assessment['weaknesses'].append("Poor MCMC convergence")
                assessment['recommendations'].extend(
                    self.convergence_diagnostics['recommendations']
                )
        
        # Calibration assessment
        if self.calibration_metrics:
            brier_score = self.calibration_metrics.get('brier_score', 1.0)
            if brier_score < 0.1:
                assessment['strengths'].append("Well-calibrated probability predictions")
            elif brier_score > 0.25:
                assessment['weaknesses'].append("Poor probability calibration")
                assessment['recommendations'].append("Consider model recalibration")
        
        # Business metrics assessment
        if self.business_metrics:
            supporter_segments = self.business_metrics.get('supporter_segments', {})
            high_value_pct = supporter_segments.get('high_value_percentage', 0)
            
            if high_value_pct > 20:
                assessment['strengths'].append("Good identification of high-value supporters")
            elif high_value_pct < 5:
                assessment['weaknesses'].append("Few high-value supporters identified")
        
        # Overall quality assessment
        if len(assessment['weaknesses']) == 0:
            assessment['overall_quality'] = 'Excellent'
        elif len(assessment['weaknesses']) <= 2:
            assessment['overall_quality'] = 'Good'
        else:
            assessment['overall_quality'] = 'Needs Improvement'
        
        return assessment
    
    def plot_diagnostics(
        self, 
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Create diagnostic plots for model evaluation.
        
        Args:
            save_path: Optional path to save plots
            figsize: Figure size for plots
        """
        if not self.convergence_diagnostics:
            logger.warning("No convergence diagnostics available. Run evaluate_convergence() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('BG/NBD Model Diagnostic Plots', fontsize=16)
        
        # Plot 1: Trace plots
        if self.model.trace is not None:
            az.plot_trace(self.model.trace, axes=axes[0, :2])
            axes[0, 0].set_title('Parameter Traces')
        
        # Plot 2: R-hat values
        rhat_data = self.convergence_diagnostics['rhat']['by_parameter']
        params = list(rhat_data.keys())
        rhat_values = list(rhat_data.values())
        
        axes[0, 2].bar(range(len(params)), rhat_values)
        axes[0, 2].axhline(y=1.01, color='r', linestyle='--', label='Threshold')
        axes[0, 2].set_xticks(range(len(params)))
        axes[0, 2].set_xticklabels(params, rotation=45)
        axes[0, 2].set_title('R-hat Values')
        axes[0, 2].legend()
        
        # Plot 3: ESS values
        ess_data = self.convergence_diagnostics['ess']['bulk_by_parameter']
        ess_values = list(ess_data.values())
        
        axes[1, 0].bar(range(len(params)), ess_values)
        axes[1, 0].axhline(y=400, color='r', linestyle='--', label='Threshold')
        axes[1, 0].set_xticks(range(len(params)))
        axes[1, 0].set_xticklabels(params, rotation=45)
        axes[1, 0].set_title('Effective Sample Size (Bulk)')
        axes[1, 0].legend()
        
        # Plot 4: Calibration curve (if available)
        if self.calibration_metrics:
            cal_data = self.calibration_metrics['calibration_curve']
            axes[1, 1].plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'], 'o-', label='Model')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[1, 1].set_xlabel('Mean Predicted Probability')
            axes[1, 1].set_ylabel('Fraction of Positives')
            axes[1, 1].set_title('Calibration Curve')
            axes[1, 1].legend()
        
        # Plot 5: Business metrics summary
        if self.business_metrics:
            segments = self.business_metrics['supporter_segments']['engagement_distribution']
            segment_names = list(segments.keys())
            segment_counts = list(segments.values())
            
            axes[1, 2].pie(segment_counts, labels=segment_names, autopct='%1.1f%%')
            axes[1, 2].set_title('Engagement Segments')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostic plots saved to {save_path}")
        
        plt.show()


def create_model_evaluator(model: BGNBDModel) -> BGNBDModelEvaluator:
    """
    Factory function to create a model evaluator.
    
    Args:
        model: Fitted BGNBDModel instance
        
    Returns:
        BGNBDModelEvaluator instance
    """
    return BGNBDModelEvaluator(model)


def evaluate_model_performance(
    model: BGNBDModel,
    test_data: Optional[pd.DataFrame] = None,
    prediction_period: int = 180,
    include_calibration: bool = True,
    include_business_metrics: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive model performance evaluation.
    
    Args:
        model: Fitted BGNBDModel instance
        test_data: Optional test dataset for validation
        prediction_period: Prediction period in days
        include_calibration: Whether to include calibration evaluation
        include_business_metrics: Whether to include business metrics
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = create_model_evaluator(model)
    
    # Always evaluate convergence
    convergence_results = evaluator.evaluate_convergence(verbose=verbose)
    
    results = {
        'convergence': convergence_results
    }
    
    # Evaluate on test data if provided
    if test_data is not None:
        if verbose:
            print(f"\nEvaluating on {len(test_data)} test samples...")
        
        # Predictive accuracy
        predictive_results = evaluator.evaluate_predictive_accuracy(
            test_data, prediction_period
        )
        results['predictive_accuracy'] = predictive_results
        
        # Calibration (if actual outcomes available)
        if include_calibration and 'actual_alive' in test_data.columns:
            calibration_results = evaluator.evaluate_calibration(test_data)
            results['calibration'] = calibration_results
    
    # Business metrics
    if include_business_metrics:
        data_for_business = test_data if test_data is not None else model.training_data
        business_results = evaluator.evaluate_business_metrics(
            data_for_business, prediction_period
        )
        results['business_metrics'] = business_results
    
    return results


def compare_models(
    models: List[BGNBDModel],
    model_names: List[str],
    test_data: pd.DataFrame,
    prediction_period: int = 180,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compare multiple BG/NBD models on the same test dataset.
    
    Args:
        models: List of fitted BGNBDModel instances
        model_names: Names for the models
        test_data: Test dataset for comparison
        prediction_period: Prediction period in days
        metrics: List of metrics to compare
        
    Returns:
        DataFrame with model comparison results
    """
    if len(models) != len(model_names):
        raise ModelEvaluationError("Number of models must match number of model names")
    
    if metrics is None:
        metrics = ['rhat_max', 'ess_bulk_min', 'brier_score', 'calibration_error']
    
    comparison_results = []
    
    for model, name in zip(models, model_names):
        logger.info(f"Evaluating model: {name}")
        
        try:
            evaluator = create_model_evaluator(model)
            
            # Get convergence diagnostics
            convergence = evaluator.evaluate_convergence(verbose=False)
            
            # Get calibration metrics (if possible)
            calibration = {}
            if 'actual_alive' in test_data.columns:
                calibration = evaluator.evaluate_calibration(test_data)
            
            # Compile results
            result_row = {
                'model_name': name,
                'n_supporters': len(model.training_data),
                'hierarchical': model.hierarchical,
                'rhat_max': convergence['rhat']['max'],
                'ess_bulk_min': convergence['ess']['bulk_min'],
                'converged': convergence['assessment']['converged'],
                'brier_score': calibration.get('brier_score', np.nan),
                'calibration_error': calibration.get('calibration_error', np.nan)
            }
            
            comparison_results.append(result_row)
            
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {e}")
            # Add row with error information
            comparison_results.append({
                'model_name': name,
                'error': str(e)
            })
    
    return pd.DataFrame(comparison_results)


def benchmark_model_performance(
    model: BGNBDModel,
    data: pd.DataFrame,
    n_iterations: int = 10,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Dict:
    """
    Benchmark model performance across multiple random samples.
    
    Args:
        model: Fitted BGNBDModel instance
        data: Dataset for benchmarking
        n_iterations: Number of benchmark iterations
        sample_size: Size of each sample (if None, uses full dataset)
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking model performance over {n_iterations} iterations")
    
    np.random.seed(random_state)
    
    if sample_size is None:
        sample_size = len(data)
    
    benchmark_results = {
        'iterations': [],
        'summary_stats': {}
    }
    
    for i in range(n_iterations):
        # Sample data
        if sample_size < len(data):
            sample_data = data.sample(n=sample_size, random_state=random_state + i)
        else:
            sample_data = data
        
        # Get segment info if hierarchical
        segment_col = None
        if model.hierarchical:
            segment_col = sample_data[model.segment_column].values
        
        # Generate predictions
        prob_alive = model.predict_probability_alive(
            sample_data['x'].values,
            sample_data['t_x'].values,
            sample_data['T'].values,
            segment_col
        )
        
        expected_transactions = model.predict_expected_transactions(
            180,  # 6 months
            sample_data['x'].values,
            sample_data['t_x'].values,
            sample_data['T'].values,
            segment_col
        )
        
        # Store iteration results
        iteration_result = {
            'iteration': i + 1,
            'sample_size': len(sample_data),
            'prob_alive_mean': float(prob_alive.mean()),
            'prob_alive_std': float(prob_alive.std()),
            'expected_transactions_mean': float(expected_transactions.mean()),
            'expected_transactions_std': float(expected_transactions.std()),
            'high_value_percentage': float((prob_alive >= 0.7).mean() * 100)
        }
        
        benchmark_results['iterations'].append(iteration_result)
    
    # Calculate summary statistics
    iterations_df = pd.DataFrame(benchmark_results['iterations'])
    
    benchmark_results['summary_stats'] = {
        'prob_alive_mean': {
            'mean': float(iterations_df['prob_alive_mean'].mean()),
            'std': float(iterations_df['prob_alive_mean'].std()),
            'min': float(iterations_df['prob_alive_mean'].min()),
            'max': float(iterations_df['prob_alive_mean'].max())
        },
        'expected_transactions_mean': {
            'mean': float(iterations_df['expected_transactions_mean'].mean()),
            'std': float(iterations_df['expected_transactions_mean'].std()),
            'min': float(iterations_df['expected_transactions_mean'].min()),
            'max': float(iterations_df['expected_transactions_mean'].max())
        },
        'high_value_percentage': {
            'mean': float(iterations_df['high_value_percentage'].mean()),
            'std': float(iterations_df['high_value_percentage'].std()),
            'min': float(iterations_df['high_value_percentage'].min()),
            'max': float(iterations_df['high_value_percentage'].max())
        }
    }
    
    return benchmark_results