"""
BG/NBD (Beta-Geometric/Negative Binomial Distribution) model implementation.

This module implements the BG/NBD model using PyMC for Bayesian inference
to predict supporter engagement in non-profit organizations.

The BG/NBD model assumes:
- Supporters make transactions according to a Poisson process with rate λ
- λ varies across supporters according to a Gamma distribution
- Each supporter has probability p of "dying" (churning) after each transaction
- p varies across supporters according to a Beta distribution

Mathematical Foundation:
- Beta-Geometric component: Models supporter "death" (churn) probability
- Negative Binomial component: Models engagement frequency while "alive"
- Parameters: r, alpha (transaction rate), a, b (dropout rate parameters)
"""

import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import special
from sklearn.model_selection import train_test_split

from ..config import get_config

logger = logging.getLogger(__name__)

# Suppress PyMC warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pymc")


class BGNBDModelError(Exception):
    """Custom exception for BG/NBD model errors."""
    pass


class BGNBDModel:
    """
    BG/NBD model implementation using PyMC for Bayesian inference.
    
    This class implements the Beta-Geometric/Negative Binomial Distribution model
    for predicting supporter engagement patterns in non-profit organizations.
    """
    
    def __init__(
        self,
        hierarchical: bool = False,
        segment_column: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the BG/NBD model.
        
        Args:
            hierarchical: Whether to use hierarchical modeling for segments
            segment_column: Column name for supporter segments (required if hierarchical=True)
            random_seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.hierarchical = hierarchical
        self.segment_column = segment_column
        self.random_seed = random_seed
        
        # Model components
        self.model = None
        self.trace = None
        self.posterior_predictive = None
        
        # Model parameters
        self.params = {}
        self.param_summary = None
        
        # Training data
        self.training_data = None
        self.segments = None
        
        # Model metadata
        self.model_metadata = {
            'created_at': None,
            'trained_at': None,
            'model_version': '1.0.0',
            'hierarchical': hierarchical,
            'segment_column': segment_column,
            'n_supporters': None,
            'n_segments': None
        }
        
        logger.info(f"Initialized BG/NBD model (hierarchical={hierarchical})")
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data for BG/NBD modeling."""
        required_columns = ['x', 't_x', 'T']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise BGNBDModelError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise BGNBDModelError("Input data is empty")
        
        # Check for invalid values
        if (data['x'] < 0).any():
            raise BGNBDModelError("Frequency (x) values must be non-negative")
        
        if (data['t_x'] < 0).any():
            raise BGNBDModelError("Recency (t_x) values must be non-negative")
        
        if (data['T'] <= 0).any():
            raise BGNBDModelError("Observation period (T) values must be positive")
        
        if (data['t_x'] > data['T']).any():
            raise BGNBDModelError("Recency (t_x) cannot exceed observation period (T)")
        
        # Check for hierarchical modeling requirements
        if self.hierarchical:
            if self.segment_column is None:
                raise BGNBDModelError("segment_column must be specified for hierarchical modeling")
            
            if self.segment_column not in data.columns:
                raise BGNBDModelError(f"Segment column '{self.segment_column}' not found in data")
    
    def _create_bgnbd_likelihood(self, x, t_x, T, r, alpha, a, b):
        """
        Create the BG/NBD likelihood function.
        
        This implements the mathematical likelihood function for the BG/NBD model
        combining the Beta-Geometric and Negative Binomial components.
        """
        # Beta function for normalization
        beta_ab = pm.math.exp(pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b))
        
        # Case 1: x = 0 (no repeat purchases)
        likelihood_x0 = beta_ab / pm.math.exp(pt.gammaln(a) + pt.gammaln(b + 1) - pt.gammaln(a + b + 1))
        
        # Case 2: x > 0 (repeat purchases occurred)
        # Negative Binomial component
        nb_term = (
            pm.math.exp(pt.gammaln(r + x) - pt.gammaln(r) - pt.gammaln(x + 1)) *
            (alpha / (alpha + T)) ** r *
            (T / (alpha + T)) ** x
        )
        
        # Beta-Geometric component
        bg_term1 = (
            pm.math.exp(pt.gammaln(a + 1) + pt.gammaln(b + x) - pt.gammaln(a + b + x + 1)) /
            beta_ab
        )
        
        bg_term2 = (
            pm.math.exp(pt.gammaln(a) + pt.gammaln(b + x + 1) - pt.gammaln(a + b + x + 1)) /
            beta_ab *
            (alpha / (alpha + t_x)) ** r *
            (t_x / (alpha + t_x)) ** x
        )
        
        likelihood_x_pos = nb_term * (bg_term1 - bg_term2)
        
        # Combine cases
        likelihood = pm.math.switch(
            pm.math.eq(x, 0),
            likelihood_x0,
            likelihood_x_pos
        )
        
        return likelihood
    
    def _build_basic_model(self, data: pd.DataFrame) -> pm.Model:
        """Build basic (non-hierarchical) BG/NBD model."""
        with pm.Model() as model:
            # Priors for model parameters
            # Transaction rate parameters (Gamma distribution for λ)
            r = pm.Exponential('r', lam=1.0)
            alpha = pm.Exponential('alpha', lam=1.0)
            
            # Dropout probability parameters (Beta distribution for p)
            a = pm.Exponential('a', lam=1.0)
            b = pm.Exponential('b', lam=1.0)
            
            # Extract data
            x = pm.ConstantData('x', data['x'].values)
            t_x = pm.ConstantData('t_x', data['t_x'].values)
            T = pm.ConstantData('T', data['T'].values)
            
            # Likelihood
            likelihood = self._create_bgnbd_likelihood(x, t_x, T, r, alpha, a, b)
            
            # Observed data
            obs = pm.Potential('obs', pm.math.log(likelihood))
            
        return model
    
    def _build_hierarchical_model(self, data: pd.DataFrame) -> pm.Model:
        """Build hierarchical BG/NBD model with segment-specific parameters."""
        # Prepare segment data
        segments = data[self.segment_column].unique()
        n_segments = len(segments)
        segment_idx = pd.Categorical(data[self.segment_column]).codes
        
        with pm.Model() as model:
            # Hyperpriors for segment-level parameters
            # Transaction rate hyperpriors
            mu_r = pm.Exponential('mu_r', lam=1.0)
            sigma_r = pm.HalfNormal('sigma_r', sigma=1.0)
            mu_alpha = pm.Exponential('mu_alpha', lam=1.0)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1.0)
            
            # Dropout probability hyperpriors
            mu_a = pm.Exponential('mu_a', lam=1.0)
            sigma_a = pm.HalfNormal('sigma_a', sigma=1.0)
            mu_b = pm.Exponential('mu_b', lam=1.0)
            sigma_b = pm.HalfNormal('sigma_b', sigma=1.0)
            
            # Segment-specific parameters
            r_seg = pm.Lognormal('r_seg', mu=pm.math.log(mu_r), sigma=sigma_r, shape=n_segments)
            alpha_seg = pm.Lognormal('alpha_seg', mu=pm.math.log(mu_alpha), sigma=sigma_alpha, shape=n_segments)
            a_seg = pm.Lognormal('a_seg', mu=pm.math.log(mu_a), sigma=sigma_a, shape=n_segments)
            b_seg = pm.Lognormal('b_seg', mu=pm.math.log(mu_b), sigma=sigma_b, shape=n_segments)
            
            # Extract data
            x = pm.ConstantData('x', data['x'].values)
            t_x = pm.ConstantData('t_x', data['t_x'].values)
            T = pm.ConstantData('T', data['T'].values)
            seg_idx = pm.ConstantData('seg_idx', segment_idx)
            
            # Select parameters for each observation based on segment
            r = r_seg[seg_idx]
            alpha = alpha_seg[seg_idx]
            a = a_seg[seg_idx]
            b = b_seg[seg_idx]
            
            # Likelihood
            likelihood = self._create_bgnbd_likelihood(x, t_x, T, r, alpha, a, b)
            
            # Observed data
            obs = pm.Potential('obs', pm.math.log(likelihood))
            
        return model
    
    def fit(
        self,
        data: pd.DataFrame,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        max_treedepth: int = 10,
        **kwargs
    ) -> 'BGNBDModel':
        """
        Fit the BG/NBD model to data using MCMC sampling.
        
        Args:
            data: DataFrame with BG/NBD variables (x, t_x, T)
            draws: Number of MCMC draws per chain
            tune: Number of tuning steps
            chains: Number of MCMC chains
            target_accept: Target acceptance rate for NUTS sampler
            max_treedepth: Maximum tree depth for NUTS sampler
            **kwargs: Additional arguments passed to pm.sample()
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting BG/NBD model fitting")
        
        # Validate data
        self._validate_data(data)
        
        # Store training data
        self.training_data = data.copy()
        
        if self.hierarchical:
            self.segments = data[self.segment_column].unique()
            self.model_metadata['n_segments'] = len(self.segments)
        
        self.model_metadata['n_supporters'] = len(data)
        self.model_metadata['created_at'] = datetime.now()
        
        try:
            # Build model
            if self.hierarchical:
                self.model = self._build_hierarchical_model(data)
            else:
                self.model = self._build_basic_model(data)
            
            # Sample from posterior
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    max_treedepth=max_treedepth,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    **kwargs
                )
            
            # Extract parameter summaries
            self.param_summary = az.summary(self.trace)
            self._extract_parameters()
            
            self.model_metadata['trained_at'] = datetime.now()
            
            logger.info("BG/NBD model fitting completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting BG/NBD model: {e}")
            raise BGNBDModelError(f"Model fitting failed: {e}")
    
    def _extract_parameters(self) -> None:
        """Extract parameter estimates from the trace."""
        if self.trace is None:
            return
        
        # Extract posterior means as point estimates
        posterior = self.trace.posterior
        
        if self.hierarchical:
            # Extract segment-specific parameters
            self.params = {
                'r_seg': posterior['r_seg'].mean(dim=['chain', 'draw']).values,
                'alpha_seg': posterior['alpha_seg'].mean(dim=['chain', 'draw']).values,
                'a_seg': posterior['a_seg'].mean(dim=['chain', 'draw']).values,
                'b_seg': posterior['b_seg'].mean(dim=['chain', 'draw']).values,
                'mu_r': float(posterior['mu_r'].mean().values),
                'mu_alpha': float(posterior['mu_alpha'].mean().values),
                'mu_a': float(posterior['mu_a'].mean().values),
                'mu_b': float(posterior['mu_b'].mean().values),
            }
        else:
            # Extract global parameters
            self.params = {
                'r': float(posterior['r'].mean().values),
                'alpha': float(posterior['alpha'].mean().values),
                'a': float(posterior['a'].mean().values),
                'b': float(posterior['b'].mean().values),
            }
    
    def predict_probability_alive(
        self,
        x: Union[int, np.ndarray],
        t_x: Union[float, np.ndarray],
        T: Union[int, np.ndarray],
        segment: Optional[Union[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Predict probability that supporters are still "alive" (active).
        
        Args:
            x: Number of repeat transactions
            t_x: Recency (time of last transaction)
            T: Observation period length
            segment: Segment identifier (required for hierarchical models)
            
        Returns:
            Array of P(Alive) probabilities
        """
        if self.params is None:
            raise BGNBDModelError("Model must be fitted before making predictions")
        
        # Convert inputs to arrays
        x = np.atleast_1d(x)
        t_x = np.atleast_1d(t_x)
        T = np.atleast_1d(T)
        
        if self.hierarchical:
            if segment is None:
                raise BGNBDModelError("Segment must be specified for hierarchical models")
            
            segment = np.atleast_1d(segment)
            segment_indices = [np.where(self.segments == s)[0][0] for s in segment]
            
            r = self.params['r_seg'][segment_indices]
            alpha = self.params['alpha_seg'][segment_indices]
            a = self.params['a_seg'][segment_indices]
            b = self.params['b_seg'][segment_indices]
        else:
            r = self.params['r']
            alpha = self.params['alpha']
            a = self.params['a']
            b = self.params['b']
        
        # Calculate P(Alive) using BG/NBD formula
        # For x = 0 case
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
        
        # Combine cases
        prob_alive = np.where(x == 0, prob_alive_x0, prob_alive_x_pos)
        
        return np.clip(prob_alive, 0, 1)  # Ensure probabilities are in [0, 1]
    
    def predict_expected_transactions(
        self,
        t: Union[int, np.ndarray],
        x: Union[int, np.ndarray],
        t_x: Union[float, np.ndarray],
        T: Union[int, np.ndarray],
        segment: Optional[Union[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Predict expected number of transactions in future period.
        
        Args:
            t: Length of future prediction period
            x: Number of repeat transactions in observation period
            t_x: Recency (time of last transaction)
            T: Observation period length
            segment: Segment identifier (required for hierarchical models)
            
        Returns:
            Array of expected transaction counts
        """
        if self.params is None:
            raise BGNBDModelError("Model must be fitted before making predictions")
        
        # Convert inputs to arrays
        t = np.atleast_1d(t)
        x = np.atleast_1d(x)
        t_x = np.atleast_1d(t_x)
        T = np.atleast_1d(T)
        
        if self.hierarchical:
            if segment is None:
                raise BGNBDModelError("Segment must be specified for hierarchical models")
            
            segment = np.atleast_1d(segment)
            segment_indices = [np.where(self.segments == s)[0][0] for s in segment]
            
            r = self.params['r_seg'][segment_indices]
            alpha = self.params['alpha_seg'][segment_indices]
            a = self.params['a_seg'][segment_indices]
            b = self.params['b_seg'][segment_indices]
        else:
            r = self.params['r']
            alpha = self.params['alpha']
            a = self.params['a']
            b = self.params['b']
        
        # Get P(Alive)
        prob_alive = self.predict_probability_alive(x, t_x, T, segment)
        
        # Expected transaction rate for alive customers
        expected_rate = r / alpha
        
        # Expected transactions = P(Alive) * rate * time
        expected_transactions = prob_alive * expected_rate * t
        
        return expected_transactions
    
    def predict_clv(
        self,
        t: Union[int, np.ndarray],
        x: Union[int, np.ndarray],
        t_x: Union[float, np.ndarray],
        T: Union[int, np.ndarray],
        monetary_value: Union[float, np.ndarray],
        segment: Optional[Union[str, np.ndarray]] = None,
        discount_rate: float = 0.1
    ) -> np.ndarray:
        """
        Predict Customer Lifetime Value (CLV) for supporters.
        
        Args:
            t: Length of future prediction period
            x: Number of repeat transactions in observation period
            t_x: Recency (time of last transaction)
            T: Observation period length
            monetary_value: Average monetary value per transaction
            segment: Segment identifier (required for hierarchical models)
            discount_rate: Annual discount rate for NPV calculation
            
        Returns:
            Array of predicted CLV values
        """
        # Get expected transactions
        expected_transactions = self.predict_expected_transactions(
            t, x, t_x, T, segment
        )
        
        # Convert inputs to arrays
        monetary_value = np.atleast_1d(monetary_value)
        t = np.atleast_1d(t)
        
        # Calculate CLV with discounting
        # Assuming transactions are spread evenly over the period
        discount_factor = 1 / (1 + discount_rate * t / 365)  # Daily discounting
        clv = expected_transactions * monetary_value * discount_factor
        
        return clv
    
    def segment_supporters(
        self,
        data: pd.DataFrame,
        method: str = 'probability',
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Segment supporters based on model predictions.
        
        Args:
            data: DataFrame with supporter data
            method: Segmentation method ('probability', 'expected_transactions', 'clv')
            threshold: Threshold for binary segmentation
            
        Returns:
            DataFrame with segmentation results
        """
        if self.params is None:
            raise BGNBDModelError("Model must be fitted before segmentation")
        
        results = data.copy()
        
        # Get segment info if hierarchical
        segment_col = None
        if self.hierarchical:
            segment_col = results[self.segment_column].values
        
        if method == 'probability':
            # Segment based on P(Alive)
            prob_alive = self.predict_probability_alive(
                results['x'].values,
                results['t_x'].values,
                results['T'].values,
                segment_col
            )
            results['prob_alive'] = prob_alive
            results['segment_prediction'] = np.where(
                prob_alive >= threshold, 'Active', 'Inactive'
            )
            
        elif method == 'expected_transactions':
            # Segment based on expected future transactions
            prediction_period = self.config.model.prediction_period_days
            expected_trans = self.predict_expected_transactions(
                prediction_period,
                results['x'].values,
                results['t_x'].values,
                results['T'].values,
                segment_col
            )
            results['expected_transactions'] = expected_trans
            results['segment_prediction'] = np.where(
                expected_trans >= threshold, 'High_Engagement', 'Low_Engagement'
            )
            
        elif method == 'clv':
            # Segment based on CLV
            if 'monetary' not in results.columns:
                raise BGNBDModelError("Monetary column required for CLV segmentation")
            
            prediction_period = self.config.model.prediction_period_days
            clv = self.predict_clv(
                prediction_period,
                results['x'].values,
                results['t_x'].values,
                results['T'].values,
                results['monetary'].values,
                segment_col
            )
            results['predicted_clv'] = clv
            results['segment_prediction'] = np.where(
                clv >= threshold, 'High_Value', 'Low_Value'
            )
        
        else:
            raise BGNBDModelError(f"Unknown segmentation method: {method}")
        
        return results
    
    def get_model_diagnostics(self) -> Dict:
        """Get model convergence and fit diagnostics."""
        if self.trace is None:
            raise BGNBDModelError("Model must be fitted before getting diagnostics")
        
        diagnostics = {}
        
        # Convergence diagnostics
        try:
            rhat = az.rhat(self.trace)
            
            # Extract R-hat values across all parameters
            all_rhat_values = []
            for var_name in rhat.data_vars:
                var_rhat = rhat[var_name]
                if hasattr(var_rhat, 'values'):
                    all_rhat_values.extend(var_rhat.values.flatten())
                else:
                    all_rhat_values.append(float(var_rhat))
            
            if all_rhat_values:
                rhat_max = float(max(all_rhat_values))
                rhat_mean = float(sum(all_rhat_values) / len(all_rhat_values))
            else:
                rhat_max = 1.0
                rhat_mean = 1.0
                
        except Exception as e:
            logger.warning(f"Could not compute R-hat: {e}")
            rhat_max = 1.0
            rhat_mean = 1.0
        
        # Get ESS using the current ArviZ API
        try:
            ess = az.ess(self.trace)
            
            # Extract minimum ESS values across all parameters
            all_ess_values = []
            for var_name in ess.data_vars:
                var_ess = ess[var_name]
                if hasattr(var_ess, 'values'):
                    all_ess_values.extend(var_ess.values.flatten())
                else:
                    all_ess_values.append(float(var_ess))
            
            if all_ess_values:
                ess_bulk_min = float(min(all_ess_values))
                ess_tail_min = float(min(all_ess_values))  # Using same value for both
            else:
                ess_bulk_min = 1000.0
                ess_tail_min = 1000.0
                
        except Exception as e:
            logger.warning(f"Could not compute ESS: {e}")
            ess_bulk_min = 1000.0  # Conservative fallback
            ess_tail_min = 1000.0
        
        diagnostics['convergence'] = {
            'rhat_max': rhat_max,
            'rhat_mean': rhat_mean,
            'ess_bulk_min': ess_bulk_min,
            'ess_tail_min': ess_tail_min,
            'converged': rhat_max < 1.01,
            'adequate_ess': ess_bulk_min > 400
        }
        
        # Parameter summary
        diagnostics['parameters'] = self.param_summary.to_dict()
        
        # Model metadata
        diagnostics['metadata'] = self.model_metadata.copy()
        
        return diagnostics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.trace is None:
            raise BGNBDModelError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data for serialization
        model_data = {
            'trace': self.trace,
            'params': self.params,
            'param_summary': self.param_summary,
            'training_data': self.training_data,
            'segments': self.segments,
            'model_metadata': self.model_metadata,
            'hierarchical': self.hierarchical,
            'segment_column': self.segment_column,
            'random_seed': self.random_seed
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise BGNBDModelError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BGNBDModel':
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BGNBDModel instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise BGNBDModelError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new model instance
            model = cls(
                hierarchical=model_data['hierarchical'],
                segment_column=model_data['segment_column'],
                random_seed=model_data['random_seed']
            )
            
            # Restore model state
            model.trace = model_data['trace']
            model.params = model_data['params']
            model.param_summary = model_data['param_summary']
            model.training_data = model_data['training_data']
            model.segments = model_data['segments']
            model.model_metadata = model_data['model_metadata']
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise BGNBDModelError(f"Failed to load model: {e}")
    
    def get_parameter_interpretation(self) -> Dict:
        """
        Get interpretation of model parameters.
        
        Returns:
            Dictionary with parameter interpretations
        """
        if self.params is None:
            raise BGNBDModelError("Model must be fitted before parameter interpretation")
        
        interpretation = {
            'description': 'BG/NBD Model Parameter Interpretation',
            'parameters': {}
        }
        
        if self.hierarchical:
            interpretation['parameters'] = {
                'r_seg': 'Shape parameter of Gamma distribution for transaction rates (per segment)',
                'alpha_seg': 'Rate parameter of Gamma distribution for transaction rates (per segment)',
                'a_seg': 'Shape parameter of Beta distribution for dropout probability (per segment)',
                'b_seg': 'Shape parameter of Beta distribution for dropout probability (per segment)',
                'mu_r': 'Population mean for r parameter',
                'mu_alpha': 'Population mean for alpha parameter',
                'mu_a': 'Population mean for a parameter',
                'mu_b': 'Population mean for b parameter'
            }
            
            # Add segment-specific interpretations
            for i, segment in enumerate(self.segments):
                r_val = self.params['r_seg'][i]
                alpha_val = self.params['alpha_seg'][i]
                a_val = self.params['a_seg'][i]
                b_val = self.params['b_seg'][i]
                
                interpretation['parameters'][f'segment_{segment}'] = {
                    'expected_transaction_rate': r_val / alpha_val,
                    'expected_lifetime': (a_val + b_val - 1) / (a_val - 1) if a_val > 1 else 'infinite',
                    'dropout_probability_mean': a_val / (a_val + b_val)
                }
        else:
            r_val = self.params['r']
            alpha_val = self.params['alpha']
            a_val = self.params['a']
            b_val = self.params['b']
            
            interpretation['parameters'] = {
                'r': f'Shape parameter of Gamma distribution for transaction rates (value: {r_val:.3f})',
                'alpha': f'Rate parameter of Gamma distribution for transaction rates (value: {alpha_val:.3f})',
                'a': f'Shape parameter of Beta distribution for dropout probability (value: {a_val:.3f})',
                'b': f'Shape parameter of Beta distribution for dropout probability (value: {b_val:.3f})',
                'expected_transaction_rate': r_val / alpha_val,
                'expected_lifetime': (a_val + b_val - 1) / (a_val - 1) if a_val > 1 else 'infinite',
                'dropout_probability_mean': a_val / (a_val + b_val)
            }
        
        return interpretation


def create_bgnbd_model(
    hierarchical: bool = False,
    segment_column: Optional[str] = None,
    random_seed: int = 42
) -> BGNBDModel:
    """
    Factory function to create a BG/NBD model instance.
    
    Args:
        hierarchical: Whether to use hierarchical modeling
        segment_column: Column name for segments (required if hierarchical=True)
        random_seed: Random seed for reproducibility
        
    Returns:
        BGNBDModel instance
    """
    return BGNBDModel(
        hierarchical=hierarchical,
        segment_column=segment_column,
        random_seed=random_seed
    )


def fit_bgnbd_model(
    data: pd.DataFrame,
    hierarchical: bool = False,
    segment_column: Optional[str] = None,
    **fit_kwargs
) -> BGNBDModel:
    """
    Convenience function to create and fit a BG/NBD model.
    
    Args:
        data: DataFrame with BG/NBD variables (x, t_x, T)
        hierarchical: Whether to use hierarchical modeling
        segment_column: Column name for segments (required if hierarchical=True)
        **fit_kwargs: Additional arguments passed to fit() method
        
    Returns:
        Fitted BGNBDModel instance
    """
    model = create_bgnbd_model(
        hierarchical=hierarchical,
        segment_column=segment_column
    )
    
    return model.fit(data, **fit_kwargs)