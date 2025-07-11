#!/usr/bin/env python3
"""
Main execution script for the non-profit BG/NBD engagement model.

This script provides a comprehensive CLI interface for running the complete
BG/NBD modeling pipeline including data extraction, preprocessing, model training,
prediction, and visualization.

Usage:
    python scripts/run_model.py [COMMAND] [OPTIONS]

Commands:
    extract      - Extract supporter actions and donations from database
    preprocess   - Transform raw data into BG/NBD format
    train        - Fit BG/NBD model using PyMC
    predict      - Generate predictions using trained model
    visualize    - Create diagnostic and business intelligence plots
    full-pipeline - Execute complete end-to-end workflow

Examples:
    python scripts/run_model.py extract --start-date 2023-01-01 --end-date 2024-01-01
    python scripts/run_model.py train --model-type hierarchical --draws 2000
    python scripts/run_model.py predict --model-path models/bgnbd_model.pkl
    python scripts/run_model.py visualize --plot-types diagnostic,business
    python scripts/run_model.py full-pipeline --config .env.production
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, test_database_connection, get_database_health
from src.data import (
    create_data_extractor, 
    create_bgnbd_processor, 
    process_supporter_data_pipeline
)
from src.models import (
    create_bgnbd_model, 
    fit_bgnbd_model, 
    create_model_evaluator,
    evaluate_model_performance,
    BGNBDModel
)
from src.visualization import (
    create_plotter, 
    plot_model_diagnostics, 
    plot_predictions,
    create_comprehensive_report
)


class ModelExecutionError(Exception):
    """Custom exception for model execution errors."""
    pass


class BGNBDModelRunner:
    """Main runner class for BG/NBD model execution."""
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize the model runner.
        
        Args:
            config_path: Optional path to configuration file
            verbose: Enable verbose logging
        """
        # Set up logging
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            import os
            from dotenv import load_dotenv
            load_dotenv(config_path)
            self.logger.info(f"Loaded configuration from {config_path}")
        
        self.config = get_config()
        
        # Validate configuration
        if not self.config.validate_config():
            raise ModelExecutionError("Configuration validation failed")
        
        self.logger.info("BG/NBD Model Runner initialized successfully")
    
    def setup_logging(self, verbose: bool) -> None:
        """Set up logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / f"run_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def validate_environment(self) -> bool:
        """Validate the execution environment."""
        self.logger.info("Validating execution environment...")
        
        try:
            # Test database connection
            if not test_database_connection():
                self.logger.error("Database connection test failed")
                return False
            
            # Check database health
            health = get_database_health()
            if not health.get('healthy', False):
                self.logger.error(f"Database health check failed: {health}")
                return False
            
            # Validate output directories
            for directory in [
                self.config.model.model_output_dir,
                self.config.model.data_output_dir,
                self.config.model.visualization_output_dir
            ]:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Validated directory: {directory}")
            
            self.logger.info("Environment validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return False
    
    def extract_data(
        self,
        start_date: datetime,
        end_date: datetime,
        min_donation: Optional[float] = None,
        use_cache: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Extract supporter actions and donations from database.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            min_donation: Minimum donation amount filter
            use_cache: Enable/disable data caching
            output_dir: Custom output directory
            
        Returns:
            Dictionary with extraction results and metadata
        """
        self.logger.info(f"Starting data extraction from {start_date} to {end_date}")
        
        try:
            # Create data extractor
            extractor = create_data_extractor()
            
            # Extract supporter actions
            self.logger.info("Extracting supporter actions...")
            with tqdm(desc="Extracting actions", unit="records") as pbar:
                actions_df = extractor.extract_supporter_actions(
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                pbar.update(len(actions_df))
            
            # Extract donations
            self.logger.info("Extracting donations...")
            with tqdm(desc="Extracting donations", unit="records") as pbar:
                donations_df = extractor.extract_donations(
                    start_date=start_date,
                    end_date=end_date,
                    min_amount=min_donation,
                    use_cache=use_cache
                )
                pbar.update(len(donations_df))
            
            # Generate data quality report
            self.logger.info("Generating data quality report...")
            quality_report = extractor.get_data_quality_report(actions_df, donations_df)
            
            # Save results
            output_path = Path(output_dir) if output_dir else self.config.model.data_output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save data files
            actions_file = output_path / f"actions_{timestamp}.csv"
            donations_file = output_path / f"donations_{timestamp}.csv"
            quality_file = output_path / f"quality_report_{timestamp}.json"
            
            actions_df.to_csv(actions_file, index=False)
            donations_df.to_csv(donations_file, index=False)
            
            with open(quality_file, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            # Prepare results
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'extraction_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'data_summary': {
                    'actions_count': len(actions_df),
                    'donations_count': len(donations_df),
                    'unique_supporters': actions_df['supporter_id'].nunique() if not actions_df.empty else 0,
                    'date_range': {
                        'actions': {
                            'min': actions_df['action_date'].min().isoformat() if not actions_df.empty else None,
                            'max': actions_df['action_date'].max().isoformat() if not actions_df.empty else None
                        },
                        'donations': {
                            'min': donations_df['donation_date'].min().isoformat() if not donations_df.empty else None,
                            'max': donations_df['donation_date'].max().isoformat() if not donations_df.empty else None
                        }
                    }
                },
                'output_files': {
                    'actions': str(actions_file),
                    'donations': str(donations_file),
                    'quality_report': str(quality_file)
                },
                'quality_report': quality_report
            }
            
            self.logger.info(f"Data extraction completed successfully")
            self.logger.info(f"Extracted {len(actions_df)} actions and {len(donations_df)} donations")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise ModelExecutionError(f"Data extraction failed: {e}")
    
    def preprocess_data(
        self,
        start_date: datetime,
        end_date: datetime,
        cutoff_date: Optional[datetime] = None,
        min_actions: int = 1,
        include_donations: bool = True,
        use_cache: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Transform raw data into BG/NBD format.
        
        Args:
            start_date: Start of observation period
            end_date: End of observation period
            cutoff_date: Analysis cutoff date
            min_actions: Minimum number of actions required per supporter
            include_donations: Whether to include donations as engagement events
            use_cache: Whether to use cached processed data
            output_dir: Custom output directory
            
        Returns:
            Dictionary with preprocessing results and metadata
        """
        self.logger.info(f"Starting data preprocessing from {start_date} to {end_date}")
        
        try:
            # Use full pipeline function for comprehensive processing
            self.logger.info("Processing supporter data into BG/NBD format...")
            
            with tqdm(desc="Processing data", unit="supporters") as pbar:
                bgnbd_df, summary_stats = process_supporter_data_pipeline(
                    start_date=start_date,
                    end_date=end_date,
                    cutoff_date=cutoff_date,
                    min_actions=min_actions,
                    include_donations=include_donations,
                    save_results=True
                )
                pbar.update(len(bgnbd_df))
            
            # Save additional outputs if custom directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save processed data
                processed_file = output_path / f"bgnbd_data_{timestamp}.csv"
                bgnbd_df.to_csv(processed_file, index=False)
                
                # Save summary statistics
                summary_file = output_path / f"preprocessing_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary_stats, f, indent=2, default=str)
                
                summary_stats['output_files'] = {
                    'processed_data': str(processed_file),
                    'summary': str(summary_file)
                }
            
            # Prepare results
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'processing_parameters': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'cutoff_date': cutoff_date.isoformat() if cutoff_date else end_date.isoformat(),
                    'min_actions': min_actions,
                    'include_donations': include_donations
                },
                'data_summary': {
                    'processed_supporters': len(bgnbd_df),
                    'observation_period_days': (end_date - start_date).days,
                    'bgnbd_variables': ['x', 't_x', 'T', 'frequency', 'monetary'] if not bgnbd_df.empty else []
                },
                'summary_statistics': summary_stats
            }
            
            self.logger.info(f"Data preprocessing completed successfully")
            self.logger.info(f"Processed {len(bgnbd_df)} supporters into BG/NBD format")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise ModelExecutionError(f"Data preprocessing failed: {e}")
    
    def train_model(
        self,
        data_path: str,
        model_type: str = 'basic',
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        segment_column: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Fit BG/NBD model using PyMC.
        
        Args:
            data_path: Path to processed BG/NBD data
            model_type: 'basic' or 'hierarchical'
            draws: Number of MCMC draws
            tune: Number of tuning steps
            chains: Number of MCMC chains
            segment_column: Column for hierarchical modeling
            output_dir: Custom output directory
            
        Returns:
            Dictionary with training results and metadata
        """
        self.logger.info(f"Starting model training with {model_type} model")
        
        try:
            # Load data
            self.logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            if data.empty:
                raise ModelExecutionError("No data found in the specified file")
            
            # Validate required columns
            required_cols = ['x', 't_x', 'T']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ModelExecutionError(f"Missing required columns: {missing_cols}")
            
            # Create and fit model
            hierarchical = model_type.lower() == 'hierarchical'
            
            if hierarchical and not segment_column:
                # Try to find a suitable segment column
                potential_cols = ['engagement_segment', 'segment', 'group']
                for col in potential_cols:
                    if col in data.columns:
                        segment_column = col
                        self.logger.info(f"Using '{col}' as segment column for hierarchical model")
                        break
                
                if not segment_column:
                    self.logger.warning("No segment column found, falling back to basic model")
                    hierarchical = False
            
            self.logger.info(f"Creating {model_type} BG/NBD model...")
            model = create_bgnbd_model(
                hierarchical=hierarchical,
                segment_column=segment_column
            )
            
            # Fit model with progress tracking
            self.logger.info(f"Fitting model with {draws} draws, {tune} tune, {chains} chains...")
            
            with tqdm(desc="MCMC Sampling", total=draws * chains) as pbar:
                # Note: PyMC doesn't support progress callbacks directly,
                # so we'll update after completion
                model.fit(
                    data=data,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=0.95,
                    max_treedepth=10
                )
                pbar.update(draws * chains)
            
            # Evaluate model convergence
            self.logger.info("Evaluating model convergence...")
            evaluator = create_model_evaluator(model)
            convergence_results = evaluator.evaluate_convergence(verbose=False)
            
            # Save model
            output_path = Path(output_dir) if output_dir else self.config.model.model_output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = output_path / f"bgnbd_model_{model_type}_{timestamp}.pkl"
            
            model.save_model(model_file)
            
            # Save convergence diagnostics
            diagnostics_file = output_path / f"convergence_diagnostics_{timestamp}.json"
            with open(diagnostics_file, 'w') as f:
                json.dump(convergence_results, f, indent=2, default=str)
            
            # Prepare results
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'model_configuration': {
                    'type': model_type,
                    'hierarchical': hierarchical,
                    'segment_column': segment_column,
                    'mcmc_parameters': {
                        'draws': draws,
                        'tune': tune,
                        'chains': chains
                    }
                },
                'training_data': {
                    'n_supporters': len(data),
                    'n_segments': data[segment_column].nunique() if segment_column and segment_column in data.columns else 1,
                    'data_path': data_path
                },
                'convergence_diagnostics': convergence_results,
                'output_files': {
                    'model': str(model_file),
                    'diagnostics': str(diagnostics_file)
                },
                'model_parameters': model.get_parameter_interpretation() if model.params else {}
            }
            
            # Log convergence status
            if convergence_results['assessment']['overall_quality']:
                self.logger.info("Model training completed successfully with good convergence")
            else:
                self.logger.warning("Model training completed but convergence may be poor")
                for rec in convergence_results['recommendations']:
                    self.logger.warning(f"Recommendation: {rec}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelExecutionError(f"Model training failed: {e}")
    
    def predict(
        self,
        model_path: str,
        data_path: Optional[str] = None,
        prediction_period: int = 180,
        output_format: str = 'csv',
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Generate predictions using trained model.
        
        Args:
            model_path: Path to trained model file
            data_path: Path to data for predictions (if None, uses training data)
            prediction_period: Prediction horizon in days
            output_format: 'csv', 'json', or 'both'
            output_dir: Custom output directory
            
        Returns:
            Dictionary with prediction results and metadata
        """
        self.logger.info(f"Starting prediction generation with {prediction_period} day horizon")
        
        try:
            # Load model
            self.logger.info(f"Loading model from {model_path}")
            model = BGNBDModel.load_model(model_path)
            
            # Load data
            if data_path:
                self.logger.info(f"Loading prediction data from {data_path}")
                data = pd.read_csv(data_path)
            else:
                self.logger.info("Using training data for predictions")
                data = model.training_data
                
                if data is None:
                    raise ModelExecutionError("No training data available in model and no data path provided")
            
            # Get segment info if hierarchical
            segment_col = None
            if model.hierarchical and model.segment_column in data.columns:
                segment_col = data[model.segment_column].values
            
            # Generate predictions
            self.logger.info("Generating P(Alive) predictions...")
            with tqdm(desc="Calculating P(Alive)", unit="supporters") as pbar:
                prob_alive = model.predict_probability_alive(
                    data['x'].values,
                    data['t_x'].values,
                    data['T'].values,
                    segment_col
                )
                pbar.update(len(data))
            
            self.logger.info("Generating expected transactions predictions...")
            with tqdm(desc="Calculating expected transactions", unit="supporters") as pbar:
                expected_transactions = model.predict_expected_transactions(
                    prediction_period,
                    data['x'].values,
                    data['t_x'].values,
                    data['T'].values,
                    segment_col
                )
                pbar.update(len(data))
            
            # Generate CLV predictions if monetary data available
            predicted_clv = None
            if 'monetary' in data.columns:
                self.logger.info("Generating CLV predictions...")
                with tqdm(desc="Calculating CLV", unit="supporters") as pbar:
                    predicted_clv = model.predict_clv(
                        prediction_period,
                        data['x'].values,
                        data['t_x'].values,
                        data['T'].values,
                        data['monetary'].values,
                        segment_col
                    )
                    pbar.update(len(data))
            
            # Create predictions dataframe
            predictions_df = data.copy()
            predictions_df['prob_alive'] = prob_alive
            predictions_df['expected_transactions'] = expected_transactions
            
            if predicted_clv is not None:
                predictions_df['predicted_clv'] = predicted_clv
            
            # Add segmentation based on predictions
            predictions_df['predicted_segment'] = model.segment_supporters(
                data, method='probability', threshold=0.5
            )['segment_prediction']
            
            # Save predictions
            output_path = Path(output_dir) if output_dir else self.config.model.data_output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_files = {}
            
            if output_format in ['csv', 'both']:
                csv_file = output_path / f"predictions_{timestamp}.csv"
                predictions_df.to_csv(csv_file, index=False)
                output_files['csv'] = str(csv_file)
            
            if output_format in ['json', 'both']:
                json_file = output_path / f"predictions_{timestamp}.json"
                predictions_df.to_json(json_file, orient='records', indent=2, default_handler=str)
                output_files['json'] = str(json_file)
            
            # Generate prediction summary
            summary = {
                'total_supporters': len(predictions_df),
                'prediction_period_days': prediction_period,
                'prob_alive_stats': {
                    'mean': float(prob_alive.mean()),
                    'median': float(prob_alive.median()),
                    'std': float(prob_alive.std()),
                    'active_supporters': int((prob_alive > 0.5).sum()),
                    'high_value_supporters': int((prob_alive > 0.7).sum())
                },
                'expected_transactions_stats': {
                    'total': float(expected_transactions.sum()),
                    'mean': float(expected_transactions.mean()),
                    'median': float(expected_transactions.median()),
                    'std': float(expected_transactions.std())
                }
            }
            
            if predicted_clv is not None:
                summary['clv_stats'] = {
                    'total': float(predicted_clv.sum()),
                    'mean': float(predicted_clv.mean()),
                    'median': float(predicted_clv.median()),
                    'std': float(predicted_clv.std())
                }
            
            # Save summary
            summary_file = output_path / f"prediction_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            output_files['summary'] = str(summary_file)
            
            # Prepare results
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'prediction_parameters': {
                    'model_path': model_path,
                    'data_path': data_path,
                    'prediction_period_days': prediction_period,
                    'model_type': 'hierarchical' if model.hierarchical else 'basic'
                },
                'prediction_summary': summary,
                'output_files': output_files
            }
            
            self.logger.info(f"Prediction generation completed successfully")
            self.logger.info(f"Generated predictions for {len(predictions_df)} supporters")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            raise ModelExecutionError(f"Prediction generation failed: {e}")
    
    def visualize(
        self,
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        plot_types: List[str] = None,
        output_format: str = 'png',
        interactive: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Create diagnostic and business intelligence plots.
        
        Args:
            model_path: Path to trained model (optional for data-only plots)
            data_path: Path to data for visualization
            plot_types: List of plot types to generate
            output_format: 'png', 'pdf', 'svg', or 'all'
            interactive: Enable interactive plots
            output_dir: Custom output directory
            
        Returns:
            Dictionary with visualization results and metadata
        """
        if plot_types is None:
            plot_types = ['diagnostic', 'prediction', 'business', 'data-quality']
        
        self.logger.info(f"Starting visualization generation for plot types: {plot_types}")
        
        try:
            # Load model if provided
            model = None
            if model_path:
                self.logger.info(f"Loading model from {model_path}")
                model = BGNBDModel.load_model(model_path)
            
            # Load data
            data = None
            if data_path:
                self.logger.info(f"Loading data from {data_path}")
                data = pd.read_csv(data_path)
            elif model and model.training_data is not None:
                self.logger.info("Using training data from model")
                data = model.training_data
            else:
                raise ModelExecutionError("No data available for visualization")
            
            # Create plotter
            plotter = create_plotter()
            
            # Set up output directory
            output_path = Path(output_dir) if output_dir else self.config.model.visualization_output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_dir = output_path / f"plots_{timestamp}"
            plot_dir.mkdir(exist_ok=True)
            
            generated_plots = {}
            
            # Generate diagnostic plots
            if 'diagnostic' in plot_types and model:
                self.logger.info("Generating diagnostic plots...")
                
                if model.trace is not None:
                    # Trace diagnostics
                    trace_path = plot_dir / f"trace_diagnostics.{output_format}"
                    plotter.plot_trace_diagnostics(model, save_path=trace_path, show_plot=interactive)
                    generated_plots['trace_diagnostics'] = str(trace_path)
                    
                    # Convergence summary
                    convergence_path = plot_dir / f"convergence_summary.{output_format}"
                    plotter.plot_convergence_summary(model, save_path=convergence_path, show_plot=interactive)
                    generated_plots['convergence_summary'] = str(convergence_path)
            
            # Generate prediction plots
            if 'prediction' in plot_types and model:
                self.logger.info("Generating prediction plots...")
                
                # P(Alive) distribution
                prob_alive_path = plot_dir / f"probability_alive.{output_format}"
                plotter.plot_probability_alive(model, data, save_path=prob_alive_path, show_plot=interactive)
                generated_plots['probability_alive'] = str(prob_alive_path)
                
                # Expected transactions
                expected_trans_path = plot_dir / f"expected_transactions.{output_format}"
                plotter.plot_expected_transactions(model, data, save_path=expected_trans_path, show_plot=interactive)
                generated_plots['expected_transactions'] = str(expected_trans_path)
                
                # CLV analysis (if monetary data available)
                if 'monetary' in data.columns:
                    clv_path = plot_dir / f"clv_analysis.{output_format}"
                    plotter.plot_clv_analysis(model, data, save_path=clv_path, show_plot=interactive)
                    generated_plots['clv_analysis'] = str(clv_path)
            
            # Generate business intelligence plots
            if 'business' in plot_types:
                self.logger.info("Generating business intelligence plots...")
                
                # Supporter segments
                if 'engagement_segment' in data.columns:
                    segments_path = plot_dir / f"supporter_segments.{output_format}"
                    plotter.plot_supporter_segments(data, save_path=segments_path, show_plot=interactive)
                    generated_plots['supporter_segments'] = str(segments_path)
                
                # Campaign targeting (if model available)
                if model:
                    targeting_path = plot_dir / f"campaign_targeting.{output_format}"
                    plotter.plot_campaign_targeting(model, data, save_path=targeting_path, show_plot=interactive)
                    generated_plots['campaign_targeting'] = str(targeting_path)
                
                # Engagement trends (if date data available)
                if 'first_event_date' in data.columns:
                    trends_path = plot_dir / f"engagement_trends.{output_format}"
                    plotter.plot_engagement_trends(data, save_path=trends_path, show_plot=interactive)
                    generated_plots['engagement_trends'] = str(trends_path)
            
            # Generate data quality plots
            if 'data-quality' in plot_types:
                self.logger.info("Generating data quality plots...")
                
                quality_path = plot_dir / f"data_quality_report.{output_format}"
                plotter.plot_data_quality_report(data, save_path=quality_path, show_plot=interactive)
                generated_plots['data_quality_report'] = str(quality_path)
            
            # Generate comprehensive dashboard
            if model and len(plot_types) > 1:
                self.logger.info("Generating comprehensive dashboard...")
                dashboard_path = plot_dir / f"dashboard.{output_format}"
                plotter.create_dashboard(model, data, save_path=dashboard_path, show_plot=interactive)
                generated_plots['dashboard'] = str(dashboard_path)
            
            # Create visualization summary
            summary = {
                'total_plots': len(generated_plots),
                'plot_types_generated': list(generated_plots.keys()),
                'output_format': output_format,
                'interactive': interactive,
                'data_summary': {
                    'n_supporters': len(data),
                    'columns': list(data.columns)
                }
            }
            
            # Save visualization summary
            summary_file = plot_dir / f"visualization_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Prepare results
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'visualization_parameters': {
                    'model_path': model_path,
                    'data_path': data_path,
                    'plot_types': plot_types,
                    'output_format': output_format,
                    'interactive': interactive
                },
                'visualization_summary': summary,
                'output_files': generated_plots,
                'output_directory': str(plot_dir),
                'summary_file': str(summary_file)
            }
            
            self.logger.info(f"Visualization generation completed successfully")
            self.logger.info(f"Generated {len(generated_plots)} plots in {plot_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise ModelExecutionError(f"Visualization generation failed: {e}")
    
    def full_pipeline(
        self,
        start_date: datetime,
        end_date: datetime,
        cutoff_date: Optional[datetime] = None,
        min_actions: int = 1,
        model_type: str = 'basic',
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        prediction_period: int = 180,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Execute complete end-to-end workflow.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            cutoff_date: Analysis cutoff date
            min_actions: Minimum actions per supporter
            model_type: 'basic' or 'hierarchical'
            draws: MCMC draws
            tune: MCMC tuning steps
            chains: MCMC chains
            prediction_period: Prediction horizon in days
            output_dir: Custom output directory
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info("Starting full BG/NBD modeling pipeline")
        
        pipeline_start = datetime.now()
        
        try:
            # Set up pipeline output directory
            if output_dir:
                pipeline_dir = Path(output_dir)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pipeline_dir = Path("outputs") / f"pipeline_{timestamp}"
            
            pipeline_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline_results = {
                'status': 'success',
                'pipeline_start': pipeline_start.isoformat(),
                'output_directory': str(pipeline_dir),
                'stages': {}
            }
            
            # Stage 1: Data Extraction
            self.logger.info("=== STAGE 1: DATA EXTRACTION ===")
            extraction_dir = pipeline_dir / "01_extraction"
            extraction_results = self.extract_data(
                start_date=start_date,
                end_date=end_date,
                output_dir=str(extraction_dir)
            )
            pipeline_results['stages']['extraction'] = extraction_results
            
            # Stage 2: Data Preprocessing
            self.logger.info("=== STAGE 2: DATA PREPROCESSING ===")
            preprocessing_dir = pipeline_dir / "02_preprocessing"
            preprocessing_results = self.preprocess_data(
                start_date=start_date,
                end_date=end_date,
                cutoff_date=cutoff_date,
                min_actions=min_actions,
                output_dir=str(preprocessing_dir)
            )
            pipeline_results['stages']['preprocessing'] = preprocessing_results
            
            # Get processed data path
            processed_data_path = preprocessing_results['summary_statistics'].get('output_file')
            if not processed_data_path:
                # Find the most recent processed data file
                processed_files = list(preprocessing_dir.glob("bgnbd_data_*.csv"))
                if not processed_files:
                    raise ModelExecutionError("No processed data file found")
                processed_data_path = str(max(processed_files, key=lambda p: p.stat().st_mtime))
            
            # Stage 3: Model Training
            self.logger.info("=== STAGE 3: MODEL TRAINING ===")
            training_dir = pipeline_dir / "03_training"
            training_results = self.train_model(
                data_path=processed_data_path,
                model_type=model_type,
                draws=draws,
                tune=tune,
                chains=chains,
                output_dir=str(training_dir)
            )
            pipeline_results['stages']['training'] = training_results
            
            # Get model path
            model_path = training_results['output_files']['model']
            
            # Stage 4: Prediction Generation
            self.logger.info("=== STAGE 4: PREDICTION GENERATION ===")
            prediction_dir = pipeline_dir / "04_predictions"
            prediction_results = self.predict(
                model_path=model_path,
                data_path=processed_data_path,
                prediction_period=prediction_period,
                output_format='both',
                output_dir=str(prediction_dir)
            )
            pipeline_results['stages']['predictions'] = prediction_results
            
            # Stage 5: Visualization
            self.logger.info("=== STAGE 5: VISUALIZATION ===")
            visualization_dir = pipeline_dir / "05_visualizations"
            visualization_results = self.visualize(
                model_path=model_path,
                data_path=processed_data_path,
                plot_types=['diagnostic', 'prediction', 'business', 'data-quality'],
                output_dir=str(visualization_dir)
            )
            pipeline_results['stages']['visualization'] = visualization_results
            
            # Generate comprehensive report
            self.logger.info("=== GENERATING COMPREHENSIVE REPORT ===")
            report_dir = pipeline_dir / "06_report"
            
            # Load model and data for report
            model = BGNBDModel.load_model(model_path)
            data = pd.read_csv(processed_data_path)
            
            report_results = create_comprehensive_report(
                model=model,
                data=data,
                output_dir=report_dir,
                prediction_period=prediction_period
            )
            pipeline_results['stages']['report'] = report_results
            
            # Calculate pipeline duration
            pipeline_end = datetime.now()
            pipeline_duration = pipeline_end - pipeline_start
            
            pipeline_results.update({
                'pipeline_end': pipeline_end.isoformat(),
                'total_duration_seconds': pipeline_duration.total_seconds(),
                'total_duration_formatted': str(pipeline_duration),
                'summary': {
                    'supporters_processed': preprocessing_results['data_summary']['processed_supporters'],
                    'model_type': training_results['model_configuration']['type'],
                    'model_converged': training_results['convergence_diagnostics']['assessment']['overall_quality'],
                    'predictions_generated': prediction_results['prediction_summary']['total_supporters'],
                    'plots_created': visualization_results['visualization_summary']['total_plots']
                }
            })
            
            # Save pipeline results
            pipeline_summary_file = pipeline_dir / "pipeline_summary.json"
            with open(pipeline_summary_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            self.logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            self.logger.info(f"Total duration: {pipeline_duration}")
            self.logger.info(f"Results saved to: {pipeline_dir}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise ModelExecutionError(f"Pipeline execution failed: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="BG/NBD Non-Profit Engagement Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data for 2023
  python scripts/run_model.py extract --start-date 2023-01-01 --end-date 2024-01-01
  
  # Train hierarchical model
  python scripts/run_model.py train data/processed/bgnbd_data.csv --model-type hierarchical --draws 2000
  
  # Generate predictions
  python scripts/run_model.py predict --model-path models/bgnbd_model.pkl --prediction-period 180
  
  # Create visualizations
  python scripts/run_model.py visualize --model-path models/bgnbd_model.pkl --plot-types diagnostic,business
  
  # Run full pipeline
  python scripts/run_model.py full-pipeline --start-date 2023-01-01 --end-date 2024-01-01 --model-type hierarchical
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Path to configuration file (.env)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract supporter data from database')
    extract_parser.add_argument('--start-date', type=str, required=True,
                               help='Analysis start date (YYYY-MM-DD)')
    extract_parser.add_argument('--end-date', type=str, required=True,
                               help='Analysis end date (YYYY-MM-DD)')
    extract_parser.add_argument('--min-donation', type=float,
                               help='Minimum donation amount filter')
    extract_parser.add_argument('--cache/--no-cache', dest='use_cache', default=True,
                               help='Enable/disable data caching')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Transform data into BG/NBD format')
    preprocess_parser.add_argument('--start-date', type=str, required=True,
                                  help='Observation period start date (YYYY-MM-DD)')
    preprocess_parser.add_argument('--end-date', type=str, required=True,
                                  help='Observation period end date (YYYY-MM-DD)')
    preprocess_parser.add_argument('--cutoff-date', type=str,
                                  help='Analysis cutoff date (YYYY-MM-DD)')
    preprocess_parser.add_argument('--min-actions', type=int, default=1,
                                  help='Minimum number of actions per supporter')
    preprocess_parser.add_argument('--include-donations/--exclude-donations',
                                  dest='include_donations', default=True,
                                  help='Include/exclude donations as engagement events')
    preprocess_parser.add_argument('--cache/--no-cache', dest='use_cache', default=True,
                                  help='Enable/disable data caching')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train BG/NBD model')
    train_parser.add_argument('data_path', type=str, help='Path to processed BG/NBD data')
    train_parser.add_argument('--model-type', choices=['basic', 'hierarchical'], default='basic',
                             help='Model type to train')
    train_parser.add_argument('--draws', type=int, default=2000,
                             help='Number of MCMC draws')
    train_parser.add_argument('--tune', type=int, default=1000,
                             help='Number of tuning steps')
    train_parser.add_argument('--chains', type=int, default=4,
                             help='Number of MCMC chains')
    train_parser.add_argument('--segment-column', type=str,
                             help='Column name for hierarchical modeling')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model file')
    predict_parser.add_argument('--data-path', type=str,
                               help='Path to data for predictions (uses training data if not provided)')
    predict_parser.add_argument('--prediction-period', type=int, default=180,
                               help='Prediction horizon in days')
    predict_parser.add_argument('--output-format', choices=['csv', 'json', 'both'], default='csv',
                               help='Output format for predictions')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations')
    visualize_parser.add_argument('--model-path', type=str,
                                 help='Path to trained model (optional for data-only plots)')
    visualize_parser.add_argument('--data-path', type=str,
                                 help='Path to data for visualization')
    visualize_parser.add_argument('--plot-types', type=str, default='diagnostic,prediction,business,data-quality',
                                 help='Comma-separated list of plot types')
    visualize_parser.add_argument('--output-format', choices=['png', 'pdf', 'svg', 'all'], default='png',
                                 help='Output format for plots')
    visualize_parser.add_argument('--interactive', action='store_true',
                                 help='Enable interactive plots')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('full-pipeline', help='Execute complete workflow')
    pipeline_parser.add_argument('--start-date', type=str, required=True,
                                help='Analysis start date (YYYY-MM-DD)')
    pipeline_parser.add_argument('--end-date', type=str, required=True,
                                help='Analysis end date (YYYY-MM-DD)')
    pipeline_parser.add_argument('--cutoff-date', type=str,
                                help='Analysis cutoff date (YYYY-MM-DD)')
    pipeline_parser.add_argument('--min-actions', type=int, default=1,
                                help='Minimum actions per supporter')
    pipeline_parser.add_argument('--model-type', choices=['basic', 'hierarchical'], default='basic',
                                help='Model type to train')
    pipeline_parser.add_argument('--draws', type=int, default=2000,
                                help='MCMC draws')
    pipeline_parser.add_argument('--tune', type=int, default=1000,
                                help='MCMC tuning steps')
    pipeline_parser.add_argument('--chains', type=int, default=4,
                                help='MCMC chains')
    pipeline_parser.add_argument('--prediction-period', type=int, default=180,
                                help='Prediction horizon in days')
    
    return parser


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize runner
        runner = BGNBDModelRunner(config_path=args.config, verbose=args.verbose)
        
        # Validate environment
        if not runner.validate_environment():
            print("❌ Environment validation failed. Please check your configuration.")
            sys.exit(1)
        
        print("✅ Environment validation successful")
        
        # Execute command
        if args.command == 'extract':
            start_date = parse_date(args.start_date)
            end_date = parse_date(args.end_date)
            
            results = runner.extract_data(
                start_date=start_date,
                end_date=end_date,
                min_donation=args.min_donation,
                use_cache=args.use_cache,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Data extraction completed successfully!")
            print(f"📊 Extracted {results['data_summary']['actions_count']} actions and {results['data_summary']['donations_count']} donations")
            print(f"👥 {results['data_summary']['unique_supporters']} unique supporters")
            print(f"📁 Output files saved to: {Path(list(results['output_files'].values())[0]).parent}")
        
        elif args.command == 'preprocess':
            start_date = parse_date(args.start_date)
            end_date = parse_date(args.end_date)
            cutoff_date = parse_date(args.cutoff_date) if args.cutoff_date else None
            
            results = runner.preprocess_data(
                start_date=start_date,
                end_date=end_date,
                cutoff_date=cutoff_date,
                min_actions=args.min_actions,
                include_donations=args.include_donations,
                use_cache=args.use_cache,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Data preprocessing completed successfully!")
            print(f"👥 Processed {results['data_summary']['processed_supporters']} supporters")
            print(f"📅 Observation period: {results['processing_parameters']['observation_period_days']} days")
            print(f"📊 Summary statistics available in output files")
        
        elif args.command == 'train':
            results = runner.train_model(
                data_path=args.data_path,
                model_type=args.model_type,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                segment_column=args.segment_column,
                output_dir=args.output_dir
            )
            
            convergence_status = "✅ Good" if results['convergence_diagnostics']['assessment']['overall_quality'] else "⚠️ Poor"
            
            print(f"\n✅ Model training completed successfully!")
            print(f"🤖 Model type: {results['model_configuration']['type']}")
            print(f"👥 Training data: {results['training_data']['n_supporters']} supporters")
            print(f"🔄 Convergence: {convergence_status}")
            print(f"💾 Model saved to: {results['output_files']['model']}")
        
        elif args.command == 'predict':
            results = runner.predict(
                model_path=args.model_path,
                data_path=args.data_path,
                prediction_period=args.prediction_period,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Prediction generation completed successfully!")
            print(f"👥 Predictions for {results['prediction_summary']['total_supporters']} supporters")
            print(f"📅 Prediction period: {results['prediction_parameters']['prediction_period_days']} days")
            print(f"🎯 Active supporters: {results['prediction_summary']['prob_alive_stats']['active_supporters']}")
            print(f"💎 High-value supporters: {results['prediction_summary']['prob_alive_stats']['high_value_supporters']}")
            print(f"📁 Output files: {list(results['output_files'].keys())}")
        
        elif args.command == 'visualize':
            plot_types = args.plot_types.split(',') if args.plot_types else None
            
            results = runner.visualize(
                model_path=args.model_path,
                data_path=args.data_path,
                plot_types=plot_types,
                output_format=args.output_format,
                interactive=args.interactive,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Visualization generation completed successfully!")
            print(f"📊 Generated {results['visualization_summary']['total_plots']} plots")
            print(f"🎨 Plot types: {', '.join(results['visualization_summary']['plot_types_generated'])}")
            print(f"📁 Plots saved to: {results['output_directory']}")
        
        elif args.command == 'full-pipeline':
            start_date = parse_date(args.start_date)
            end_date = parse_date(args.end_date)
            cutoff_date = parse_date(args.cutoff_date) if args.cutoff_date else None
            
            results = runner.full_pipeline(
                start_date=start_date,
                end_date=end_date,
                cutoff_date=cutoff_date,
                min_actions=args.min_actions,
                model_type=args.model_type,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                prediction_period=args.prediction_period,
                output_dir=args.output_dir
            )
            
            print(f"\n🎉 Full pipeline completed successfully!")
            print(f"⏱️  Total duration: {results['total_duration_formatted']}")
            print(f"👥 Supporters processed: {results['summary']['supporters_processed']}")
            print(f"🤖 Model type: {results['summary']['model_type']}")
            print(f"🔄 Model converged: {'✅ Yes' if results['summary']['model_converged'] else '⚠️ No'}")
            print(f"🎯 Predictions generated: {results['summary']['predictions_generated']}")
            print(f"📊 Plots created: {results['summary']['plots_created']}")
            print(f"📁 All results saved to: {results['output_directory']}")
        
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except ModelExecutionError as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📋 Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()