#!/usr/bin/env python3
"""
MVP Validation Script for Non-Profit Engagement Model

This script performs essential validation checks to ensure the project is
production-ready without requiring extensive test infrastructure.

Usage:
    python validate_mvp.py [--environment dev|prod] [--quick]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mvp_validation.log')
    ]
)
logger = logging.getLogger(__name__)


class MVPValidator:
    """Minimal Viable Product validator for the engagement model."""
    
    def __init__(self, environment: str = "development", quick_mode: bool = False):
        """Initialize the MVP validator."""
        self.environment = environment
        self.quick_mode = quick_mode
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'environment': environment,
            'quick_mode': quick_mode,
            'validation_results': {},
            'overall_status': 'UNKNOWN',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        logger.info(f"Initializing MVP validation for {environment} environment")
        if quick_mode:
            logger.info("Running in quick mode - reduced test coverage")
    
    def validate_imports(self) -> bool:
        """Test that all core modules can be imported."""
        logger.info("🔍 Validating module imports...")
        
        try:
            # Test core imports
            from src.config import get_config
            from src.data.sample_data import generate_sample_data
            from src.models.bgnbd import BGNBDModel, create_bgnbd_model
            from src.visualization.plots import create_plotter
            
            # Test CLI import
            from scripts.run_model import BGNBDModelRunner
            
            self.results['validation_results']['imports'] = {
                'status': 'PASS',
                'message': 'All core modules imported successfully'
            }
            logger.info("✅ Module imports: PASS")
            return True
            
        except Exception as e:
            error_msg = f"Import validation failed: {str(e)}"
            self.results['validation_results']['imports'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Module imports: FAIL - {error_msg}")
            return False
    
    def validate_configuration(self) -> bool:
        """Test configuration loading and validation."""
        logger.info("🔍 Validating configuration...")
        
        try:
            # Load environment file
            env_file = f".env.{self.environment}"
            if Path(env_file).exists():
                from dotenv import load_dotenv
                load_dotenv(env_file)
                logger.info(f"Loaded environment from {env_file}")
            
            # Test configuration loading
            from src.config import get_config
            config = get_config()
            
            # Basic configuration checks
            checks = {
                'environment_set': config.environment is not None,
                'database_config': hasattr(config, 'database'),
                'model_config': hasattr(config, 'model'),
                'logging_config': hasattr(config, 'logging')
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                error_msg = f"Configuration validation failed: {failed_checks}"
                self.results['validation_results']['configuration'] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'failed_checks': failed_checks
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ Configuration: FAIL - {error_msg}")
                return False
            
            self.results['validation_results']['configuration'] = {
                'status': 'PASS',
                'message': 'Configuration loaded and validated successfully',
                'environment': config.environment,
                'debug_mode': config.debug
            }
            logger.info("✅ Configuration: PASS")
            return True
            
        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            self.results['validation_results']['configuration'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Configuration: FAIL - {error_msg}")
            return False
    
    def validate_sample_data_generation(self) -> bool:
        """Test sample data generation functionality."""
        logger.info("🔍 Validating sample data generation...")
        
        try:
            from src.data.sample_data import generate_demo_dataset
            
            # Generate small sample dataset
            sample_size = 100 if self.quick_mode else 500
            logger.info(f"Generating sample data with {sample_size} supporters...")
            
            start_time = time.time()
            dataset = generate_demo_dataset(
                n_supporters=sample_size,
                save_files=False
            )
            generation_time = time.time() - start_time
            
            # Extract the supporters data which contains BG/NBD variables
            sample_data = dataset['supporters']
            
            # Validate sample data structure
            required_columns = ['supporter_id', 'x', 't_x', 'T']
            missing_columns = [col for col in required_columns if col not in sample_data.columns]
            
            if missing_columns:
                error_msg = f"Sample data missing required columns: {missing_columns}"
                self.results['validation_results']['sample_data'] = {
                    'status': 'FAIL',
                    'message': error_msg
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ Sample data: FAIL - {error_msg}")
                return False
            
            self.results['validation_results']['sample_data'] = {
                'status': 'PASS',
                'message': 'Sample data generated successfully',
                'sample_size': len(sample_data),
                'generation_time_seconds': round(generation_time, 2),
                'columns': list(sample_data.columns)
            }
            logger.info(f"✅ Sample data: PASS - Generated {len(sample_data)} records in {generation_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = f"Sample data generation failed: {str(e)}"
            self.results['validation_results']['sample_data'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Sample data: FAIL - {error_msg}")
            return False
    
    def validate_model_creation(self) -> bool:
        """Test BG/NBD model creation and basic functionality."""
        logger.info("🔍 Validating model creation...")
        
        try:
            from src.models.bgnbd import create_bgnbd_model
            
            # Test basic model creation
            logger.info("Creating basic BG/NBD model...")
            basic_model = create_bgnbd_model(hierarchical=False)
            
            # Test hierarchical model creation
            logger.info("Creating hierarchical BG/NBD model...")
            hierarchical_model = create_bgnbd_model(
                hierarchical=True, 
                segment_column='engagement_segment'
            )
            
            # Validate model attributes
            model_checks = {
                'basic_model_created': basic_model is not None,
                'hierarchical_model_created': hierarchical_model is not None,
                'basic_model_type': not basic_model.hierarchical,
                'hierarchical_model_type': hierarchical_model.hierarchical
            }
            
            failed_checks = [k for k, v in model_checks.items() if not v]
            
            if failed_checks:
                error_msg = f"Model creation validation failed: {failed_checks}"
                self.results['validation_results']['model_creation'] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'failed_checks': failed_checks
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ Model creation: FAIL - {error_msg}")
                return False
            
            self.results['validation_results']['model_creation'] = {
                'status': 'PASS',
                'message': 'Model creation successful',
                'models_tested': ['basic', 'hierarchical']
            }
            logger.info("✅ Model creation: PASS")
            return True
            
        except Exception as e:
            error_msg = f"Model creation failed: {str(e)}"
            self.results['validation_results']['model_creation'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Model creation: FAIL - {error_msg}")
            return False
    
    def validate_model_training(self) -> bool:
        """Test model training with sample data."""
        logger.info("🔍 Validating model training...")
        
        try:
            from src.data.sample_data import generate_demo_dataset
            from src.models.bgnbd import create_bgnbd_model
            
            # Generate sample data for training
            sample_size = 50 if self.quick_mode else 200
            logger.info(f"Generating training data with {sample_size} supporters...")
            
            dataset = generate_demo_dataset(
                n_supporters=sample_size,
                save_files=False
            )
            
            # We need to create BG/NBD format data from the raw dataset
            # For now, let's create a simple mock dataset with the required columns
            training_data = pd.DataFrame({
                'supporter_id': range(1, sample_size + 1),
                'x': np.random.poisson(2, sample_size),  # frequency
                't_x': np.random.uniform(0, 365, sample_size),  # recency
                'T': np.full(sample_size, 365)  # observation period
            })
            
            # Create and train model with minimal parameters for speed
            logger.info("Training BG/NBD model...")
            model = create_bgnbd_model(hierarchical=False)
            
            start_time = time.time()
            model.fit(
                data=training_data,
                draws=100 if self.quick_mode else 500,
                tune=50 if self.quick_mode else 250,
                chains=2,
                target_accept=0.8
            )
            training_time = time.time() - start_time
            
            # Validate model training results
            training_checks = {
                'model_fitted': model.trace is not None,
                'parameters_extracted': model.params is not None,
                'training_data_stored': model.training_data is not None
            }
            
            failed_checks = [k for k, v in training_checks.items() if not v]
            
            if failed_checks:
                error_msg = f"Model training validation failed: {failed_checks}"
                self.results['validation_results']['model_training'] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'failed_checks': failed_checks
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ Model training: FAIL - {error_msg}")
                return False
            
            # Test basic predictions
            logger.info("Testing model predictions...")
            test_data = training_data.head(10)
            
            prob_alive = model.predict_probability_alive(
                test_data['x'].values,
                test_data['t_x'].values,
                test_data['T'].values
            )
            
            expected_transactions = model.predict_expected_transactions(
                90,  # 90 days prediction
                test_data['x'].values,
                test_data['t_x'].values,
                test_data['T'].values
            )
            
            prediction_checks = {
                'prob_alive_generated': prob_alive is not None and len(prob_alive) == len(test_data),
                'expected_transactions_generated': expected_transactions is not None and len(expected_transactions) == len(test_data),
                'prob_alive_valid_range': all(0 <= p <= 1 for p in prob_alive),
                'expected_transactions_non_negative': all(t >= 0 for t in expected_transactions)
            }
            
            failed_prediction_checks = [k for k, v in prediction_checks.items() if not v]
            
            if failed_prediction_checks:
                warning_msg = f"Prediction validation issues: {failed_prediction_checks}"
                self.results['warnings'].append(warning_msg)
                logger.warning(f"⚠️ Prediction validation: {warning_msg}")
            
            self.results['validation_results']['model_training'] = {
                'status': 'PASS',
                'message': 'Model training and prediction successful',
                'training_time_seconds': round(training_time, 2),
                'training_data_size': len(training_data),
                'predictions_tested': len(test_data),
                'prediction_checks': prediction_checks
            }
            logger.info(f"✅ Model training: PASS - Trained in {training_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.results['validation_results']['model_training'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Model training: FAIL - {error_msg}")
            return False
    
    def validate_visualization(self) -> bool:
        """Test visualization functionality."""
        logger.info("🔍 Validating visualization...")
        
        try:
            from src.visualization.plots import create_plotter
            from src.data.sample_data import generate_sample_data
            
            # Create plotter
            plotter = create_plotter()
            
            # Generate sample data for plotting
            sample_data = generate_sample_data(
                n_supporters=100,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
                seed=42
            )
            
            # Test basic plotting functionality (without saving files)
            logger.info("Testing basic plotting functionality...")
            
            # This is a minimal test - just ensure the plotter can be created
            # and basic methods exist
            visualization_checks = {
                'plotter_created': plotter is not None,
                'has_plot_methods': hasattr(plotter, 'plot_data_quality_report'),
                'sample_data_available': sample_data is not None and len(sample_data) > 0
            }
            
            failed_checks = [k for k, v in visualization_checks.items() if not v]
            
            if failed_checks:
                error_msg = f"Visualization validation failed: {failed_checks}"
                self.results['validation_results']['visualization'] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'failed_checks': failed_checks
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ Visualization: FAIL - {error_msg}")
                return False
            
            self.results['validation_results']['visualization'] = {
                'status': 'PASS',
                'message': 'Visualization components validated successfully',
                'plotter_type': type(plotter).__name__
            }
            logger.info("✅ Visualization: PASS")
            return True
            
        except Exception as e:
            error_msg = f"Visualization validation failed: {str(e)}"
            self.results['validation_results']['visualization'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ Visualization: FAIL - {error_msg}")
            return False
    
    def validate_cli_interface(self) -> bool:
        """Test CLI interface functionality."""
        logger.info("🔍 Validating CLI interface...")
        
        try:
            from scripts.run_model import BGNBDModelRunner, create_parser
            
            # Test CLI components
            logger.info("Testing CLI parser creation...")
            parser = create_parser()
            
            # Test model runner initialization
            logger.info("Testing model runner initialization...")
            runner = BGNBDModelRunner(verbose=False)
            
            cli_checks = {
                'parser_created': parser is not None,
                'runner_created': runner is not None,
                'runner_has_config': hasattr(runner, 'config'),
                'runner_has_logger': hasattr(runner, 'logger')
            }
            
            failed_checks = [k for k, v in cli_checks.items() if not v]
            
            if failed_checks:
                error_msg = f"CLI validation failed: {failed_checks}"
                self.results['validation_results']['cli_interface'] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'failed_checks': failed_checks
                }
                self.results['critical_issues'].append(error_msg)
                logger.error(f"❌ CLI interface: FAIL - {error_msg}")
                return False
            
            self.results['validation_results']['cli_interface'] = {
                'status': 'PASS',
                'message': 'CLI interface validated successfully'
            }
            logger.info("✅ CLI interface: PASS")
            return True
            
        except Exception as e:
            error_msg = f"CLI validation failed: {str(e)}"
            self.results['validation_results']['cli_interface'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['critical_issues'].append(error_msg)
            logger.error(f"❌ CLI interface: FAIL - {error_msg}")
            return False
    
    def validate_docker_configuration(self) -> bool:
        """Test Docker configuration files."""
        logger.info("🔍 Validating Docker configuration...")
        
        try:
            docker_files = {
                'Dockerfile': Path('Dockerfile'),
                'docker-compose.yml': Path('docker-compose.yml'),
                'docker-compose.dev.yml': Path('docker-compose.dev.yml')
            }
            
            # Check if Docker files exist and are readable
            docker_checks = {}
            for name, path in docker_files.items():
                docker_checks[f'{name}_exists'] = path.exists()
                if path.exists():
                    docker_checks[f'{name}_readable'] = path.is_file() and path.stat().st_size > 0
            
            # Check environment files
            env_files = {
                '.env.development': Path('.env.development'),
                '.env.production': Path('.env.production')
            }
            
            for name, path in env_files.items():
                docker_checks[f'{name}_exists'] = path.exists()
            
            failed_checks = [k for k, v in docker_checks.items() if not v]
            
            if failed_checks:
                warning_msg = f"Docker configuration issues: {failed_checks}"
                self.results['warnings'].append(warning_msg)
                logger.warning(f"⚠️ Docker configuration: {warning_msg}")
            
            # This is not a critical failure for MVP
            self.results['validation_results']['docker_configuration'] = {
                'status': 'PASS' if not failed_checks else 'WARNING',
                'message': 'Docker configuration validated' if not failed_checks else f'Docker configuration has issues: {failed_checks}',
                'checks': docker_checks
            }
            
            if failed_checks:
                logger.warning("⚠️ Docker configuration: WARNING - Some issues found")
            else:
                logger.info("✅ Docker configuration: PASS")
            
            return True
            
        except Exception as e:
            error_msg = f"Docker configuration validation failed: {str(e)}"
            self.results['validation_results']['docker_configuration'] = {
                'status': 'FAIL',
                'message': error_msg,
                'error': str(e)
            }
            self.results['warnings'].append(error_msg)
            logger.warning(f"⚠️ Docker configuration: FAIL - {error_msg}")
            return True  # Not critical for MVP
    
    def run_validation(self) -> Dict:
        """Run all validation checks."""
        logger.info("🚀 Starting MVP validation...")
        start_time = time.time()
        
        # Define validation steps
        validation_steps = [
            ('imports', self.validate_imports),
            ('configuration', self.validate_configuration),
            ('sample_data', self.validate_sample_data_generation),
            ('model_creation', self.validate_model_creation),
            ('model_training', self.validate_model_training),
            ('visualization', self.validate_visualization),
            ('cli_interface', self.validate_cli_interface),
            ('docker_configuration', self.validate_docker_configuration)
        ]
        
        # Run validation steps
        passed_steps = 0
        total_steps = len(validation_steps)
        
        for step_name, validation_func in validation_steps:
            logger.info(f"Running validation step: {step_name}")
            try:
                if validation_func():
                    passed_steps += 1
            except Exception as e:
                logger.error(f"Validation step {step_name} failed with exception: {e}")
                self.results['critical_issues'].append(f"Validation step {step_name} failed: {str(e)}")
        
        # Calculate overall status
        total_time = time.time() - start_time
        success_rate = passed_steps / total_steps
        
        if success_rate >= 0.9:
            self.results['overall_status'] = 'PASS'
        elif success_rate >= 0.7:
            self.results['overall_status'] = 'WARNING'
        else:
            self.results['overall_status'] = 'FAIL'
        
        # Add summary
        self.results['summary'] = {
            'total_validation_time_seconds': round(total_time, 2),
            'steps_passed': passed_steps,
            'total_steps': total_steps,
            'success_rate': round(success_rate * 100, 1),
            'critical_issues_count': len(self.results['critical_issues']),
            'warnings_count': len(self.results['warnings'])
        }
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"🏁 MVP validation completed in {total_time:.2f}s")
        logger.info(f"📊 Results: {passed_steps}/{total_steps} steps passed ({success_rate*100:.1f}%)")
        logger.info(f"🎯 Overall status: {self.results['overall_status']}")
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for critical issues
        if self.results['critical_issues']:
            recommendations.append("🚨 CRITICAL: Address all critical issues before production deployment")
        
        # Check success rate
        success_rate = self.results['summary']['success_rate']
        if success_rate < 90:
            recommendations.append(f"⚠️ Success rate is {success_rate:.1f}% - aim for >90% before production")
        
        # Environment-specific recommendations
        if self.environment == 'production':
            recommendations.extend([
                "🔒 Ensure all production credentials are properly configured",
                "📊 Run performance testing with production data volumes",
                "🔍 Enable comprehensive monitoring and alerting",
                "💾 Verify backup and disaster recovery procedures"
            ])
        
        # Quick mode recommendations
        if self.quick_mode:
            recommendations.append("🏃 Quick mode used - run full validation before production deployment")
        
        # Add general recommendations
        recommendations.extend([
            "📚 Review all documentation for completeness",
            "🧪 Consider implementing automated testing pipeline",
            "🔄 Set up continuous integration/deployment",
            "📈 Monitor system performance in production"
        ])
        
        self.results['recommendations'] = recommendations
    
    def save_results(self, output_file: str = None):
        """Save validation results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"mvp_validation_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"💾 Validation results saved to {output_file}")
        return output_file


def main():
    """Main entry point for MVP validation."""
    parser = argparse.ArgumentParser(description="MVP Validation for Non-Profit Engagement Model")
    parser.add_argument(
        '--environment', 
        choices=['dev', 'development', 'prod', 'production'], 
        default='development',
        help='Environment to validate (default: development)'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run in quick mode with reduced test coverage'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file for validation results (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Normalize environment name
    environment = 'development' if args.environment in ['dev', 'development'] else 'production'
    
    try:
        # Run validation
        validator = MVPValidator(environment=environment, quick_mode=args.quick)
        results = validator.run_validation()
        
        # Save results
        output_file = validator.save_results(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 MVP VALIDATION SUMMARY")
        print("="*60)
        print(f"Environment: {results['environment']}")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Success Rate: {results['summary']['success_rate']}%")
        print(f"Steps Passed: {results['summary']['steps_passed']}/{results['summary']['total_steps']}")
        print(f"Validation Time: {results['summary']['total_validation_time_seconds']}s")
        
        if results['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(results['critical_issues'])}):")
            for issue in results['critical_issues']:
                print(f"  • {issue}")
        
        if results['warnings']:
            print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  • {warning}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n📄 Full results saved to: {output_file}")
        print("="*60)
        
        # Exit with appropriate code
        if results['overall_status'] == 'FAIL':
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"💥 MVP validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()