"""
Settings module for the non-profit engagement model.

This module handles loading and validation of environment variables,
and provides configuration classes for different aspects of the application.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class DatabaseConfig:
    """Database configuration settings."""
    
    def __init__(self):
        self.server = self._get_required_env("AZURE_SQL_SERVER")
        self.database = self._get_required_env("AZURE_SQL_DATABASE")
        self.username = self._get_required_env("AZURE_SQL_USERNAME")
        self.password = self._get_required_env("AZURE_SQL_PASSWORD")
        self.driver = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
        
        # SSL/TLS Configuration
        self.encrypt = os.getenv("AZURE_SQL_ENCRYPT", "yes").lower()
        self.trust_server_certificate = os.getenv("AZURE_SQL_TRUST_SERVER_CERTIFICATE", "no").lower()
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Connection timeout settings
        self.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
        self.command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", "30"))
        
        # Retry settings
        self.max_retries = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("DB_RETRY_DELAY", "1.0"))
    
    @staticmethod
    def _get_required_env(key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value


class ModelConfig:
    """BG/NBD model configuration settings."""
    
    def __init__(self):
        self.prediction_period_days = int(os.getenv("PREDICTION_PERIOD_DAYS", "365"))
        self.engagement_threshold = float(os.getenv("ENGAGEMENT_THRESHOLD", "0.5"))
        self.sample_size = int(os.getenv("SAMPLE_SIZE", "10000"))
        
        # Model output paths
        self.model_output_dir = Path(os.getenv("MODEL_OUTPUT_DIR", "models/"))
        self.data_output_dir = Path(os.getenv("DATA_OUTPUT_DIR", "data/processed/"))
        self.visualization_output_dir = Path(
            os.getenv("VISUALIZATION_OUTPUT_DIR", "outputs/visualizations/")
        )
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create output directories if they don't exist."""
        for directory in [
            self.model_output_dir,
            self.data_output_dir,
            self.visualization_output_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class LoggingConfig:
    """Logging configuration settings."""
    
    def __init__(self):
        self.level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_file = os.getenv("LOG_FILE", "logs/app.log")
        self.format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Ensure log directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def configure_logging(self):
        """Configure application logging."""
        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format,
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )


class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        # Load environment variables
        self._load_environment()
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        
        # Application settings
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.dev_mode = os.getenv("DEV_MODE", "True").lower() == "true"
        
        # Configure logging
        self.logging.configure_logging()
        
        # Log configuration loaded
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded for environment: {self.environment}")
        logger.info(f"Debug mode: {self.debug}")
        logger.info(f"Development mode: {self.dev_mode}")
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        # Look for .env file in project root
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logging.getLogger(__name__).info(f"Loaded environment from {env_path}")
        else:
            logging.getLogger(__name__).info("No .env file found, using system environment variables")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def validate_config(self) -> bool:
        """Validate that all required configuration is present."""
        try:
            # Validate database config by accessing required properties
            _ = self.database.server
            _ = self.database.database
            _ = self.database.username
            _ = self.database.password
            
            logger = logging.getLogger(__name__)
            logger.info("Configuration validation successful")
            return True
            
        except ValueError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = AppConfig()
    return config


def reload_config() -> AppConfig:
    """Reload the configuration from environment variables."""
    global config
    config = AppConfig()
    return config