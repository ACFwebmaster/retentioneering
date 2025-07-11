"""
Configuration module for the non-profit engagement model.

This module handles environment variables, database connections,
and model configuration settings.
"""

from .settings import (
    AppConfig,
    DatabaseConfig,
    ModelConfig,
    LoggingConfig,
    get_config,
    reload_config,
)

from .database import (
    DatabaseManager,
    DatabaseHealthCheck,
    get_database_manager,
    get_session,
    session_scope,
    test_database_connection,
    get_database_health,
    close_database_connections,
)

# Convenience imports for common usage
__all__ = [
    # Settings classes
    "AppConfig",
    "DatabaseConfig", 
    "ModelConfig",
    "LoggingConfig",
    # Settings functions
    "get_config",
    "reload_config",
    # Database classes
    "DatabaseManager",
    "DatabaseHealthCheck",
    # Database functions
    "get_database_manager",
    "get_session",
    "session_scope",
    "test_database_connection",
    "get_database_health",
    "close_database_connections",
]

# Initialize configuration on import
config = get_config()