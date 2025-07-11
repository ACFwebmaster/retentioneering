"""
Example usage of the configuration and database modules.

This demonstrates how to use the configuration and database functionality
in your non-profit engagement model application.
"""

import logging
from pathlib import Path

# Import the configuration and database modules
from config import (
    get_config,
    get_database_manager,
    session_scope,
    test_database_connection,
    get_database_health,
)


def example_configuration_usage():
    """Example of how to use the configuration module."""
    print("=== Configuration Usage Example ===")
    
    # Get the global configuration
    config = get_config()
    
    # Access database configuration
    print(f"Database Server: {config.database.server}")
    print(f"Database Name: {config.database.database}")
    print(f"Connection Pool Size: {config.database.pool_size}")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    
    # Access model configuration
    print(f"Prediction Period: {config.model.prediction_period_days} days")
    print(f"Engagement Threshold: {config.model.engagement_threshold}")
    print(f"Model Output Directory: {config.model.model_output_dir}")
    
    # Check environment type
    if config.is_development:
        print("Running in development mode")
    elif config.is_production:
        print("Running in production mode")
    
    print()


def example_database_usage():
    """Example of how to use the database module."""
    print("=== Database Usage Example ===")
    
    try:
        # Get database manager
        db_manager = get_database_manager()
        
        # Get connection information (for debugging)
        conn_info = db_manager.get_connection_info()
        print(f"Connected to: {conn_info['server']}/{conn_info['database']}")
        
        # Test database connection
        print("Testing database connection...")
        if test_database_connection():
            print("✓ Database connection successful")
        else:
            print("✗ Database connection failed")
        
        # Get database health status
        health = get_database_health()
        print(f"Database Status: {health['status']}")
        if health['status'] == 'healthy':
            pool_status = health['pool_status']
            print(f"Connection Pool - Size: {pool_status['size']}, "
                  f"Active: {pool_status['checked_out']}, "
                  f"Available: {pool_status['checked_in']}")
        
        print()
        
    except Exception as e:
        print(f"Database connection error: {e}")
        print("This is expected if you haven't set up your .env file with valid credentials")
        print()


def example_database_session_usage():
    """Example of how to use database sessions."""
    print("=== Database Session Usage Example ===")
    
    try:
        # Example 1: Using session scope context manager
        with session_scope() as session:
            # Your database operations would go here
            # For example:
            # result = session.execute(text("SELECT COUNT(*) FROM supporters"))
            # count = result.scalar()
            # print(f"Total supporters: {count}")
            
            print("Session created successfully (no actual queries executed)")
            # Session will be automatically committed and closed
        
        # Example 2: Using database manager with retry logic
        db_manager = get_database_manager()
        
        def sample_database_operation():
            """Sample database operation that could be retried."""
            with session_scope() as session:
                # This would be your actual database operation
                # result = session.execute(text("SELECT 1"))
                # return result.scalar()
                return 1
        
        # Execute with automatic retry on failure
        result = db_manager.execute_with_retry(sample_database_operation)
        print(f"Database operation result: {result}")
        
        print()
        
    except Exception as e:
        print(f"Database session error: {e}")
        print("This is expected if you haven't set up your .env file with valid credentials")
        print()


def example_logging_usage():
    """Example of how logging is configured."""
    print("=== Logging Usage Example ===")
    
    # Get a logger for your module
    logger = logging.getLogger(__name__)
    
    # Log at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("Check your log file for the logged messages")
    print()


def main():
    """Main example function."""
    print("Non-Profit Engagement Model - Configuration & Database Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_configuration_usage()
    example_database_usage()
    example_database_session_usage()
    example_logging_usage()
    
    print("Examples completed!")
    print()
    print("Next steps:")
    print("1. Copy .env.example to .env and fill in your Azure SQL credentials")
    print("2. Install dependencies: poetry install")
    print("3. Test your configuration: python -c 'from config import test_database_connection; print(test_database_connection())'")


if __name__ == "__main__":
    main()