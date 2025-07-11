"""
Test script for configuration and database modules.

This script tests the configuration loading and database connection
functionality without requiring actual database credentials.
"""

import logging
import os
import tempfile
from pathlib import Path

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from config.settings import AppConfig, get_config, reload_config
from config.database import get_database_manager, DatabaseHealthCheck


def test_configuration_loading():
    """Test configuration loading and validation."""
    print("Testing configuration loading...")
    
    # Create a temporary .env file for testing
    test_env_content = """
# Test environment variables
AZURE_SQL_SERVER=test-server.database.windows.net
AZURE_SQL_DATABASE=test-database
AZURE_SQL_USERNAME=test-user
AZURE_SQL_PASSWORD=test-password
LOG_LEVEL=DEBUG
ENVIRONMENT=development
DEBUG=True
DEV_MODE=True
PREDICTION_PERIOD_DAYS=180
ENGAGEMENT_THRESHOLD=0.7
"""
    
    # Save current environment
    original_env = {}
    test_vars = [
        'AZURE_SQL_SERVER', 'AZURE_SQL_DATABASE', 'AZURE_SQL_USERNAME', 
        'AZURE_SQL_PASSWORD', 'LOG_LEVEL', 'ENVIRONMENT', 'DEBUG', 'DEV_MODE',
        'PREDICTION_PERIOD_DAYS', 'ENGAGEMENT_THRESHOLD'
    ]
    
    for var in test_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
    
    try:
        # Set test environment variables
        os.environ.update({
            'AZURE_SQL_SERVER': 'test-server.database.windows.net',
            'AZURE_SQL_DATABASE': 'test-database',
            'AZURE_SQL_USERNAME': 'test-user',
            'AZURE_SQL_PASSWORD': 'test-password',
            'LOG_LEVEL': 'DEBUG',
            'ENVIRONMENT': 'development',
            'DEBUG': 'True',
            'DEV_MODE': 'True',
            'PREDICTION_PERIOD_DAYS': '180',
            'ENGAGEMENT_THRESHOLD': '0.7'
        })
        
        # Test configuration loading
        config = AppConfig()
        
        # Test database configuration
        assert config.database.server == 'test-server.database.windows.net'
        assert config.database.database == 'test-database'
        assert config.database.username == 'test-user'
        assert config.database.password == 'test-password'
        assert config.database.pool_size == 5  # default value
        
        # Test model configuration
        assert config.model.prediction_period_days == 180
        assert config.model.engagement_threshold == 0.7
        
        # Test application configuration
        assert config.debug == True
        assert config.environment == 'development'
        assert config.dev_mode == True
        assert config.is_development == True
        assert config.is_production == False
        
        # Test configuration validation
        assert config.validate_config() == True
        
        print("✓ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False
        
    finally:
        # Restore original environment
        for var in test_vars:
            if var in original_env:
                os.environ[var] = original_env[var]
            elif var in os.environ:
                del os.environ[var]


def test_database_configuration():
    """Test database configuration without actual connection."""
    print("Testing database configuration...")
    
    try:
        # Set test environment variables
        os.environ.update({
            'AZURE_SQL_SERVER': 'test-server.database.windows.net',
            'AZURE_SQL_DATABASE': 'test-database',
            'AZURE_SQL_USERNAME': 'test-user',
            'AZURE_SQL_PASSWORD': 'test-password@123',
            'DB_POOL_SIZE': '10',
            'DB_MAX_OVERFLOW': '20',
            'DB_CONNECTION_TIMEOUT': '60'
        })
        
        # Test database manager initialization (without actual connection)
        config = AppConfig()
        
        # Test connection string building (we can't test actual connection without credentials)
        db_config = config.database
        assert db_config.server == 'test-server.database.windows.net'
        assert db_config.database == 'test-database'
        assert db_config.username == 'test-user'
        assert db_config.password == 'test-password@123'
        assert db_config.pool_size == 10
        assert db_config.max_overflow == 20
        assert db_config.connection_timeout == 60
        
        print("✓ Database configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Database configuration test failed: {e}")
        return False


def test_missing_required_config():
    """Test behavior when required configuration is missing."""
    print("Testing missing required configuration...")
    
    # Save current environment
    original_server = os.environ.get('AZURE_SQL_SERVER')
    
    try:
        # Remove required environment variable
        if 'AZURE_SQL_SERVER' in os.environ:
            del os.environ['AZURE_SQL_SERVER']
        
        # This should raise a ValueError
        try:
            config = AppConfig()
            print("✗ Missing required config test failed: Should have raised ValueError")
            return False
        except ValueError as e:
            if "AZURE_SQL_SERVER" in str(e):
                print("✓ Missing required config test passed")
                return True
            else:
                print(f"✗ Missing required config test failed: Wrong error message: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Missing required config test failed: {e}")
        return False
        
    finally:
        # Restore original environment
        if original_server:
            os.environ['AZURE_SQL_SERVER'] = original_server


def run_all_tests():
    """Run all configuration tests."""
    print("Running configuration and database tests...\n")
    
    tests = [
        test_configuration_loading,
        test_database_configuration,
        test_missing_required_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    # Configure basic logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)