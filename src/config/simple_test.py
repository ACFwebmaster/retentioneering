"""
Simple test script for configuration module only.

This script tests just the settings module without database dependencies.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

def test_settings_only():
    """Test settings module without database dependencies."""
    print("Testing settings module...")
    
    # Set test environment variables
    test_env = {
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
    }
    
    # Save original environment
    original_env = {}
    for key in test_env:
        if key in os.environ:
            original_env[key] = os.environ[key]
    
    try:
        # Set test environment
        os.environ.update(test_env)
        
        # Import and test settings
        from config.settings import AppConfig, DatabaseConfig, ModelConfig, LoggingConfig
        
        # Test DatabaseConfig
        db_config = DatabaseConfig()
        assert db_config.server == 'test-server.database.windows.net'
        assert db_config.database == 'test-database'
        assert db_config.username == 'test-user'
        assert db_config.password == 'test-password'
        print("✓ DatabaseConfig test passed")
        
        # Test ModelConfig
        model_config = ModelConfig()
        assert model_config.prediction_period_days == 180
        assert model_config.engagement_threshold == 0.7
        print("✓ ModelConfig test passed")
        
        # Test LoggingConfig
        logging_config = LoggingConfig()
        assert logging_config.level == 'DEBUG'
        print("✓ LoggingConfig test passed")
        
        # Test AppConfig
        app_config = AppConfig()
        assert app_config.debug == True
        assert app_config.environment == 'development'
        assert app_config.is_development == True
        assert app_config.is_production == False
        print("✓ AppConfig test passed")
        
        # Test validation
        assert app_config.validate_config() == True
        print("✓ Configuration validation test passed")
        
        print("\n🎉 All settings tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original environment
        for key in test_env:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]


def test_missing_config():
    """Test behavior with missing required configuration."""
    print("\nTesting missing required configuration...")
    
    # Save original environment
    original_server = os.environ.get('AZURE_SQL_SERVER')
    
    try:
        # Remove required environment variable
        if 'AZURE_SQL_SERVER' in os.environ:
            del os.environ['AZURE_SQL_SERVER']
        
        from config.settings import DatabaseConfig
        
        # This should raise a ValueError
        try:
            db_config = DatabaseConfig()
            print("✗ Missing config test failed: Should have raised ValueError")
            return False
        except ValueError as e:
            if "AZURE_SQL_SERVER" in str(e):
                print("✓ Missing required config test passed")
                return True
            else:
                print(f"✗ Wrong error message: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Missing config test failed: {e}")
        return False
        
    finally:
        # Restore original environment
        if original_server:
            os.environ['AZURE_SQL_SERVER'] = original_server


if __name__ == "__main__":
    print("Running simple configuration tests...\n")
    
    success1 = test_settings_only()
    success2 = test_missing_config()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)