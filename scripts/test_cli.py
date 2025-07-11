#!/usr/bin/env python3
"""
Test script for the BG/NBD model CLI interface.

This script tests the basic functionality of the run_model.py CLI
without requiring actual database connections or data.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def test_help_commands():
    """Test help functionality for all commands."""
    print("Testing help commands...")
    
    commands = [
        "python scripts/run_model.py --help",
        "python scripts/run_model.py extract --help",
        "python scripts/run_model.py preprocess --help", 
        "python scripts/run_model.py train --help",
        "python scripts/run_model.py predict --help",
        "python scripts/run_model.py visualize --help",
        "python scripts/run_model.py full-pipeline --help"
    ]
    
    for cmd in commands:
        print(f"  Testing: {cmd}")
        returncode, stdout, stderr = run_command(cmd)
        
        if returncode == 0:
            print(f"    ✅ Success")
        else:
            print(f"    ❌ Failed (return code: {returncode})")
            if stderr:
                print(f"    Error: {stderr[:200]}...")
    
    print()

def test_argument_parsing():
    """Test argument parsing without execution."""
    print("Testing argument parsing...")
    
    # Test invalid commands
    invalid_commands = [
        "python scripts/run_model.py invalid-command",
        "python scripts/run_model.py extract",  # Missing required args
        "python scripts/run_model.py train",   # Missing required args
    ]
    
    for cmd in invalid_commands:
        print(f"  Testing invalid: {cmd}")
        returncode, stdout, stderr = run_command(cmd)
        
        if returncode != 0:
            print(f"    ✅ Correctly rejected")
        else:
            print(f"    ❌ Should have failed but didn't")
    
    print()

def test_import_functionality():
    """Test that all imports work correctly."""
    print("Testing import functionality...")
    
    test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.run_model import BGNBDModelRunner, create_parser, parse_date
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test parser creation
try:
    parser = create_parser()
    print("✅ Parser creation successful")
except Exception as e:
    print(f"❌ Parser creation failed: {e}")
    sys.exit(1)

# Test date parsing
try:
    from datetime import datetime
    date = parse_date("2023-01-01")
    assert date == datetime(2023, 1, 1)
    print("✅ Date parsing successful")
except Exception as e:
    print(f"❌ Date parsing failed: {e}")
    sys.exit(1)

print("All tests passed!")
"""
    
    # Write test script to temporary file
    test_file = Path("test_imports.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    try:
        returncode, stdout, stderr = run_command(f"python {test_file}")
        print(stdout)
        if stderr:
            print(f"Warnings: {stderr}")
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
    
    print()

def test_configuration_validation():
    """Test configuration validation without database."""
    print("Testing configuration validation...")
    
    # Create a minimal test configuration
    test_env = """
AZURE_SQL_SERVER=test-server.database.windows.net
AZURE_SQL_DATABASE=test-database
AZURE_SQL_USERNAME=test-user
AZURE_SQL_PASSWORD=test-password
ENVIRONMENT=development
DEV_MODE=True
"""
    
    env_file = Path(".env.test")
    with open(env_file, 'w') as f:
        f.write(test_env)
    
    try:
        # Test configuration loading (this will fail at database connection, which is expected)
        cmd = f"python scripts/run_model.py --config {env_file} extract --start-date 2023-01-01 --end-date 2023-01-02"
        returncode, stdout, stderr = run_command(cmd)
        
        # We expect this to fail at database connection, not at config loading
        if "Configuration validation failed" in stderr:
            print("    ❌ Configuration validation failed")
        elif "Database connection test failed" in stderr or "Environment validation failed" in stderr:
            print("    ✅ Configuration loaded successfully (failed at database connection as expected)")
        else:
            print(f"    ⚠️  Unexpected result: {stderr[:200]}...")
    
    finally:
        # Clean up
        if env_file.exists():
            env_file.unlink()
    
    print()

def main():
    """Run all tests."""
    print("🧪 Testing BG/NBD Model CLI Interface")
    print("=" * 50)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    test_help_commands()
    test_argument_parsing()
    test_import_functionality()
    test_configuration_validation()
    
    print("🎉 CLI testing completed!")
    print("\nNote: Full functionality testing requires:")
    print("  - Valid database connection")
    print("  - Actual data for processing")
    print("  - Complete environment setup")
    print("\nFor full testing, use the examples in scripts/README.md")

if __name__ == "__main__":
    main()