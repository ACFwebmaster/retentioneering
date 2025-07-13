"""
Test script for data extraction and preprocessing modules.

This script tests the integration with config modules and validates
the functionality of the data processing pipeline.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from ..config import get_config, test_database_connection, get_database_health
from . import (
    DataExtractor,
    BGNBDDataProcessor,
    create_data_extractor,
    create_bgnbd_processor,
    DataExtractionError,
    DataPreprocessingError,
)


def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_config_integration():
    """Test integration with config modules."""
    print("\n=== Testing Config Integration ===")
    
    try:
        # Test config loading
        config = get_config()
        print(f"✅ Config loaded successfully")
        print(f"   Environment: {config.environment}")
        print(f"   Debug mode: {config.debug}")
        print(f"   Development mode: {config.dev_mode}")
        
        # Test database config
        db_config = config.database
        print(f"✅ Database config loaded")
        print(f"   Server: {db_config.server}")
        print(f"   Database: {db_config.database}")
        print(f"   Pool size: {db_config.pool_size}")
        
        # Test model config
        model_config = config.model
        print(f"✅ Model config loaded")
        print(f"   Data output dir: {model_config.data_output_dir}")
        print(f"   Prediction period: {model_config.prediction_period_days} days")
        
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False


def test_database_connection():
    """Test database connection and health."""
    print("\n=== Testing Database Connection ===")
    
    try:
        # Test basic connection
        connection_ok = test_database_connection()
        if connection_ok:
            print("✅ Database connection test passed")
        else:
            print("❌ Database connection test failed")
            return False
        
        # Test database health
        health_status = get_database_health()
        print(f"✅ Database health check completed")
        print(f"   Status: {health_status['status']}")
        print(f"   Connection test: {health_status['connection_test']}")
        
        if health_status['pool_status']:
            pool = health_status['pool_status']
            print(f"   Pool size: {pool['size']}")
            print(f"   Checked out: {pool['checked_out']}")
        
        return health_status['status'] in ['healthy', 'unknown']
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False


def test_data_extractor_creation():
    """Test DataExtractor creation and initialization."""
    print("\n=== Testing DataExtractor Creation ===")
    
    try:
        # Test factory function
        extractor = create_data_extractor()
        print("✅ DataExtractor created via factory function")
        
        # Test direct instantiation
        extractor2 = DataExtractor()
        print("✅ DataExtractor created via direct instantiation")
        
        # Test configuration access
        config = extractor.config
        print(f"✅ Config accessible from extractor")
        print(f"   Cache enabled: {extractor.cache_enabled}")
        print(f"   Raw data dir: {extractor.raw_data_dir}")
        
        # Test directory creation
        assert extractor.raw_data_dir.exists(), "Raw data directory should exist"
        print("✅ Raw data directory created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ DataExtractor creation test failed: {e}")
        return False


def test_bgnbd_processor_creation():
    """Test BGNBDDataProcessor creation and initialization."""
    print("\n=== Testing BGNBDDataProcessor Creation ===")
    
    try:
        # Test factory function
        processor = create_bgnbd_processor()
        print("✅ BGNBDDataProcessor created via factory function")
        
        # Test with custom extractor
        extractor = create_data_extractor()
        processor2 = create_bgnbd_processor(extractor)
        print("✅ BGNBDDataProcessor created with custom extractor")
        
        # Test configuration access
        config = processor.config
        print(f"✅ Config accessible from processor")
        print(f"   Cache enabled: {processor.cache_enabled}")
        print(f"   Processed data dir: {processor.processed_data_dir}")
        
        # Test engagement weights
        weights = processor.engagement_weights
        print(f"✅ Engagement weights configured")
        print(f"   Donation weight: {weights['donation']}")
        print(f"   Volunteer weight: {weights['volunteer']}")
        
        # Test directory creation
        assert processor.processed_data_dir.exists(), "Processed data directory should exist"
        print("✅ Processed data directory created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ BGNBDDataProcessor creation test failed: {e}")
        return False


def test_query_building():
    """Test SQL query building functionality."""
    print("\n=== Testing Query Building ===")
    
    try:
        extractor = create_data_extractor()
        
        # Test actions query building
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        supporter_ids = [1, 2, 3]
        action_types = ['email_open', 'donation']
        
        query, params = extractor._build_actions_query(
            start_date, end_date, supporter_ids, action_types
        )
        
        print("✅ Actions query built successfully")
        print(f"   Parameters: {len(params)} items")
        assert 'start_date' in params
        assert 'end_date' in params
        assert 'id_0' in params
        assert 'type_0' in params
        
        # Test donations query building
        query2, params2 = extractor._build_donations_query(
            start_date, end_date, supporter_ids, 10.0
        )
        
        print("✅ Donations query built successfully")
        print(f"   Parameters: {len(params2)} items")
        assert 'min_amount' in params2
        
        # Test summary query building
        query3, params3 = extractor._build_summary_query(start_date, end_date)
        
        print("✅ Summary query built successfully")
        print(f"   Parameters: {len(params3)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Query building test failed: {e}")
        return False


def test_data_validation():
    """Test data validation functionality."""
    print("\n=== Testing Data Validation ===")
    
    try:
        extractor = create_data_extractor()
        
        # Test valid actions data
        valid_actions = pd.DataFrame({
            'supporter_id': [1, 2, 3],
            'action_date': [datetime.now(), datetime.now(), datetime.now()],
            'action_type': ['email_open', 'donation', 'volunteer'],
            'tags': ['tag1', 'tag2', '']
        })
        
        extractor._validate_actions_data(valid_actions)
        print("✅ Valid actions data validation passed")
        
        # Test valid donations data
        valid_donations = pd.DataFrame({
            'supporter_id': [1, 2, 3],
            'donation_date': [datetime.now(), datetime.now(), datetime.now()],
            'amount': [25.0, 50.0, 100.0]
        })
        
        extractor._validate_donations_data(valid_donations)
        print("✅ Valid donations data validation passed")
        
        # Test invalid data (should raise exception)
        try:
            invalid_actions = pd.DataFrame({
                'supporter_id': [1, None, 3],  # Null supporter_id
                'action_date': [datetime.now(), datetime.now(), datetime.now()],
                'action_type': ['email_open', 'donation', 'volunteer'],
                'tags': ['tag1', 'tag2', '']
            })
            extractor._validate_actions_data(invalid_actions)
            print("❌ Invalid data validation should have failed")
            return False
        except DataExtractionError:
            print("✅ Invalid data validation correctly failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False


def test_cache_functionality():
    """Test caching functionality."""
    print("\n=== Testing Cache Functionality ===")
    
    try:
        extractor = create_data_extractor()
        processor = create_bgnbd_processor()
        
        # Test cache key generation
        cache_key = extractor._generate_cache_key(
            "test", datetime(2023, 1, 1), [1, 2, 3], "param"
        )
        print(f"✅ Cache key generated: {cache_key}")
        
        # Test cache save/load with sample data
        test_data = pd.DataFrame({
            'supporter_id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        # Save to cache
        extractor._save_to_cache(test_data, "test_key")
        print("✅ Data saved to cache")
        
        # Load from cache
        loaded_data = extractor._load_from_cache("test_key")
        if loaded_data is not None and len(loaded_data) == len(test_data):
            print("✅ Data loaded from cache successfully")
        else:
            print("❌ Cache load failed or returned incorrect data")
            return False
        
        # Test cache clearing
        extractor.clear_cache("test_key")
        print("✅ Cache cleared successfully")
        
        # Verify cache was cleared
        cleared_data = extractor._load_from_cache("test_key")
        if cleared_data is None:
            print("✅ Cache clearing verified")
        else:
            print("❌ Cache was not properly cleared")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Cache functionality test failed: {e}")
        return False


def test_bgnbd_calculations():
    """Test BG/NBD variable calculations."""
    print("\n=== Testing BG/NBD Calculations ===")
    
    try:
        processor = create_bgnbd_processor()
        
        # Create sample events data
        events_data = pd.DataFrame({
            'supporter_id': [1, 1, 1, 2, 2, 3],
            'event_date': [
                datetime(2023, 1, 15),
                datetime(2023, 3, 15),
                datetime(2023, 6, 15),
                datetime(2023, 2, 15),
                datetime(2023, 4, 15),
                datetime(2023, 1, 15)
            ],
            'event_type': ['email_open', 'donation', 'volunteer', 'email_open', 'donation', 'email_open'],
            'event_value': [1.0, 2.0, 1.5, 1.0, 2.0, 1.0],
            'weighted_value': [1.0, 2.0, 1.5, 1.0, 2.0, 1.0],
            'is_donation': [False, True, False, False, True, False],
            'amount': [0, 50, 0, 0, 25, 0],
            'tags': ['', '', '', '', '', '']
        })
        
        # Test BG/NBD calculation
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        cutoff_date = datetime(2023, 6, 30)
        
        bgnbd_df = processor._calculate_bgnbd_variables(
            events_data, start_date, end_date, cutoff_date, min_actions=1
        )
        
        print(f"✅ BG/NBD variables calculated for {len(bgnbd_df)} supporters")
        
        # Verify calculations
        if not bgnbd_df.empty:
            supporter_1 = bgnbd_df[bgnbd_df['supporter_id'] == 1].iloc[0]
            print(f"   Supporter 1: x={supporter_1['x']}, t_x={supporter_1['t_x']}, T={supporter_1['T']}")
            
            # Supporter 1 should have x=2 (3 events - 1), t_x should be days from start to last event
            assert supporter_1['x'] == 2, f"Expected x=2, got {supporter_1['x']}"
            assert supporter_1['T'] == 364, f"Expected T=364, got {supporter_1['T']}"  # 2023 is not leap year
            
            print("✅ BG/NBD calculations verified")
        
        return True
        
    except Exception as e:
        print(f"❌ BG/NBD calculations test failed: {e}")
        return False


def test_engagement_scoring():
    """Test engagement scoring functionality."""
    print("\n=== Testing Engagement Scoring ===")
    
    try:
        processor = create_bgnbd_processor()
        
        # Create sample BG/NBD data
        bgnbd_data = pd.DataFrame({
            'supporter_id': [1, 2, 3, 4],
            'x': [5, 2, 0, 8],
            't_x': [100, 50, 0, 200],
            'T': [365, 365, 365, 365],
            'frequency': [6, 3, 1, 9],
            'total_donation_amount': [100, 0, 0, 500],
            'recency_ratio': [100/365, 50/365, 0, 200/365]
        })
        
        # Test engagement score calculation
        engagement_scores = processor._calculate_engagement_score(bgnbd_data)
        print(f"✅ Engagement scores calculated: {len(engagement_scores)} scores")
        
        # Test engagement segment assignment
        bgnbd_data['engagement_score'] = engagement_scores
        segments = processor._assign_engagement_segment(bgnbd_data)
        print(f"✅ Engagement segments assigned")
        
        segment_counts = segments.value_counts()
        print(f"   Segment distribution: {dict(segment_counts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Engagement scoring test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    try:
        processor = create_bgnbd_processor()
        
        # Test with invalid date parameters
        try:
            start_date = datetime(2023, 6, 1)
            end_date = datetime(2023, 1, 1)  # End before start
            processor._validate_date_parameters(start_date, end_date, end_date)
            print("❌ Date validation should have failed")
            return False
        except DataPreprocessingError:
            print("✅ Invalid date parameters correctly rejected")
        
        # Test with empty data
        empty_df = processor._create_empty_bgnbd_dataframe()
        print(f"✅ Empty BG/NBD DataFrame created with {len(empty_df.columns)} columns")
        
        # Test summary with empty data
        summary = processor.generate_summary_statistics(empty_df)
        assert 'error' in summary, "Summary should indicate error for empty data"
        print("✅ Empty data summary handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all integration and functionality tests."""
    print("=== Non-Profit Engagement Model - Data Module Tests ===")
    
    setup_test_logging()
    
    tests = [
        ("Config Integration", test_config_integration),
        ("Database Connection", test_database_connection),
        ("DataExtractor Creation", test_data_extractor_creation),
        ("BGNBDDataProcessor Creation", test_bgnbd_processor_creation),
        ("Query Building", test_query_building),
        ("Data Validation", test_data_validation),
        ("Cache Functionality", test_cache_functionality),
        ("BG/NBD Calculations", test_bgnbd_calculations),
        ("Engagement Scoring", test_engagement_scoring),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n=== Test Results Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data modules are ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


def main():
    """Main test function."""
    success = run_all_tests()
    
    if success:
        print("\n=== Next Steps ===")
        print("1. Run the example usage script: python -m src.data.example_usage")
        print("2. Set up your .env file with Azure SQL credentials")
        print("3. Test with real data extraction and processing")
        print("4. Use the processed data for BG/NBD model training")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())