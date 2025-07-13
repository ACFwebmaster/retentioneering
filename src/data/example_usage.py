"""
Example usage of the data extraction and preprocessing modules.

This script demonstrates how to use the DataExtractor and BGNBDDataProcessor
classes to extract supporter data from Azure SQL and process it into BG/NBD format.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from ..config import get_config, test_database_connection
from . import (
    create_data_extractor,
    create_bgnbd_processor,
    process_supporter_data_pipeline,
    DataExtractionError,
    DataPreprocessingError,
)


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_extraction():
    """Example of basic data extraction."""
    print("\n=== Basic Data Extraction Example ===")
    
    try:
        # Create data extractor
        extractor = create_data_extractor()
        
        # Define date range (last 6 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        print(f"Extracting data from {start_date.date()} to {end_date.date()}")
        
        # Extract supporter actions
        print("Extracting supporter actions...")
        actions_df = extractor.extract_supporter_actions(
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        print(f"Extracted {len(actions_df)} supporter actions")
        
        # Extract donations
        print("Extracting donations...")
        donations_df = extractor.extract_donations(
            start_date=start_date,
            end_date=end_date,
            min_amount=1.0,  # Minimum $1 donation
            use_cache=True
        )
        print(f"Extracted {len(donations_df)} donations")
        
        # Generate data quality report
        print("Generating data quality report...")
        quality_report = extractor.get_data_quality_report(actions_df, donations_df)
        
        print("\nData Quality Summary:")
        print(f"  Actions: {quality_report['actions']['total_records']} records")
        print(f"  Unique supporters (actions): {quality_report['actions']['unique_supporters']}")
        print(f"  Donations: {quality_report['donations']['total_records']} records")
        print(f"  Unique supporters (donations): {quality_report['donations']['unique_supporters']}")
        print(f"  Total donation amount: ${quality_report['donations']['amount_stats']['total']:,.2f}")
        
        return actions_df, donations_df
        
    except DataExtractionError as e:
        print(f"Data extraction failed: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during extraction: {e}")
        return None, None


def example_bgnbd_processing():
    """Example of BG/NBD data processing."""
    print("\n=== BG/NBD Data Processing Example ===")
    
    try:
        # Define analysis period (last year)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        cutoff_date = end_date - timedelta(days=30)  # 30 days ago for recency calculation
        
        print(f"Processing BG/NBD data:")
        print(f"  Observation period: {start_date.date()} to {end_date.date()}")
        print(f"  Cutoff date: {cutoff_date.date()}")
        
        # Use the complete pipeline
        bgnbd_df, summary_stats = process_supporter_data_pipeline(
            start_date=start_date,
            end_date=end_date,
            cutoff_date=cutoff_date,
            min_actions=2,  # Require at least 2 actions
            include_donations=True,
            save_results=True
        )
        
        if bgnbd_df.empty:
            print("No data processed - check your database connection and data availability")
            return None
        
        print(f"\nProcessed {len(bgnbd_df)} supporters into BG/NBD format")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(f"  Observation period: {summary_stats['observation_period']['total_days']} days")
        print(f"  Average frequency: {summary_stats['frequency_stats']['mean']:.2f}")
        print(f"  Median frequency: {summary_stats['frequency_stats']['median']:.2f}")
        print(f"  Average recency: {summary_stats['recency_stats']['mean']:.1f} days")
        print(f"  Total donations: ${summary_stats['monetary_stats']['total_donations']:,.2f}")
        print(f"  Donor percentage: {summary_stats['monetary_stats']['donor_percentage']:.1f}%")
        
        print("\nEngagement Segments:")
        for segment, count in summary_stats['engagement_segments'].items():
            print(f"  {segment}: {count} supporters")
        
        print(f"\nRepeat rate: {summary_stats['data_quality']['repeat_rate']:.1f}%")
        print(f"Average event diversity: {summary_stats['data_quality']['avg_event_diversity']:.2f}")
        
        # Display sample of processed data
        print("\nSample of processed data:")
        sample_cols = ['supporter_id', 'x', 't_x', 'T', 'frequency', 'monetary', 'engagement_segment']
        print(bgnbd_df[sample_cols].head(10).to_string(index=False))
        
        return bgnbd_df, summary_stats
        
    except DataPreprocessingError as e:
        print(f"Data preprocessing failed: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
        return None, None


def example_advanced_processing():
    """Example of advanced data processing with custom parameters."""
    print("\n=== Advanced Processing Example ===")
    
    try:
        # Create processor with custom settings
        processor = create_bgnbd_processor()
        
        # Custom engagement weights
        processor.engagement_weights.update({
            'donation': 3.0,  # Higher weight for donations
            'volunteer': 2.0,
            'event_attendance': 1.5,
            'email_click': 0.8,
            'website_visit': 0.3
        })
        
        # Define analysis parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        cutoff_date = end_date - timedelta(days=90)  # 3 months ago
        
        print(f"Advanced processing with custom weights:")
        print(f"  Observation period: {start_date.date()} to {end_date.date()}")
        print(f"  Analysis cutoff: {cutoff_date.date()}")
        
        # Process with higher minimum actions threshold
        bgnbd_df = processor.process_supporter_data(
            start_date=start_date,
            end_date=end_date,
            cutoff_date=cutoff_date,
            min_actions=3,  # Require at least 3 actions
            include_donations=True,
            use_cache=True
        )
        
        if bgnbd_df.empty:
            print("No data processed with advanced criteria")
            return None
        
        print(f"Processed {len(bgnbd_df)} supporters with advanced criteria")
        
        # Generate and save summary
        summary = processor.generate_summary_statistics(bgnbd_df)
        
        # Save processed data
        output_path = processor.save_processed_data(
            bgnbd_df, 
            filename="advanced_bgnbd_analysis.csv"
        )
        print(f"Saved advanced analysis to: {output_path}")
        
        # Display high-value supporters
        high_value = bgnbd_df[
            (bgnbd_df['engagement_segment'] == 'High') & 
            (bgnbd_df['total_donation_amount'] > 0)
        ].sort_values('total_donation_amount', ascending=False)
        
        if not high_value.empty:
            print(f"\nTop 5 high-value supporters:")
            top_cols = ['supporter_id', 'frequency', 'total_donation_amount', 'engagement_score']
            print(high_value[top_cols].head().to_string(index=False))
        
        return bgnbd_df
        
    except Exception as e:
        print(f"Advanced processing failed: {e}")
        return None


def example_data_validation():
    """Example of data validation and quality checks."""
    print("\n=== Data Validation Example ===")
    
    try:
        extractor = create_data_extractor()
        
        # Test with specific supporter IDs
        test_supporter_ids = [1, 2, 3, 4, 5]  # Replace with actual IDs
        
        print(f"Testing extraction for specific supporters: {test_supporter_ids}")
        
        # Extract data for specific supporters
        actions_df = extractor.extract_supporter_actions(
            supporter_ids=test_supporter_ids,
            use_cache=False  # Force fresh extraction
        )
        
        donations_df = extractor.extract_donations(
            supporter_ids=test_supporter_ids,
            use_cache=False
        )
        
        print(f"Extracted {len(actions_df)} actions and {len(donations_df)} donations")
        
        if not actions_df.empty:
            print("\nAction types found:")
            print(actions_df['action_type'].value_counts().to_string())
        
        if not donations_df.empty:
            print(f"\nDonation summary:")
            print(f"  Total amount: ${donations_df['amount'].sum():,.2f}")
            print(f"  Average donation: ${donations_df['amount'].mean():,.2f}")
            print(f"  Date range: {donations_df['donation_date'].min().date()} to {donations_df['donation_date'].max().date()}")
        
        return True
        
    except Exception as e:
        print(f"Data validation failed: {e}")
        return False


def main():
    """Main example function."""
    setup_logging()
    
    print("=== Non-Profit Engagement Model - Data Processing Examples ===")
    
    # Check configuration
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug}")
    print(f"Development mode: {config.dev_mode}")
    
    # Test database connection
    print("\nTesting database connection...")
    if not test_database_connection():
        print("❌ Database connection failed. Please check your configuration.")
        print("Make sure your .env file is properly configured with Azure SQL credentials.")
        return
    
    print("✅ Database connection successful")
    
    # Run examples
    try:
        # Basic extraction example
        actions_df, donations_df = example_basic_extraction()
        
        if actions_df is not None and donations_df is not None:
            # BG/NBD processing example
            bgnbd_df, summary = example_bgnbd_processing()
            
            if bgnbd_df is not None:
                # Advanced processing example
                advanced_df = example_advanced_processing()
        
        # Data validation example
        example_data_validation()
        
        print("\n=== Examples completed successfully! ===")
        print("\nNext steps:")
        print("1. Check the data/processed/ directory for output files")
        print("2. Review the generated summary statistics")
        print("3. Use the processed BG/NBD data for model training")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        logging.exception("Detailed error information:")


if __name__ == "__main__":
    main()