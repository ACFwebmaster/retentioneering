# Data Processing Modules

This directory contains the data extraction and preprocessing modules for the non-profit BG/NBD engagement model project.

## Overview

The data processing pipeline consists of two main components:

1. **Data Extraction** (`extraction.py`) - Extracts supporter actions and donations from Azure SQL Database
2. **Data Preprocessing** (`preprocessing.py`) - Transforms raw data into BG/NBD model format

## Modules

### DataExtractor (`extraction.py`)

Handles extraction of supporter data from Azure SQL Database with the following features:

- **Parameterized Queries**: Prevents SQL injection with safe parameter binding
- **Data Caching**: Caches raw extracts in `data/raw/` for development efficiency
- **Data Validation**: Validates extracted data quality and completeness
- **Date Range Filtering**: Supports flexible date-based filtering
- **Quality Reports**: Generates comprehensive data quality reports

#### Key Methods:
- `extract_supporter_actions()` - Extract supporter actions with filtering
- `extract_donations()` - Extract donation data with amount filtering
- `extract_supporter_summary()` - Extract supporter summary statistics
- `get_data_quality_report()` - Generate data quality metrics

### BGNBDDataProcessor (`preprocessing.py`)

Transforms supporter data into BG/NBD model format with the following features:

- **BG/NBD Variables**: Calculates x (frequency), t_x (recency), T (observation period)
- **Event Weighting**: Applies configurable weights to different engagement types
- **Engagement Scoring**: Calculates engagement scores and segments
- **Data Quality Filters**: Removes invalid or low-quality records
- **Summary Statistics**: Generates comprehensive processing statistics

#### Key Methods:
- `process_supporter_data()` - Main processing pipeline
- `generate_summary_statistics()` - Generate processing statistics
- `save_processed_data()` - Save results to CSV files

## BG/NBD Data Format

The processed data includes the following variables for each supporter:

- **x**: Number of repeat engagement events (frequency - 1)
- **t_x**: Time of last engagement event in days from start (recency)
- **T**: Total observation period length in days
- **frequency**: Total number of engagement events
- **monetary**: Total donation amount
- **engagement_score**: Calculated engagement score (0-1)
- **engagement_segment**: Engagement segment (High/Medium/Low/Inactive)

## Usage Examples

### Basic Data Extraction

```python
from src.data import create_data_extractor
from datetime import datetime, timedelta

# Create extractor
extractor = create_data_extractor()

# Define date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Extract supporter actions
actions_df = extractor.extract_supporter_actions(
    start_date=start_date,
    end_date=end_date,
    use_cache=True
)

# Extract donations
donations_df = extractor.extract_donations(
    start_date=start_date,
    end_date=end_date,
    min_amount=1.0
)

# Generate quality report
quality_report = extractor.get_data_quality_report(actions_df, donations_df)
```

### BG/NBD Data Processing

```python
from src.data import create_bgnbd_processor
from datetime import datetime, timedelta

# Create processor
processor = create_bgnbd_processor()

# Define analysis period
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
cutoff_date = end_date - timedelta(days=30)

# Process data
bgnbd_df = processor.process_supporter_data(
    start_date=start_date,
    end_date=end_date,
    cutoff_date=cutoff_date,
    min_actions=2,
    include_donations=True
)

# Generate summary statistics
summary = processor.generate_summary_statistics(bgnbd_df)

# Save results
output_path = processor.save_processed_data(bgnbd_df)
```

### Complete Pipeline

```python
from src.data import process_supporter_data_pipeline
from datetime import datetime, timedelta

# Define parameters
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Run complete pipeline
bgnbd_df, summary_stats = process_supporter_data_pipeline(
    start_date=start_date,
    end_date=end_date,
    min_actions=2,
    include_donations=True,
    save_results=True
)

print(f"Processed {len(bgnbd_df)} supporters")
print(f"Average frequency: {summary_stats['frequency_stats']['mean']:.2f}")
```

## Configuration

The modules integrate with the existing configuration system:

- **Database Connection**: Uses `DatabaseManager` from `src.config.database`
- **Application Settings**: Uses `AppConfig` from `src.config.settings`
- **Logging**: Follows established logging patterns
- **Caching**: Enabled in development mode (`config.dev_mode`)

## Data Schema Assumptions

The modules assume the following database schema:

### supporter_level_action_role_tags
- `supporter_id` (int) - Unique supporter identifier
- `action_date` (datetime) - Date of the action
- `action_type` (string) - Type of action performed
- `tags` (string) - Additional tags or metadata

### donations
- `supporter_id` (int) - Unique supporter identifier
- `donation_date` (datetime) - Date of the donation
- `amount` (float) - Donation amount

## Engagement Weights

Default engagement weights for different action types:

- **donation**: 2.0
- **volunteer**: 1.5
- **event_attendance**: 1.2
- **email_click**: 0.5
- **social_media**: 0.4
- **email_open**: 0.3
- **website_visit**: 0.2
- **default**: 1.0

These weights can be customized by modifying the `engagement_weights` dictionary in the processor.

## Caching Strategy

### Raw Data Cache (`data/raw/`)
- Caches extracted data with timestamps
- Cache expires after 24 hours
- Includes extraction parameters in cache key
- Disabled in production mode

### Processed Data Cache (`data/processed/`)
- Caches processed BG/NBD data
- Includes processing parameters in cache key
- Can be manually cleared using `clear_cache()` method

## Data Quality Validation

### Extraction Validation
- Checks for required columns
- Validates date ranges and formats
- Identifies null values in critical fields
- Warns about suspicious data patterns

### Processing Validation
- Filters supporters with insufficient actions
- Removes invalid BG/NBD calculations (t_x > T)
- Validates date parameter consistency
- Applies configurable quality thresholds

## Error Handling

- **DataExtractionError**: Raised for database extraction issues
- **DataPreprocessingError**: Raised for data processing issues
- **Comprehensive Logging**: All operations are logged with appropriate levels
- **Graceful Degradation**: Cache failures don't stop processing

## Testing

### Unit Tests
Run the comprehensive test suite:
```bash
python -m src.data.test_data_modules
```

### Integration Tests
Test with example usage:
```bash
python -m src.data.example_usage
```

### Manual Testing
```python
# Test basic functionality
from src.data import create_data_extractor, create_bgnbd_processor

extractor = create_data_extractor()
processor = create_bgnbd_processor()

# Test configuration integration
print(f"Cache enabled: {extractor.cache_enabled}")
print(f"Engagement weights: {len(processor.engagement_weights)}")
```

## File Structure

```
src/data/
├── __init__.py              # Module exports
├── extraction.py            # Data extraction from Azure SQL
├── preprocessing.py         # BG/NBD data processing
├── example_usage.py         # Usage examples
├── test_data_modules.py     # Comprehensive tests
└── README.md               # This documentation

data/
├── raw/                    # Cached raw extracts
└── processed/              # Processed BG/NBD data
```

## Dependencies

Required packages (defined in `pyproject.toml`):
- `pandas ^2.0.0` - Data manipulation
- `numpy ^1.24.0` - Numerical operations
- `sqlalchemy ^2.0.0` - Database connectivity
- `pyodbc ^4.0.0` - SQL Server driver
- `python-dotenv ^1.0.0` - Environment variables

## Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy sqlalchemy pyodbc python-dotenv
   ```

2. **Configure Database**:
   Create `.env` file with Azure SQL credentials:
   ```
   AZURE_SQL_SERVER=your-server.database.windows.net
   AZURE_SQL_DATABASE=your-database
   AZURE_SQL_USERNAME=your-username
   AZURE_SQL_PASSWORD=your-password
   ```

3. **Test Connection**:
   ```python
   from src.config import test_database_connection
   print(test_database_connection())
   ```

## Performance Considerations

- **Batch Processing**: Process data in reasonable date ranges
- **Memory Usage**: Large datasets may require chunked processing
- **Database Load**: Use caching to reduce database queries
- **Index Optimization**: Ensure database tables are properly indexed

## Next Steps

1. **Model Training**: Use processed BG/NBD data for model training
2. **Visualization**: Create engagement analysis visualizations
3. **Automation**: Set up scheduled data processing pipelines
4. **Monitoring**: Implement data quality monitoring and alerts

## Support

For issues or questions:
1. Check the test suite for examples
2. Review the example usage script
3. Examine the comprehensive logging output
4. Validate your database schema and configuration