# BG/NBD Model Execution Script

This directory contains the main execution script for the non-profit BG/NBD engagement model project.

## Overview

The `run_model.py` script provides a comprehensive CLI interface for running the complete BG/NBD modeling pipeline including:

- **Data Extraction**: Extract supporter actions and donations from Azure SQL Database
- **Data Preprocessing**: Transform raw data into BG/NBD format with engagement scoring
- **Model Training**: Fit BG/NBD models using PyMC with Bayesian inference
- **Prediction Generation**: Generate P(Alive), expected transactions, and CLV predictions
- **Visualization**: Create diagnostic and business intelligence plots
- **Full Pipeline**: Execute complete end-to-end workflow

## Installation & Setup

### Prerequisites

1. **Python Environment**: Python 3.8+ with required packages (see `pyproject.toml`)
2. **Database Access**: Azure SQL Database with supporter data
3. **Configuration**: Environment variables or `.env` file with database credentials

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# Database Configuration
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password
AZURE_SQL_DRIVER=ODBC Driver 18 for SQL Server

# Model Configuration
PREDICTION_PERIOD_DAYS=365
ENGAGEMENT_THRESHOLD=0.5
MODEL_OUTPUT_DIR=models/
DATA_OUTPUT_DIR=data/processed/
VISUALIZATION_OUTPUT_DIR=outputs/visualizations/

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Environment Settings
ENVIRONMENT=development
DEBUG=False
DEV_MODE=True
```

## Usage

### Basic Command Structure

```bash
python scripts/run_model.py [COMMAND] [OPTIONS]
```

### Available Commands

#### 1. Data Extraction (`extract`)

Extract supporter actions and donations from the database.

```bash
# Basic extraction for 2023
python scripts/run_model.py extract --start-date 2023-01-01 --end-date 2024-01-01

# With minimum donation filter and custom output
python scripts/run_model.py extract \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --min-donation 10.0 \
    --output-dir data/raw/2023_data \
    --verbose

# Disable caching
python scripts/run_model.py extract \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --no-cache
```

**Options:**
- `--start-date`: Analysis start date (YYYY-MM-DD) [Required]
- `--end-date`: Analysis end date (YYYY-MM-DD) [Required]
- `--min-donation`: Minimum donation amount filter
- `--cache/--no-cache`: Enable/disable data caching (default: enabled)

#### 2. Data Preprocessing (`preprocess`)

Transform raw data into BG/NBD format with engagement scoring.

```bash
# Basic preprocessing
python scripts/run_model.py preprocess \
    --start-date 2023-01-01 \
    --end-date 2024-01-01

# With custom cutoff date and minimum actions
python scripts/run_model.py preprocess \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --cutoff-date 2023-10-01 \
    --min-actions 2 \
    --exclude-donations

# Custom output directory
python scripts/run_model.py preprocess \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --output-dir data/processed/custom \
    --verbose
```

**Options:**
- `--start-date`: Observation period start date (YYYY-MM-DD) [Required]
- `--end-date`: Observation period end date (YYYY-MM-DD) [Required]
- `--cutoff-date`: Analysis cutoff date (YYYY-MM-DD)
- `--min-actions`: Minimum number of actions per supporter (default: 1)
- `--include-donations/--exclude-donations`: Include/exclude donations as engagement events
- `--cache/--no-cache`: Enable/disable data caching

#### 3. Model Training (`train`)

Train BG/NBD models using PyMC with Bayesian inference.

```bash
# Basic model training
python scripts/run_model.py train data/processed/bgnbd_data.csv

# Hierarchical model with custom parameters
python scripts/run_model.py train data/processed/bgnbd_data.csv \
    --model-type hierarchical \
    --segment-column engagement_segment \
    --draws 3000 \
    --tune 1500 \
    --chains 4

# High-performance training
python scripts/run_model.py train data/processed/bgnbd_data.csv \
    --model-type basic \
    --draws 5000 \
    --tune 2000 \
    --chains 6 \
    --output-dir models/high_performance \
    --verbose
```

**Options:**
- `data_path`: Path to processed BG/NBD data [Required]
- `--model-type`: Model type ('basic' or 'hierarchical', default: 'basic')
- `--draws`: Number of MCMC draws (default: 2000)
- `--tune`: Number of tuning steps (default: 1000)
- `--chains`: Number of MCMC chains (default: 4)
- `--segment-column`: Column name for hierarchical modeling

#### 4. Prediction Generation (`predict`)

Generate predictions using trained models.

```bash
# Basic predictions
python scripts/run_model.py predict --model-path models/bgnbd_model.pkl

# Custom prediction period and data
python scripts/run_model.py predict \
    --model-path models/bgnbd_model_hierarchical.pkl \
    --data-path data/processed/new_supporters.csv \
    --prediction-period 365 \
    --output-format both

# CLV predictions with custom output
python scripts/run_model.py predict \
    --model-path models/bgnbd_model.pkl \
    --prediction-period 180 \
    --output-format json \
    --output-dir predictions/q1_2024 \
    --verbose
```

**Options:**
- `--model-path`: Path to trained model file [Required]
- `--data-path`: Path to data for predictions (uses training data if not provided)
- `--prediction-period`: Prediction horizon in days (default: 180)
- `--output-format`: Output format ('csv', 'json', or 'both', default: 'csv')

#### 5. Visualization (`visualize`)

Create diagnostic and business intelligence plots.

```bash
# All plot types
python scripts/run_model.py visualize \
    --model-path models/bgnbd_model.pkl \
    --data-path data/processed/bgnbd_data.csv

# Specific plot types
python scripts/run_model.py visualize \
    --model-path models/bgnbd_model.pkl \
    --data-path data/processed/bgnbd_data.csv \
    --plot-types diagnostic,prediction \
    --output-format pdf

# Interactive plots
python scripts/run_model.py visualize \
    --model-path models/bgnbd_model.pkl \
    --data-path data/processed/bgnbd_data.csv \
    --plot-types business,data-quality \
    --interactive \
    --output-dir visualizations/interactive

# Data-only plots (no model required)
python scripts/run_model.py visualize \
    --data-path data/processed/bgnbd_data.csv \
    --plot-types data-quality \
    --output-format png
```

**Options:**
- `--model-path`: Path to trained model (optional for data-only plots)
- `--data-path`: Path to data for visualization
- `--plot-types`: Comma-separated list of plot types:
  - `diagnostic`: Model convergence and trace plots
  - `prediction`: P(Alive), expected transactions, CLV plots
  - `business`: Segmentation, targeting, trends plots
  - `data-quality`: Data quality assessment plots
- `--output-format`: Output format ('png', 'pdf', 'svg', or 'all', default: 'png')
- `--interactive`: Enable interactive plots

#### 6. Full Pipeline (`full-pipeline`)

Execute the complete end-to-end workflow.

```bash
# Basic full pipeline
python scripts/run_model.py full-pipeline \
    --start-date 2023-01-01 \
    --end-date 2024-01-01

# Advanced pipeline with hierarchical model
python scripts/run_model.py full-pipeline \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --cutoff-date 2023-10-01 \
    --model-type hierarchical \
    --draws 3000 \
    --tune 1500 \
    --prediction-period 365 \
    --min-actions 2

# Production pipeline with custom output
python scripts/run_model.py full-pipeline \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --model-type hierarchical \
    --draws 5000 \
    --tune 2000 \
    --chains 6 \
    --prediction-period 365 \
    --output-dir outputs/production_run_2024 \
    --config .env.production \
    --verbose
```

**Options:**
- `--start-date`: Analysis start date (YYYY-MM-DD) [Required]
- `--end-date`: Analysis end date (YYYY-MM-DD) [Required]
- `--cutoff-date`: Analysis cutoff date (YYYY-MM-DD)
- `--min-actions`: Minimum actions per supporter (default: 1)
- `--model-type`: Model type ('basic' or 'hierarchical', default: 'basic')
- `--draws`: MCMC draws (default: 2000)
- `--tune`: MCMC tuning steps (default: 1000)
- `--chains`: MCMC chains (default: 4)
- `--prediction-period`: Prediction horizon in days (default: 180)

### Global Options

These options are available for all commands:

- `--config`: Path to configuration file (.env)
- `--verbose`, `-v`: Enable verbose logging
- `--output-dir`: Custom output directory

## Output Structure

The script creates organized output directories with the following structure:

### Individual Commands
```
outputs/
├── data/
│   ├── raw/                    # Extracted data
│   └── processed/              # BG/NBD formatted data
├── models/                     # Trained model files
├── predictions/                # Prediction results
├── visualizations/             # Generated plots
└── logs/                       # Execution logs
```

### Full Pipeline
```
outputs/pipeline_YYYYMMDD_HHMMSS/
├── 01_extraction/              # Data extraction results
├── 02_preprocessing/           # Data preprocessing results
├── 03_training/                # Model training results
├── 04_predictions/             # Prediction results
├── 05_visualizations/          # Generated plots
├── 06_report/                  # Comprehensive report
├── pipeline_summary.json      # Pipeline execution summary
└── logs/                       # Execution logs
```

## Examples

### Example 1: Quick Analysis

Run a complete analysis for 2023 data:

```bash
python scripts/run_model.py full-pipeline \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --verbose
```

### Example 2: Production Model Training

Train a high-quality hierarchical model:

```bash
# First, preprocess the data
python scripts/run_model.py preprocess \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --min-actions 3 \
    --output-dir data/production

# Train hierarchical model
python scripts/run_model.py train data/production/bgnbd_data.csv \
    --model-type hierarchical \
    --draws 5000 \
    --tune 2000 \
    --chains 6 \
    --output-dir models/production

# Generate predictions
python scripts/run_model.py predict \
    --model-path models/production/bgnbd_model_hierarchical.pkl \
    --prediction-period 365 \
    --output-format both \
    --output-dir predictions/2024_forecast
```

### Example 3: Model Comparison

Compare basic vs hierarchical models:

```bash
# Train basic model
python scripts/run_model.py train data/processed/bgnbd_data.csv \
    --model-type basic \
    --output-dir models/comparison/basic

# Train hierarchical model
python scripts/run_model.py train data/processed/bgnbd_data.csv \
    --model-type hierarchical \
    --output-dir models/comparison/hierarchical

# Generate visualizations for both
python scripts/run_model.py visualize \
    --model-path models/comparison/basic/bgnbd_model_basic.pkl \
    --data-path data/processed/bgnbd_data.csv \
    --output-dir visualizations/basic_model

python scripts/run_model.py visualize \
    --model-path models/comparison/hierarchical/bgnbd_model_hierarchical.pkl \
    --data-path data/processed/bgnbd_data.csv \
    --output-dir visualizations/hierarchical_model
```

## Error Handling & Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify database credentials in `.env` file
   - Check network connectivity to Azure SQL
   - Ensure ODBC driver is installed

2. **Memory Issues During Training**
   - Reduce `--draws` and `--tune` parameters
   - Use fewer `--chains`
   - Consider using basic model instead of hierarchical

3. **Convergence Issues**
   - Increase `--tune` parameter
   - Increase `--draws` parameter
   - Check data quality and preprocessing

4. **Missing Data Columns**
   - Verify data preprocessing completed successfully
   - Check that required columns (x, t_x, T) exist
   - Ensure segment column exists for hierarchical models

### Validation Checks

The script performs automatic validation:

- ✅ Environment configuration
- ✅ Database connectivity
- ✅ Data quality and format
- ✅ Model convergence
- ✅ Output directory creation

### Logging

Detailed logs are saved to:
- Console output (INFO level by default)
- Log files in `logs/` directory
- Use `--verbose` for DEBUG level logging

## Performance Considerations

### Recommended Settings

**Development/Testing:**
```bash
--draws 1000 --tune 500 --chains 2
```

**Production/Research:**
```bash
--draws 5000 --tune 2000 --chains 4
```

**High-Performance:**
```bash
--draws 10000 --tune 3000 --chains 6
```

### Resource Requirements

- **Memory**: 4-16 GB depending on data size and model complexity
- **CPU**: Multi-core recommended for parallel MCMC chains
- **Storage**: 1-10 GB for outputs depending on data size
- **Time**: 10 minutes to several hours depending on model complexity

## Integration

### Batch Processing

The script can be integrated into batch processing workflows:

```bash
#!/bin/bash
# Monthly model update script

DATE=$(date +%Y-%m-%d)
LAST_MONTH=$(date -d "1 month ago" +%Y-%m-%d)

python scripts/run_model.py full-pipeline \
    --start-date 2023-01-01 \
    --end-date $DATE \
    --model-type hierarchical \
    --output-dir "outputs/monthly_update_$DATE" \
    --config .env.production
```

### API Integration

Results can be consumed by other systems through:
- JSON output files
- CSV prediction files
- Programmatic access to saved models

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files for detailed error messages
3. Verify environment configuration
4. Consult the project documentation in the main README

## Version History

- **v1.0.0**: Initial release with full CLI functionality
- Comprehensive pipeline support
- Multiple execution modes
- Production-ready error handling
- Extensive visualization capabilities