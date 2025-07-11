# Non-Profit Engagement Model

A Python analytics project implementing BG/NBD (Beta-Geometric/Negative Binomial Distribution) model for predicting supporter engagement in non-profit organizations.

## Overview

This project uses Bayesian inference with PyMC to model supporter behavior patterns and predict future engagement. The BG/NBD model is particularly well-suited for non-profit organizations as it accounts for:

- **Heterogeneity** in supporter engagement patterns
- **Dropout behavior** where supporters may become inactive
- **Frequency and recency** of past engagement activities
- **Probabilistic predictions** with uncertainty quantification

## Features

- **Azure SQL Database Integration**: Secure connection to cloud-based supporter data
- **Bayesian Modeling**: PyMC implementation of BG/NBD model with MCMC sampling
- **Engagement Prediction**: Probabilistic forecasts of supporter activity
- **Visualization**: Comprehensive plotting and analysis tools
- **Scalable Architecture**: Modular design for easy extension and maintenance

## Project Structure

```
nonprofit-engagement-model/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   ├── data/                     # Data processing modules
│   ├── models/                   # BG/NBD model implementation
│   └── visualization/            # Plotting and visualization
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Executable scripts
├── data/                         # Data storage (gitignored)
├── models/                       # Trained model storage (gitignored)
├── pyproject.toml               # Poetry configuration
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## Requirements

- Python 3.11+
- Poetry for dependency management
- Azure SQL Database access
- Jupyter Lab (for notebooks)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd nonprofit-engagement-model
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Azure SQL Database credentials and configuration
   ```

5. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

#### Azure SQL Database
- `AZURE_SQL_SERVER`: Your Azure SQL server name
- `AZURE_SQL_DATABASE`: Database name
- `AZURE_SQL_USERNAME`: Database username
- `AZURE_SQL_PASSWORD`: Database password

#### Model Configuration
- `PREDICTION_PERIOD_DAYS`: Forecast horizon (default: 365)
- `ENGAGEMENT_THRESHOLD`: Minimum engagement score (default: 0.5)
- `SAMPLE_SIZE`: Maximum number of supporters to analyze (default: 10000)

#### Development Settings
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, WARNING, ERROR)
- `DEBUG`: Enable debug mode (True/False)
- `ENVIRONMENT`: Current environment (development, staging, production)

## Usage

### Quick Start

1. **Data Processing**:
   ```bash
   python scripts/data_processing/extract_supporter_data.py
   ```

2. **Model Training**:
   ```bash
   python scripts/model_training/train_bgnbd_model.py
   ```

3. **Generate Predictions**:
   ```bash
   python scripts/prediction/generate_engagement_predictions.py
   ```

### Jupyter Notebooks

Start Jupyter Lab for interactive analysis:

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and explore:
- Data exploration and preprocessing
- Model development and validation
- Results analysis and visualization

### Python API

```python
from src.models.bgnbd import BGNBDModel
from src.data.loader import SupporterDataLoader

# Load data
loader = SupporterDataLoader()
data = loader.load_supporter_transactions()

# Train model
model = BGNBDModel()
model.fit(data)

# Generate predictions
predictions = model.predict_engagement(
    period_days=365,
    supporters=data['supporter_id'].unique()
)
```

## Model Details

### BG/NBD Model

The Beta-Geometric/Negative Binomial Distribution model assumes:

1. **Transaction Process**: Each supporter has a transaction rate λ that follows a Gamma distribution
2. **Dropout Process**: Each supporter has a dropout probability p that follows a Beta distribution
3. **Heterogeneity**: Population-level parameters capture variation across supporters

### Key Metrics

- **Probability Alive**: Likelihood that a supporter is still active
- **Expected Transactions**: Predicted number of future engagements
- **Customer Lifetime Value**: Estimated total value of supporter relationship

## Development

### Code Style

This project uses Black and isort for code formatting:

```bash
# Format code
poetry run black src/ scripts/
poetry run isort src/ scripts/

# Check formatting
poetry run black --check src/ scripts/
poetry run isort --check-only src/ scripts/
```

### Testing

```bash
# Run tests (when implemented)
poetry run pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run code formatting and tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Create an issue in the repository
- Contact the development team
- Review the documentation in `notebooks/` and `scripts/` directories

## Acknowledgments

- PyMC development team for the Bayesian modeling framework
- Azure team for cloud database services
- Non-profit sector for inspiring this analytical approach