# MVP Testing Guide
## Non-Profit Engagement Model

This guide documents the MVP (Minimum Viable Product) testing approach for the Non-Profit Engagement Model project, providing a streamlined validation strategy focused on core functionality and production readiness.

---

## Overview

The MVP testing approach prioritizes **essential functionality validation** over comprehensive test coverage, enabling rapid assessment of production readiness while identifying critical issues that must be addressed before deployment.

### Testing Philosophy

- **Focus on Core Functionality:** Test the most critical components that directly impact user value
- **Production Readiness:** Validate deployment infrastructure and configuration
- **Quick Feedback:** Provide rapid validation results to enable fast iteration
- **Risk-Based:** Prioritize testing of high-risk, high-impact components

---

## Testing Strategy

### 1. Component-Based Validation

The MVP testing approach validates 8 core components:

| Component | Priority | Description |
|-----------|----------|-------------|
| **Module Imports** | Critical | Verify all core modules can be imported |
| **Configuration** | Critical | Validate environment and settings loading |
| **Sample Data** | High | Test data generation and format compliance |
| **Model Creation** | Critical | Verify BG/NBD model instantiation |
| **Model Training** | Critical | Test MCMC sampling and parameter estimation |
| **Visualization** | Medium | Validate plotting and chart generation |
| **CLI Interface** | High | Test command-line interface functionality |
| **Docker Config** | Medium | Verify containerization setup |

### 2. Quick Mode vs Full Mode

#### Quick Mode (Default for MVP)
- **Duration:** 2-5 minutes
- **Sample Size:** 50-100 supporters
- **MCMC Draws:** 100-500 (reduced for speed)
- **Use Case:** Rapid development validation

#### Full Mode
- **Duration:** 10-30 minutes
- **Sample Size:** 500-1000 supporters
- **MCMC Draws:** 2000+ (production quality)
- **Use Case:** Pre-production validation

---

## Validation Script Usage

### Basic Usage

```bash
# Quick validation (development)
python validate_mvp.py --environment development --quick

# Full validation (pre-production)
python validate_mvp.py --environment production

# Custom output location
python validate_mvp.py --output validation_results.json
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--environment` | Target environment (dev/prod) | development |
| `--quick` | Enable quick mode | False |
| `--output` | Output file for results | Auto-generated |

### Environment Setup

1. **Development Environment:**
   ```bash
   # Use development configuration
   cp .env.development .env
   python validate_mvp.py --environment development
   ```

2. **Production Environment:**
   ```bash
   # Use production configuration
   cp .env.production .env
   # Configure actual database credentials
   python validate_mvp.py --environment production
   ```

---

## Validation Components

### 1. Module Imports Validation

**Purpose:** Verify all core modules can be imported without errors

**Tests:**
- Import configuration management
- Import data processing modules
- Import BG/NBD model classes
- Import visualization components
- Import CLI interface

**Success Criteria:**
- All imports complete without exceptions
- No missing dependencies
- Configuration loads successfully

### 2. Configuration Validation

**Purpose:** Ensure environment configuration is properly loaded

**Tests:**
- Environment variable loading
- Database configuration validation
- Model parameter configuration
- Logging configuration setup

**Success Criteria:**
- Configuration object created successfully
- All required settings present
- Environment-specific values loaded

### 3. Sample Data Validation

**Purpose:** Test synthetic data generation for development and testing

**Tests:**
- Generate sample supporter data
- Validate data structure and format
- Check BG/NBD variable presence (x, t_x, T)
- Verify data quality and completeness

**Success Criteria:**
- Sample data generated within time limits
- Required columns present
- Data values within expected ranges

### 4. Model Creation Validation

**Purpose:** Verify BG/NBD model instantiation and configuration

**Tests:**
- Create basic BG/NBD model
- Create hierarchical BG/NBD model
- Validate model parameters and structure
- Test model configuration options

**Success Criteria:**
- Both model types instantiate successfully
- Model attributes properly initialized
- No configuration errors

### 5. Model Training Validation

**Purpose:** Test MCMC sampling and parameter estimation

**Tests:**
- Generate training data
- Fit BG/NBD model using PyMC
- Validate convergence diagnostics
- Test prediction generation

**Success Criteria:**
- Model training completes without errors
- Parameters extracted successfully
- Predictions generated correctly
- Convergence within acceptable limits

### 6. Visualization Validation

**Purpose:** Verify plotting and chart generation capabilities

**Tests:**
- Create plotter instance
- Test basic plotting functionality
- Validate visualization components
- Check plot generation methods

**Success Criteria:**
- Plotter instantiates successfully
- Core plotting methods available
- No import or dependency errors

### 7. CLI Interface Validation

**Purpose:** Test command-line interface functionality

**Tests:**
- Create argument parser
- Initialize model runner
- Validate CLI configuration
- Test command structure

**Success Criteria:**
- Parser creates successfully
- Model runner initializes
- Configuration validation passes
- Help system functional

### 8. Docker Configuration Validation

**Purpose:** Verify containerization and deployment setup

**Tests:**
- Check Dockerfile presence and structure
- Validate docker-compose configurations
- Verify environment file setup
- Test configuration file accessibility

**Success Criteria:**
- All Docker files present
- Configuration files readable
- Environment setup complete

---

## Interpreting Results

### Success Rates

| Success Rate | Status | Action Required |
|--------------|--------|-----------------|
| **90-100%** | ✅ PASS | Ready for production |
| **70-89%** | ⚠️ WARNING | Address issues before production |
| **< 70%** | ❌ FAIL | Critical issues must be resolved |

### Common Issues and Solutions

#### Import Failures
- **Cause:** Missing dependencies
- **Solution:** Install required packages with pip/poetry
- **Prevention:** Use virtual environments

#### Configuration Errors
- **Cause:** Missing environment variables
- **Solution:** Copy and configure .env files
- **Prevention:** Validate environment setup

#### Model Training Failures
- **Cause:** PyMC API changes or data format issues
- **Solution:** Update API calls and data preprocessing
- **Prevention:** Pin dependency versions

#### Data Format Issues
- **Cause:** Sample data doesn't match BG/NBD requirements
- **Solution:** Implement data transformation pipeline
- **Prevention:** Validate data schemas

---

## Continuous Integration Integration

### GitHub Actions Example

```yaml
name: MVP Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run MVP validation
      run: |
        python validate_mvp.py --environment development --quick
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: validation-results
        path: mvp_validation_results_*.json
```

### Docker Testing

```bash
# Build and test in Docker
docker build -t nonprofit-engagement-model .
docker run --rm nonprofit-engagement-model python validate_mvp.py --quick
```

---

## Performance Benchmarks

### Expected Performance (Quick Mode)

| Component | Expected Time | Acceptable Range |
|-----------|---------------|------------------|
| **Total Validation** | 2-5 minutes | < 10 minutes |
| **Sample Data Generation** | 0.1-0.5 seconds | < 2 seconds |
| **Model Creation** | 0.5-2 seconds | < 5 seconds |
| **Model Training** | 30-120 seconds | < 300 seconds |

### Resource Usage

| Resource | Expected Usage | Maximum Acceptable |
|----------|----------------|-------------------|
| **Memory** | 500MB-1GB | < 2GB |
| **CPU** | 1-2 cores | < 4 cores |
| **Disk** | 100MB | < 500MB |

---

## Extending the MVP Testing

### Adding New Validation Components

1. **Create validation method** in `MVPValidator` class
2. **Add to validation steps** in `run_validation()` method
3. **Update documentation** with new component details
4. **Test the new validation** with known good/bad cases

### Custom Validation Scripts

```python
from validate_mvp import MVPValidator

# Create custom validator
validator = MVPValidator(environment="development", quick_mode=True)

# Run specific validation
result = validator.validate_model_training()
print(f"Model training: {'PASS' if result else 'FAIL'}")
```

---

## Troubleshooting

### Common Problems

1. **"No module named 'sqlalchemy'"**
   - Install missing dependencies: `pip install sqlalchemy`

2. **"Required environment variable not set"**
   - Copy appropriate .env file: `cp .env.development .env`

3. **"Model training failed: gammaln"**
   - Update PyMC API calls in model code

4. **"Sample data missing required columns"**
   - Implement BG/NBD data transformation

### Debug Mode

```bash
# Enable verbose logging
python validate_mvp.py --environment development --quick --verbose
```

### Manual Component Testing

```python
# Test individual components
python -c "
from src.config import get_config
config = get_config()
print('Configuration loaded successfully')
"
```

---

## Best Practices

### Development Workflow

1. **Run MVP validation** before committing code
2. **Use quick mode** for rapid iteration
3. **Run full validation** before creating pull requests
4. **Address critical issues** immediately

### Production Deployment

1. **Run full validation** in production environment
2. **Achieve 90%+ success rate** before deployment
3. **Document any known issues** and workarounds
4. **Set up monitoring** for ongoing validation

### Maintenance

1. **Update validation script** when adding new features
2. **Review validation results** regularly
3. **Update benchmarks** as system evolves
4. **Keep documentation current**

---

## Conclusion

The MVP testing approach provides a practical, efficient method for validating the Non-Profit Engagement Model's core functionality and production readiness. By focusing on essential components and providing rapid feedback, this approach enables confident deployment decisions while maintaining development velocity.

Regular use of this validation approach ensures consistent quality and reduces the risk of production issues, making it an essential tool in the development and deployment workflow.