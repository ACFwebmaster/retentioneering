# Deployment Documentation - Non-Profit Engagement Model

## Overview

This document provides comprehensive deployment instructions for the Non-Profit Engagement Model project. The system is designed to run in containerized environments with support for both development and production deployments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Configuration Management](#configuration-management)
5. [Database Setup](#database-setup)
6. [Monitoring and Health Checks](#monitoring-and-health-checks)
7. [Scaling and Performance](#scaling-and-performance)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Azure SQL Database** access (or PostgreSQL for development)
- **Git** for repository management
- **4GB+ RAM** and **2+ CPU cores** recommended

### 30-Second Production Deployment

```bash
# Clone repository
git clone <repository-url>
cd nonprofit-engagement-model

# Configure environment
cp .env.production .env
# Edit .env with your database credentials

# Deploy
docker-compose up -d

# Verify
docker-compose ps
```

### 30-Second Development Setup

```bash
# Clone repository
git clone <repository-url>
cd nonprofit-engagement-model

# Configure development environment
cp .env.development .env

# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Access services:
# - Application: http://localhost:8000
# - Jupyter Lab: http://localhost:8888
```

## Environment Setup

### Environment Files

The project includes three environment configuration files:

| File | Purpose | Use Case |
|------|---------|----------|
| [`.env.example`](.env.example) | Template with all variables | Reference and initial setup |
| [`.env.production`](.env.production) | Production-ready settings | Production deployments |
| [`.env.development`](.env.development) | Development-friendly settings | Local development |

### Configuration Steps

1. **Choose Environment Template**
   ```bash
   # For production
   cp .env.production .env
   
   # For development
   cp .env.development .env
   ```

2. **Configure Database Connection**
   ```bash
   # Edit .env file
   AZURE_SQL_SERVER=your-server.database.windows.net
   AZURE_SQL_DATABASE=your-database-name
   AZURE_SQL_USERNAME=your-username
   AZURE_SQL_PASSWORD=your-secure-password
   ```

3. **Set Environment Type**
   ```bash
   ENVIRONMENT=production  # or development
   DEBUG=False            # True for development
   ```

## Docker Deployment

### Production Deployment

The production deployment uses optimized containers with security hardening:

```bash
# Build and start services
docker-compose up -d

# View running services
docker-compose ps

# Check logs
docker-compose logs -f app
```

#### Production Services

- **App**: Main application container
- **Redis**: Caching and session storage
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Development Deployment

Development deployment includes additional tools and hot reloading:

```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or run in background
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

#### Development Services

- **App**: Application with hot reloading
- **Jupyter**: Jupyter Lab for interactive development
- **Dev DB**: PostgreSQL database for local development
- **Redis**: Local Redis instance
- **MailHog**: Email testing service
- **Linter**: Code quality tools

### Service Access

| Service | Production URL | Development URL | Credentials |
|---------|---------------|-----------------|-------------|
| Application | http://localhost:8000 | http://localhost:8000 | - |
| Jupyter Lab | - | http://localhost:8888 | Token: `dev-token-123` |
| Grafana | http://localhost:3000 | http://localhost:3000 | admin/admin |
| MailHog | - | http://localhost:8025 | - |

## Configuration Management

### Environment Variables

#### Core Configuration

```bash
# Application
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Database
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password

# Model Parameters
PREDICTION_PERIOD_DAYS=365
SAMPLE_SIZE=50000
MCMC_DRAWS=2000
```

#### Performance Tuning

```bash
# Memory and CPU
MAX_MEMORY_USAGE_GB=8
MAX_WORKER_THREADS=8
MODEL_BATCH_SIZE=1000

# Database Connection Pool
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=60

# Caching
ENABLE_REDIS_CACHE=True
CACHE_TTL_SECONDS=3600
```

### Configuration Validation

Test your configuration before deployment:

```bash
# Validate configuration
docker-compose exec app python -c "
from src.config.settings import get_config
config = get_config()
print('Configuration valid:', config.validate_config())
"

# Test database connection
docker-compose exec app python -c "
from src.config.database import test_database_connection
print('Database connection:', test_database_connection())
"
```

## Database Setup

### Azure SQL Database (Production)

1. **Create Azure SQL Database**
   ```bash
   # Using Azure CLI
   az sql server create --name your-server --resource-group your-rg --location eastus --admin-user your-admin --admin-password your-password
   az sql db create --resource-group your-rg --server your-server --name nonprofit-engagement
   ```

2. **Configure Firewall**
   ```bash
   # Allow Azure services
   az sql server firewall-rule create --resource-group your-rg --server your-server --name AllowAzureServices --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
   
   # Allow your IP
   az sql server firewall-rule create --resource-group your-rg --server your-server --name AllowMyIP --start-ip-address YOUR_IP --end-ip-address YOUR_IP
   ```

3. **Initialize Database Schema**
   ```bash
   # Run database migrations
   docker-compose exec app python scripts/init_database.py
   ```

### PostgreSQL (Development)

For local development, the system can use PostgreSQL:

```bash
# Start development database
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up dev_db

# Connect to database
docker-compose exec dev_db psql -U dev_user -d nonprofit_engagement_dev
```

### Database Migrations

```bash
# Check current schema
docker-compose exec app python scripts/check_schema.py

# Run migrations
docker-compose exec app python scripts/migrate_database.py

# Seed with sample data (development only)
docker-compose exec app python scripts/seed_sample_data.py
```

## Monitoring and Health Checks

### Application Health

```bash
# Check application status
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Service Health

```bash
# Check all services
docker-compose ps

# View service logs
docker-compose logs app
docker-compose logs redis
docker-compose logs nginx

# Monitor resource usage
docker stats
```

### Monitoring Dashboard

Access Grafana dashboard at http://localhost:3000:

- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request rates, response times
- **Database Metrics**: Connection pool, query performance
- **Model Metrics**: Training time, prediction accuracy

### Log Management

```bash
# View application logs
docker-compose logs -f app

# View specific log files
docker-compose exec app tail -f logs/production.log

# Search logs
docker-compose exec app grep "ERROR" logs/production.log
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale application instances
docker-compose up --scale app=3

# Update load balancer configuration
# Edit nginx/nginx.conf to include all app instances
```

### Performance Optimization

#### Model Performance
```bash
# Adjust model parameters for performance
MCMC_DRAWS=1000          # Reduce for faster training
MCMC_CHAINS=2            # Reduce for less CPU usage
MODEL_BATCH_SIZE=500     # Adjust based on memory
```

#### Database Performance
```bash
# Optimize connection pool
DB_POOL_SIZE=20          # Increase for high concurrency
DB_MAX_OVERFLOW=40       # Allow burst connections
DB_POOL_RECYCLE=7200     # Recycle connections every 2 hours
```

#### Caching
```bash
# Enable Redis caching
ENABLE_REDIS_CACHE=True
CACHE_TTL_SECONDS=3600   # Cache for 1 hour
REDIS_MAX_MEMORY=512mb   # Adjust based on available memory
```

### Resource Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 20GB SSD
- **Network**: 100 Mbps

#### Recommended Production
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 100GB+ SSD
- **Network**: 1 Gbps

## Troubleshooting

### Common Issues

#### 1. SSL Certificate Verification Errors

**Symptoms**:
- `SSL Provider: [error:0A000086:SSL routines::certificate verify failed:self-signed certificate]`
- `certificate verify failed` errors in logs
- Database connection failures with SSL-related messages

**Root Cause**: Azure SQL Database SSL certificate validation issues, commonly occurring with ODBC Driver 18 for SQL Server.

**Solutions**:

**For Development Environment (Quick Fix)**:
```bash
# Add SSL configuration to your .env file
echo "AZURE_SQL_ENCRYPT=yes" >> .env
echo "AZURE_SQL_TRUST_SERVER_CERTIFICATE=yes" >> .env

# Restart the application
docker-compose restart app
```

**For Production Environment (Secure)**:
```bash
# Use proper certificate validation (recommended)
AZURE_SQL_ENCRYPT=yes
AZURE_SQL_TRUST_SERVER_CERTIFICATE=no

# Ensure your Azure SQL server has proper SSL certificates
# Contact your Azure administrator if certificate issues persist
```

**Environment Variable Reference**:
```bash
# SSL Configuration Options
AZURE_SQL_ENCRYPT=yes                    # Enable SSL encryption (required)
AZURE_SQL_TRUST_SERVER_CERTIFICATE=yes   # Skip certificate validation (dev only)
AZURE_SQL_TRUST_SERVER_CERTIFICATE=no    # Validate certificates (production)
```

**Testing SSL Configuration**:
```bash
# Test database connection with SSL settings
docker-compose exec app python -c "
from src.config.database import test_database_connection
result = test_database_connection()
print('Database connection successful:', result)
"

# Check SSL configuration
docker-compose exec app python -c "
from src.config.settings import get_config
config = get_config()
print('Encrypt:', config.database.encrypt)
print('Trust Server Certificate:', config.database.trust_server_certificate)
"
```

**⚠️ Security Warning**:
- `TrustServerCertificate=yes` bypasses SSL certificate validation
- Only use this setting in development environments
- Always use `TrustServerCertificate=no` in production with proper certificates

#### 2. Database Connection Failed

**Symptoms**: Application fails to start, database connection errors

**Solutions**:
```bash
# Check database credentials
grep -E "AZURE_SQL_" .env

# Test network connectivity
docker-compose exec app ping your-server.database.windows.net

# Verify firewall rules
az sql server firewall-rule list --resource-group your-rg --server your-server

# Check connection string format
docker-compose exec app python -c "
from src.config.database import DatabaseManager
dm = DatabaseManager()
print(dm.get_connection_info())
"
```

#### 2. Out of Memory Errors

**Symptoms**: Application crashes, OOM killer messages

**Solutions**:
```bash
# Reduce model parameters
SAMPLE_SIZE=10000        # Reduce dataset size
MCMC_DRAWS=500          # Reduce MCMC iterations
MODEL_BATCH_SIZE=100    # Reduce batch size

# Increase container memory limits
# Edit docker-compose.yml memory limits

# Monitor memory usage
docker stats
```

#### 3. Slow Performance

**Symptoms**: Long response times, high CPU usage

**Solutions**:
```bash
# Enable caching
ENABLE_REDIS_CACHE=True

# Optimize database queries
# Check slow query logs

# Scale horizontally
docker-compose up --scale app=2

# Adjust model parameters
MCMC_CHAINS=2           # Reduce parallel chains
```

#### 4. Port Conflicts

**Symptoms**: Services fail to start, port binding errors

**Solutions**:
```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use different external port

# Or set environment variables
APP_PORT=8001
JUPYTER_PORT=8889
```

### Debug Commands

```bash
# Enter application container
docker-compose exec app bash

# Check Python environment
docker-compose exec app python --version
docker-compose exec app pip list

# Test configuration loading
docker-compose exec app python -c "
from src.config.settings import get_config
config = get_config()
print('Environment:', config.environment)
print('Debug mode:', config.debug)
"

# Test database connection
docker-compose exec app python -c "
from src.config.database import test_database_connection
print('Database OK:', test_database_connection())
"

# Check model functionality
docker-compose exec app python -c "
from src.models.bgnbd import BGNBDModel
model = BGNBDModel()
print('Model initialized successfully')
"
```

### Log Analysis

```bash
# Application errors
docker-compose logs app | grep ERROR

# Database connection issues
docker-compose logs app | grep -i "database\|connection"

# Performance issues
docker-compose logs app | grep -i "slow\|timeout\|memory"

# Export logs for analysis
docker-compose logs app > app_logs.txt
```

### Getting Help

1. **Check this documentation** for common solutions
2. **Review application logs** for specific error messages
3. **Verify configuration** using the debug commands above
4. **Check resource usage** with `docker stats`
5. **Consult the project README** for additional context

### Maintenance Tasks

#### Daily
- Monitor application health and performance
- Check log files for errors or warnings
- Verify backup completion

#### Weekly
- Review resource usage trends
- Update Docker images if needed
- Clean up old log files

#### Monthly
- Review and rotate credentials
- Update dependencies
- Performance optimization review

---

## Additional Resources

- **Project README**: [README.md](README.md)
- **Security Guide**: [security/SECURITY.md](security/SECURITY.md)
- **Deployment Guide**: [security/deployment.md](security/deployment.md)
- **Configuration Reference**: [src/config/settings.py](src/config/settings.py)

For additional support or questions, please refer to the project documentation or contact the development team.