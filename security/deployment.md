# Deployment Guide for Non-Profit Engagement Model

## Quick Start Deployment

### Prerequisites

- Docker and Docker Compose installed
- Azure SQL Database access (or local PostgreSQL for development)
- Basic understanding of environment variables

### Production Deployment

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd nonprofit-engagement-model
   ```

2. **Configure Environment**
   ```bash
   cp .env.production .env
   # Edit .env with your actual database credentials and settings
   ```

3. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Verify Deployment**
   ```bash
   docker-compose ps
   docker-compose logs app
   ```

### Development Deployment

1. **Setup Development Environment**
   ```bash
   cp .env.development .env
   # Edit .env with your development database settings
   ```

2. **Start Development Services**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
   ```

3. **Access Services**
   - Application: http://localhost:8000
   - Jupyter Lab: http://localhost:8888 (token: dev-token-123)
   - MailHog: http://localhost:8025

## Environment Configuration

### Required Environment Variables

#### Database Configuration
```bash
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database-name
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password
```

#### Application Settings
```bash
ENVIRONMENT=production  # or development
DEBUG=False            # True for development
LOG_LEVEL=INFO         # DEBUG for development
```

### Optional Configuration

#### Performance Tuning
```bash
MCMC_DRAWS=2000        # Reduce for faster development
SAMPLE_SIZE=50000      # Reduce for development
MODEL_BATCH_SIZE=1000  # Adjust based on memory
```

#### Caching (Production)
```bash
ENABLE_REDIS_CACHE=True
REDIS_HOST=redis
REDIS_PASSWORD=your-redis-password
```

## Deployment Environments

### Local Development
- Uses SQLite or local PostgreSQL
- Hot reloading enabled
- Debug logging
- Sample data generation

### Staging
- Production-like environment
- Real database connections
- Performance monitoring
- Limited data set

### Production
- Full security measures
- Performance optimization
- Monitoring and alerting
- Backup and recovery

## Health Checks and Monitoring

### Application Health
```bash
# Check application status
curl http://localhost:8000/health

# View logs
docker-compose logs -f app

# Monitor resource usage
docker stats
```

### Database Health
```bash
# Test database connection
docker-compose exec app python -c "from src.config.database import test_database_connection; print(test_database_connection())"
```

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale application instances
docker-compose up --scale app=3

# Update load balancer configuration
# (Configure nginx upstream servers)
```

### Performance Optimization
- Adjust `MCMC_DRAWS` and `MCMC_CHAINS` based on available CPU
- Configure `MODEL_BATCH_SIZE` based on available memory
- Enable Redis caching for production workloads

## Backup and Recovery

### Database Backup
```bash
# Manual backup
docker-compose exec app python scripts/backup_database.py

# Automated backup (configure in cron)
0 2 * * * /path/to/backup_script.sh
```

### Application Data Backup
```bash
# Backup models and outputs
tar -czf backup-$(date +%Y%m%d).tar.gz models/ outputs/ logs/
```

## Troubleshooting

### Common Issues

#### Database Connection Failed
1. Check database credentials in `.env`
2. Verify network connectivity
3. Check firewall rules
4. Validate connection string format

#### Application Won't Start
1. Check Docker logs: `docker-compose logs app`
2. Verify environment variables
3. Check port conflicts
4. Ensure sufficient resources

#### Performance Issues
1. Monitor resource usage: `docker stats`
2. Check database query performance
3. Adjust model parameters
4. Enable caching

### Debug Commands
```bash
# Enter application container
docker-compose exec app bash

# Check Python environment
docker-compose exec app python --version
docker-compose exec app pip list

# Test configuration
docker-compose exec app python -c "from src.config.settings import get_config; print(get_config().validate_config())"
```

## Maintenance

### Regular Tasks
- Monitor disk space and clean up old logs
- Update Docker images regularly
- Review and rotate credentials
- Monitor application performance
- Check for security updates

### Updates and Patches
```bash
# Update application
git pull origin main
docker-compose build --no-cache
docker-compose up -d

# Update dependencies
docker-compose exec app poetry update
```

## Support and Documentation

### Getting Help
- Check application logs first
- Review this deployment guide
- Consult the main README.md
- Check Docker Compose documentation

### Useful Commands
```bash
# View all services
docker-compose ps

# Restart specific service
docker-compose restart app

# View resource usage
docker system df

# Clean up unused resources
docker system prune