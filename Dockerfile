# =============================================================================
# Multi-stage Dockerfile for Non-Profit Engagement Model
# =============================================================================
# This Dockerfile creates a production-ready container with security hardening,
# optimized layers, and proper dependency management.
# =============================================================================

# =============================================================================
# Stage 1: Base Python Image with System Dependencies
# =============================================================================
FROM python:3.11-slim-bookworm as base

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.6.1

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential system packages
    curl \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    # Microsoft ODBC Driver for SQL Server
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    # Security updates
    && apt-get upgrade -y \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# =============================================================================
# Stage 2: Poetry Installation and Dependency Resolution
# =============================================================================
FROM base as poetry-base

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Stage 3: Development Stage (for development builds)
# =============================================================================
FROM poetry-base as development

# Install development dependencies
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Create non-root user for development
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set up application directory
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs data models outputs

# Expose development ports
EXPOSE 8000 8888

# Development entrypoint
CMD ["poetry", "run", "python", "-m", "src.main"]

# =============================================================================
# Stage 4: Production Stage (optimized for production)
# =============================================================================
FROM base as production

# Create non-root user with specific UID/GID for security
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /bin/bash --create-home appuser \
    && mkdir -p /app /app/logs /app/data /app/models /app/outputs \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from poetry stage
COPY --from=poetry-base --chown=appuser:appuser /.venv /.venv

# Add virtual environment to PATH
ENV PATH="/.venv/bin:$PATH"

# Copy application code with proper ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser pyproject.toml ./

# Create additional required directories
RUN mkdir -p \
    /app/logs/production \
    /app/data/production/processed \
    /app/models/production \
    /app/outputs/production/visualizations \
    /app/backups/database \
    /app/security/keys \
    && chown -R appuser:appuser /app

# Set proper permissions
RUN chmod -R 755 /app \
    && chmod -R 700 /app/security \
    && chmod -R 750 /app/logs \
    && chmod -R 750 /app/data

# Switch to non-root user
USER appuser

# Set production environment variables
ENV ENVIRONMENT=production \
    DEBUG=False \
    DEV_MODE=False \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.config.database import test_database_connection; exit(0 if test_database_connection() else 1)"

# Expose application port
EXPOSE 8000

# Production entrypoint with proper signal handling
ENTRYPOINT ["python", "-m", "src.main"]

# =============================================================================
# Stage 5: Testing Stage (for CI/CD pipelines)
# =============================================================================
FROM development as testing

# Install additional testing dependencies
RUN poetry install --with dev --no-root

# Copy test files
COPY --chown=appuser:appuser tests/ ./tests/

# Set testing environment
ENV ENVIRONMENT=testing \
    DEBUG=True

# Run tests by default
CMD ["poetry", "run", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html"]

# =============================================================================
# Build Arguments and Labels
# =============================================================================
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Non-Profit Engagement Model Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="nonprofit-engagement-model" \
      org.label-schema.description="BG/NBD model for predicting supporter engagement" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/nonprofit-engagement-model" \
      org.label-schema.schema-version="1.0"

# =============================================================================
# Security Hardening
# =============================================================================
# The following security measures are implemented:
# 1. Non-root user execution (UID 10001)
# 2. Minimal base image (slim-bookworm)
# 3. No unnecessary packages installed
# 4. Proper file permissions
# 5. Health checks enabled
# 6. Multi-stage build to reduce attack surface
# 7. No secrets in image layers
# 8. Proper signal handling
# =============================================================================