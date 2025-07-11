"""
Database module for the non-profit engagement model.

This module handles Azure SQL database connections using SQLAlchemy 2.0+
with proper connection pooling, retry logic, and error handling.
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from .settings import get_config


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.config = get_config().database
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the SQLAlchemy engine with connection pooling."""
        try:
            connection_string = self._build_connection_string()
            
            self._engine = create_engine(
                connection_string,
                # Connection pool settings
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Validate connections before use
                
                # Connection settings
                connect_args={
                    "timeout": self.config.connection_timeout,
                    "autocommit": False,
                },
                
                # Echo SQL queries in development
                echo=get_config().debug,
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build Azure SQL connection string with SSL configuration."""
        # URL encode password to handle special characters
        encoded_password = quote_plus(self.config.password)
        encoded_username = quote_plus(self.config.username)
        
        # Build connection string for Azure SQL with SSL parameters
        connection_string = (
            f"mssql+pyodbc://{encoded_username}:{encoded_password}"
            f"@{self.config.server}/{self.config.database}"
            f"?driver={quote_plus(self.config.driver)}"
            f"&Encrypt={self.config.encrypt}"
            f"&TrustServerCertificate={self.config.trust_server_certificate}"
            f"&Connection Timeout={self.config.connection_timeout}"
        )
        
        logger.debug("Connection string built successfully with SSL configuration")
        logger.debug(f"SSL settings - Encrypt: {self.config.encrypt}, TrustServerCertificate: {self.config.trust_server_certificate}")
        return connection_string
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")
        return self._engine
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection health with SSL-specific error handling."""
        try:
            with self.session_scope() as session:
                # Simple query to test connection
                result = session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                
                if test_value == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed: unexpected result")
                    return False
                    
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for SSL certificate errors
            if "certificate verify failed" in error_msg or "ssl" in error_msg:
                logger.error(f"SSL Certificate Error: {e}")
                logger.error("TROUBLESHOOTING SSL ISSUES:")
                logger.error("1. For development: Set AZURE_SQL_TRUST_SERVER_CERTIFICATE=yes in your .env file")
                logger.error("2. For production: Ensure proper SSL certificates are configured")
                logger.error("3. Check that AZURE_SQL_ENCRYPT=yes is set")
                logger.error("4. Verify your Azure SQL server allows the connection")
            else:
                logger.error(f"Database connection test failed: {e}")
            
            return False
    
    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return operation(*args, **kwargs)
                
            except SQLAlchemyError as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Database operation failed after {self.config.max_retries + 1} attempts")
        
        # Re-raise the last exception if all retries failed
        raise last_exception
    
    def get_connection_info(self) -> dict:
        """Get connection information for debugging."""
        return {
            "server": self.config.server,
            "database": self.config.database,
            "username": self.config.username,
            "driver": self.config.driver,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "connection_timeout": self.config.connection_timeout,
        }
    
    def close(self):
        """Close database connections and clean up resources."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


class DatabaseHealthCheck:
    """Database health check utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def check_connection(self) -> dict:
        """Perform comprehensive connection health check."""
        health_status = {
            "status": "unknown",
            "timestamp": time.time(),
            "connection_test": False,
            "pool_status": {},
            "error": None
        }
        
        try:
            # Test basic connection
            health_status["connection_test"] = self.db_manager.test_connection()
            
            # Get pool status
            pool = self.db_manager.engine.pool
            health_status["pool_status"] = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            
            # Overall status
            if health_status["connection_test"]:
                health_status["status"] = "healthy"
            else:
                health_status["status"] = "unhealthy"
                
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def log_health_status(self):
        """Log current health status."""
        status = self.check_connection()
        logger.info(f"Database health check: {status}")
        return status


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session() -> Session:
    """Get a new database session."""
    return get_database_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    with get_database_manager().session_scope() as session:
        yield session


def test_database_connection() -> bool:
    """Test database connection."""
    return get_database_manager().test_connection()


def get_database_health() -> dict:
    """Get database health status."""
    db_manager = get_database_manager()
    health_checker = DatabaseHealthCheck(db_manager)
    return health_checker.check_connection()


def close_database_connections():
    """Close all database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None