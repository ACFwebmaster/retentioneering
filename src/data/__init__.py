"""
Data processing module for the non-profit engagement model.

This module handles data extraction, transformation, and loading (ETL)
operations for supporter engagement data from Azure SQL Database.
"""

from .extraction import (
    DataExtractor,
    DataExtractionError,
    create_data_extractor,
)

from .preprocessing import (
    BGNBDDataProcessor,
    DataPreprocessingError,
    create_bgnbd_processor,
    process_supporter_data_pipeline,
)

# Convenience imports for common usage
__all__ = [
    # Extraction classes and functions
    "DataExtractor",
    "DataExtractionError", 
    "create_data_extractor",
    # Preprocessing classes and functions
    "BGNBDDataProcessor",
    "DataPreprocessingError",
    "create_bgnbd_processor",
    "process_supporter_data_pipeline",
]

# Version information
__version__ = "0.1.0"