"""
Data extraction module for the non-profit engagement model.

This module handles extracting supporter actions and donations from Azure SQL Database
with proper parameterized queries, caching, and data validation for BG/NBD modeling.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..config import get_config, session_scope


logger = logging.getLogger(__name__)


class DataExtractionError(Exception):
    """Custom exception for data extraction errors."""
    pass


class DataExtractor:
    """Handles extraction of supporter data from Azure SQL Database."""
    
    def __init__(self):
        """Initialize the data extractor with configuration."""
        self.config = get_config()
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_enabled = self.config.dev_mode
        self.cache_expiry_hours = 24
        
        logger.info("DataExtractor initialized")
    
    def extract_supporter_actions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        supporter_ids: Optional[List[int]] = None,
        action_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract supporter actions from the database.
        
        Args:
            start_date: Start date for filtering actions
            end_date: End date for filtering actions
            supporter_ids: List of specific supporter IDs to extract
            action_types: List of action types to filter by
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: supporter_id, action_date, action_type, tags
            
        Raises:
            DataExtractionError: If extraction fails
        """
        cache_key = self._generate_cache_key(
            "actions", start_date, end_date, supporter_ids, action_types
        )
        
        # Try to load from cache first
        if use_cache and self.cache_enabled:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded supporter actions from cache: {cache_key}")
                return cached_data
        
        try:
            # Build parameterized query
            query, params = self._build_actions_query(
                start_date, end_date, supporter_ids, action_types
            )
            
            # Execute query
            with session_scope() as session:
                logger.info("Executing supporter actions query")
                result = session.execute(text(query), params)
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if df.empty:
                    logger.warning("No supporter actions found for the given criteria")
                    return pd.DataFrame(columns=['supporter_id', 'action_date', 'action_type', 'tags'])
                
                # Data type conversions
                df['supporter_id'] = df['supporter_id'].astype(int)
                df['action_date'] = pd.to_datetime(df['action_date'])
                df['action_type'] = df['action_type'].astype(str)
                df['tags'] = df['tags'].fillna('').astype(str)
                
                # Validate extracted data
                self._validate_actions_data(df)
                
                # Cache the results
                if self.cache_enabled:
                    self._save_to_cache(df, cache_key)
                
                logger.info(f"Extracted {len(df)} supporter actions")
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"Database error during actions extraction: {e}")
            raise DataExtractionError(f"Failed to extract supporter actions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during actions extraction: {e}")
            raise DataExtractionError(f"Unexpected error extracting supporter actions: {e}")
    
    def extract_donations(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        supporter_ids: Optional[List[int]] = None,
        min_amount: Optional[float] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract donation data from the database.
        
        Args:
            start_date: Start date for filtering donations
            end_date: End date for filtering donations
            supporter_ids: List of specific supporter IDs to extract
            min_amount: Minimum donation amount to include
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: supporter_id, donation_date, amount
            
        Raises:
            DataExtractionError: If extraction fails
        """
        cache_key = self._generate_cache_key(
            "donations", start_date, end_date, supporter_ids, min_amount
        )
        
        # Try to load from cache first
        if use_cache and self.cache_enabled:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded donations from cache: {cache_key}")
                return cached_data
        
        try:
            # Build parameterized query
            query, params = self._build_donations_query(
                start_date, end_date, supporter_ids, min_amount
            )
            
            # Execute query
            with session_scope() as session:
                logger.info("Executing donations query")
                result = session.execute(text(query), params)
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if df.empty:
                    logger.warning("No donations found for the given criteria")
                    return pd.DataFrame(columns=['supporter_id', 'donation_date', 'amount'])
                
                # Data type conversions
                df['supporter_id'] = df['supporter_id'].astype(int)
                df['donation_date'] = pd.to_datetime(df['donation_date'])
                df['amount'] = df['amount'].astype(float)
                
                # Validate extracted data
                self._validate_donations_data(df)
                
                # Cache the results
                if self.cache_enabled:
                    self._save_to_cache(df, cache_key)
                
                logger.info(f"Extracted {len(df)} donations")
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"Database error during donations extraction: {e}")
            raise DataExtractionError(f"Failed to extract donations: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during donations extraction: {e}")
            raise DataExtractionError(f"Unexpected error extracting donations: {e}")
    
    def extract_supporter_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract supporter summary statistics.
        
        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with supporter summary statistics
            
        Raises:
            DataExtractionError: If extraction fails
        """
        cache_key = self._generate_cache_key("summary", start_date, end_date)
        
        # Try to load from cache first
        if use_cache and self.cache_enabled:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded supporter summary from cache: {cache_key}")
                return cached_data
        
        try:
            # Build parameterized query for summary statistics
            query, params = self._build_summary_query(start_date, end_date)
            
            # Execute query
            with session_scope() as session:
                logger.info("Executing supporter summary query")
                result = session.execute(text(query), params)
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if df.empty:
                    logger.warning("No supporter summary data found")
                    return pd.DataFrame()
                
                # Data type conversions
                df['supporter_id'] = df['supporter_id'].astype(int)
                df['first_action_date'] = pd.to_datetime(df['first_action_date'])
                df['last_action_date'] = pd.to_datetime(df['last_action_date'])
                df['total_actions'] = df['total_actions'].astype(int)
                df['total_donations'] = df['total_donations'].fillna(0).astype(float)
                
                # Cache the results
                if self.cache_enabled:
                    self._save_to_cache(df, cache_key)
                
                logger.info(f"Extracted summary for {len(df)} supporters")
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"Database error during summary extraction: {e}")
            raise DataExtractionError(f"Failed to extract supporter summary: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during summary extraction: {e}")
            raise DataExtractionError(f"Unexpected error extracting supporter summary: {e}")
    
    def _build_actions_query(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        supporter_ids: Optional[List[int]],
        action_types: Optional[List[str]]
    ) -> Tuple[str, Dict]:
        """Build parameterized query for supporter actions."""
        base_query = """
        SELECT 
            signup_id AS supporter_id,
            nbuild_auscon.signup_taggings.created_at AS action_date,
            split_tag.action_type,
            name AS tags,
            supporter_level
        FROM nbuild_auscon.signup_taggings
        JOIN nbuild_auscon.signup_tags ON nbuild_auscon.signup_taggings.tag_id = nbuild_auscon.signup_tags.id
        CROSS APPLY lark.fn_split_tag(name) AS split_tag
        JOIN lark.supporter_level_action_role_tags ON name LIKE [prefix_name] + '%' AND lark.supporter_level_action_role_tags.tag_type = 'Action'
        WHERE 1=1
        """
        
        params = {}
        conditions = []
        
        if start_date:
            conditions.append("AND nbuild_auscon.signup_taggings.created_at >= :start_date")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("AND nbuild_auscon.signup_taggings.created_at <= :end_date")
            params['end_date'] = end_date
        
        if supporter_ids:
            # Handle list of IDs safely
            id_placeholders = ','.join([f':id_{i}' for i in range(len(supporter_ids))])
            conditions.append(f"AND signup_id IN ({id_placeholders})")
            for i, supporter_id in enumerate(supporter_ids):
                params[f'id_{i}'] = supporter_id
        
        if action_types:
            # Handle list of action types safely
            type_placeholders = ','.join([f':type_{i}' for i in range(len(action_types))])
            conditions.append(f"AND action_type IN ({type_placeholders})")
            for i, action_type in enumerate(action_types):
                params[f'type_{i}'] = action_type
        
        query = base_query + ' '.join(conditions) #+ " ORDER BY signup_id, nbuild_auscon.signup_taggings.created_at"
        
        return query, params
    
    def _build_donations_query(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        supporter_ids: Optional[List[int]],
        min_amount: Optional[float]
    ) -> Tuple[str, Dict]:
        """Build parameterized query for donations."""
        base_query = """
        SELECT 
            supporter_id,
            donation_date,
            amount
        FROM donations
        WHERE 1=1
        """
        
        params = {}
        conditions = []
        
        if start_date:
            conditions.append("AND donation_date >= :start_date")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("AND donation_date <= :end_date")
            params['end_date'] = end_date
        
        if supporter_ids:
            # Handle list of IDs safely
            id_placeholders = ','.join([f':id_{i}' for i in range(len(supporter_ids))])
            conditions.append(f"AND supporter_id IN ({id_placeholders})")
            for i, supporter_id in enumerate(supporter_ids):
                params[f'id_{i}'] = supporter_id
        
        if min_amount is not None:
            conditions.append("AND amount >= :min_amount")
            params['min_amount'] = min_amount
        
        query = base_query + ' '.join(conditions) + " ORDER BY supporter_id, donation_date"
        
        return query, params
    
    def _build_summary_query(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Tuple[str, Dict]:
        """Build parameterized query for supporter summary."""
        query = """
        SELECT 
            a.supporter_id,
            MIN(a.action_date) as first_action_date,
            MAX(a.action_date) as last_action_date,
            COUNT(a.action_date) as total_actions,
            COALESCE(SUM(d.amount), 0) as total_donations
        FROM supporter_level_action_role_tags a
        LEFT JOIN donations d ON a.supporter_id = d.supporter_id
        WHERE 1=1
        """
        
        params = {}
        conditions = []
        
        if start_date:
            conditions.append("AND a.action_date >= :start_date")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("AND a.action_date <= :end_date")
            params['end_date'] = end_date
        
        query += ' '.join(conditions)
        query += " GROUP BY a.supporter_id ORDER BY a.supporter_id"
        
        return query, params
    
    def _validate_actions_data(self, df: pd.DataFrame) -> None:
        """Validate supporter actions data quality."""
        required_columns = ['supporter_id', 'action_date', 'action_type', 'tags']
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataExtractionError(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        null_supporters = df['supporter_id'].isnull().sum()
        null_dates = df['action_date'].isnull().sum()
        null_types = df['action_type'].isnull().sum()
        
        if null_supporters > 0:
            raise DataExtractionError(f"Found {null_supporters} null supporter_id values")
        if null_dates > 0:
            raise DataExtractionError(f"Found {null_dates} null action_date values")
        if null_types > 0:
            raise DataExtractionError(f"Found {null_types} null action_type values")
        
        # Check date range validity
        if not df.empty:
            min_date = df['action_date'].min()
            max_date = df['action_date'].max()
            
            if min_date > datetime.now():
                logger.warning(f"Minimum action date is in the future: {min_date}")
            
            if (max_date - min_date).days > 3650:  # 10 years
                logger.warning(f"Action date range spans more than 10 years: {min_date} to {max_date}")
        
        logger.info("Actions data validation passed")
    
    def _validate_donations_data(self, df: pd.DataFrame) -> None:
        """Validate donations data quality."""
        required_columns = ['supporter_id', 'donation_date', 'amount']
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataExtractionError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_supporters = df['supporter_id'].isnull().sum()
        null_dates = df['donation_date'].isnull().sum()
        null_amounts = df['amount'].isnull().sum()
        
        if null_supporters > 0:
            raise DataExtractionError(f"Found {null_supporters} null supporter_id values")
        if null_dates > 0:
            raise DataExtractionError(f"Found {null_dates} null donation_date values")
        if null_amounts > 0:
            raise DataExtractionError(f"Found {null_amounts} null amount values")
        
        # Check for negative amounts
        negative_amounts = (df['amount'] < 0).sum()
        if negative_amounts > 0:
            logger.warning(f"Found {negative_amounts} negative donation amounts")
        
        # Check for unreasonably large amounts
        large_amounts = (df['amount'] > 1000000).sum()  # $1M threshold
        if large_amounts > 0:
            logger.warning(f"Found {large_amounts} donations over $1M")
        
        logger.info("Donations data validation passed")
    
    def _generate_cache_key(self, data_type: str, *args) -> str:
        """Generate a cache key based on extraction parameters."""
        key_parts = [data_type]
        
        for arg in args:
            if isinstance(arg, datetime):
                key_parts.append(arg.strftime('%Y%m%d'))
            elif isinstance(arg, list):
                key_parts.append('_'.join(map(str, sorted(arg))))
            elif arg is not None:
                key_parts.append(str(arg))
            else:
                key_parts.append('none')
        
        return '_'.join(key_parts)
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache with metadata."""
        try:
            cache_file = self.raw_data_dir / f"{cache_key}.pkl"
            cache_data = {
                'data': data,
                'timestamp': datetime.now(),
                'cache_key': cache_key
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Saved data to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save data to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        try:
            cache_file = self.raw_data_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check cache expiry
            cache_age = datetime.now() - cache_data['timestamp']
            if cache_age.total_seconds() > (self.cache_expiry_hours * 3600):
                logger.debug(f"Cache expired for key: {cache_key}")
                return None
            
            logger.debug(f"Loaded data from cache: {cache_file}")
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load data from cache: {e}")
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear cached data."""
        try:
            if cache_key:
                # Clear specific cache file
                cache_file = self.raw_data_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cleared cache: {cache_key}")
            else:
                # Clear all cache files
                for cache_file in self.raw_data_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("Cleared all cached data")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_data_quality_report(
        self,
        actions_df: pd.DataFrame,
        donations_df: pd.DataFrame
    ) -> Dict:
        """Generate a data quality report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'actions': {
                'total_records': len(actions_df),
                'unique_supporters': actions_df['supporter_id'].nunique() if not actions_df.empty else 0,
                'date_range': {
                    'min_date': actions_df['action_date'].min().isoformat() if not actions_df.empty else None,
                    'max_date': actions_df['action_date'].max().isoformat() if not actions_df.empty else None
                },
                'action_types': actions_df['action_type'].value_counts().to_dict() if not actions_df.empty else {},
                'null_values': {
                    'supporter_id': actions_df['supporter_id'].isnull().sum(),
                    'action_date': actions_df['action_date'].isnull().sum(),
                    'action_type': actions_df['action_type'].isnull().sum(),
                    'tags': actions_df['tags'].isnull().sum()
                } if not actions_df.empty else {}
            },
            'donations': {
                'total_records': len(donations_df),
                'unique_supporters': donations_df['supporter_id'].nunique() if not donations_df.empty else 0,
                'date_range': {
                    'min_date': donations_df['donation_date'].min().isoformat() if not donations_df.empty else None,
                    'max_date': donations_df['donation_date'].max().isoformat() if not donations_df.empty else None
                },
                'amount_stats': {
                    'total': donations_df['amount'].sum() if not donations_df.empty else 0,
                    'mean': donations_df['amount'].mean() if not donations_df.empty else 0,
                    'median': donations_df['amount'].median() if not donations_df.empty else 0,
                    'min': donations_df['amount'].min() if not donations_df.empty else 0,
                    'max': donations_df['amount'].max() if not donations_df.empty else 0
                },
                'null_values': {
                    'supporter_id': donations_df['supporter_id'].isnull().sum(),
                    'donation_date': donations_df['donation_date'].isnull().sum(),
                    'amount': donations_df['amount'].isnull().sum()
                } if not donations_df.empty else {}
            }
        }
        
        return report


def create_data_extractor() -> DataExtractor:
    """Factory function to create a DataExtractor instance."""
    return DataExtractor()