"""
Fixed SixMonthRetentionProcessor - resolves IntCastingNaNError and schema issues.

This implementation fixes the critical issues with the retentioneering integration
by properly handling schema objects and ensuring data type consistency.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from retentioneering.eventstream import Eventstream

logger = logging.getLogger(__name__)


class SixMonthRetentionProcessor:
    """
    Fixed implementation of six-month retention calculation using retentioneering.
    
    This processor calculates supporter retention by:
    1. Filtering supporters who were active in a reference period
    2. Checking their activity in a following period
    3. Creating labeled eventstreams for visualization
    
    Key fixes:
    - Proper schema handling to avoid 'list' object errors
    - Data type validation to prevent IntCastingNaNError
    - Robust NaN handling throughout the pipeline
    """
    
    def __init__(self, eventstream: Eventstream):
        """
        Initialize the retention processor.
        
        Args:
            eventstream: Retentioneering Eventstream object with supporter data
        """
        self.eventstream = eventstream
        self.excluded_action_types = ['is_regular_giver', 'is_mid_value_donor']
        
    def calculate_metric_1(
        self,
        last_period_start: datetime,
        last_period_end: datetime,
        following_period_start: datetime,
        following_period_end: datetime
    ) -> Dict[str, Any]:
        """
        Calculate 6-month retention metric.
        
        Args:
            last_period_start: Start of reference period
            last_period_end: End of reference period
            following_period_start: Start of following period
            following_period_end: End of following period
            
        Returns:
            Dictionary containing retention metrics and labeled eventstream
        """
        logger.info("Starting 6-month retention calculation")
        
        # Store period metadata for labeling logic
        self.analysis_periods = {
            'reference_period': {
                'start': last_period_start,
                'end': last_period_end
            },
            'following_period': {
                'start': following_period_start,
                'end': following_period_end
            }
        }

        try:
            # Step 1: Filter eventstream for analysis period
            filtered_stream = self._filter_eventstream_for_analysis(
                last_period_start, following_period_end
            )
            
            # Step 2: Identify supporters active in reference period
            reference_supporters = self._get_supporters_in_period(
                filtered_stream, last_period_start, last_period_end
            )
            
            # Step 3: Identify supporters active in following period
            following_supporters = self._get_supporters_in_period(
                filtered_stream, following_period_start, following_period_end
            )
            
            # Step 4: Calculate retention
            retained_supporters = set(reference_supporters) & set(following_supporters)
            non_retained_supporters = set(reference_supporters) - retained_supporters
            
            retention_rate = (len(retained_supporters) / len(reference_supporters) * 100 
                            if reference_supporters else 0)
            
            # Step 5: Create simple labeled eventstream (without retentioneering processors)
            labeled_stream = self._create_simple_labeled_eventstream(
                filtered_stream, retained_supporters, non_retained_supporters
            )
            
            result = {
                'retention_rate': retention_rate,
                'supporters_retained': len(retained_supporters),
                'supporters_not_retained': len(non_retained_supporters),
                'total_supporters_last_period': len(reference_supporters),
                'retained_supporter_ids': list(retained_supporters),
                'non_retained_supporter_ids': list(non_retained_supporters),
                'eventstream_with_labels': labeled_stream,
                'analysis_periods': {
                    'reference_period': {
                        'start': last_period_start,
                        'end': last_period_end
                    },
                    'following_period': {
                        'start': following_period_start,
                        'end': following_period_end
                    }
                }
            }
            
            logger.info(f"Retention calculation completed: {retention_rate:.1f}% retention rate")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating retention metric: {e}")
            raise
    
    def _filter_eventstream_for_analysis(
        self, 
        analysis_start: datetime, 
        analysis_end: datetime
    ) -> Eventstream:
        """
        Filter eventstream to analysis period and exclude status events.
        
        Args:
            analysis_start: Start of analysis period
            analysis_end: End of analysis period
            
        Returns:
            Filtered Eventstream
        """
        logger.debug("Filtering eventstream for analysis period")
        
        # Get the dataframe from eventstream
        df = self.eventstream.to_dataframe().copy()
        
        # Get schema information
        schema = self.eventstream.schema
        event_col = schema.event_name
        timestamp_col = schema.event_timestamp
        user_col = schema.user_id
        
        # Filter by date range
        df_filtered = df[
            (df[timestamp_col] >= analysis_start) & 
            (df[timestamp_col] <= analysis_end)
        ].copy()
        
        # Exclude status-type events
        df_filtered = df_filtered[
            ~df_filtered[event_col].isin(self.excluded_action_types)
        ].copy()
        
        # Ensure data types are correct to prevent IntCastingNaNError
        df_filtered = self._ensure_clean_data_types(df_filtered, schema)
        
        logger.debug(f"Filtered to {len(df_filtered)} events from {df_filtered[user_col].nunique()} supporters")
        
        # Create new eventstream with filtered data
        return Eventstream(df_filtered, schema=schema, add_start_end_events=False)
    
    def _ensure_clean_data_types(self, df: pd.DataFrame, schema) -> pd.DataFrame:
        """
        Ensure data types are clean and won't cause IntCastingNaNError.
        
        Args:
            df: DataFrame to clean
            schema: Eventstream schema
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Get column names from schema
        user_col = schema.user_id
        event_col = schema.event_name
        timestamp_col = schema.event_timestamp
        
        # Remove rows with NaN user_ids (critical for retentioneering)
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[user_col])
        dropped_user_rows = initial_rows - len(cleaned_df)
        if dropped_user_rows > 0:
            logger.warning(f"Dropped {dropped_user_rows} rows with NaN user_ids")
        
        # Remove rows with NaN event names
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[event_col])
        dropped_event_rows = initial_rows - len(cleaned_df)
        if dropped_event_rows > 0:
            logger.warning(f"Dropped {dropped_event_rows} rows with NaN event names")
        
        # Ensure user_id is proper type (not float with NaN)
        if cleaned_df[user_col].dtype == 'float64':
            # Convert to int if no decimals
            if cleaned_df[user_col].apply(lambda x: x.is_integer() if pd.notna(x) else False).all():
                cleaned_df[user_col] = cleaned_df[user_col].astype('int64')
            else:
                # Convert to string if mixed types
                cleaned_df[user_col] = cleaned_df[user_col].astype('str')
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(cleaned_df[timestamp_col]):
            cleaned_df[timestamp_col] = pd.to_datetime(cleaned_df[timestamp_col])
        
        # Ensure event names are strings
        cleaned_df[event_col] = cleaned_df[event_col].astype('str')
        
        return cleaned_df
    
    def _get_supporters_in_period(
        self, 
        eventstream: Eventstream, 
        start_date: datetime, 
        end_date: datetime
    ) -> List:
        """
        Get list of supporters who had activity in specified period.
        
        Args:
            eventstream: Filtered eventstream
            start_date: Period start date
            end_date: Period end date
            
        Returns:
            List of supporter IDs active in period
        """
        df = eventstream.to_dataframe()
        schema = eventstream.schema
        
        # Filter to period
        period_df = df[
            (df[schema.event_timestamp] >= start_date) & 
            (df[schema.event_timestamp] <= end_date)
        ]
        
        # Get unique supporters
        supporters = period_df[schema.user_id].unique().tolist()
        
        logger.debug(f"Found {len(supporters)} supporters active between {start_date.date()} and {end_date.date()}")
        return supporters
    
    def _create_simple_labeled_eventstream(
        self,
        stream: Eventstream,
        retained_supporters: set,
        non_retained_supporters: set
    ) -> Eventstream:
        """
        Create a labeled eventstream by adding a 'retained' or 'not_retained' event 
        based on supporter activity in the following period.
        """
        logger.debug("Creating simple labeled eventstream")
        
        df = stream.to_dataframe().copy()
        schema = stream.schema
        
        # Ensure clean types before filtering
        df = self._ensure_clean_data_types(df, schema)
        
        # Create retention events manually
        retention_events = []

        # Extract timestamp boundaries from analysis_periods if available
        following_period_start = self.analysis_periods["following_period"]["start"]
        following_period_end = self.analysis_periods["following_period"]["end"]
        
        # Only need events from the following period for retained supporters
        following_period_df = df[
            (df[schema.event_timestamp] >= following_period_start) &
            (df[schema.event_timestamp] <= following_period_end)
        ]

        # Get first event per retained supporter in the following period
        first_following_actions = (
            following_period_df[following_period_df[schema.user_id].isin(retained_supporters)]
            .sort_values(by=[schema.user_id, schema.event_timestamp])
            .groupby(schema.user_id)
            .first()
            .reset_index()
        )

        for _, row in first_following_actions.iterrows():
            retention_events.append({
                schema.user_id: row[schema.user_id],
                schema.event_name: 'retained',
                schema.event_timestamp: row[schema.event_timestamp]
            })

        # Add artificial event just after the last event for non-retained supporters
        full_df_last_event_time = df.groupby(schema.user_id)[schema.event_timestamp].max()
        
        for supporter_id in non_retained_supporters:
            if supporter_id in full_df_last_event_time:
                retention_events.append({
                    schema.user_id: supporter_id,
                    schema.event_name: 'not_retained',
                    schema.event_timestamp: full_df_last_event_time[supporter_id] + timedelta(seconds=1)
                })

        # Combine events
        if retention_events:
            retention_df = pd.DataFrame(retention_events)
            combined_df = pd.concat([df, retention_df], ignore_index=True)
        else:
            combined_df = df

        # Sort and clean
        combined_df = combined_df.sort_values(schema.event_timestamp).reset_index(drop=True)
        combined_df = self._ensure_clean_data_types(combined_df, schema)

        logger.debug(f"Created labeled eventstream with {len(retention_events)} retention events")

        return Eventstream(combined_df, schema=schema, add_start_end_events=False).add_positive_events(targets=['retained'])


def create_sample_retention_data(n_supporters: int = 200) -> pd.DataFrame:
    """
    Create sample supporter data for testing retention calculations.
    
    Args:
        n_supporters: Number of supporters to generate
        
    Returns:
        DataFrame with supporter action data
    """
    np.random.seed(42)
    
    action_types = ['donation', 'petition_signed', 'event_attended', 'newsletter_signup', 'volunteer_signup']
    excluded_statuses = ['is_regular_giver', 'is_mid_value_donor']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for supporter_id in range(1, n_supporters + 1):
        n_actions = np.random.poisson(4) + 1
        supporter_start = start_date + timedelta(days=np.random.randint(0, 365))
        
        for action_num in range(n_actions):
            days_offset = np.random.exponential(30) * action_num
            action_date = supporter_start + timedelta(days=int(days_offset))
            
            # Mix of real actions and status attributes
            if np.random.random() < 0.1:
                action_type = np.random.choice(excluded_statuses)
            else:
                action_type = np.random.choice(action_types)
            
            data.append({
                'supporter_id': f'supporter_{supporter_id:03d}',
                'action_type': action_type,
                'action_date': action_date
            })
    
    return pd.DataFrame(data)


# Example usage function
def run_retention_analysis_example():
    """
    Example of how to run the fixed retention analysis.
    """
    # Generate sample data
    supporter_data = create_sample_retention_data(200)
    
    # Create eventstream
    raw_data_schema = {
        'user_id': 'supporter_id',
        'event_name': 'action_type',
        'event_timestamp': 'action_date'
    }
    
    eventstream = Eventstream(supporter_data, raw_data_schema=raw_data_schema)
    
    # Initialize processor
    processor = SixMonthRetentionProcessor(eventstream)
    
    # Define analysis periods
    analysis_date = datetime(2024, 1, 1)
    last_period_start = analysis_date - timedelta(days=180)
    last_period_end = analysis_date
    following_period_start = analysis_date
    following_period_end = analysis_date + timedelta(days=180)
    
    # Calculate retention
    result = processor.calculate_metric_1(
        last_period_start=last_period_start,
        last_period_end=last_period_end,
        following_period_start=following_period_start,
        following_period_end=following_period_end
    )
    
    return result