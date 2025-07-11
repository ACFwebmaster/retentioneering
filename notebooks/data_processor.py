"""
Standalone data processor for notebook demonstrations.
This module provides BG/NBD data processing without database dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class SimpleDataProcessor:
    """Simple data processor for converting sample data to BG/NBD format."""
    
    def __init__(self):
        """Initialize the simple data processor."""
        pass
    
    def prepare_bgnbd_data(
        self,
        actions_df: pd.DataFrame,
        donations_df: pd.DataFrame,
        observation_period_days: int = 365,
        min_actions: int = 1
    ) -> pd.DataFrame:
        """
        Convert actions and donations data to BG/NBD format.
        
        Args:
            actions_df: DataFrame with supporter actions
            donations_df: DataFrame with supporter donations
            observation_period_days: Length of observation period in days
            min_actions: Minimum number of actions required per supporter
            
        Returns:
            DataFrame with BG/NBD variables (supporter_id, frequency, recency, T)
        """
        # Combine actions and donations into events
        events_df = self._combine_events(actions_df, donations_df)
        
        if events_df.empty:
            return self._create_empty_bgnbd_dataframe()
        
        # Calculate observation period
        end_date = events_df['event_date'].max()
        start_date = end_date - timedelta(days=observation_period_days)
        
        # Filter events within observation period
        obs_events = events_df[
            (events_df['event_date'] >= start_date) &
            (events_df['event_date'] <= end_date)
        ].copy()
        
        # Calculate BG/NBD variables for each supporter
        bgnbd_data = []
        
        for supporter_id, supporter_events in obs_events.groupby('supporter_id'):
            supporter_events = supporter_events.sort_values('event_date')
            
            if len(supporter_events) < min_actions:
                continue
            
            # Calculate days from start for each event
            supporter_events['days_from_start'] = (
                supporter_events['event_date'] - start_date
            ).dt.days
            
            # BG/NBD variables
            frequency = len(supporter_events)  # Total number of events
            x = frequency - 1  # Number of repeat events (excluding first)
            
            if x > 0:
                recency = supporter_events['days_from_start'].iloc[-1]  # Last event timing
            else:
                recency = 0
            
            T = observation_period_days  # Observation period length
            
            # Additional metrics
            total_donations = supporter_events[
                supporter_events['event_type'] == 'donation'
            ]['amount'].sum()
            
            bgnbd_data.append({
                'supporter_id': supporter_id,
                'frequency': frequency,
                'recency': recency,
                'T': T,
                'x': x,
                't_x': recency,  # Same as recency for compatibility
                'monetary': total_donations,
                'first_event_date': supporter_events['event_date'].min(),
                'last_event_date': supporter_events['event_date'].max(),
                'observation_start': start_date,
                'observation_end': end_date
            })
        
        if not bgnbd_data:
            return self._create_empty_bgnbd_dataframe()
        
        bgnbd_df = pd.DataFrame(bgnbd_data)
        
        # Add engagement segments
        bgnbd_df = self._add_engagement_segments(bgnbd_df)
        
        return bgnbd_df
    
    def _combine_events(
        self, 
        actions_df: pd.DataFrame, 
        donations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine actions and donations into unified event stream."""
        events = []
        
        # Process actions
        if not actions_df.empty:
            actions_events = actions_df.copy()
            actions_events['event_date'] = pd.to_datetime(actions_events['action_date'])
            actions_events['event_type'] = actions_events['action_type']
            actions_events['amount'] = 0.0
            
            events.append(actions_events[['supporter_id', 'event_date', 'event_type', 'amount']])
        
        # Process donations
        if not donations_df.empty:
            donation_events = donations_df.copy()
            donation_events['event_date'] = pd.to_datetime(donation_events['donation_date'])
            donation_events['event_type'] = 'donation'
            
            events.append(donation_events[['supporter_id', 'event_date', 'event_type', 'amount']])
        
        if not events:
            return pd.DataFrame(columns=['supporter_id', 'event_date', 'event_type', 'amount'])
        
        # Combine and sort
        combined_df = pd.concat(events, ignore_index=True)
        combined_df = combined_df.sort_values(['supporter_id', 'event_date'])
        
        return combined_df
    
    def _add_engagement_segments(self, bgnbd_df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement segments based on frequency and monetary value."""
        if bgnbd_df.empty:
            return bgnbd_df
        
        # Calculate engagement score
        freq_norm = bgnbd_df['frequency'] / bgnbd_df['frequency'].max() if bgnbd_df['frequency'].max() > 0 else 0
        monetary_norm = bgnbd_df['monetary'] / bgnbd_df['monetary'].max() if bgnbd_df['monetary'].max() > 0 else 0
        
        bgnbd_df['engagement_score'] = 0.6 * freq_norm + 0.4 * monetary_norm
        
        # Assign segments based on quartiles
        q25 = bgnbd_df['engagement_score'].quantile(0.25)
        q50 = bgnbd_df['engagement_score'].quantile(0.50)
        q75 = bgnbd_df['engagement_score'].quantile(0.75)
        
        def assign_segment(score):
            if score >= q75:
                return 'High'
            elif score >= q50:
                return 'Medium'
            elif score >= q25:
                return 'Low'
            else:
                return 'Inactive'
        
        bgnbd_df['engagement_segment'] = bgnbd_df['engagement_score'].apply(assign_segment)
        
        return bgnbd_df
    
    def _create_empty_bgnbd_dataframe(self) -> pd.DataFrame:
        """Create empty BG/NBD DataFrame with proper columns."""
        return pd.DataFrame(columns=[
            'supporter_id', 'frequency', 'recency', 'T', 'x', 't_x', 'monetary',
            'first_event_date', 'last_event_date', 'observation_start', 
            'observation_end', 'engagement_score', 'engagement_segment'
        ])
    
    def get_summary_statistics(self, bgnbd_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for BG/NBD data."""
        if bgnbd_df.empty:
            return {'error': 'No data to summarize'}
        
        return {
            'total_supporters': len(bgnbd_df),
            'frequency_stats': {
                'mean': float(bgnbd_df['frequency'].mean()),
                'median': float(bgnbd_df['frequency'].median()),
                'std': float(bgnbd_df['frequency'].std()),
                'min': int(bgnbd_df['frequency'].min()),
                'max': int(bgnbd_df['frequency'].max())
            },
            'recency_stats': {
                'mean': float(bgnbd_df['recency'].mean()),
                'median': float(bgnbd_df['recency'].median()),
                'std': float(bgnbd_df['recency'].std()),
                'min': float(bgnbd_df['recency'].min()),
                'max': float(bgnbd_df['recency'].max())
            },
            'monetary_stats': {
                'total_donations': float(bgnbd_df['monetary'].sum()),
                'mean_donation': float(bgnbd_df['monetary'].mean()),
                'donors_count': int((bgnbd_df['monetary'] > 0).sum()),
                'donor_percentage': float((bgnbd_df['monetary'] > 0).mean() * 100)
            },
            'engagement_segments': bgnbd_df['engagement_segment'].value_counts().to_dict()
        }