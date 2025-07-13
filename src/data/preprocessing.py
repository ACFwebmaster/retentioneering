"""
Data preprocessing module for the non-profit engagement model.

This module transforms supporter action and donation data into the format required
for BG/NBD modeling, calculating engagement variables (x, t_x, T) and handling
data quality issues.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config
from .extraction import DataExtractor, create_data_extractor


logger = logging.getLogger(__name__)


class DataPreprocessingError(Exception):
    """Custom exception for data preprocessing errors."""
    pass


class BGNBDDataProcessor:
    """Processes supporter data into BG/NBD model format."""
    
    def __init__(self, data_extractor: Optional[DataExtractor] = None):
        """
        Initialize the BG/NBD data processor.
        
        Args:
            data_extractor: Optional DataExtractor instance. If None, creates a new one.
        """
        self.config = get_config()
        self.data_extractor = data_extractor or create_data_extractor()
        
        # Output directories
        self.processed_data_dir = Path("data/processed")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters - engagement weights based on supporter_level
        # supporter_level values: 1, 2, 3, 4 (from the action data)
        self.engagement_weights = {
            'donation': 2.0,  # Keep donation separate as it's not an action type
            'default': 1.0    # Default fallback weight
        }
        
        # Action type prefix to engagement level mapping
        # The actual weight comes from supporter_level column in the data
        self.action_type_prefixes = {
            'A-PET': 'petition',
            'A-SU-': 'signup',
            'A-EV_RSVP': 'event_rsvp',
            'A-VOL_EOI': 'volunteer_eoi',
            'A-SU_ValX': 'signup_valx',
            'A-EV_Attendee': 'event_attendee',
            'A-ADV_SubmissionPF': 'advocacy_submission_pf',
            'A-ADV_LobbyPF': 'advocacy_lobby_pf',
            'A-PLE': 'pledge',
            'A-SMP': 'sample',
            'A-SC': 'social_connect',
            'A-CMP': 'campaign',
            'A-SUR': 'survey',
            'A-SU_Connect': 'signup_connect',
            'A-VOL_Slack': 'volunteer_slack',
            'A-ADV_Submission-': 'advocacy_submission',
            'A-ADV_Lobby-': 'advocacy_lobby',
            'A-ADV_LetterToEditor': 'advocacy_letter',
            'A-RB': 'response_boost',
            'A-ADV_LobbyCalls': 'advocacy_calls',
            'A-EV_Host': 'event_host',
            'A-VOL_Volunteered': 'volunteer_active',
            'A-TND': 'thank_donate',
            'A-ADV_Meeting': 'advocacy_meeting',
            'A-CC': 'community_connect',
            'A-NOM': 'nomination',
            'A-TIP': 'tip',
            'A-ADV_Approach': 'advocacy_approach',
            'A-VOL_ModeratesSocial': 'volunteer_moderate',
            'A-VOL_SupporterBaseBroadcasted': 'volunteer_broadcast',
            'A-TNR': 'thank_refer',
            'A-VOL_ProvidedStrategicInput': 'volunteer_strategic'
        }
        
        # Cache settings
        self.cache_enabled = self.config.dev_mode
        
        logger.info("BGNBDDataProcessor initialized")
    
    def process_supporter_data(
        self,
        start_date: datetime,
        end_date: datetime,
        cutoff_date: Optional[datetime] = None,
        min_actions: int = 1,
        include_donations: bool = True,
        supporter_ids: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Process supporter data into BG/NBD format.
        
        Args:
            start_date: Start of observation period
            end_date: End of observation period
            cutoff_date: Analysis cutoff date (for calculating recency)
            min_actions: Minimum number of actions required per supporter
            include_donations: Whether to include donations as engagement events
            supporter_ids: Optional list of specific supporter IDs to process
            use_cache: Whether to use cached processed data
            
        Returns:
            DataFrame with BG/NBD variables: supporter_id, x, t_x, T, frequency, monetary
            
        Raises:
            DataPreprocessingError: If processing fails
        """
        if cutoff_date is None:
            cutoff_date = end_date
        
        # Validate date parameters
        self._validate_date_parameters(start_date, end_date, cutoff_date)
        
        # Check cache first
        cache_key = self._generate_cache_key(
            "bgnbd", start_date, end_date, cutoff_date, min_actions, 
            include_donations, supporter_ids
        )
        
        if use_cache and self.cache_enabled:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded processed BG/NBD data from cache: {cache_key}")
                return cached_data
        
        try:
            # Extract raw data
            logger.info("Extracting supporter actions and donations")
            actions_df = self.data_extractor.extract_supporter_actions(
                start_date=start_date,
                end_date=end_date,
                supporter_ids=supporter_ids,
                use_cache=use_cache
            )
            
            donations_df = pd.DataFrame()
            if include_donations:
                donations_df = self.data_extractor.extract_donations(
                    start_date=start_date,
                    end_date=end_date,
                    supporter_ids=supporter_ids,
                    use_cache=use_cache
                )
            
            # Combine actions and donations into unified event stream
            events_df = self._combine_events(actions_df, donations_df)
            
            if events_df.empty:
                logger.warning("No events found for processing")
                return self._create_empty_bgnbd_dataframe()
            
            # Calculate BG/NBD variables
            bgnbd_df = self._calculate_bgnbd_variables(
                events_df, start_date, end_date, cutoff_date, min_actions
            )
            
            # Add engagement scoring
            bgnbd_df = self._add_engagement_metrics(bgnbd_df, events_df)
            
            # Apply data quality filters
            bgnbd_df = self._apply_quality_filters(bgnbd_df, min_actions)
            
            # Cache the results
            if self.cache_enabled:
                self._save_to_cache(bgnbd_df, cache_key)
            
            logger.info(f"Processed {len(bgnbd_df)} supporters into BG/NBD format")
            return bgnbd_df
            
        except Exception as e:
            logger.error(f"Error processing supporter data: {e}")
            raise DataPreprocessingError(f"Failed to process supporter data: {e}")
    
    def _combine_events(
        self, 
        actions_df: pd.DataFrame, 
        donations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine actions and donations into a unified event stream."""
        events = []
        
        # Process actions
        if not actions_df.empty:
            actions_events = actions_df.copy()
            actions_events['event_date'] = actions_events['action_date']
            actions_events['event_type'] = actions_events['action_type']
            actions_events['event_value'] = 1.0  # Base weight
            actions_events['is_donation'] = False
            actions_events['amount'] = 0.0
            
            events.append(actions_events[[
                'supporter_id', 'event_date', 'event_type', 'event_value', 
                'is_donation', 'amount', 'tags'
            ]])
        
        # Process donations
        if not donations_df.empty:
            donation_events = donations_df.copy()
            donation_events['event_date'] = donation_events['donation_date']
            donation_events['event_type'] = 'donation'
            donation_events['event_value'] = self.engagement_weights.get('donation', 2.0)
            donation_events['is_donation'] = True
            donation_events['tags'] = ''
            
            events.append(donation_events[[
                'supporter_id', 'event_date', 'event_type', 'event_value', 
                'is_donation', 'amount', 'tags'
            ]])
        
        if not events:
            return pd.DataFrame(columns=[
                'supporter_id', 'event_date', 'event_type', 'event_value', 
                'is_donation', 'amount', 'tags'
            ])
        
        # Combine all events
        combined_df = pd.concat(events, ignore_index=True)
        
        # Sort by supporter and date
        combined_df = combined_df.sort_values(['supporter_id', 'event_date'])
        
        # Apply engagement weights based on event type and supporter_level
        # Note: supporter_level should be available in the actions_df data
        combined_df['weighted_value'] = combined_df.apply(
            lambda row: self._calculate_event_weight(
                row['event_type'],
                row['amount'],
                row.get('supporter_level', 1)  # Default to 1 if not available
            ),
            axis=1
        )
        
        logger.info(f"Combined {len(combined_df)} events from {combined_df['supporter_id'].nunique()} supporters")
        return combined_df
    
    def _calculate_event_weight(self, event_type: str, amount: float = 0.0, supporter_level: int = 1) -> float:
        """
        Calculate weighted value for an event based on supporter_level.
        
        Args:
            event_type: The type of event/action
            amount: Donation amount (if applicable)
            supporter_level: The engagement level from the data (1, 2, 3, 4)
        
        Returns:
            Weighted value for the event
        """
        # For donations, use the donation weight plus amount-based weighting
        if event_type.lower() == 'donation' and amount > 0:
            base_weight = self.engagement_weights.get('donation', 2.0)
            # Log scale for donation amounts to prevent extreme outliers
            amount_weight = np.log10(max(amount, 1)) / 3  # Normalize to reasonable scale
            return base_weight + amount_weight
        
        # For actions, use the supporter_level as the weight
        # supporter_level comes from the data and represents engagement intensity
        return float(supporter_level) if supporter_level > 0 else self.engagement_weights['default']
    
    def _calculate_bgnbd_variables(
        self,
        events_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        cutoff_date: datetime,
        min_actions: int
    ) -> pd.DataFrame:
        """Calculate BG/NBD variables (x, t_x, T) for each supporter."""
        # Calculate observation period length in days
        T = (end_date - start_date).days
        
        # Group events by supporter
        supporter_stats = []
        
        for supporter_id, supporter_events in events_df.groupby('supporter_id'):
            # Filter events within observation period
            obs_events = supporter_events[
                (supporter_events['event_date'] >= start_date) &
                (supporter_events['event_date'] <= cutoff_date)
            ].copy()
            
            if len(obs_events) < min_actions:
                continue
            
            # Calculate days from start for each event
            obs_events['days_from_start'] = (
                obs_events['event_date'] - start_date
            ).dt.days
            
            # BG/NBD variables
            x = len(obs_events) - 1  # Number of repeat events (excluding first)
            
            if x > 0:
                t_x = obs_events['days_from_start'].iloc[-1]  # Recency of last event
            else:
                t_x = 0
            
            # Additional metrics
            first_event_date = obs_events['event_date'].min()
            last_event_date = obs_events['event_date'].max()
            total_weighted_value = obs_events['weighted_value'].sum()
            total_donation_amount = obs_events[obs_events['is_donation']]['amount'].sum()
            
            supporter_stats.append({
                'supporter_id': supporter_id,
                'x': x,
                't_x': t_x,
                'T': T,
                'frequency': len(obs_events),
                'first_event_date': first_event_date,
                'last_event_date': last_event_date,
                'total_weighted_value': total_weighted_value,
                'total_donation_amount': total_donation_amount,
                'observation_start': start_date,
                'observation_end': end_date,
                'cutoff_date': cutoff_date
            })
        
        if not supporter_stats:
            return self._create_empty_bgnbd_dataframe()
        
        bgnbd_df = pd.DataFrame(supporter_stats)
        
        # Data type conversions
        bgnbd_df['supporter_id'] = bgnbd_df['supporter_id'].astype(int)
        bgnbd_df['x'] = bgnbd_df['x'].astype(int)
        bgnbd_df['t_x'] = bgnbd_df['t_x'].astype(float)
        bgnbd_df['T'] = bgnbd_df['T'].astype(int)
        bgnbd_df['frequency'] = bgnbd_df['frequency'].astype(int)
        
        logger.info(f"Calculated BG/NBD variables for {len(bgnbd_df)} supporters")
        return bgnbd_df
    
    def _add_engagement_metrics(
        self,
        bgnbd_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add additional engagement metrics to the BG/NBD data."""
        if bgnbd_df.empty:
            return bgnbd_df
        
        try:
            # Validate required columns exist
            required_columns = ['t_x', 'T', 'total_donation_amount', 'frequency', 'supporter_id']
            missing_columns = [col for col in required_columns if col not in bgnbd_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for engagement metrics: {missing_columns}")
                raise DataPreprocessingError(f"Missing required columns: {missing_columns}")
            
            # Calculate recency ratio first (needed for engagement score)
            bgnbd_df['recency_ratio'] = bgnbd_df['t_x'] / bgnbd_df['T']
            
            # Calculate monetary metrics
            bgnbd_df['monetary'] = bgnbd_df['total_donation_amount']
            bgnbd_df['avg_donation'] = bgnbd_df.apply(
                lambda row: row['total_donation_amount'] / max(row['frequency'], 1), axis=1
            )
            
            # Calculate engagement scores and segments (now that recency_ratio exists)
            bgnbd_df['engagement_score'] = self._calculate_engagement_score(bgnbd_df)
            bgnbd_df['engagement_segment'] = self._assign_engagement_segment(bgnbd_df)
            
            # Add event diversity metrics
            diversity_metrics = self._calculate_event_diversity(events_df, bgnbd_df['supporter_id'].tolist())
            bgnbd_df = bgnbd_df.merge(diversity_metrics, on='supporter_id', how='left')
            
            logger.debug(f"Added engagement metrics for {len(bgnbd_df)} supporters")
            return bgnbd_df
            
        except Exception as e:
            logger.error(f"Error adding engagement metrics: {e}")
            raise DataPreprocessingError(f"Failed to add engagement metrics: {e}")
    
    def _calculate_engagement_score(self, bgnbd_df: pd.DataFrame) -> pd.Series:
        """Calculate engagement score based on frequency, recency, and monetary value."""
        if bgnbd_df.empty:
            return pd.Series(dtype=float)
        
        # Validate required columns exist
        required_columns = ['frequency', 'recency_ratio', 'total_donation_amount']
        missing_columns = [col for col in required_columns if col not in bgnbd_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for engagement score calculation: {missing_columns}")
            raise DataPreprocessingError(f"Missing required columns for engagement score: {missing_columns}")
        
        try:
            # Normalize components to 0-1 scale
            freq_max = bgnbd_df['frequency'].max()
            freq_norm = bgnbd_df['frequency'] / freq_max if freq_max > 0 else pd.Series(0, index=bgnbd_df.index)
            
            recency_norm = bgnbd_df['recency_ratio']
            
            monetary_max = bgnbd_df['total_donation_amount'].max()
            monetary_norm = bgnbd_df['total_donation_amount'] / monetary_max if monetary_max > 0 else pd.Series(0, index=bgnbd_df.index)
            
            # Weighted combination
            engagement_score = (
                0.4 * freq_norm +
                0.3 * recency_norm +
                0.3 * monetary_norm
            )
            
            logger.debug(f"Calculated engagement scores with range: {engagement_score.min():.3f} - {engagement_score.max():.3f}")
            return engagement_score
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            raise DataPreprocessingError(f"Failed to calculate engagement score: {e}")
    
    def _assign_engagement_segment(self, bgnbd_df: pd.DataFrame) -> pd.Series:
        """Assign engagement segments based on engagement score."""
        if bgnbd_df.empty:
            return pd.Series(dtype=str)
        
        # Define segments based on engagement score quartiles
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
        
        return bgnbd_df['engagement_score'].apply(assign_segment)
    
    def _calculate_event_diversity(
        self, 
        events_df: pd.DataFrame, 
        supporter_ids: List[int]
    ) -> pd.DataFrame:
        """Calculate event type diversity metrics for supporters."""
        diversity_stats = []
        
        for supporter_id in supporter_ids:
            supporter_events = events_df[events_df['supporter_id'] == supporter_id]
            
            if supporter_events.empty:
                diversity_stats.append({
                    'supporter_id': supporter_id,
                    'event_type_count': 0,
                    'event_diversity_index': 0.0
                })
                continue
            
            # Count unique event types
            event_types = supporter_events['event_type'].value_counts()
            event_type_count = len(event_types)
            
            # Calculate Shannon diversity index
            total_events = len(supporter_events)
            diversity_index = 0.0
            
            if total_events > 1:
                for count in event_types.values:
                    proportion = count / total_events
                    if proportion > 0:
                        diversity_index -= proportion * np.log(proportion)
            
            diversity_stats.append({
                'supporter_id': supporter_id,
                'event_type_count': event_type_count,
                'event_diversity_index': diversity_index
            })
        
        return pd.DataFrame(diversity_stats)
    
    def _apply_quality_filters(
        self, 
        bgnbd_df: pd.DataFrame, 
        min_actions: int
    ) -> pd.DataFrame:
        """Apply data quality filters to the BG/NBD data."""
        if bgnbd_df.empty:
            return bgnbd_df
        
        initial_count = len(bgnbd_df)
        
        # Filter by minimum actions
        bgnbd_df = bgnbd_df[bgnbd_df['frequency'] >= min_actions]
        
        # Remove supporters with invalid recency (t_x > T)
        bgnbd_df = bgnbd_df[bgnbd_df['t_x'] <= bgnbd_df['T']]
        
        # Remove supporters with negative values
        bgnbd_df = bgnbd_df[
            (bgnbd_df['x'] >= 0) & 
            (bgnbd_df['t_x'] >= 0) & 
            (bgnbd_df['T'] > 0)
        ]
        
        final_count = len(bgnbd_df)
        filtered_count = initial_count - final_count
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} supporters due to quality issues")
        
        return bgnbd_df
    
    def _validate_date_parameters(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        cutoff_date: datetime
    ) -> None:
        """Validate date parameters for processing."""
        if start_date >= end_date:
            raise DataPreprocessingError("Start date must be before end date")
        
        if cutoff_date > end_date:
            raise DataPreprocessingError("Cutoff date cannot be after end date")
        
        if cutoff_date < start_date:
            raise DataPreprocessingError("Cutoff date cannot be before start date")
        
        # Check for reasonable observation period
        observation_days = (end_date - start_date).days
        if observation_days < 30:
            logger.warning(f"Short observation period: {observation_days} days")
        elif observation_days > 1095:  # 3 years
            logger.warning(f"Long observation period: {observation_days} days")
    
    def _create_empty_bgnbd_dataframe(self) -> pd.DataFrame:
        """Create an empty BG/NBD DataFrame with proper columns."""
        return pd.DataFrame(columns=[
            'supporter_id', 'x', 't_x', 'T', 'frequency', 'monetary',
            'first_event_date', 'last_event_date', 'total_weighted_value',
            'total_donation_amount', 'engagement_score', 'engagement_segment',
            'avg_donation', 'recency_ratio', 'event_type_count', 
            'event_diversity_index', 'observation_start', 'observation_end', 'cutoff_date'
        ])
    
    def generate_summary_statistics(self, bgnbd_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the processed BG/NBD data."""
        if bgnbd_df.empty:
            return {'error': 'No data to summarize'}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_supporters': len(bgnbd_df),
            'observation_period': {
                'start_date': bgnbd_df['observation_start'].iloc[0].isoformat(),
                'end_date': bgnbd_df['observation_end'].iloc[0].isoformat(),
                'cutoff_date': bgnbd_df['cutoff_date'].iloc[0].isoformat(),
                'total_days': int(bgnbd_df['T'].iloc[0])
            },
            'frequency_stats': {
                'mean': float(bgnbd_df['frequency'].mean()),
                'median': float(bgnbd_df['frequency'].median()),
                'std': float(bgnbd_df['frequency'].std()),
                'min': int(bgnbd_df['frequency'].min()),
                'max': int(bgnbd_df['frequency'].max()),
                'q25': float(bgnbd_df['frequency'].quantile(0.25)),
                'q75': float(bgnbd_df['frequency'].quantile(0.75))
            },
            'recency_stats': {
                'mean': float(bgnbd_df['t_x'].mean()),
                'median': float(bgnbd_df['t_x'].median()),
                'std': float(bgnbd_df['t_x'].std()),
                'min': float(bgnbd_df['t_x'].min()),
                'max': float(bgnbd_df['t_x'].max())
            },
            'monetary_stats': {
                'total_donations': float(bgnbd_df['total_donation_amount'].sum()),
                'mean_donation': float(bgnbd_df['total_donation_amount'].mean()),
                'median_donation': float(bgnbd_df['total_donation_amount'].median()),
                'donors_count': int((bgnbd_df['total_donation_amount'] > 0).sum()),
                'donor_percentage': float((bgnbd_df['total_donation_amount'] > 0).mean() * 100)
            },
            'engagement_segments': bgnbd_df['engagement_segment'].value_counts().to_dict(),
            'data_quality': {
                'supporters_with_repeat_events': int((bgnbd_df['x'] > 0).sum()),
                'repeat_rate': float((bgnbd_df['x'] > 0).mean() * 100),
                'avg_event_diversity': float(bgnbd_df['event_diversity_index'].mean()),
                'avg_event_types_per_supporter': float(bgnbd_df['event_type_count'].mean())
            }
        }
        
        return summary
    
    def save_processed_data(
        self, 
        bgnbd_df: pd.DataFrame, 
        filename: Optional[str] = None
    ) -> Path:
        """Save processed BG/NBD data to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bgnbd_data_{timestamp}.csv"
        
        output_path = self.processed_data_dir / filename
        
        try:
            bgnbd_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise DataPreprocessingError(f"Failed to save processed data: {e}")
    
    def _generate_cache_key(self, data_type: str, *args) -> str:
        """Generate a cache key based on processing parameters."""
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
        """Save processed data to cache."""
        try:
            cache_file = self.processed_data_dir / f"{cache_key}.pkl"
            cache_data = {
                'data': data,
                'timestamp': datetime.now(),
                'cache_key': cache_key
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Saved processed data to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save processed data to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load processed data from cache if valid."""
        try:
            cache_file = self.processed_data_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check cache expiry (24 hours)
            cache_age = datetime.now() - cache_data['timestamp']
            if cache_age.total_seconds() > (24 * 3600):
                logger.debug(f"Cache expired for key: {cache_key}")
                return None
            
            logger.debug(f"Loaded processed data from cache: {cache_file}")
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load processed data from cache: {e}")
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear cached processed data."""
        try:
            if cache_key:
                # Clear specific cache file
                cache_file = self.processed_data_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cleared processed data cache: {cache_key}")
            else:
                # Clear all cache files
                for cache_file in self.processed_data_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("Cleared all processed data cache")
                
        except Exception as e:
            logger.error(f"Failed to clear processed data cache: {e}")


def create_bgnbd_processor(data_extractor: Optional[DataExtractor] = None) -> BGNBDDataProcessor:
    """Factory function to create a BGNBDDataProcessor instance."""
    return BGNBDDataProcessor(data_extractor)


def process_supporter_data_pipeline(
    start_date: datetime,
    end_date: datetime,
    cutoff_date: Optional[datetime] = None,
    min_actions: int = 1,
    include_donations: bool = True,
    supporter_ids: Optional[List[int]] = None,
    save_results: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete pipeline for processing supporter data into BG/NBD format.
    
    Args:
        start_date: Start of observation period
        end_date: End of observation period
        cutoff_date: Analysis cutoff date
        min_actions: Minimum number of actions required per supporter
        include_donations: Whether to include donations as engagement events
        supporter_ids: Optional list of specific supporter IDs to process
        save_results: Whether to save results to file
        
    Returns:
        Tuple of (processed_dataframe, summary_statistics)
    """
    processor = create_bgnbd_processor()
    
    # Process the data
    bgnbd_df = processor.process_supporter_data(
        start_date=start_date,
        end_date=end_date,
        cutoff_date=cutoff_date,
        min_actions=min_actions,
        include_donations=include_donations,
        supporter_ids=supporter_ids
    )
    
    # Generate summary statistics
    summary_stats = processor.generate_summary_statistics(bgnbd_df)
    
    # Save results if requested
    if save_results and not bgnbd_df.empty:
        output_path = processor.save_processed_data(bgnbd_df)
        summary_stats['output_file'] = str(output_path)
    
    return bgnbd_df, summary_stats