"""
Sample data generation module for the non-profit engagement model.

This module creates realistic synthetic data that mimics non-profit supporter behavior
for demonstration and testing purposes when database access is not available.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """
    Generates realistic sample data for non-profit supporter engagement modeling.
    
    This class creates synthetic data that follows realistic patterns of supporter
    behavior including different engagement segments, seasonal patterns, and
    donation behaviors.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the sample data generator.
        
        Args:
            random_seed: Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define supporter segments with different behavior patterns
        self.segments = {
            'Champions': {
                'proportion': 0.15,
                'avg_frequency': 8.0,
                'frequency_std': 3.0,
                'donation_rate': 0.8,
                'avg_donation': 150.0,
                'donation_std': 75.0,
                'churn_rate': 0.05
            },
            'Loyal_Supporters': {
                'proportion': 0.25,
                'avg_frequency': 5.0,
                'frequency_std': 2.0,
                'donation_rate': 0.6,
                'avg_donation': 75.0,
                'donation_std': 40.0,
                'churn_rate': 0.10
            },
            'Potential_Loyalists': {
                'proportion': 0.30,
                'avg_frequency': 3.0,
                'frequency_std': 1.5,
                'donation_rate': 0.4,
                'avg_donation': 50.0,
                'donation_std': 25.0,
                'churn_rate': 0.20
            },
            'At_Risk': {
                'proportion': 0.20,
                'avg_frequency': 1.5,
                'frequency_std': 1.0,
                'donation_rate': 0.2,
                'avg_donation': 30.0,
                'donation_std': 15.0,
                'churn_rate': 0.40
            },
            'Lost': {
                'proportion': 0.10,
                'avg_frequency': 0.5,
                'frequency_std': 0.5,
                'donation_rate': 0.1,
                'avg_donation': 20.0,
                'donation_std': 10.0,
                'churn_rate': 0.70
            }
        }
        
        # Action types with different weights and frequencies
        self.action_types = {
            'email_open': {'weight': 0.3, 'frequency': 0.4},
            'email_click': {'weight': 0.5, 'frequency': 0.2},
            'website_visit': {'weight': 0.2, 'frequency': 0.3},
            'social_media': {'weight': 0.4, 'frequency': 0.15},
            'event_attendance': {'weight': 1.2, 'frequency': 0.05},
            'volunteer': {'weight': 1.5, 'frequency': 0.03},
            'petition_sign': {'weight': 0.8, 'frequency': 0.08},
            'newsletter_signup': {'weight': 0.6, 'frequency': 0.02}
        }
        
        logger.info("SampleDataGenerator initialized")
    
    def generate_supporters(self, n_supporters: int = 1000) -> pd.DataFrame:
        """
        Generate supporter demographic data.
        
        Args:
            n_supporters: Number of supporters to generate
            
        Returns:
            DataFrame with supporter information
        """
        logger.info(f"Generating {n_supporters} supporters")
        
        supporters = []
        supporter_id = 1
        
        for segment_name, segment_config in self.segments.items():
            # Calculate number of supporters for this segment
            segment_size = int(n_supporters * segment_config['proportion'])
            
            for _ in range(segment_size):
                supporter = {
                    'supporter_id': supporter_id,
                    'segment': segment_name,
                    'acquisition_date': self._generate_acquisition_date(),
                    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], 
                                                p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1]),
                    'location_type': np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.5, 0.35, 0.15]),
                    'communication_preference': np.random.choice(['Email', 'Mail', 'Phone', 'SMS'], 
                                                               p=[0.6, 0.25, 0.1, 0.05])
                }
                supporters.append(supporter)
                supporter_id += 1
        
        return pd.DataFrame(supporters)
    
    def generate_actions(
        self, 
        supporters_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate supporter actions based on their segments and behavior patterns.
        
        Args:
            supporters_df: DataFrame with supporter information
            start_date: Start date for action generation
            end_date: End date for action generation
            
        Returns:
            DataFrame with supporter actions
        """
        logger.info(f"Generating actions from {start_date} to {end_date}")
        
        actions = []
        total_days = (end_date - start_date).days
        
        for _, supporter in supporters_df.iterrows():
            segment_config = self.segments[supporter['segment']]
            
            # Determine if supporter is still active (not churned)
            churn_probability = segment_config['churn_rate']
            if np.random.random() < churn_probability:
                # Supporter churned - generate fewer actions and earlier last action
                active_period = int(total_days * np.random.uniform(0.2, 0.7))
                frequency_multiplier = 0.3
            else:
                active_period = total_days
                frequency_multiplier = 1.0
            
            # Generate number of actions based on segment
            base_frequency = max(1, int(np.random.normal(
                segment_config['avg_frequency'] * frequency_multiplier,
                segment_config['frequency_std']
            )))
            
            # Generate action dates with some clustering (supporters tend to be active in bursts)
            action_dates = []
            remaining_actions = base_frequency
            
            while remaining_actions > 0:
                # Create a burst of activity
                burst_start = start_date + timedelta(days=np.random.randint(0, active_period))
                burst_size = min(remaining_actions, np.random.poisson(2) + 1)
                
                for i in range(burst_size):
                    action_date = burst_start + timedelta(days=np.random.exponential(2))
                    if action_date <= end_date:
                        action_dates.append(action_date)
                
                remaining_actions -= burst_size
            
            # Generate specific actions
            for action_date in action_dates:
                # Choose action type based on probabilities
                frequencies = [config['frequency'] for config in self.action_types.values()]
                # Normalize probabilities to sum to 1
                frequencies = np.array(frequencies)
                frequencies = frequencies / frequencies.sum()
                
                action_type = np.random.choice(
                    list(self.action_types.keys()),
                    p=frequencies
                )
                
                # Add some seasonal variation
                seasonal_factor = self._get_seasonal_factor(action_date)
                
                # Generate tags based on action type and season
                tags = self._generate_action_tags(action_type, seasonal_factor)
                
                action = {
                    'supporter_id': supporter['supporter_id'],
                    'action_date': action_date,
                    'action_type': action_type,
                    'tags': tags
                }
                actions.append(action)
        
        actions_df = pd.DataFrame(actions)
        
        # Sort by supporter and date
        if not actions_df.empty:
            actions_df = actions_df.sort_values(['supporter_id', 'action_date'])
        
        logger.info(f"Generated {len(actions_df)} actions")
        return actions_df
    
    def generate_donations(
        self,
        supporters_df: pd.DataFrame,
        actions_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate donation data correlated with supporter segments and actions.
        
        Args:
            supporters_df: DataFrame with supporter information
            actions_df: DataFrame with supporter actions
            start_date: Start date for donation generation
            end_date: End date for donation generation
            
        Returns:
            DataFrame with donation data
        """
        logger.info("Generating donation data")
        
        donations = []
        
        for _, supporter in supporters_df.iterrows():
            segment_config = self.segments[supporter['segment']]
            
            # Determine if supporter makes donations
            if np.random.random() > segment_config['donation_rate']:
                continue
            
            # Get supporter's actions to correlate donations
            supporter_actions = actions_df[actions_df['supporter_id'] == supporter['supporter_id']]
            
            if supporter_actions.empty:
                continue
            
            # Generate donations correlated with high-engagement actions
            high_engagement_actions = supporter_actions[
                supporter_actions['action_type'].isin(['event_attendance', 'volunteer', 'petition_sign'])
            ]
            
            # Base number of donations on segment
            base_donations = max(1, int(np.random.poisson(
                segment_config['donation_rate'] * 3
            )))
            
            # Additional donations correlated with high engagement
            bonus_donations = len(high_engagement_actions) // 2
            
            total_donations = base_donations + bonus_donations
            
            for _ in range(total_donations):
                # Choose donation date - either random or correlated with action
                if not supporter_actions.empty and np.random.random() < 0.6:
                    # Correlate with an action (within 30 days)
                    base_action = supporter_actions.sample(1).iloc[0]
                    donation_date = base_action['action_date'] + timedelta(
                        days=np.random.randint(0, 30)
                    )
                else:
                    # Random date
                    donation_date = start_date + timedelta(
                        days=np.random.randint(0, (end_date - start_date).days)
                    )
                
                if donation_date > end_date:
                    continue
                
                # Generate donation amount based on segment and seasonal factors
                seasonal_factor = self._get_seasonal_factor(donation_date)
                base_amount = np.random.normal(
                    segment_config['avg_donation'],
                    segment_config['donation_std']
                )
                
                # Apply seasonal multiplier (higher donations in December, lower in summer)
                amount = max(5.0, base_amount * seasonal_factor)
                
                # Round to nearest dollar
                amount = round(amount, 2)
                
                donation = {
                    'supporter_id': supporter['supporter_id'],
                    'donation_date': donation_date,
                    'amount': amount
                }
                donations.append(donation)
        
        donations_df = pd.DataFrame(donations)
        
        # Sort by supporter and date
        if not donations_df.empty:
            donations_df = donations_df.sort_values(['supporter_id', 'donation_date'])
        
        logger.info(f"Generated {len(donations_df)} donations")
        return donations_df
    
    def _generate_acquisition_date(self) -> datetime:
        """Generate a realistic supporter acquisition date."""
        # Most supporters acquired in the last 2 years, with some older ones
        days_ago = np.random.exponential(365)  # Exponential distribution favoring recent
        days_ago = min(days_ago, 1095)  # Cap at 3 years
        
        return datetime.now() - timedelta(days=int(days_ago))
    
    def _get_seasonal_factor(self, date: datetime) -> float:
        """Get seasonal factor for actions/donations based on date."""
        month = date.month
        
        # Higher activity in fall/winter (fundraising season)
        seasonal_factors = {
            1: 1.2,   # January - New Year giving
            2: 0.9,   # February
            3: 0.9,   # March
            4: 1.0,   # April
            5: 0.8,   # May
            6: 0.7,   # June - summer low
            7: 0.7,   # July - summer low
            8: 0.8,   # August
            9: 1.1,   # September - back to school
            10: 1.2,  # October - fall campaigns
            11: 1.4,  # November - Thanksgiving
            12: 1.6   # December - year-end giving
        }
        
        return seasonal_factors.get(month, 1.0)
    
    def _generate_action_tags(self, action_type: str, seasonal_factor: float) -> str:
        """Generate realistic tags for actions."""
        base_tags = {
            'email_open': ['newsletter', 'campaign', 'update', 'appeal'],
            'email_click': ['donation_link', 'event_link', 'petition_link', 'learn_more'],
            'website_visit': ['homepage', 'about', 'donate', 'events', 'blog'],
            'social_media': ['facebook', 'twitter', 'instagram', 'share', 'like'],
            'event_attendance': ['fundraiser', 'awareness', 'community', 'virtual', 'in_person'],
            'volunteer': ['community_service', 'event_help', 'administrative', 'outreach'],
            'petition_sign': ['policy_change', 'awareness', 'advocacy', 'campaign'],
            'newsletter_signup': ['monthly', 'weekly', 'campaign_updates', 'general']
        }
        
        # Add seasonal tags
        seasonal_tags = []
        if seasonal_factor > 1.3:  # High season (Nov-Dec)
            seasonal_tags = ['year_end', 'holiday', 'giving_season']
        elif seasonal_factor < 0.8:  # Low season (summer)
            seasonal_tags = ['summer', 'vacation', 'low_activity']
        
        # Combine base and seasonal tags
        available_tags = base_tags.get(action_type, ['general'])
        if seasonal_tags:
            available_tags.extend(seasonal_tags)
        
        # Select 1-3 tags
        num_tags = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        selected_tags = np.random.choice(available_tags, size=min(num_tags, len(available_tags)), replace=False)
        
        return ', '.join(selected_tags)
    
    def generate_complete_dataset(
        self,
        n_supporters: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_to_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete dataset with supporters, actions, and donations.
        
        Args:
            n_supporters: Number of supporters to generate
            start_date: Start date for data generation (default: 2 years ago)
            end_date: End date for data generation (default: today)
            save_to_csv: Whether to save data to CSV files
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary containing supporters, actions, and donations DataFrames
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Generating complete dataset: {n_supporters} supporters, {start_date} to {end_date}")
        
        # Generate supporters
        supporters_df = self.generate_supporters(n_supporters)
        
        # Generate actions
        actions_df = self.generate_actions(supporters_df, start_date, end_date)
        
        # Generate donations
        donations_df = self.generate_donations(supporters_df, actions_df, start_date, end_date)
        
        dataset = {
            'supporters': supporters_df,
            'actions': actions_df,
            'donations': donations_df
        }
        
        # Save to CSV if requested
        if save_to_csv:
            if output_dir is None:
                output_dir = "data/sample"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for name, df in dataset.items():
                csv_path = output_path / f"sample_{name}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {name} data to {csv_path}")
        
        # Log summary statistics
        logger.info(f"Dataset generated successfully:")
        logger.info(f"  - Supporters: {len(supporters_df)}")
        logger.info(f"  - Actions: {len(actions_df)}")
        logger.info(f"  - Donations: {len(donations_df)}")
        logger.info(f"  - Date range: {start_date.date()} to {end_date.date()}")
        
        return dataset
    
    def get_dataset_summary(self, dataset: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            dataset: Dictionary containing supporters, actions, and donations DataFrames
            
        Returns:
            Dictionary with summary statistics
        """
        supporters_df = dataset['supporters']
        actions_df = dataset['actions']
        donations_df = dataset['donations']
        
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'supporters': {
                'total_count': len(supporters_df),
                'segments': supporters_df['segment'].value_counts().to_dict(),
                'age_groups': supporters_df['age_group'].value_counts().to_dict(),
                'location_types': supporters_df['location_type'].value_counts().to_dict()
            },
            'actions': {
                'total_count': len(actions_df),
                'unique_supporters': actions_df['supporter_id'].nunique() if not actions_df.empty else 0,
                'action_types': actions_df['action_type'].value_counts().to_dict() if not actions_df.empty else {},
                'date_range': {
                    'start': actions_df['action_date'].min().isoformat() if not actions_df.empty else None,
                    'end': actions_df['action_date'].max().isoformat() if not actions_df.empty else None
                },
                'actions_per_supporter': {
                    'mean': actions_df.groupby('supporter_id').size().mean() if not actions_df.empty else 0,
                    'median': actions_df.groupby('supporter_id').size().median() if not actions_df.empty else 0,
                    'std': actions_df.groupby('supporter_id').size().std() if not actions_df.empty else 0
                }
            },
            'donations': {
                'total_count': len(donations_df),
                'unique_donors': donations_df['supporter_id'].nunique() if not donations_df.empty else 0,
                'total_amount': donations_df['amount'].sum() if not donations_df.empty else 0,
                'amount_stats': {
                    'mean': donations_df['amount'].mean() if not donations_df.empty else 0,
                    'median': donations_df['amount'].median() if not donations_df.empty else 0,
                    'std': donations_df['amount'].std() if not donations_df.empty else 0,
                    'min': donations_df['amount'].min() if not donations_df.empty else 0,
                    'max': donations_df['amount'].max() if not donations_df.empty else 0
                },
                'donor_rate': donations_df['supporter_id'].nunique() / len(supporters_df) * 100 if len(supporters_df) > 0 else 0
            }
        }
        
        return summary


def create_sample_data_generator(random_seed: int = 42) -> SampleDataGenerator:
    """
    Factory function to create a SampleDataGenerator instance.
    
    Args:
        random_seed: Random seed for reproducible data generation
        
    Returns:
        SampleDataGenerator instance
    """
    return SampleDataGenerator(random_seed)


def generate_demo_dataset(
    n_supporters: int = 1000,
    save_files: bool = True,
    output_dir: str = "data/sample"
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate a demo dataset.
    
    Args:
        n_supporters: Number of supporters to generate
        save_files: Whether to save data to CSV files
        output_dir: Directory to save files
        
    Returns:
        Dictionary containing the generated dataset
    """
    generator = create_sample_data_generator()
    
    return generator.generate_complete_dataset(
        n_supporters=n_supporters,
        save_to_csv=save_files,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Generate demo dataset when run directly
    print("Generating demo dataset...")
    dataset = generate_demo_dataset(n_supporters=500, save_files=True)
    
    generator = create_sample_data_generator()
    summary = generator.get_dataset_summary(dataset)
    
    print("\nDataset Summary:")
    print(f"Supporters: {summary['supporters']['total_count']}")
    print(f"Actions: {summary['actions']['total_count']}")
    print(f"Donations: {summary['donations']['total_count']}")
    print(f"Donor Rate: {summary['donations']['donor_rate']:.1f}%")