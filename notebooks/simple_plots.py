"""
Simple plotting utilities for notebook demonstrations.
This module provides basic visualization without external dependencies.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class SimplePlotter:
    """Simple plotting utilities for BG/NBD analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the simple plotter."""
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_frequency_distribution(self, bgnbd_df: pd.DataFrame, title: str = "Frequency Distribution") -> plt.Figure:
        """Plot frequency distribution of supporter engagement."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(bgnbd_df['frequency'], bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.set_xlabel('Frequency (Number of Events)')
        ax1.set_ylabel('Number of Supporters')
        ax1.set_title('Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(bgnbd_df['frequency'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor=self.colors[0], alpha=0.7))
        ax2.set_ylabel('Frequency')
        ax2.set_title('Frequency Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_recency_distribution(self, bgnbd_df: pd.DataFrame, title: str = "Recency Distribution") -> plt.Figure:
        """Plot recency distribution of supporter engagement."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(bgnbd_df['recency'], bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax1.set_xlabel('Recency (Days)')
        ax1.set_ylabel('Number of Supporters')
        ax1.set_title('Recency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: Frequency vs Recency
        ax2.scatter(bgnbd_df['recency'], bgnbd_df['frequency'], alpha=0.6, color=self.colors[1])
        ax2.set_xlabel('Recency (Days)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Frequency vs Recency')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_engagement_segments(self, bgnbd_df: pd.DataFrame, title: str = "Engagement Segments") -> plt.Figure:
        """Plot engagement segment distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        segment_counts = bgnbd_df['engagement_segment'].value_counts()
        ax1.bar(segment_counts.index, segment_counts.values, color=self.colors[:len(segment_counts)])
        ax1.set_xlabel('Engagement Segment')
        ax1.set_ylabel('Number of Supporters')
        ax1.set_title('Supporters by Engagement Segment')
        ax1.grid(True, alpha=0.3)
        
        # Pie chart
        ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
               colors=self.colors[:len(segment_counts)])
        ax2.set_title('Engagement Segment Distribution')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_monetary_analysis(self, bgnbd_df: pd.DataFrame, title: str = "Monetary Analysis") -> plt.Figure:
        """Plot monetary value analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Donation amount distribution
        donors = bgnbd_df[bgnbd_df['monetary'] > 0]
        if not donors.empty:
            ax1.hist(donors['monetary'], bins=20, alpha=0.7, color=self.colors[2], edgecolor='black')
            ax1.set_xlabel('Total Donation Amount')
            ax1.set_ylabel('Number of Donors')
            ax1.set_title('Donation Amount Distribution')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No donations found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Donation Amount Distribution')
        
        # Donors vs Non-donors
        donor_counts = bgnbd_df['monetary'].apply(lambda x: 'Donor' if x > 0 else 'Non-donor').value_counts()
        ax2.bar(donor_counts.index, donor_counts.values, color=self.colors[2:4])
        ax2.set_ylabel('Number of Supporters')
        ax2.set_title('Donors vs Non-donors')
        ax2.grid(True, alpha=0.3)
        
        # Frequency vs Monetary
        ax3.scatter(bgnbd_df['frequency'], bgnbd_df['monetary'], alpha=0.6, color=self.colors[2])
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Total Donation Amount')
        ax3.set_title('Frequency vs Monetary Value')
        ax3.grid(True, alpha=0.3)
        
        # Monetary by segment
        if 'engagement_segment' in bgnbd_df.columns:
            segment_monetary = bgnbd_df.groupby('engagement_segment')['monetary'].mean()
            ax4.bar(segment_monetary.index, segment_monetary.values, color=self.colors[:len(segment_monetary)])
            ax4.set_xlabel('Engagement Segment')
            ax4.set_ylabel('Average Donation Amount')
            ax4.set_title('Average Donations by Segment')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No segment data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Donations by Segment')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_data_overview(self, dataset: Dict, title: str = "Dataset Overview") -> plt.Figure:
        """Plot overview of the dataset."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        supporters_df = dataset['supporters']
        actions_df = dataset['actions']
        donations_df = dataset['donations']
        
        # Supporters by segment
        if 'segment' in supporters_df.columns:
            segment_counts = supporters_df['segment'].value_counts()
            ax1.bar(segment_counts.index, segment_counts.values, color=self.colors[:len(segment_counts)])
            ax1.set_xlabel('Supporter Segment')
            ax1.set_ylabel('Number of Supporters')
            ax1.set_title('Supporters by Segment')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No segment data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Supporters by Segment')
        
        # Actions by type
        if 'action_type' in actions_df.columns:
            action_counts = actions_df['action_type'].value_counts()
            ax2.bar(action_counts.index, action_counts.values, color=self.colors[:len(action_counts)])
            ax2.set_xlabel('Action Type')
            ax2.set_ylabel('Number of Actions')
            ax2.set_title('Actions by Type')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No action data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Actions by Type')
        
        # Actions over time
        if 'action_date' in actions_df.columns:
            actions_df['action_date'] = pd.to_datetime(actions_df['action_date'])
            daily_actions = actions_df.groupby(actions_df['action_date'].dt.date).size()
            ax3.plot(daily_actions.index, daily_actions.values, color=self.colors[0], linewidth=2)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Number of Actions')
            ax3.set_title('Actions Over Time')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No date data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Actions Over Time')
        
        # Donation amounts
        if not donations_df.empty and 'amount' in donations_df.columns:
            ax4.hist(donations_df['amount'], bins=20, alpha=0.7, color=self.colors[2], edgecolor='black')
            ax4.set_xlabel('Donation Amount')
            ax4.set_ylabel('Number of Donations')
            ax4.set_title('Donation Amount Distribution')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No donation data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Donation Amount Distribution')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, bgnbd_df: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
        """Plot correlation matrix of BG/NBD variables."""
        # Select numeric columns
        numeric_cols = bgnbd_df.select_dtypes(include=[np.number]).columns
        correlation_data = bgnbd_df[numeric_cols]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_summary_plot(self, bgnbd_df: pd.DataFrame, dataset: Dict) -> plt.Figure:
        """Create a comprehensive summary plot."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Frequency distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(bgnbd_df['frequency'], bins=15, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Count')
        ax1.set_title('Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Recency distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(bgnbd_df['recency'], bins=15, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.set_xlabel('Recency (Days)')
        ax2.set_ylabel('Count')
        ax2.set_title('Recency Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Frequency vs Recency
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(bgnbd_df['recency'], bgnbd_df['frequency'], alpha=0.6, color=self.colors[2])
        ax3.set_xlabel('Recency (Days)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Frequency vs Recency')
        ax3.grid(True, alpha=0.3)
        
        # Engagement segments
        ax4 = fig.add_subplot(gs[1, 0])
        if 'engagement_segment' in bgnbd_df.columns:
            segment_counts = bgnbd_df['engagement_segment'].value_counts()
            ax4.bar(segment_counts.index, segment_counts.values, color=self.colors[:len(segment_counts)])
            ax4.set_xlabel('Engagement Segment')
            ax4.set_ylabel('Count')
            ax4.set_title('Engagement Segments')
            ax4.grid(True, alpha=0.3)
        
        # Monetary distribution
        ax5 = fig.add_subplot(gs[1, 1])
        donors = bgnbd_df[bgnbd_df['monetary'] > 0]
        if not donors.empty:
            ax5.hist(donors['monetary'], bins=15, alpha=0.7, color=self.colors[3], edgecolor='black')
            ax5.set_xlabel('Donation Amount')
            ax5.set_ylabel('Count')
            ax5.set_title('Donation Distribution')
            ax5.grid(True, alpha=0.3)
        
        # Action types
        ax6 = fig.add_subplot(gs[1, 2])
        if 'actions' in dataset and 'action_type' in dataset['actions'].columns:
            action_counts = dataset['actions']['action_type'].value_counts()
            ax6.bar(range(len(action_counts)), action_counts.values, color=self.colors[:len(action_counts)])
            ax6.set_xticks(range(len(action_counts)))
            ax6.set_xticklabels(action_counts.index, rotation=45, ha='right')
            ax6.set_ylabel('Count')
            ax6.set_title('Action Types')
            ax6.grid(True, alpha=0.3)
        
        # Summary statistics text
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate summary stats
        total_supporters = len(bgnbd_df)
        avg_frequency = bgnbd_df['frequency'].mean()
        avg_recency = bgnbd_df['recency'].mean()
        total_donations = bgnbd_df['monetary'].sum()
        donor_count = (bgnbd_df['monetary'] > 0).sum()
        
        summary_text = f"""
        Dataset Summary:
        • Total Supporters: {total_supporters:,}
        • Average Frequency: {avg_frequency:.2f} events per supporter
        • Average Recency: {avg_recency:.1f} days since last event
        • Total Donations: ${total_donations:,.2f}
        • Number of Donors: {donor_count:,} ({donor_count/total_supporters*100:.1f}%)
        • Observation Period: {bgnbd_df['T'].iloc[0]} days
        """
        
        ax7.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle('BG/NBD Analysis Summary', fontsize=18, fontweight='bold')
        return fig