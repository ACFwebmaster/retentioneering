"""
Board Report Visualizations

This module provides visualization functions for board retention reports,
creating professional charts suitable for board presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

# Set style for professional looking charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class BoardReportVisualizer:
    """Creates professional visualizations for board retention reports."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for charts
            dpi: Resolution for saved charts
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#48C9B0', 
            'accent': '#F39C12',
            'warning': '#E74C3C',
            'success': '#27AE60',
            'neutral': '#85929E'
        }
        
    def create_retention_trend_chart(
        self, 
        report: Dict[str, Any], 
        save_path: Optional[Path] = None,
        show_chart: bool = True
    ) -> plt.Figure:
        """
        Create a retention rate trend chart across financial quarters.
        
        Args:
            report: Board retention report dictionary
            save_path: Optional path to save the chart
            show_chart: Whether to display the chart
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract data from report
        periods_data = report['retention_periods']
        if not periods_data:
            ax.text(0.5, 0.5, 'No retention data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
            
        # Prepare data
        period_labels = [p['reference_quarter'] for p in periods_data]
        retention_rates = [p['retention_rate'] for p in periods_data]
        supporter_counts = [p['total_supporters_reference'] for p in periods_data]
        
        # Create the main line plot
        line = ax.plot(range(len(period_labels)), retention_rates, 
                      marker='o', linewidth=3, markersize=8, 
                      color=self.colors['primary'], label='Retention Rate')
        
        # Add data labels
        for i, (rate, count) in enumerate(zip(retention_rates, supporter_counts)):
            ax.annotate(f'{rate:.1f}%\n({count:,} supporters)', 
                       (i, rate), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize chart
        ax.set_xticks(range(len(period_labels)))
        ax.set_xticklabels(period_labels, rotation=45, ha='right')
        ax.set_ylabel('Retention Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Financial Quarter', fontsize=12, fontweight='bold')
        ax.set_title('6-Month Supporter Retention Rate by Quarter', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add trend line if enough data points
        if len(retention_rates) >= 3:
            z = np.polyfit(range(len(retention_rates)), retention_rates, 1)
            p = np.poly1d(z)
            ax.plot(range(len(period_labels)), p(range(len(period_labels))), 
                   linestyle='--', alpha=0.7, color=self.colors['accent'], 
                   label=f'Trend ({"↗" if z[0] > 0 else "↘"})')
            
        # Add average line
        avg_retention = np.mean(retention_rates)
        ax.axhline(y=avg_retention, color=self.colors['neutral'], 
                  linestyle=':', alpha=0.8, label=f'Average ({avg_retention:.1f}%)')
        
        # Formatting
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(retention_rates) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Retention trend chart saved to {save_path}")
            
        if show_chart:
            plt.show()
            
        return fig
    
    def create_supporter_volume_chart(
        self,
        report: Dict[str, Any],
        save_path: Optional[Path] = None,
        show_chart: bool = True
    ) -> plt.Figure:
        """
        Create a chart showing supporter volumes across quarters.
        
        Args:
            report: Board retention report dictionary
            save_path: Optional path to save the chart
            show_chart: Whether to display the chart
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), 
                                      dpi=self.dpi, sharex=True)
        
        periods_data = report['retention_periods']
        if not periods_data:
            ax1.text(0.5, 0.5, 'No retention data available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            return fig
            
        # Prepare data
        period_labels = [p['reference_quarter'] for p in periods_data]
        total_supporters = [p['total_supporters_reference'] for p in periods_data]
        retained_supporters = [p['supporters_retained'] for p in periods_data]
        not_retained_supporters = [p['supporters_not_retained'] for p in periods_data]
        
        # Chart 1: Stacked bar chart of retained vs not retained
        x = range(len(period_labels))
        width = 0.6
        
        bars1 = ax1.bar(x, retained_supporters, width, label='Retained', 
                       color=self.colors['success'], alpha=0.8)
        bars2 = ax1.bar(x, not_retained_supporters, width, bottom=retained_supporters,
                       label='Not Retained', color=self.colors['warning'], alpha=0.8)
        
        # Add value labels on bars
        for i, (ret, not_ret, total) in enumerate(zip(retained_supporters, not_retained_supporters, total_supporters)):
            ax1.text(i, total + total*0.02, f'{total:,}', ha='center', va='bottom', fontweight='bold')
            
        ax1.set_ylabel('Number of Supporters', fontsize=12, fontweight='bold')
        ax1.set_title('Supporter Retention Breakdown by Quarter', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Total supporter volumes
        line = ax2.plot(x, total_supporters, marker='s', linewidth=3, markersize=8,
                       color=self.colors['primary'], label='Total Supporters')
        
        # Add data labels
        for i, count in enumerate(total_supporters):
            ax2.annotate(f'{count:,}', (i, count), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(period_labels, rotation=45, ha='right')
        ax2.set_ylabel('Total Supporters', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Financial Quarter', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Supporter volume chart saved to {save_path}")
            
        if show_chart:
            plt.show()
            
        return fig
    
    def create_summary_dashboard(
        self,
        report: Dict[str, Any],
        save_path: Optional[Path] = None,
        show_chart: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            report: Board retention report dictionary
            save_path: Optional path to save the chart
            show_chart: Whether to display the chart
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        periods_data = report['retention_periods']
        summary_stats = report.get('summary_statistics', {})
        
        if not periods_data:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No retention data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=20)
            return fig
        
        # 1. Main retention trend (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_retention_trend_subplot(ax1, periods_data)
        
        # 2. Summary statistics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_summary_stats_subplot(ax2, summary_stats)
        
        # 3. Retention rate distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_retention_distribution_subplot(ax3, periods_data)
        
        # 4. Supporter volumes (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_supporter_volumes_subplot(ax4, periods_data)
        
        # 5. Performance indicators (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_performance_indicators_subplot(ax5, summary_stats)
        
        # 6. Quarter comparison table (bottom row, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_comparison_table_subplot(ax6, periods_data)
        
        # Add main title
        fig.suptitle('Board Retention Report Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Summary dashboard saved to {save_path}")
            
        if show_chart:
            plt.show()
            
        return fig
    
    def _plot_retention_trend_subplot(self, ax, periods_data):
        """Plot retention trend in subplot."""
        period_labels = [p['reference_quarter'] for p in periods_data]
        retention_rates = [p['retention_rate'] for p in periods_data]
        
        x = range(len(period_labels))
        ax.plot(x, retention_rates, marker='o', linewidth=2, markersize=6, 
               color=self.colors['primary'])
        
        # Add average line
        avg_retention = np.mean(retention_rates)
        ax.axhline(y=avg_retention, color=self.colors['neutral'], 
                  linestyle='--', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels([label.split()[1] for label in period_labels], rotation=45)
        ax.set_ylabel('Retention Rate (%)')
        ax.set_title('Retention Trend Over Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats_subplot(self, ax, summary_stats):
        """Plot key summary statistics."""
        ax.axis('off')
        
        if not summary_stats or summary_stats.get('total_periods_analyzed', 0) == 0:
            ax.text(0.5, 0.5, 'No statistics\navailable', ha='center', va='center', 
                   fontsize=12, transform=ax.transAxes)
            return
            
        stats_text = f"""Key Metrics

Average: {summary_stats.get('average_retention_rate', 0):.1f}%
Median: {summary_stats.get('median_retention_rate', 0):.1f}%
Range: {summary_stats.get('min_retention_rate', 0):.1f}% - {summary_stats.get('max_retention_rate', 0):.1f}%

Trend: {summary_stats.get('retention_rate_trend', 'N/A').title()}

Total Supporters: {summary_stats.get('total_supporters_across_periods', 0):,}
Periods Analyzed: {summary_stats.get('total_periods_analyzed', 0)}"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    def _plot_retention_distribution_subplot(self, ax, periods_data):
        """Plot retention rate distribution."""
        retention_rates = [p['retention_rate'] for p in periods_data]
        
        ax.hist(retention_rates, bins=min(5, len(retention_rates)), 
               color=self.colors['secondary'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Retention Rate (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Retention Rate\nDistribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_supporter_volumes_subplot(self, ax, periods_data):
        """Plot supporter volumes."""
        total_supporters = [p['total_supporters_reference'] for p in periods_data]
        period_labels = [p['reference_quarter'].split()[1] for p in periods_data]
        
        bars = ax.bar(range(len(period_labels)), total_supporters, 
                     color=self.colors['accent'], alpha=0.7)
        
        ax.set_xticks(range(len(period_labels)))
        ax.set_xticklabels(period_labels, rotation=45)
        ax.set_ylabel('Supporters')
        ax.set_title('Supporter Volumes\nby Quarter', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_performance_indicators_subplot(self, ax, summary_stats):
        """Plot performance indicators."""
        ax.axis('off')
        
        if not summary_stats or summary_stats.get('total_periods_analyzed', 0) == 0:
            return
            
        # Create simple gauge-like visualization
        avg_retention = summary_stats.get('average_retention_rate', 0)
        trend = summary_stats.get('retention_rate_trend', 'stable')
        
        # Color based on performance
        if avg_retention >= 70:
            perf_color = self.colors['success']
            perf_text = 'EXCELLENT'
        elif avg_retention >= 50:
            perf_color = self.colors['accent']
            perf_text = 'GOOD'
        else:
            perf_color = self.colors['warning']
            perf_text = 'NEEDS ATTENTION'
        
        # Trend indicator
        trend_symbol = {'improving': '↗', 'declining': '↘', 'stable': '→'}.get(trend, '→')
        trend_color = {'improving': self.colors['success'], 
                      'declining': self.colors['warning'], 
                      'stable': self.colors['neutral']}.get(trend, self.colors['neutral'])
        
        ax.text(0.5, 0.7, f'{avg_retention:.1f}%', ha='center', va='center',
               fontsize=24, fontweight='bold', color=perf_color, 
               transform=ax.transAxes)
        ax.text(0.5, 0.5, perf_text, ha='center', va='center',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.3, f'Trend: {trend_symbol}', ha='center', va='center',
               fontsize=16, color=trend_color, transform=ax.transAxes)
        ax.text(0.5, 0.1, 'Overall Performance', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
    
    def _plot_comparison_table_subplot(self, ax, periods_data):
        """Plot comparison table."""
        ax.axis('off')
        
        # Prepare table data
        headers = ['Quarter', 'Retention Rate', 'Retained', 'Not Retained', 'Total']
        rows = []
        
        for p in periods_data:
            rows.append([
                p['reference_quarter'].split()[1],  # Just the quarter part
                f"{p['retention_rate']:.1f}%",
                f"{p['supporters_retained']:,}",
                f"{p['supporters_not_retained']:,}",
                f"{p['total_supporters_reference']:,}"
            ])
        
        # Create table
        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Quarter-by-Quarter Comparison', fontweight='bold', pad=20)


def create_all_board_visualizations(
    report: Dict[str, Any], 
    output_dir: Path,
    report_name: str = "board_retention"
) -> List[Path]:
    """
    Create all standard board visualizations for a retention report.
    
    Args:
        report: Board retention report dictionary
        output_dir: Directory to save charts
        report_name: Base name for chart files
        
    Returns:
        List of paths to saved charts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = BoardReportVisualizer()
    saved_files = []
    
    # 1. Retention trend chart
    trend_path = output_dir / f"{report_name}_trend.png"
    visualizer.create_retention_trend_chart(report, trend_path, show_chart=False)
    saved_files.append(trend_path)
    
    # 2. Supporter volume chart
    volume_path = output_dir / f"{report_name}_volumes.png"
    visualizer.create_supporter_volume_chart(report, volume_path, show_chart=False)
    saved_files.append(volume_path)
    
    # 3. Summary dashboard
    dashboard_path = output_dir / f"{report_name}_dashboard.png"
    visualizer.create_summary_dashboard(report, dashboard_path, show_chart=False)
    saved_files.append(dashboard_path)
    
    logger.info(f"Created {len(saved_files)} visualizations in {output_dir}")
    return saved_files


# Example usage
if __name__ == "__main__":
    # This would typically be used with real data
    from src.retention_metrics.board_retention_reporter import create_sample_board_report
    
    # Generate sample report
    reports = create_sample_board_report()
    annual_report = reports['annual_report']
    
    # Create visualizations
    visualizer = BoardReportVisualizer()
    
    # Show individual charts
    visualizer.create_retention_trend_chart(annual_report)
    visualizer.create_supporter_volume_chart(annual_report)
    visualizer.create_summary_dashboard(annual_report)