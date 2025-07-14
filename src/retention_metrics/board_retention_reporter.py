"""
Board Retention Reporting Module

This module creates board-level retention reports across arbitrary date ranges,
broken down by financial year quarters and calculating 6-month retention metrics
for each period.

Financial Year Quarters:
- Q1: July 1 to September 30
- Q2: October 1 to December 31  
- Q3: January 1 to March 31
- Q4: April 1 to June 30
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from retentioneering.eventstream import Eventstream

from src.retention_metrics.six_month_retention_processor import SixMonthRetentionProcessor

logger = logging.getLogger(__name__)


@dataclass
class FinancialQuarter:
    """Represents a financial quarter with start and end dates."""
    year: int
    quarter: int  # 1, 2, 3, or 4
    start_date: date
    end_date: date
    
    def __str__(self) -> str:
        return f"FY{self.year} Q{self.quarter}"
    
    @property
    def label(self) -> str:
        """Human-readable quarter label."""
        return f"FY{self.year} Q{self.quarter} ({self.start_date.strftime('%b %d')} - {self.end_date.strftime('%b %d')})"


@dataclass
class RetentionPeriod:
    """Represents a retention analysis period with reference and following periods."""
    reference_quarter: FinancialQuarter
    following_quarter: FinancialQuarter
    analysis_id: str
    
    @property
    def label(self) -> str:
        """Human-readable period label."""
        return f"{self.reference_quarter} → {self.following_quarter}"


@dataclass
class RetentionResult:
    """Results from a retention analysis period."""
    period: RetentionPeriod
    retention_rate: float
    supporters_retained: int
    supporters_not_retained: int
    total_supporters_reference: int
    retained_supporter_ids: List[str]
    non_retained_supporter_ids: List[str]
    calculation_timestamp: datetime
    
    @property
    def summary_text(self) -> str:
        """Summary text for reporting."""
        return (f"{self.period.label}: {self.retention_rate:.1f}% retention "
                f"({self.supporters_retained}/{self.total_supporters_reference} supporters)")


class FinancialYearCalculator:
    """Utility class for financial year calculations."""
    
    @staticmethod
    def get_financial_year(input_date: Union[date, datetime]) -> int:
        """
        Get the financial year for a given date.
        Financial year runs July 1 to June 30.
        
        Args:
            input_date: Date to determine financial year for
            
        Returns:
            Financial year (e.g., 2024 for FY2024 = July 1, 2023 to June 30, 2024)
        """
        if isinstance(input_date, datetime):
            input_date = input_date.date()
            
        if input_date.month >= 7:  # July onwards = start of new FY
            return input_date.year + 1
        else:  # January to June = continuation of FY that started previous year
            return input_date.year
    
    @staticmethod
    def get_quarter_number(input_date: Union[date, datetime]) -> int:
        """
        Get the quarter number (1-4) for a given date.
        
        Returns:
            1: July-September, 2: October-December, 3: January-March, 4: April-June
        """
        if isinstance(input_date, datetime):
            input_date = input_date.date()
            
        month = input_date.month
        if 7 <= month <= 9:
            return 1
        elif 10 <= month <= 12:
            return 2
        elif 1 <= month <= 3:
            return 3
        else:  # 4 <= month <= 6
            return 4
    
    @staticmethod
    def get_quarter_dates(financial_year: int, quarter: int) -> Tuple[date, date]:
        """
        Get start and end dates for a specific financial quarter.
        
        Args:
            financial_year: Financial year (e.g., 2024)
            quarter: Quarter number (1-4)
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if quarter == 1:  # July-September
            start_date = date(financial_year - 1, 7, 1)
            end_date = date(financial_year - 1, 9, 30)
        elif quarter == 2:  # October-December
            start_date = date(financial_year - 1, 10, 1)
            end_date = date(financial_year - 1, 12, 31)
        elif quarter == 3:  # January-March
            start_date = date(financial_year, 1, 1)
            end_date = date(financial_year, 3, 31)
        elif quarter == 4:  # April-June
            start_date = date(financial_year, 4, 1)
            end_date = date(financial_year, 6, 30)
        else:
            raise ValueError(f"Invalid quarter: {quarter}. Must be 1-4.")
            
        return start_date, end_date
    
    @staticmethod
    def create_financial_quarter(financial_year: int, quarter: int) -> FinancialQuarter:
        """Create a FinancialQuarter object."""
        start_date, end_date = FinancialYearCalculator.get_quarter_dates(financial_year, quarter)
        return FinancialQuarter(
            year=financial_year,
            quarter=quarter,
            start_date=start_date,
            end_date=end_date
        )


class BoardRetentionReporter:
    """
    Main class for generating board retention reports across arbitrary date ranges.
    """
    
    def __init__(self, eventstream: Eventstream):
        """
        Initialize the board retention reporter.
        
        Args:
            eventstream: Retentioneering Eventstream with supporter data
        """
        self.eventstream = eventstream
        self.retention_processor = SixMonthRetentionProcessor(eventstream)
        self.fy_calculator = FinancialYearCalculator()
        
    def generate_board_report(
        self,
        start_date: Union[date, datetime],
        end_date: Union[date, datetime],
        include_partial_quarters: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive board retention report for the specified date range.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            include_partial_quarters: Whether to include quarters that are only partially
                                    within the date range
                                    
        Returns:
            Dictionary containing complete board report with retention metrics
        """
        logger.info(f"Generating board retention report from {start_date} to {end_date}")
        
        # Convert to date objects if necessary
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            
        # Get all financial quarters in the date range
        quarters = self._get_quarters_in_range(start_date, end_date, include_partial_quarters)
        
        # Create retention analysis periods (each quarter → next quarter)
        retention_periods = self._create_retention_periods(quarters)
        
        # Calculate retention for each period
        retention_results = []
        for period in retention_periods:
            try:
                result = self._calculate_period_retention(period)
                retention_results.append(result)
                logger.info(f"Calculated retention for {period.label}: {result.retention_rate:.1f}%")
            except Exception as e:
                logger.error(f"Failed to calculate retention for {period.label}: {e}")
                # Continue with other periods rather than failing completely
                
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(retention_results)
        
        # Create the comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now(),
                'date_range': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'quarters_analyzed': len(quarters),
                'retention_periods_calculated': len(retention_results),
                'include_partial_quarters': include_partial_quarters
            },
            'quarters_in_range': [
                {
                    'quarter': q.label,
                    'start_date': q.start_date,
                    'end_date': q.end_date,
                    'financial_year': q.year,
                    'quarter_number': q.quarter
                } for q in quarters
            ],
            'retention_periods': [
                {
                    'period_label': result.period.label,
                    'reference_quarter': result.period.reference_quarter.label,
                    'following_quarter': result.period.following_quarter.label,
                    'retention_rate': result.retention_rate,
                    'supporters_retained': result.supporters_retained,
                    'supporters_not_retained': result.supporters_not_retained,
                    'total_supporters_reference': result.total_supporters_reference,
                    'summary': result.summary_text
                } for result in retention_results
            ],
            'summary_statistics': summary_stats,
            'detailed_results': retention_results  # Full objects for further analysis
        }
        
        logger.info(f"Board report generated successfully with {len(retention_results)} retention periods")
        return report
    
    def _get_quarters_in_range(
        self,
        start_date: date,
        end_date: date,
        include_partial: bool
    ) -> List[FinancialQuarter]:
        """Get all financial quarters within the specified date range."""
        quarters = []
        
        # Start from the financial quarter containing start_date
        current_fy = self.fy_calculator.get_financial_year(start_date)
        current_q = self.fy_calculator.get_quarter_number(start_date)
        
        # If we don't want partial quarters, start from the next complete quarter
        if not include_partial:
            current_quarter = self.fy_calculator.create_financial_quarter(current_fy, current_q)
            if start_date > current_quarter.start_date:
                # Move to next quarter
                current_q += 1
                if current_q > 4:
                    current_q = 1
                    current_fy += 1
        
        # Iterate through quarters until we exceed end_date
        while True:
            quarter = self.fy_calculator.create_financial_quarter(current_fy, current_q)
            
            # Check if this quarter extends beyond our end date
            if quarter.start_date > end_date:
                break
                
            # If we don't want partial quarters, skip if quarter end is beyond range
            if not include_partial and quarter.end_date > end_date:
                break
                
            quarters.append(quarter)
            
            # Move to next quarter
            current_q += 1
            if current_q > 4:
                current_q = 1
                current_fy += 1
                
        return quarters
    
    def _create_retention_periods(self, quarters: List[FinancialQuarter]) -> List[RetentionPeriod]:
        """Create retention analysis periods from consecutive quarters."""
        retention_periods = []
        
        for i in range(len(quarters) - 1):
            reference_quarter = quarters[i]
            following_quarter = quarters[i + 1]
            
            # Create analysis ID
            analysis_id = f"{reference_quarter}_to_{following_quarter}"
            
            period = RetentionPeriod(
                reference_quarter=reference_quarter,
                following_quarter=following_quarter,
                analysis_id=analysis_id
            )
            
            retention_periods.append(period)
            
        return retention_periods
    
    def _calculate_period_retention(self, period: RetentionPeriod) -> RetentionResult:
        """Calculate retention metrics for a specific period."""
        # Convert dates to datetime for the retention processor
        ref_start = datetime.combine(period.reference_quarter.start_date, datetime.min.time())
        ref_end = datetime.combine(period.reference_quarter.end_date, datetime.max.time())
        follow_start = datetime.combine(period.following_quarter.start_date, datetime.min.time())
        follow_end = datetime.combine(period.following_quarter.end_date, datetime.max.time())
        
        # Calculate retention using the existing processor
        retention_result = self.retention_processor.calculate_metric_1(
            last_period_start=ref_start,
            last_period_end=ref_end,
            following_period_start=follow_start,
            following_period_end=follow_end
        )
        
        # Create our result object
        return RetentionResult(
            period=period,
            retention_rate=retention_result['retention_rate'],
            supporters_retained=retention_result['supporters_retained'],
            supporters_not_retained=retention_result['supporters_not_retained'],
            total_supporters_reference=retention_result['total_supporters_last_period'],
            retained_supporter_ids=retention_result['retained_supporter_ids'],
            non_retained_supporter_ids=retention_result['non_retained_supporter_ids'],
            calculation_timestamp=datetime.now()
        )
    
    def _generate_summary_statistics(self, results: List[RetentionResult]) -> Dict[str, Any]:
        """Generate summary statistics across all retention periods."""
        if not results:
            return {
                'message': 'No retention periods calculated',
                'total_periods': 0
            }
            
        retention_rates = [r.retention_rate for r in results]
        total_supporters = [r.total_supporters_reference for r in results]
        
        return {
            'total_periods_analyzed': len(results),
            'average_retention_rate': np.mean(retention_rates),
            'median_retention_rate': np.median(retention_rates),
            'min_retention_rate': min(retention_rates),
            'max_retention_rate': max(retention_rates),
            'std_retention_rate': np.std(retention_rates),
            'total_supporters_across_periods': sum(total_supporters),
            'average_supporters_per_period': np.mean(total_supporters),
            'retention_rate_trend': self._calculate_trend(retention_rates),
            'periods_summary': [
                {
                    'period': result.period.label,
                    'retention_rate': result.retention_rate,
                    'supporters': result.total_supporters_reference
                } for result in results
            ]
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend description."""
        if len(values) < 2:
            return "insufficient_data"
            
        # Simple linear trend
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.5:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"


# Utility functions for common board reporting scenarios

def generate_annual_board_report(
    eventstream: Eventstream,
    financial_year: int
) -> Dict[str, Any]:
    """
    Generate a full annual board report for a specific financial year.
    
    Args:
        eventstream: Supporter eventstream
        financial_year: Financial year to analyze (e.g., 2024)
        
    Returns:
        Complete annual board report
    """
    # Financial year runs July 1 (prev year) to June 30 (this year)
    start_date = date(financial_year - 1, 7, 1)
    end_date = date(financial_year, 6, 30)
    
    reporter = BoardRetentionReporter(eventstream)
    return reporter.generate_board_report(start_date, end_date, include_partial_quarters=False)


def generate_quarterly_board_update(
    eventstream: Eventstream,
    report_date: Union[date, datetime] = None
) -> Dict[str, Any]:
    """
    Generate a quarterly board update report for the most recent complete quarters.
    
    Args:
        eventstream: Supporter eventstream
        report_date: Date of the report (defaults to today)
        
    Returns:
        Quarterly board update report
    """
    if report_date is None:
        report_date = date.today()
    elif isinstance(report_date, datetime):
        report_date = report_date.date()
        
    # Go back 2 quarters to ensure we have at least one complete retention period
    fy_calc = FinancialYearCalculator()
    current_fy = fy_calc.get_financial_year(report_date)
    current_q = fy_calc.get_quarter_number(report_date)
    
    # Calculate the start of the period (2 quarters back)
    start_q = current_q - 2
    start_fy = current_fy
    if start_q <= 0:
        start_q += 4
        start_fy -= 1
        
    start_quarter = fy_calc.create_financial_quarter(start_fy, start_q)
    
    reporter = BoardRetentionReporter(eventstream)
    return reporter.generate_board_report(
        start_quarter.start_date, 
        report_date, 
        include_partial_quarters=False
    )


# Example usage and testing functions

def create_sample_board_report():
    """Example function showing how to use the board reporting system."""
    from src.retention_metrics.six_month_retention_processor import create_sample_retention_data
    
    # Create sample data
    supporter_data = create_sample_retention_data(500)
    
    # Create eventstream
    raw_data_schema = {
        'user_id': 'supporter_id',
        'event_name': 'action_type',
        'event_timestamp': 'action_date'
    }
    
    eventstream = Eventstream(supporter_data, raw_data_schema=raw_data_schema)
    
    # Generate an annual report for FY2024
    annual_report = generate_annual_board_report(eventstream, 2024)
    
    # Generate a quarterly update
    quarterly_report = generate_quarterly_board_update(eventstream)
    
    return {
        'annual_report': annual_report,
        'quarterly_report': quarterly_report
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    reports = create_sample_board_report()
    print("Board reports generated successfully!")
    print(f"Annual report periods: {len(reports['annual_report']['retention_periods'])}")
    print(f"Quarterly report periods: {len(reports['quarterly_report']['retention_periods'])}")