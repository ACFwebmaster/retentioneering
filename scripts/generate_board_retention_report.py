#!/usr/bin/env python3
"""
Generate Board Retention Reports - CLI Script

This script generates board-level retention reports with financial quarter breakdowns.
Supports various reporting scenarios including annual reports and quarterly updates.

Usage:
    python scripts/reporting/generate_board_retention_report.py --help
    python scripts/reporting/generate_board_retention_report.py --annual 2024
    python scripts/reporting/generate_board_retention_report.py --quarterly
    python scripts/reporting/generate_board_retention_report.py --custom 2023-07-01 2024-06-30
"""

import argparse
import sys
import logging
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.retention_metrics.board_retention_reporter import (
    BoardRetentionReporter,
    generate_annual_board_report,
    generate_quarterly_board_update
)
from src.data.loader import SupporterDataLoader  # Assuming this exists
from retentioneering.eventstream import Eventstream


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('board_retention_report.log')
        ]
    )


def load_supporter_eventstream() -> Eventstream:
    """
    Load supporter data and create eventstream.
    This function should be customized based on your data loading requirements.
    """
    try:
        # Option 1: Load from database (customize as needed)
        # loader = SupporterDataLoader()
        # supporter_data = loader.load_supporter_transactions()
        
        # Option 2: For testing, use sample data
        from src.retention_metrics.six_month_retention_processor import create_sample_retention_data
        supporter_data = create_sample_retention_data(1000)
        
        # Create eventstream
        raw_data_schema = {
            'user_id': 'supporter_id',
            'event_name': 'action_type',
            'event_timestamp': 'action_date'
        }
        
        eventstream = Eventstream(supporter_data, raw_data_schema=raw_data_schema)
        logging.info(f"Loaded eventstream with {len(supporter_data)} events")
        return eventstream
        
    except Exception as e:
        logging.error(f"Failed to load supporter data: {e}")
        raise


def save_report(report: Dict[str, Any], output_path: Path, format_type: str = 'json') -> None:
    """Save report to file in specified format."""
    try:
        if format_type.lower() == 'json':
            # Convert datetime objects to strings for JSON serialization
            report_serializable = convert_datetimes_to_strings(report)
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report_serializable, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        logging.info(f"Report saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save report: {e}")
        raise


def convert_datetimes_to_strings(obj):
    """Recursively convert datetime objects to strings for JSON serialization."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_datetimes_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes_to_strings(item) for item in obj]
    else:
        return obj


def print_report_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the report to console."""
    print("\n" + "="*80)
    print("BOARD RETENTION REPORT SUMMARY")
    print("="*80)
    
    # Report metadata
    metadata = report['report_metadata']
    print(f"Generated: {metadata['generated_at']}")
    print(f"Date Range: {metadata['date_range']['start_date']} to {metadata['date_range']['end_date']}")
    print(f"Quarters Analyzed: {metadata['quarters_analyzed']}")
    print(f"Retention Periods: {metadata['retention_periods_calculated']}")
    
    # Summary statistics
    if 'summary_statistics' in report and report['summary_statistics'].get('total_periods_analyzed', 0) > 0:
        stats = report['summary_statistics']
        print(f"\nSUMMARY STATISTICS:")
        print(f"Average Retention Rate: {stats['average_retention_rate']:.1f}%")
        print(f"Median Retention Rate: {stats['median_retention_rate']:.1f}%")
        print(f"Range: {stats['min_retention_rate']:.1f}% - {stats['max_retention_rate']:.1f}%")
        print(f"Trend: {stats['retention_rate_trend'].upper()}")
        print(f"Total Supporters Analyzed: {stats['total_supporters_across_periods']:,}")
    
    # Individual period results
    print(f"\nINDIVIDUAL PERIOD RESULTS:")
    print("-" * 80)
    for period in report['retention_periods']:
        print(f"{period['period_label']}: {period['retention_rate']:.1f}% "
              f"({period['supporters_retained']}/{period['total_supporters_reference']} supporters)")
    
    print("="*80)


def generate_annual_report(year: int, output_dir: Path) -> None:
    """Generate annual board report."""
    logging.info(f"Generating annual board report for FY{year}")
    
    # Load data
    eventstream = load_supporter_eventstream()
    
    # Generate report
    report = generate_annual_board_report(eventstream, year)
    
    # Save and display
    output_file = output_dir / f"board_retention_annual_FY{year}"
    save_report(report, output_file)
    print_report_summary(report)


def generate_quarterly_report(output_dir: Path, report_date: Optional[date] = None) -> None:
    """Generate quarterly board update."""
    logging.info("Generating quarterly board update")
    
    # Load data
    eventstream = load_supporter_eventstream()
    
    # Generate report
    report = generate_quarterly_board_update(eventstream, report_date)
    
    # Save and display
    date_str = (report_date or date.today()).strftime("%Y%m%d")
    output_file = output_dir / f"board_retention_quarterly_{date_str}"
    save_report(report, output_file)
    print_report_summary(report)


def generate_custom_report(start_date: date, end_date: date, output_dir: Path) -> None:
    """Generate custom date range report."""
    logging.info(f"Generating custom board report from {start_date} to {end_date}")
    
    # Load data
    eventstream = load_supporter_eventstream()
    
    # Generate report
    reporter = BoardRetentionReporter(eventstream)
    report = reporter.generate_board_report(start_date, end_date, include_partial_quarters=True)
    
    # Save and display
    output_file = output_dir / f"board_retention_custom_{start_date}_{end_date}"
    save_report(report, output_file)
    print_report_summary(report)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate Board Retention Reports with Financial Quarter Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate annual report for FY2024:
    python scripts/reporting/generate_board_retention_report.py --annual 2024

  Generate quarterly update for today:
    python scripts/reporting/generate_board_retention_report.py --quarterly

  Generate quarterly update for specific date:
    python scripts/reporting/generate_board_retention_report.py --quarterly --date 2024-03-31

  Generate custom date range report:
    python scripts/reporting/generate_board_retention_report.py --custom 2023-07-01 2024-06-30
        """
    )
    
    # Report type (mutually exclusive)
    report_group = parser.add_mutually_exclusive_group(required=True)
    report_group.add_argument(
        '--annual', 
        type=int, 
        metavar='YEAR',
        help='Generate annual report for specified financial year (e.g., 2024 for FY2024)'
    )
    report_group.add_argument(
        '--quarterly', 
        action='store_true',
        help='Generate quarterly board update report'
    )
    report_group.add_argument(
        '--custom', 
        nargs=2, 
        metavar=('START_DATE', 'END_DATE'),
        help='Generate custom date range report (format: YYYY-MM-DD)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--date',
        type=str,
        metavar='YYYY-MM-DD',
        help='Specific date for quarterly report (defaults to today)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reports',
        metavar='PATH',
        help='Output directory for reports (default: ./reports)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.annual:
            generate_annual_report(args.annual, output_dir)
            
        elif args.quarterly:
            report_date = None
            if args.date:
                report_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            generate_quarterly_report(output_dir, report_date)
            
        elif args.custom:
            start_date = datetime.strptime(args.custom[0], '%Y-%m-%d').date()
            end_date = datetime.strptime(args.custom[1], '%Y-%m-%d').date()
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
                
            generate_custom_report(start_date, end_date, output_dir)
        
        logging.info("Report generation completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Report generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()