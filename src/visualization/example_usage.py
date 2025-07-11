"""
Example usage of the BG/NBD visualization module.

This script demonstrates how to use the comprehensive visualization toolkit
for BG/NBD model analysis, including model diagnostics, predictions,
business insights, and data quality assessment.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import visualization components
from .plots import (
    BGNBDPlotter,
    create_plotter,
    plot_model_diagnostics,
    plot_predictions,
    create_comprehensive_report
)

# Import model components for demonstration
from ..models.bgnbd import create_bgnbd_model
from ..data.preprocessing import create_bgnbd_processor

logger = logging.getLogger(__name__)


def create_sample_data(n_supporters: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Create sample BG/NBD data for demonstration purposes.
    
    Args:
        n_supporters: Number of supporters to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sample BG/NBD data
    """
    np.random.seed(random_seed)
    
    # Generate synthetic BG/NBD data
    data = []
    
    for i in range(n_supporters):
        # Observation period (days)
        T = 365
        
        # Generate frequency (number of repeat transactions)
        # Using negative binomial distribution
        x = np.random.negative_binomial(n=2, p=0.3)
        
        # Generate recency (time of last transaction)
        if x > 0:
            t_x = np.random.uniform(0, T)
        else:
            t_x = 0
        
        # Calculate additional metrics
        frequency = x + 1  # Include first transaction
        recency_ratio = t_x / T if T > 0 else 0
        
        # Generate monetary value (for CLV analysis)
        if frequency > 0:
            monetary = np.random.lognormal(mean=3.5, sigma=1.0)  # Average ~$30
        else:
            monetary = 0
        
        # Assign engagement segments based on behavior
        if frequency >= 5 and recency_ratio >= 0.7:
            engagement_segment = 'High'
        elif frequency >= 3 and recency_ratio >= 0.5:
            engagement_segment = 'Medium'
        elif frequency >= 1 and recency_ratio >= 0.3:
            engagement_segment = 'Low'
        else:
            engagement_segment = 'Inactive'
        
        # Generate first event date
        first_event_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
        
        data.append({
            'supporter_id': i + 1,
            'x': x,
            't_x': t_x,
            'T': T,
            'frequency': frequency,
            'monetary': monetary,
            'recency_ratio': recency_ratio,
            'engagement_segment': engagement_segment,
            'first_event_date': first_event_date,
            'last_event_date': first_event_date + timedelta(days=t_x) if t_x > 0 else first_event_date,
            'total_weighted_value': frequency * 1.5,
            'total_donation_amount': monetary,
            'engagement_score': (frequency / 10) * 0.4 + recency_ratio * 0.3 + (monetary / 100) * 0.3,
            'avg_donation': monetary / frequency if frequency > 0 else 0,
            'event_type_count': np.random.randint(1, 4),
            'event_diversity_index': np.random.uniform(0, 2),
            'observation_start': datetime.now() - timedelta(days=365),
            'observation_end': datetime.now(),
            'cutoff_date': datetime.now()
        })
    
    return pd.DataFrame(data)


def demonstrate_basic_plotting():
    """Demonstrate basic plotting functionality."""
    print("\n" + "="*60)
    print("BASIC PLOTTING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(n_supporters=500)
    print(f"Generated data for {len(data)} supporters")
    
    # Create and fit a simple BG/NBD model
    print("Fitting BG/NBD model...")
    model = create_bgnbd_model(hierarchical=False)
    model.fit(data, draws=1000, tune=500, chains=2)
    print("Model fitted successfully")
    
    # Create plotter instance
    plotter = create_plotter(
        style='seaborn-v0_8',
        color_palette='husl',
        figure_size=(12, 8),
        dpi=150  # Lower DPI for faster rendering in examples
    )
    
    # Create output directory
    output_dir = Path("outputs/visualization_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving plots to {output_dir}")
    
    # 1. Model diagnostics
    print("Creating model diagnostic plots...")
    if model.trace is not None:
        plotter.plot_trace_diagnostics(
            model, 
            save_path=output_dir / "trace_diagnostics.png",
            show_plot=False
        )
        
        plotter.plot_convergence_summary(
            model,
            save_path=output_dir / "convergence_summary.png", 
            show_plot=False
        )
        print("✓ Model diagnostic plots created")
    
    # 2. Prediction visualizations
    print("Creating prediction plots...")
    plotter.plot_probability_alive(
        model, data,
        save_path=output_dir / "probability_alive.png",
        show_plot=False
    )
    
    plotter.plot_expected_transactions(
        model, data, prediction_period=180,
        save_path=output_dir / "expected_transactions.png",
        show_plot=False
    )
    
    plotter.plot_clv_analysis(
        model, data,
        save_path=output_dir / "clv_analysis.png",
        show_plot=False
    )
    print("✓ Prediction plots created")
    
    # 3. Business intelligence charts
    print("Creating business intelligence plots...")
    plotter.plot_supporter_segments(
        data,
        save_path=output_dir / "supporter_segments.png",
        show_plot=False
    )
    
    plotter.plot_campaign_targeting(
        model, data,
        budget_constraint=10000,
        save_path=output_dir / "campaign_targeting.png",
        show_plot=False
    )
    
    plotter.plot_engagement_trends(
        data,
        save_path=output_dir / "engagement_trends.png",
        show_plot=False
    )
    print("✓ Business intelligence plots created")
    
    # 4. Data quality visualizations
    print("Creating data quality plots...")
    plotter.plot_data_quality_report(
        data,
        save_path=output_dir / "data_quality_report.png",
        show_plot=False
    )
    print("✓ Data quality plots created")
    
    # 5. Comprehensive dashboard
    print("Creating comprehensive dashboard...")
    plotter.create_dashboard(
        model, data,
        save_path=output_dir / "dashboard.png",
        show_plot=False
    )
    print("✓ Dashboard created")
    
    print(f"\nAll plots saved to: {output_dir}")
    return output_dir


def demonstrate_hierarchical_model():
    """Demonstrate hierarchical model visualization."""
    print("\n" + "="*60)
    print("HIERARCHICAL MODEL DEMONSTRATION")
    print("="*60)
    
    # Create sample data with segments
    print("Creating sample data with segments...")
    data = create_sample_data(n_supporters=800)
    print(f"Generated data for {len(data)} supporters")
    print(f"Segments: {data['engagement_segment'].value_counts().to_dict()}")
    
    # Create and fit hierarchical model
    print("Fitting hierarchical BG/NBD model...")
    model = create_bgnbd_model(
        hierarchical=True,
        segment_column='engagement_segment'
    )
    model.fit(data, draws=1000, tune=500, chains=2)
    print("Hierarchical model fitted successfully")
    
    # Create plotter
    plotter = create_plotter(dpi=150)
    
    # Create output directory
    output_dir = Path("outputs/hierarchical_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving hierarchical model plots to {output_dir}")
    
    # Generate all plots for hierarchical model
    saved_plots = plotter.generate_all_plots(
        model, data, output_dir, prediction_period=180
    )
    
    print(f"✓ Generated {len(saved_plots)} plots for hierarchical model")
    print("Plot files:")
    for name, path in saved_plots.items():
        print(f"  • {name}: {path}")
    
    return output_dir


def demonstrate_convenience_functions():
    """Demonstrate convenience functions for quick plotting."""
    print("\n" + "="*60)
    print("CONVENIENCE FUNCTIONS DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(n_supporters=300)
    
    # Create and fit model
    model = create_bgnbd_model()
    model.fit(data, draws=500, tune=250, chains=2)
    
    output_dir = Path("outputs/convenience_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Using convenience functions...")
    
    # 1. Model diagnostics convenience function
    print("Creating model diagnostics...")
    diagnostic_figures = plot_model_diagnostics(
        model, output_dir=output_dir, show_plots=False
    )
    print(f"✓ Created {len(diagnostic_figures)} diagnostic plots")
    
    # 2. Predictions convenience function
    print("Creating prediction plots...")
    prediction_figures = plot_predictions(
        model, data, prediction_period=180,
        output_dir=output_dir, show_plots=False
    )
    print(f"✓ Created {len(prediction_figures)} prediction plots")
    
    # 3. Comprehensive report
    print("Creating comprehensive report...")
    report = create_comprehensive_report(
        model, data, output_dir, prediction_period=180
    )
    print("✓ Comprehensive report created")
    print(f"Report summary: {report['summary_file']}")
    print(f"Generated {len(report['plots_generated'])} plots")
    
    return output_dir


def demonstrate_customization():
    """Demonstrate plot customization options."""
    print("\n" + "="*60)
    print("CUSTOMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(n_supporters=400)
    model = create_bgnbd_model()
    model.fit(data, draws=500, tune=250, chains=2)
    
    output_dir = Path("outputs/customization_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Demonstrating different styling options...")
    
    # Different color palettes and styles
    styles = [
        ('default', 'Set1', 'Default Style'),
        ('seaborn-v0_8', 'husl', 'Seaborn Style'),
        ('ggplot', 'viridis', 'GGPlot Style')
    ]
    
    for style, palette, description in styles:
        print(f"Creating plots with {description}...")
        
        try:
            plotter = create_plotter(
                style=style,
                color_palette=palette,
                figure_size=(10, 6),
                dpi=150,
                font_scale=0.9
            )
            
            # Create a sample plot with this style
            style_dir = output_dir / style.replace('-', '_')
            style_dir.mkdir(exist_ok=True)
            
            plotter.plot_probability_alive(
                model, data,
                save_path=style_dir / "probability_alive.png",
                show_plot=False
            )
            
            plotter.plot_supporter_segments(
                data,
                save_path=style_dir / "supporter_segments.png",
                show_plot=False
            )
            
            print(f"✓ {description} plots created in {style_dir}")
            
        except Exception as e:
            print(f"✗ Error with {description}: {e}")
    
    return output_dir


def run_all_examples():
    """Run all visualization examples."""
    print("BG/NBD VISUALIZATION MODULE EXAMPLES")
    print("="*60)
    print("This script demonstrates the comprehensive visualization")
    print("capabilities of the BG/NBD engagement model.")
    print("="*60)
    
    try:
        # Run all demonstrations
        basic_dir = demonstrate_basic_plotting()
        hierarchical_dir = demonstrate_hierarchical_model()
        convenience_dir = demonstrate_convenience_functions()
        custom_dir = demonstrate_customization()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Output directories:")
        print(f"  • Basic plotting: {basic_dir}")
        print(f"  • Hierarchical model: {hierarchical_dir}")
        print(f"  • Convenience functions: {convenience_dir}")
        print(f"  • Customization: {custom_dir}")
        print("\nCheck these directories for generated plots and reports.")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\n✗ Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all examples
    run_all_examples()