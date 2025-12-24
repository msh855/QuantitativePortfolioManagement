#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap Visualization Example
================================

This script demonstrates the new bootstrap visualization functions for
plotting distributions of performance metrics like CAGR, Sharpe, Volatility, etc.

Usage:
    python example_bootstrap_visualization.py
    
Features Demonstrated:
1. Single metric distribution with confidence intervals
2. Multiple metrics comparison (violin/box/histogram plots)
3. In-sample vs out-of-sample comparison

Author: myPortfolioManagement
Date: December 2024
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("=" * 80)
print("Bootstrap Visualization Example")
print("=" * 80)

# =============================================================================
# Generate Sample Bootstrap Data
# =============================================================================
print("\nGenerating sample bootstrap data...")

np.random.seed(42)
n_samples = 1000

# Simulate realistic bootstrap distributions for portfolio metrics
bootstrap_data = {
    'cagr': np.random.normal(0.12, 0.03, n_samples),
    'volatility': np.abs(np.random.normal(0.18, 0.04, n_samples)),
    'sharpe': np.random.normal(0.65, 0.15, n_samples),
    'adjusted_sortino': np.random.normal(0.90, 0.20, n_samples),
}

bootstrap_results = pd.DataFrame(bootstrap_data)
print(f"✓ Generated {n_samples} bootstrap samples for {len(bootstrap_data)} metrics")

# =============================================================================
# Import Plotting Functions
# =============================================================================
from myPortfolioManagement.myPlots import (
    plot_bootstrap_distribution,
    plot_bootstrap_distributions,
    plot_bootstrap_comparison
)

# =============================================================================
# Example 1: Single Metric Distribution
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: Plotting Single Metric (CAGR)")
print("=" * 80)

plot_bootstrap_distribution(
    bootstrap_results=bootstrap_results,
    metric='cagr',
    confidence_level=0.95,
    figsize=(10, 6),
    color='steelblue',
    title='Bootstrap Distribution: CAGR with 95% Confidence Interval',
    savefig='example_cagr_distribution.png',
    show=False
)

print("✓ Saved: example_cagr_distribution.png")
print("  Features:")
print("  - Histogram with KDE overlay")
print("  - Mean and median markers")
print("  - 95% confidence interval bounds")

# =============================================================================
# Example 2: Multiple Metrics - Violin Plot
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Multiple Metrics Comparison (Violin Plot)")
print("=" * 80)

plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=None,  # Plot all metrics
    confidence_level=0.95,
    plot_type='violin',
    title='Bootstrap Distributions: All Metrics',
    savefig='example_metrics_violin.png',
    show=False
)

print("✓ Saved: example_metrics_violin.png")
print("  Features:")
print("  - Distribution shape for all metrics")
print("  - Mean values marked with diamonds")
print("  - Inner box plot showing quartiles")

# =============================================================================
# Example 3: Multiple Metrics - Box Plot
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: Multiple Metrics Comparison (Box Plot)")
print("=" * 80)

plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=['cagr', 'sharpe', 'volatility'],
    confidence_level=0.95,
    plot_type='box',
    title='Bootstrap Distributions: Selected Metrics',
    savefig='example_metrics_box.png',
    show=False
)

print("✓ Saved: example_metrics_box.png")
print("  Features:")
print("  - Quartile distributions")
print("  - Outlier detection")
print("  - Mean values marked")

# =============================================================================
# Example 4: Multiple Metrics - Histogram Grid
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Multiple Metrics (Histogram Grid)")
print("=" * 80)

plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=None,
    confidence_level=0.95,
    plot_type='hist',
    title='Bootstrap Distributions: Histogram Grid',
    savefig='example_metrics_hist_grid.png',
    show=False
)

print("✓ Saved: example_metrics_hist_grid.png")
print("  Features:")
print("  - Individual histograms for each metric")
print("  - Mean and confidence intervals per metric")
print("  - Grid layout for easy comparison")

# =============================================================================
# Example 5: In-Sample vs Out-of-Sample Comparison
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: In-Sample vs Out-of-Sample Comparison")
print("=" * 80)

# Simulate in-sample and out-of-sample bootstrap results
# (In real usage, these would come from bootstrap_portfolio_performance)
comparison_data = {
    'cagr_in_sample': np.random.normal(0.14, 0.03, n_samples),
    'cagr_out_sample': np.random.normal(0.10, 0.04, n_samples),
    'volatility_in_sample': np.abs(np.random.normal(0.16, 0.03, n_samples)),
    'volatility_out_sample': np.abs(np.random.normal(0.20, 0.05, n_samples)),
    'sharpe_in_sample': np.random.normal(0.75, 0.15, n_samples),
    'sharpe_out_sample': np.random.normal(0.50, 0.18, n_samples),
}

comparison_results = pd.DataFrame(comparison_data)

plot_bootstrap_comparison(
    bootstrap_results=comparison_results,
    comparison_col='sample',  # Auto-detects '_in_sample' and '_out_sample'
    metrics=None,  # Plot all metrics
    confidence_level=0.95,
    title='In-Sample vs Out-of-Sample Performance',
    savefig='example_in_out_comparison.png',
    show=False
)

print("✓ Saved: example_in_out_comparison.png")
print("  Features:")
print("  - Side-by-side distribution comparison")
print("  - Automatic detection of comparison groups")
print("  - Overlaid histograms with transparency")
print("  - Mean values for each group")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
All example plots generated successfully!

Files Created:
1. example_cagr_distribution.png      - Single metric with CI
2. example_metrics_violin.png         - Violin plot comparison
3. example_metrics_box.png            - Box plot comparison
4. example_metrics_hist_grid.png      - Histogram grid
5. example_in_out_comparison.png      - In/Out-of-sample comparison

How to Use in Your Code:
========================

1. Import the functions:
   from myPortfolioManagement.myPlots import (
       plot_bootstrap_distribution,
       plot_bootstrap_distributions,
       plot_bootstrap_comparison
   )

2. Generate bootstrap results:
   from myPortfolioManagement.myBacktesting import bootstrap_stats
   
   results = bootstrap_stats(
       returns=your_returns,
       returns_benchmark=benchmark_returns,
       rf=0.04,
       periods=252,
       n_sim=1000
   )

3. Visualize:
   # Single metric
   plot_bootstrap_distribution(results, metric='cagr')
   
   # Multiple metrics
   plot_bootstrap_distributions(results, plot_type='violin')
   
   # In-sample vs out-of-sample
   plot_bootstrap_comparison(in_out_results)

Key Features:
=============
✓ Publication-ready visualizations
✓ Customizable confidence intervals
✓ Multiple plot types (violin, box, histogram)
✓ Automatic layout and styling
✓ Error handling and validation
✓ Flexible input formats
✓ Auto-save functionality

For more information, see the documentation in myPlots.py
""")

print("=" * 80)
print("END OF EXAMPLE")
print("=" * 80)
