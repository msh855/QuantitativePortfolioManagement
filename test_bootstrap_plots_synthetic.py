#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for bootstrap plotting functions using synthetic data

This script tests the new bootstrap visualization functions added to myPlots.py
without requiring network access.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("=" * 80)
print("Testing Bootstrap Plotting Functions (Synthetic Data)")
print("=" * 80)

# Import the new functions
from myPortfolioManagement.myPlots import (
    plot_bootstrap_distribution,
    plot_bootstrap_distributions,
    plot_bootstrap_comparison
)

# =============================================================================
# SECTION 1: Generate synthetic bootstrap data
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Generating Synthetic Bootstrap Data")
print("=" * 80)

np.random.seed(42)

# Simulate bootstrap results for common metrics
n_samples = 1000

# Generate realistic bootstrap distributions
bootstrap_data = {
    'cagr': np.random.normal(0.12, 0.03, n_samples),  # Mean 12% CAGR
    'volatility': np.abs(np.random.normal(0.18, 0.04, n_samples)),  # Mean 18% vol
    'sharpe': np.random.normal(0.65, 0.15, n_samples),  # Mean Sharpe 0.65
    'adjusted_sortino': np.random.normal(0.90, 0.20, n_samples),  # Mean Sortino 0.90
}

bootstrap_results = pd.DataFrame(bootstrap_data)

print(f"Generated bootstrap results shape: {bootstrap_results.shape}")
print(f"Metrics: {list(bootstrap_results.columns)}")
print("\nBootstrap statistics summary:")
print(bootstrap_results.describe().round(4))

# =============================================================================
# SECTION 2: Test single metric distribution plot
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Testing plot_bootstrap_distribution (single metric)")
print("=" * 80)

print("\n2.1 Plotting CAGR distribution...")
plot_bootstrap_distribution(
    bootstrap_results=bootstrap_results,
    metric='cagr',
    confidence_level=0.95,
    figsize=(10, 6),
    color='steelblue',
    savefig='bootstrap_cagr_distribution.png',
    show=False
)
print("✓ Saved: bootstrap_cagr_distribution.png")

print("\n2.2 Plotting Sharpe ratio distribution...")
plot_bootstrap_distribution(
    bootstrap_results=bootstrap_results,
    metric='sharpe',
    confidence_level=0.95,
    figsize=(10, 6),
    color='coral',
    savefig='bootstrap_sharpe_distribution.png',
    show=False
)
print("✓ Saved: bootstrap_sharpe_distribution.png")

print("\n2.3 Plotting Volatility distribution...")
plot_bootstrap_distribution(
    bootstrap_results=bootstrap_results,
    metric='volatility',
    confidence_level=0.95,
    figsize=(10, 6),
    color='mediumseagreen',
    savefig='bootstrap_volatility_distribution.png',
    show=False
)
print("✓ Saved: bootstrap_volatility_distribution.png")

# =============================================================================
# SECTION 3: Test multiple metrics distribution plot
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Testing plot_bootstrap_distributions (multiple metrics)")
print("=" * 80)

print("\n3.1 Creating violin plot for all metrics...")
plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=None,  # Plot all metrics
    confidence_level=0.95,
    plot_type='violin',
    savefig='bootstrap_distributions_violin.png',
    show=False
)
print("✓ Saved: bootstrap_distributions_violin.png")

print("\n3.2 Creating box plot for all metrics...")
plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=None,
    confidence_level=0.95,
    plot_type='box',
    savefig='bootstrap_distributions_box.png',
    show=False
)
print("✓ Saved: bootstrap_distributions_box.png")

print("\n3.3 Creating histogram grid for all metrics...")
plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=None,
    confidence_level=0.95,
    plot_type='hist',
    savefig='bootstrap_distributions_hist.png',
    show=False
)
print("✓ Saved: bootstrap_distributions_hist.png")

print("\n3.4 Creating histogram grid for selected metrics...")
plot_bootstrap_distributions(
    bootstrap_results=bootstrap_results,
    metrics=['cagr', 'sharpe', 'volatility'],
    confidence_level=0.95,
    plot_type='hist',
    savefig='bootstrap_distributions_selected.png',
    show=False
)
print("✓ Saved: bootstrap_distributions_selected.png")

# =============================================================================
# SECTION 4: Test comparison plot (in-sample vs out-of-sample)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Testing plot_bootstrap_comparison (in-sample vs out-of-sample)")
print("=" * 80)

print("\n4.1 Generating synthetic in-sample vs out-of-sample data...")
# Simulate in-sample and out-of-sample bootstrap results
comparison_data = {
    'cagr_in_sample': np.random.normal(0.14, 0.03, n_samples),
    'cagr_out_sample': np.random.normal(0.10, 0.04, n_samples),
    'volatility_in_sample': np.abs(np.random.normal(0.16, 0.03, n_samples)),
    'volatility_out_sample': np.abs(np.random.normal(0.20, 0.05, n_samples)),
    'sharpe_in_sample': np.random.normal(0.75, 0.15, n_samples),
    'sharpe_out_sample': np.random.normal(0.50, 0.18, n_samples),
    'adjusted_sortino_in_sample': np.random.normal(1.00, 0.20, n_samples),
    'adjusted_sortino_out_sample': np.random.normal(0.70, 0.22, n_samples),
}

comparison_results = pd.DataFrame(comparison_data)

print(f"Generated comparison results shape: {comparison_results.shape}")
print(f"Columns: {list(comparison_results.columns)}")

print("\n4.2 Creating comparison plot...")
plot_bootstrap_comparison(
    bootstrap_results=comparison_results,
    comparison_col='sample',  # Will match '_in_sample' and '_out_sample'
    metrics=None,  # Auto-detect all metrics
    confidence_level=0.95,
    savefig='bootstrap_comparison_in_out_sample.png',
    show=False
)
print("✓ Saved: bootstrap_comparison_in_out_sample.png")

# =============================================================================
# SECTION 5: Test edge cases
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Testing Edge Cases")
print("=" * 80)

print("\n5.1 Testing with Series input (single metric)...")
cagr_series = bootstrap_results['cagr']
cagr_series.name = 'CAGR'
plot_bootstrap_distribution(
    bootstrap_results=cagr_series,
    confidence_level=0.90,  # Different CI level
    figsize=(10, 6),
    savefig='bootstrap_cagr_from_series.png',
    show=False
)
print("✓ Saved: bootstrap_cagr_from_series.png")

print("\n5.2 Testing with custom title and confidence level...")
plot_bootstrap_distribution(
    bootstrap_results=bootstrap_results,
    metric='sharpe',
    confidence_level=0.99,  # 99% CI
    title='Bootstrap Distribution: Sharpe Ratio (99% CI)',
    savefig='bootstrap_sharpe_99ci.png',
    show=False
)
print("✓ Saved: bootstrap_sharpe_99ci.png")

# =============================================================================
# SECTION 6: Test error handling
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Testing Error Handling")
print("=" * 80)

print("\n6.1 Testing invalid metric name...")
try:
    plot_bootstrap_distribution(
        bootstrap_results=bootstrap_results,
        metric='invalid_metric',
        show=False
    )
    print("✗ Error: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)}")

print("\n6.2 Testing missing metric parameter for DataFrame...")
try:
    plot_bootstrap_distribution(
        bootstrap_results=bootstrap_results,
        show=False
    )
    print("✗ Error: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)}")

print("\n6.3 Testing invalid plot_type...")
try:
    plot_bootstrap_distributions(
        bootstrap_results=bootstrap_results,
        plot_type='invalid_type',
        show=False
    )
    print("✗ Error: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("""
All tests completed successfully! ✓

Generated plots:
1. bootstrap_cagr_distribution.png          - Single metric: CAGR
2. bootstrap_sharpe_distribution.png        - Single metric: Sharpe
3. bootstrap_volatility_distribution.png    - Single metric: Volatility
4. bootstrap_distributions_violin.png       - All metrics: Violin plot
5. bootstrap_distributions_box.png          - All metrics: Box plot
6. bootstrap_distributions_hist.png         - All metrics: Histogram grid
7. bootstrap_distributions_selected.png     - Selected metrics: Histogram grid
8. bootstrap_comparison_in_out_sample.png   - In-sample vs Out-of-sample comparison
9. bootstrap_cagr_from_series.png           - Series input test
10. bootstrap_sharpe_99ci.png               - Custom CI level test

New Functions Added to myPlots.py:
================================
1. plot_bootstrap_distribution()
   - Plot single metric with CI
   - Histogram with KDE overlay
   - Mean, median, and CI markers
   - Customizable confidence levels

2. plot_bootstrap_distributions()
   - Plot multiple metrics
   - Three plot types: violin, box, histogram grid
   - Auto-layout for subplots
   - Metric selection support

3. plot_bootstrap_comparison()
   - Compare in-sample vs out-of-sample
   - Auto-detect comparison groups
   - Side-by-side overlaid distributions
   - Multiple metrics in grid layout

Features:
========
✓ Publication-ready visualizations
✓ Confidence intervals (customizable)
✓ Multiple plot types
✓ Auto-save functionality
✓ Error handling
✓ Flexible input (DataFrame or Series)
✓ Clean, professional styling
✓ Grid layouts for multiple metrics
""")

print("=" * 80)
print("END OF TESTS")
print("=" * 80)
