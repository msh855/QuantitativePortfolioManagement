#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for bootstrap plotting functions

This script tests the new bootstrap visualization functions added to myPlots.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("Testing Bootstrap Plotting Functions")
print("=" * 80)

# Import the new functions
from myPortfolioManagement.myPlots import (
    plot_bootstrap_distribution,
    plot_bootstrap_distributions,
    plot_bootstrap_comparison
)

from myPortfolioManagement.myBacktesting import (
    bootstrap_stats,
    bootstrap_portfolio_performance
)

from myPortfolioManagement.myData import get_stock_prices
import quantstats_lumi as qs

# =============================================================================
# SECTION 1: Generate sample bootstrap data
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Generating Sample Bootstrap Data")
print("=" * 80)

print("\n1.1 Fetching sample data...")
# Fetch SPY data for testing
prices = get_stock_prices(
    yahoo_tickers=['SPY'],
    start_date='2020-01-01',
    end_date='2024-01-01',
    freq='daily',
    wide_format=True
)

print(f"Fetched {len(prices)} days of price data")

# Calculate returns
returns = prices.pct_change().dropna().squeeze()
returns.name = 'SPY'

print(f"\n1.2 Running bootstrap_stats (this may take a minute)...")
# Run bootstrap to get sample data
bootstrap_results = bootstrap_stats(
    returns=returns,
    returns_benchmark=None,
    rf=0.04,
    periods=252,
    n_sim=500  # Reduced for faster testing
)

print(f"Bootstrap results shape: {bootstrap_results.shape}")
print(f"Metrics calculated: {list(bootstrap_results.columns)}")
print("\nBootstrap statistics summary:")
print(bootstrap_results.describe())

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

print("\n4.1 Running bootstrap_portfolio_performance with in/out-of-sample split...")
# Run bootstrap with in-sample/out-of-sample split
means, distributions, dist_stats = bootstrap_portfolio_performance(
    returns=returns,
    returns_benchmark=None,
    periods=252,
    rf=0.04,
    out_of_sample_date='2023-01-01',
    n_sim=500  # Reduced for faster testing
)

print(f"Distribution results shape: {distributions.shape}")
print(f"Columns: {list(distributions.columns)}")

print("\n4.2 Creating comparison plot...")
plot_bootstrap_comparison(
    bootstrap_results=distributions,
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
1. plot_bootstrap_distribution()     - Plot single metric with CI
2. plot_bootstrap_distributions()    - Plot multiple metrics (violin/box/hist)
3. plot_bootstrap_comparison()       - Compare in-sample vs out-of-sample

Features:
- Histogram with KDE overlay
- Mean, median, and confidence interval markers
- Multiple plot types (violin, box, histogram grid)
- In-sample vs out-of-sample comparison
- Customizable confidence levels
- Auto-save functionality
- Clean, publication-ready visualizations
""")

print("=" * 80)
print("END OF TESTS")
print("=" * 80)
