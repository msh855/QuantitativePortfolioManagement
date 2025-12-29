#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implied Probability Distribution Analysis Example

This script demonstrates how to:
1. Create option pricing models for stocks without option contracts
2. Extract implied probability distributions from option prices
3. Compare implied distributions with bootstrapped historical distributions
4. Identify potential mispricing opportunities

Author: MyPortfolioManagement Team
Date: December 2024
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Ensure local imports work when running this script from any directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

print("=" * 80)
print("Implied Probability Distribution Analysis")
print("=" * 80)

# =============================================================================
# SECTION 1: SETUP - Fetch Historical Data
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: FETCH HISTORICAL PRICE DATA")
print("=" * 80)

# Select a stock for analysis
ticker = 'AAPL'
print(f"\nAnalyzing: {ticker}")

# Generate synthetic data for demonstration
# In real usage, you would fetch actual data using:
# from myPortfolioManagement.myData import get_stock_prices
# from myPortfolioManagement.myReturns import calculate_returns
print("Using synthetic data for demonstration...")

np.random.seed(42)
dates = pd.date_range(start='2019-01-01', end='2024-12-01', freq='B')
returns = pd.Series(
    np.random.normal(0.0005, 0.015, len(dates)),
    index=dates,
    name=ticker
)
prices = pd.DataFrame(
    100 * (1 + returns).cumprod(),
    columns=[ticker]
)
print(f"✓ Generated synthetic data: {len(prices)} days")
print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
print(f"✓ Calculated returns: mean={returns.mean()*252:.2%}, std={returns.std()*np.sqrt(252):.2%}")

S0 = prices.iloc[-1, 0]  # Current stock price

# =============================================================================
# SECTION 2: BOOTSTRAP FUTURE DISTRIBUTION (FIRST)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: BOOTSTRAP HISTORICAL DISTRIBUTION")
print("=" * 80)

# Parameters for option pricing / horizon
T_years = 5.0  # 5-year horizon
r = 0.05  # Risk-free rate (5%)
historical_vol = returns.std() * np.sqrt(252)  # Annualized volatility

print(f"\nOption Parameters:")
print(f"  Current Price (S): ${S0:.2f}")
print(f"  Time to Expiration (T): {T_years} years")
print(f"  Risk-free Rate (r): {r:.2%}")
print(f"  Historical Volatility: {historical_vol:.2%}")

print(f"\nBootstrapping future prices using historical returns...")
print(f"  Number of simulations: 10,000")
print(f"  Bootstrap method: IID (using simple resampling)")

np.random.seed(123)
n_sim = 10000
periods = int(T_years * 252)

bootstrap_prices = []
for _ in range(n_sim):
    sampled_returns = np.random.choice(returns.values, size=periods, replace=True)
    cumulative_return = np.prod(1 + sampled_returns)
    future_price = S0 * cumulative_return
    bootstrap_prices.append(future_price)

bootstrap_dist = pd.DataFrame({
    'simulation': range(n_sim),
    'future_price': bootstrap_prices
})

print(f"✓ Bootstrapped distribution created")
print(f"\nBootstrap Distribution Statistics:")
print(f"  Mean Price: ${bootstrap_dist['future_price'].mean():.2f}")
print(f"  Std Deviation: ${bootstrap_dist['future_price'].std():.2f}")
print(f"  Min Price: ${bootstrap_dist['future_price'].min():.2f}")
print(f"  Max Price: ${bootstrap_dist['future_price'].max():.2f}")

# =============================================================================
# SECTION 3: CREATE OPTION CHAIN (USING BOOTSTRAP RANGE)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: CREATE SYNTHETIC OPTION CHAIN")
print("=" * 80)

from myPortfolioManagement.myOptionPricing import create_option_chain

# Use an asymmetric range: keep the downside from the bootstrap minimum (to avoid
# clipping the left tail), while trimming only the extreme upside via a high
# quantile to reduce the impact of a few right-tail outliers.
q_high = 0.999  # 99.9%
low_price = float(bootstrap_dist['future_price'].min())
high_price = float(bootstrap_dist['future_price'].quantile(q_high))

# Convert to spot multipliers expected by create_option_chain().
# Add a small buffer so the strikes extend slightly beyond the chosen quantiles.
buffer = 0.05
min_mult = max(0.01, (low_price / S0) * (1.0 - buffer))
max_mult = max(min_mult * 1.05, (high_price / S0) * (1.0 + buffer))

print(f"\nStrike range derived from bootstrap (min to upper-quantile):")
print(f"  Upper quantile: {q_high:.3%}")
print(f"  Bootstrapped future price range used: ${low_price:.2f} to ${high_price:.2f}")
print(f"  Strike multipliers vs spot: {min_mult:.3f}x to {max_mult:.3f}x")

option_chain = create_option_chain(
    S=S0,
    T=T_years,
    r=r,
    sigma=historical_vol,
    strike_range=(min_mult, max_mult),
    num_strikes=80
)

print(f"\n✓ Created option chain with {len(option_chain)} strikes")
print(f"  Strike range: ${option_chain['strike'].min():.2f} to ${option_chain['strike'].max():.2f}")
print("\nSample option prices:")
print(option_chain.head(5).to_string(index=False))

# =============================================================================
# SECTION 4: EXTRACT IMPLIED DISTRIBUTION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: EXTRACT IMPLIED PROBABILITY DISTRIBUTION")
print("=" * 80)

from myPortfolioManagement.myImpliedDistribution import extract_implied_distribution

# Extract implied distribution from call options
implied_dist = extract_implied_distribution(
    option_chain=option_chain,
    S=S0,
    r=r,
    T=T_years,
    option_type='call',
    return_range_mass=True
)

# Plot the density curve (histogram of density values is misleading)
plt.figure(figsize=(10, 4))
plt.plot(implied_dist['price_level'], implied_dist['probability_density'], linewidth=2)
plt.title('Implied Risk-Neutral Density (from option prices)')
plt.xlabel('Future price level (strike)')
plt.ylabel('Probability density')
plt.grid(True, alpha=0.3)

print(f"\n✓ Extracted implied distribution")
print(f"  Price range: ${implied_dist['price_level'].min():.2f} to ${implied_dist['price_level'].max():.2f}")
print(f"  Number of points: {len(implied_dist)}")
if 'probability_mass_in_range' in implied_dist.columns:
    mass_in_range = implied_dist['probability_mass_in_range'].iloc[0]
    print(f"  Probability mass in strike range: {mass_in_range:.2%}")

# Calculate implied statistics
from scipy.integrate import simpson

implied_prices = implied_dist['price_level'].values
implied_probs = implied_dist['probability_density'].values

# Normalize
prob_sum = simpson(implied_probs, x=implied_prices)
if prob_sum > 0:
    implied_probs = implied_probs / prob_sum

implied_mean = simpson(implied_prices * implied_probs, x=implied_prices)
implied_var = simpson((implied_prices - implied_mean)**2 * implied_probs, x=implied_prices)
implied_std = np.sqrt(implied_var)

print(f"\nImplied Distribution Statistics:")
print(f"  Expected Price: ${implied_mean:.2f}")
print(f"  Std Deviation: ${implied_std:.2f}")
print(f"  Expected Return: {((implied_mean / S0) ** (1/T_years) - 1):.2%} annualized")

# =============================================================================
# SECTION 5: COMPARE DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: COMPARE IMPLIED VS BOOTSTRAPPED DISTRIBUTIONS")
print("=" * 80)

from myPortfolioManagement.myImpliedDistribution import compare_distributions

comparison = compare_distributions(
    implied_dist=implied_dist,
    bootstrap_dist=bootstrap_dist,
    S0=S0,
    confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
)

print("\nDistribution Comparison:")
print(comparison.to_string(index=False))

# =============================================================================
# SECTION 6: IDENTIFY MISPRICING OPPORTUNITIES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: IDENTIFY POTENTIAL MISPRICING")
print("=" * 80)

from myPortfolioManagement.myImpliedDistribution import find_mispricing_opportunities

mispricings = find_mispricing_opportunities(
    comparison=comparison,
    threshold_pct=5.0  # Flag differences > 5%
)

print("\nPotential Mispricing Opportunities:")
if len(mispricings) > 0 and 'opportunity' in mispricings.columns:
    print(mispricings.to_string(index=False))
else:
    print("No significant mispricing detected (markets appear fairly priced)")

# =============================================================================
# SECTION 7: VISUALIZE DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: VISUALIZE DISTRIBUTION COMPARISON")
print("=" * 80)

from myPortfolioManagement.myImpliedDistribution import plot_distribution_comparison

print("\nCreating comparison plot...")

try:
    fig = plot_distribution_comparison(
        implied_dist=implied_dist,
        bootstrap_dist=bootstrap_dist,
        S0=S0,
        title=f"{ticker} - 5 Year Implied vs Bootstrapped Distribution",
        save_path=f'implied_vs_bootstrap_{ticker}.png'
    )
    print(f"✓ Plot saved as 'implied_vs_bootstrap_{ticker}.png'")
    
    # Display the plot
    plt.show()
    
except Exception as e:
    print(f"✗ Error creating plot: {e}")

# =============================================================================
# SECTION 8: SUMMARY AND INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SUMMARY AND INSIGHTS")
print("=" * 80)

print(f"\nAnalysis Summary for {ticker}:")
print(f"  Current Price: ${S0:.2f}")
print(f"  Time Horizon: {T_years} years")
print(f"  Historical Volatility: {historical_vol:.2%}")
print(f"\n  Implied Expected Price: ${implied_mean:.2f} ({((implied_mean/S0-1)*100):.1f}% change)")
print(f"  Bootstrap Expected Price: ${bootstrap_dist['future_price'].mean():.2f} ({((bootstrap_dist['future_price'].mean()/S0-1)*100):.1f}% change)")
print(f"\n  Difference: ${implied_mean - bootstrap_dist['future_price'].mean():.2f} ({((implied_mean - bootstrap_dist['future_price'].mean())/bootstrap_dist['future_price'].mean()*100):.1f}%)")

# Interpretation
mean_diff_pct = ((implied_mean - bootstrap_dist['future_price'].mean()) / 
                 bootstrap_dist['future_price'].mean() * 100)

print("\nInterpretation:")
if abs(mean_diff_pct) < 5:
    print("  • Markets appear fairly priced relative to historical patterns")
elif mean_diff_pct > 5:
    print("  • Options market implies higher future prices than historical patterns suggest")
    print("  • Potential bullish sentiment or elevated implied volatility")
    print("  • Consider: Options may be overpriced, or market expects structural changes")
elif mean_diff_pct < -5:
    print("  • Options market implies lower future prices than historical patterns suggest")
    print("  • Potential bearish sentiment or depressed implied volatility")
    print("  • Consider: Options may be underpriced, or market expects increased risk")

print("\nUse Cases:")
print("  1. Asset Allocation: Use comparison to inform position sizing")
print("  2. Options Trading: Identify over/underpriced options")
print("  3. Risk Management: Compare tail risks between distributions")
print("  4. Market Sentiment: Gauge forward-looking market expectations")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
