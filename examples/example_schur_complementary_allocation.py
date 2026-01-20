#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schur Complementary Allocation Example
======================================

This example demonstrates Schur Complementary Allocation, a portfolio optimization
method that unifies Hierarchical Risk Parity (HRP) and Minimum Variance (MVP).

Key Concepts:
- Traditional mean-variance treats marginal risk as diversification, leading to
  "diversifiers" that all tank when equities crash (fake diversification).
- Schur Complementary Allocation fixes this by using the Schur complement to
  isolate what each asset contributes CONDITIONAL on existing holdings.

The `gamma` parameter controls the trade-off:
- gamma = 0: Pure HRP (hierarchical, no off-diagonal information)
- gamma -> 1: Approaches Minimum Variance Portfolio

Based on Peter Cotton's 2024 paper: "Schur Complementary Allocation:
A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios"

Author: Generated for MyPortfolioManagement
Date: January 2025
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myReturns import calculate_returns
from myPortfolioManagement.myPortfolioOptimisation import (
    schur_complementary,
    redundancy_analysis,
    conditional_covariance,
    compare_schur_vs_hrp,
    HRP,
    port_GMV,
)

print("=" * 80)
print("SCHUR COMPLEMENTARY ALLOCATION EXAMPLE")
print("Unifying HRP and Minimum Variance Portfolios")
print("=" * 80)


# =============================================================================
# CONFIGURATION: Your Portfolio
# =============================================================================

# Mixed portfolio: Tech/Financials (core equity) + Gold/Crypto (potential diversifiers)
TICKERS = [
    "MSFT",      # Tech - Core
    "NVDA",      # Tech - Core
    "CRWD",      # Cybersecurity - Core
    "V",         # Financials - Core
    "JPM",       # Financials - Core
    "ABBV",      # Healthcare - Core
    "IGLN.L",    # Gold ETF - Satellite (diversifier)
    "BTC-USD",   # Bitcoin - Satellite (diversifier)
    "ETH-USD",   # Ethereum - Satellite (diversifier)
]

# Define what's "core" (equity-like) vs "satellite" (potential diversifiers)
CORE_ASSETS = ["MSFT", "NVDA", "CRWD", "V", "JPM", "ABBV"]
SATELLITE_ASSETS = ["IGLN.L", "BTC-USD", "ETH-USD"]

START_DATE = '2020-01-01'
END_DATE = None


# =============================================================================
# SECTION 1: FETCH DATA
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: FETCHING DATA")
print("=" * 80)

print(f"\nFetching prices for {len(TICKERS)} assets...")
print(f"Core assets (equity-like): {CORE_ASSETS}")
print(f"Satellite assets (diversifiers): {SATELLITE_ASSETS}")

prices = get_stock_prices(
    yahoo_tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    freq='daily',
    wide_format=True,
)

# Get stock info for readable names
stock_info = get_stock_info(TICKERS)
ticker_to_name = dict(zip(stock_info['yahooTicker'], stock_info['longName']))

print(f"\nPrice data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} rows)")
print("\nAssets:")
for ticker in TICKERS:
    name = ticker_to_name.get(ticker, ticker)
    role = "CORE" if ticker in CORE_ASSETS else "SATELLITE"
    print(f"  [{role:9}] {ticker}: {name}")

# Calculate returns
returns = calculate_returns(prices, log_returns=False)
returns = returns.dropna()

print(f"\nReturns calculated: {len(returns)} observations")


# =============================================================================
# SECTION 2: REDUNDANCY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: REDUNDANCY ANALYSIS")
print("Which satellites provide TRUE diversification vs being redundant?")
print("=" * 80)

print("\nComputing Schur complement to analyze diversification value...")
print(f"Core holdings: {CORE_ASSETS}")
print(f"Potential diversifiers: {SATELLITE_ASSETS}")

analysis = redundancy_analysis(
    returns_training=returns,
    core_assets=CORE_ASSETS,
    satellite_assets=SATELLITE_ASSETS,
    threshold=0.1,  # Flag if <10% variance retained
)

print("\n" + "-" * 60)
print("DIVERSIFICATION ANALYSIS RESULTS")
print("-" * 60)
print("\nVariance Retained = Conditional Var / Unconditional Var")
print("Higher = More independent from core = Better diversifier")
print()
print(analysis.to_string())

print("\n" + "-" * 60)
print("INTERPRETATION")
print("-" * 60)

for asset in analysis.index:
    var_retained = analysis.loc[asset, 'variance_retained']
    is_redundant = analysis.loc[asset, 'is_redundant']
    name = ticker_to_name.get(asset, asset)

    if var_retained > 0.8:
        verdict = "EXCELLENT diversifier - highly independent from core equities"
    elif var_retained > 0.5:
        verdict = "GOOD diversifier - moderate independence"
    elif var_retained > 0.2:
        verdict = "MARGINAL diversifier - partially correlated with core"
    else:
        verdict = "POOR diversifier - largely redundant given core holdings"

    print(f"\n{asset} ({name}):")
    print(f"  Variance retained: {var_retained:.1%}")
    print(f"  Verdict: {verdict}")


# =============================================================================
# SECTION 3: CONDITIONAL COVARIANCE MATRIX
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: CONDITIONAL COVARIANCE MATRIX (SCHUR COMPLEMENT)")
print("=" * 80)

cov_matrix = returns.cov()
cond_cov = conditional_covariance(cov_matrix, CORE_ASSETS, SATELLITE_ASSETS)

print("\nOriginal covariance matrix (satellites only):")
orig_sat_cov = cov_matrix.loc[SATELLITE_ASSETS, SATELLITE_ASSETS]
print((orig_sat_cov * 10000).round(4))  # Scale for readability
print("(scaled by 10,000 for readability)")

print("\nConditional covariance matrix (Schur complement):")
print("This is what remains AFTER removing correlation with core equities:")
print((cond_cov * 10000).round(4))

# Show reduction
print("\n% Reduction in variance (diagonal) after conditioning:")
for asset in SATELLITE_ASSETS:
    orig = orig_sat_cov.loc[asset, asset]
    cond = cond_cov.loc[asset, asset]
    reduction = 1 - (cond / orig) if orig > 0 else 0
    print(f"  {asset}: {reduction:.1%} explained by core")


# =============================================================================
# SECTION 4: SCHUR COMPLEMENTARY ALLOCATION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: SCHUR COMPLEMENTARY ALLOCATION")
print("gamma=0 (HRP) <---> gamma=1 (MVP)")
print("=" * 80)

print("\nComparing portfolio weights across gamma values...")

comparison = compare_schur_vs_hrp(
    returns_training=returns,
    gamma_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    covariance='ledoit',
    weight_min=0.02,
    weight_max=0.30,
)

print("\nPortfolio Weights by Gamma:")
print(comparison.round(4).to_string())

# Show how allocations change
print("\n" + "-" * 60)
print("ALLOCATION CHANGES FROM HRP TO MVP")
print("-" * 60)

for asset in comparison.index:
    hrp_weight = comparison.loc[asset, 'gamma=0 (HRP)']
    mvp_weight = comparison.loc[asset, 'gamma≈1 (MVP)']
    change = mvp_weight - hrp_weight
    direction = "+" if change > 0 else ""
    name = ticker_to_name.get(asset, asset)

    print(f"{asset:12} | HRP: {hrp_weight:6.1%} -> MVP: {mvp_weight:6.1%} ({direction}{change:+.1%})")


# =============================================================================
# SECTION 5: OPTIMAL SCHUR PORTFOLIO (gamma=0.5)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: RECOMMENDED SCHUR PORTFOLIO (gamma=0.5)")
print("=" * 80)

# gamma=0.5 is a good default - balances HRP's stability with MVP's efficiency
weights_schur = schur_complementary(
    returns_training=returns,
    gamma=0.5,
    covariance='ledoit',
    distance='pearson',
    linkage='ward',
    weight_min=0.02,
    weight_max=0.30,
)

print("\nSchur Complementary Portfolio (gamma=0.5):")
print(weights_schur.sort_values('port_weight', ascending=False).round(4).to_string())

# Compare with traditional HRP
weights_hrp = HRP(
    model='HRP',
    returns_training=returns,
    covariance='ledoit',
    weight_min=0.02,
    weight_max=0.30,
)

# Compare with GMV
weights_gmv = port_GMV(
    returns_training=returns,
    weight_min=0.02,
    weight_max=0.30,
)

print("\n" + "-" * 60)
print("COMPARISON: Schur vs HRP vs GMV")
print("-" * 60)

combined = pd.DataFrame({
    'Schur (γ=0.5)': weights_schur['port_weight'],
    'HRP': weights_hrp['port_weight'],
    'GMV': weights_gmv['port_min_vol'],
})
combined = combined.sort_values('Schur (γ=0.5)', ascending=False)
print(combined.round(4).to_string())


# =============================================================================
# SECTION 6: PORTFOLIO STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: PORTFOLIO STATISTICS")
print("=" * 80)

# Compute portfolio metrics
cov_annual = returns.cov() * 252
mean_annual = returns.mean() * 252

def portfolio_stats(weights_series, cov_matrix, mean_returns):
    w = weights_series.values
    var = w @ cov_matrix.values @ w
    vol = np.sqrt(var)
    ret = mean_returns.values @ w
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

portfolios = {
    'Schur (γ=0.5)': weights_schur['port_weight'],
    'Schur (γ=0)': comparison['gamma=0 (HRP)'],
    'Schur (γ=1)': comparison['gamma≈1 (MVP)'],
    'Traditional HRP': weights_hrp['port_weight'],
    'GMV': weights_gmv['port_min_vol'],
}

stats_data = []
for name, weights in portfolios.items():
    # Align weights with returns columns
    w = weights.reindex(returns.columns).fillna(0)
    ret, vol, sharpe = portfolio_stats(w, cov_annual, mean_annual)
    stats_data.append({
        'Portfolio': name,
        'Expected Return': ret,
        'Volatility': vol,
        'Sharpe Ratio': sharpe,
    })

stats_df = pd.DataFrame(stats_data).set_index('Portfolio')
print("\nAnnualized Portfolio Statistics:")
print(stats_df.round(4).to_string())


# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Weight comparison across gamma
ax1 = axes[0, 0]
comparison_plot = comparison.copy()
comparison_plot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Portfolio Weights: HRP to MVP Transition', fontsize=12)
ax1.set_xlabel('Asset')
ax1.set_ylabel('Weight')
ax1.legend(title='Gamma', loc='upper right', fontsize=8)
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0, 0.35)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Diversification value
ax2 = axes[0, 1]
div_values = analysis['diversification_value'].sort_values(ascending=True)
colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in div_values]
div_values.plot(kind='barh', ax=ax2, color=colors)
ax2.set_title('Diversification Value of Satellite Assets', fontsize=12)
ax2.set_xlabel('Variance Retained (higher = better diversifier)')
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
ax2.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Marginal threshold')
ax2.legend(fontsize=8)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Schur vs HRP vs GMV comparison
ax3 = axes[1, 0]
combined_plot = combined.copy()
combined_plot.plot(kind='bar', ax=ax3, width=0.8, color=['#2E86AB', '#A23B72', '#F18F01'])
ax3.set_title('Weight Comparison: Schur vs HRP vs GMV', fontsize=12)
ax3.set_xlabel('Asset')
ax3.set_ylabel('Weight')
ax3.legend(loc='upper right', fontsize=9)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Risk-Return scatter
ax4 = axes[1, 1]
for name, row in stats_df.iterrows():
    marker = 'o' if 'Schur' in name else 's' if 'HRP' in name else '^'
    ax4.scatter(row['Volatility'], row['Expected Return'], s=100, marker=marker, label=name)
ax4.set_title('Risk-Return Profile', fontsize=12)
ax4.set_xlabel('Volatility (Annualized)')
ax4.set_ylabel('Expected Return (Annualized)')
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('schur_complementary_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: schur_complementary_analysis.png")
plt.close()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
KEY INSIGHTS FROM SCHUR COMPLEMENTARY ALLOCATION:

1. REDUNDANCY ANALYSIS:
   - The Schur complement reveals which "diversifiers" are truly independent
   - Assets with low variance retained are mathematically redundant

2. GAMMA PARAMETER:
   - gamma=0: Pure HRP (divide-and-conquer, no correlation structure)
   - gamma=0.5: Balanced approach (recommended default)
   - gamma=1: Approaches MVP (full correlation exploitation)

3. ADVANTAGES OVER TRADITIONAL APPROACHES:
   - Unlike MVP: More stable out-of-sample (doesn't overfit correlations)
   - Unlike HRP: Uses some correlation information for efficiency
   - Smooth interpolation lets you choose your risk/stability trade-off

4. PRACTICAL RECOMMENDATIONS:
   - Start with gamma=0.5 for balanced performance
   - Use gamma closer to 0 for more stable, diversified portfolios
   - Use gamma closer to 1 if covariance matrix is well-conditioned
   - Always use Ledoit-Wolf shrinkage for covariance estimation
""")

# Show best diversifiers
best_div = analysis[analysis['diversification_value'] > 0.5].index.tolist()
poor_div = analysis[analysis['diversification_value'] < 0.3].index.tolist()

if best_div:
    print(f"BEST DIVERSIFIERS (>50% variance retained): {best_div}")
if poor_div:
    print(f"POOR DIVERSIFIERS (<30% variance retained): {poor_div}")

print("\n" + "=" * 80)
print("END OF SCHUR COMPLEMENTARY ALLOCATION EXAMPLE")
print("=" * 80)
