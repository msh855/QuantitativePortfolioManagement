#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schur Complementary Allocation Example
======================================

Demonstrates how Schur Complementary Allocation identifies TRUE vs FAKE diversifiers
and optimally weights your portfolio.

Based on Peter Cotton's 2024 paper: "Schur Complementary Allocation:
A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios"
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION - EDIT THIS FOR YOUR PORTFOLIO
# =============================================================================

TICKERS = [
    # CORE: Your main equity exposure
    "MSFT", "NVDA", "CRWD", "V", "JPM", "ABBV",
    # SATELLITES: Potential diversifiers
    "GLD", "BTC-USD", "ETH-USD",
]

CORE_ASSETS = ["MSFT", "NVDA", "CRWD", "V", "JPM", "ABBV"]
SATELLITE_ASSETS = ["GLD", "BTC-USD", "ETH-USD"]

START_DATE = '2020-01-01'

# =============================================================================
# FETCH DATA
# =============================================================================
print("=" * 70)
print("SCHUR COMPLEMENTARY ALLOCATION EXAMPLE")
print("=" * 70)

print(f"\nFetching data for: {TICKERS}")

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns

prices = get_stock_prices(
    yahoo_tickers=TICKERS,
    start_date=START_DATE,
    freq='daily',
    wide_format=True,
)

# Ensure column names are clean strings (handle MultiIndex from yfinance)
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(-1)
prices.columns = [str(c).strip() for c in prices.columns]

# Calculate returns
returns = calculate_returns(prices, log_returns=False).dropna()

print(f"Data: {returns.index[0].date()} to {returns.index[-1].date()} ({len(returns)} days)")
print(f"Columns: {returns.columns.tolist()}")

# Verify all assets are present
missing = [a for a in CORE_ASSETS + SATELLITE_ASSETS if a not in returns.columns]
if missing:
    print(f"\nWARNING: Missing assets: {missing}")
    print(f"Available: {returns.columns.tolist()}")
    print("Please update CORE_ASSETS and SATELLITE_ASSETS to match your column names.")
    raise SystemExit(1)

# =============================================================================
# 1. REDUNDANCY ANALYSIS - Which satellites are TRUE diversifiers?
# =============================================================================
print("\n" + "=" * 70)
print("1. REDUNDANCY ANALYSIS")
print("=" * 70)

from myPortfolioManagement.myPortfolioOptimisation import redundancy_analysis

analysis = redundancy_analysis(
    returns_training=returns,
    core_assets=CORE_ASSETS,
    satellite_assets=SATELLITE_ASSETS,
)

print("\nCore holdings (equity exposure):", CORE_ASSETS)
print("Satellites to analyze:", SATELLITE_ASSETS)
print("\nRESULTS:")
print("-" * 70)
print(f"{'Asset':<12} {'Var Retained':>14} {'Verdict':<30}")
print("-" * 70)

for asset in analysis.index:
    vr = analysis.loc[asset, 'variance_retained']
    if vr > 0.8:
        verdict = "TRUE DIVERSIFIER"
    elif vr > 0.5:
        verdict = "GOOD DIVERSIFIER"
    elif vr > 0.3:
        verdict = "MARGINAL"
    else:
        verdict = "REDUNDANT (fake diversifier)"
    print(f"{asset:<12} {vr:>13.1%}  {verdict}")

# =============================================================================
# 2. SCHUR COMPLEMENTARY ALLOCATION
# =============================================================================
print("\n" + "=" * 70)
print("2. PORTFOLIO ALLOCATION: Schur vs HRP")
print("=" * 70)

from myPortfolioManagement.myPortfolioOptimisation import schur_complementary, HRP

# Schur with gamma=0.5 (balanced between HRP and MVP)
weights_schur = schur_complementary(
    returns_training=returns,
    gamma=0.5,
    covariance='ledoit',
)

# Traditional HRP for comparison
weights_hrp = HRP(
    model='HRP',
    returns_training=returns,
    covariance='ledoit',
)

# Combine for comparison
comparison = pd.DataFrame({
    'Schur (g=0.5)': weights_schur['port_weight'],
    'HRP': weights_hrp['port_weight'],
})
comparison['Diff'] = comparison['Schur (g=0.5)'] - comparison['HRP']
comparison = comparison.sort_values('Schur (g=0.5)', ascending=False)

print("\nPortfolio Weights:")
print("-" * 70)
print(f"{'Asset':<12} {'Schur':>12} {'HRP':>12} {'Diff':>12}")
print("-" * 70)
for asset in comparison.index:
    s = comparison.loc[asset, 'Schur (g=0.5)']
    h = comparison.loc[asset, 'HRP']
    d = comparison.loc[asset, 'Diff']
    print(f"{asset:<12} {s:>11.1%} {h:>11.1%} {d:>+11.1%}")

# Summary by type
print("\n" + "-" * 70)
core_schur = comparison.loc[CORE_ASSETS, 'Schur (g=0.5)'].sum()
core_hrp = comparison.loc[CORE_ASSETS, 'HRP'].sum()
sat_schur = comparison.loc[SATELLITE_ASSETS, 'Schur (g=0.5)'].sum()
sat_hrp = comparison.loc[SATELLITE_ASSETS, 'HRP'].sum()

print(f"CORE total:      Schur={core_schur:.1%}  HRP={core_hrp:.1%}")
print(f"SATELLITE total: Schur={sat_schur:.1%}  HRP={sat_hrp:.1%}")

# =============================================================================
# 3. PORTFOLIO RISK
# =============================================================================
print("\n" + "=" * 70)
print("3. PORTFOLIO RISK")
print("=" * 70)

cov_annual = returns.cov() * 252

def calc_vol(weights):
    w = weights.reindex(returns.columns).fillna(0).values
    return np.sqrt(w @ cov_annual.values @ w)

vol_schur = calc_vol(weights_schur['port_weight'])
vol_hrp = calc_vol(weights_hrp['port_weight'])

print(f"\nAnnualized Volatility:")
print(f"  Schur (gamma=0.5): {vol_schur:.2%}")
print(f"  HRP:               {vol_hrp:.2%}")
print(f"  Difference:        {vol_schur - vol_hrp:+.2%}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

best = analysis[analysis['variance_retained'] > 0.5].index.tolist()
poor = analysis[analysis['variance_retained'] < 0.3].index.tolist()

print(f"""
The Schur complement reveals which assets provide TRUE diversification
by measuring independence from your core equity holdings.

Your results:
  - Best diversifiers: {best if best else 'None'}
  - Redundant (fake):  {poor if poor else 'None'}

Schur Complementary Allocation (gamma=0.5) adjusts weights to:
  - Exploit TRUE diversifiers more effectively
  - Reduce allocation to correlated "fake" diversifiers
""")

print("=" * 70)
