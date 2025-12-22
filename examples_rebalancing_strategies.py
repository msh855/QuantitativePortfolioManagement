#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples: Buy-and-Hold vs Rebalancing Strategies
=================================================

This script demonstrates the difference between buy-and-hold and rebalancing 
strategies, and the impact of transaction costs.

Key Concepts:
1. Daily Rebalancing (traditional r*w): Assumes portfolio is rebalanced to target 
   weights every single day. This is rarely realistic in practice.

2. Buy-and-Hold: Portfolio is purchased once and held without rebalancing. Weights
   drift naturally based on asset performance.

3. Periodic Rebalancing: Portfolio is rebalanced to target weights at regular 
   intervals (weekly, monthly, quarterly, etc.).

4. Transaction Costs: Every trade incurs a cost, typically measured in basis points
   (1 bps = 0.01% = 0.0001).
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from myPortfolioManagement.myReturns import (
    calculate_buy_and_hold_returns,
    calculate_rebalanced_returns,
    calculate_portfolio_returns
)

print("=" * 80)
print("Buy-and-Hold vs Rebalancing Strategies - Examples")
print("=" * 80)

# =============================================================================
# Example 1: Simple Comparison
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: Simple Two-Asset Portfolio")
print("=" * 80)

# Create sample returns where one asset outperforms
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=252, freq='D')  # One year of trading days

# Asset A: Higher returns, higher volatility
returns_A = np.random.normal(0.0008, 0.02, 252)  # ~20% annual return
# Asset B: Lower returns, lower volatility  
returns_B = np.random.normal(0.0003, 0.01, 252)  # ~7.5% annual return

returns = pd.DataFrame({
    'Tech_Stock': returns_A,
    'Bond_Fund': returns_B
}, index=dates)

weights = pd.Series({'Tech_Stock': 0.6, 'Bond_Fund': 0.4})

print(f"\nInitial Weights: {weights.to_dict()}")
print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

# Calculate returns under different strategies
bh_returns = calculate_buy_and_hold_returns(returns, weights, portfolio_name='Buy-and-Hold')
monthly_returns = calculate_rebalanced_returns(returns, weights, rebalance_freq='monthly', 
                                               portfolio_name='Monthly Rebalancing')
daily_returns = calculate_rebalanced_returns(returns, weights, rebalance_freq='daily',
                                             portfolio_name='Daily Rebalancing')

# Calculate cumulative returns
cum_bh = (1 + bh_returns).cumprod()
cum_monthly = (1 + monthly_returns).cumprod()
cum_daily = (1 + daily_returns).cumprod()

print(f"\nFinal Cumulative Returns (no transaction costs):")
print(f"  Buy-and-Hold:        {cum_bh.iloc[-1, 0]:.4f} ({(cum_bh.iloc[-1, 0]-1)*100:.2f}%)")
print(f"  Monthly Rebalancing: {cum_monthly.iloc[-1, 0]:.4f} ({(cum_monthly.iloc[-1, 0]-1)*100:.2f}%)")
print(f"  Daily Rebalancing:   {cum_daily.iloc[-1, 0]:.4f} ({(cum_daily.iloc[-1, 0]-1)*100:.2f}%)")

# =============================================================================
# Example 2: Impact of Transaction Costs
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Impact of Transaction Costs")
print("=" * 80)

# Compare monthly rebalancing with different transaction cost levels
tc_levels = [0, 5, 10, 20, 50]  # in basis points
results = {}

for tc in tc_levels:
    port_returns = calculate_rebalanced_returns(
        returns, weights, 
        rebalance_freq='monthly',
        transaction_cost_bps=tc,
        portfolio_name=f'Monthly_TC_{tc}bps'
    )
    cum_return = (1 + port_returns).prod().values[0]
    results[tc] = cum_return

print(f"\nMonthly Rebalancing - Final Returns by Transaction Cost:")
print(f"{'TC (bps)':<12} {'Cumulative Return':<20} {'Annual Return %'}")
print("-" * 55)
for tc, cum_ret in results.items():
    annual_ret = (cum_ret ** (252/len(returns)) - 1) * 100
    print(f"{tc:<12} {cum_ret:<20.4f} {annual_ret:>10.2f}%")

# =============================================================================
# Example 3: Different Rebalancing Frequencies
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: Comparing Rebalancing Frequencies")
print("=" * 80)

# Use 2 years of data for better comparison
dates_long = pd.date_range('2020-01-01', periods=504, freq='D')
returns_long = pd.DataFrame({
    'Tech_Stock': np.random.normal(0.0008, 0.02, 504),
    'Bond_Fund': np.random.normal(0.0003, 0.01, 504)
}, index=dates_long)

# Set reasonable transaction costs
tc_bps = 10  # 10 basis points per trade

frequencies = ['buy_and_hold', 'yearly', 'quarterly', 'monthly', 'weekly', 'daily']
freq_results = {}

for freq in frequencies:
    if freq == 'buy_and_hold':
        port_returns = calculate_buy_and_hold_returns(
            returns_long, weights,
            transaction_cost_bps=tc_bps,
            portfolio_name=freq
        )
    else:
        port_returns = calculate_rebalanced_returns(
            returns_long, weights,
            rebalance_freq=freq,
            transaction_cost_bps=tc_bps,
            portfolio_name=freq
        )
    
    cum_return = (1 + port_returns).prod().values[0]
    # Annualized return
    years = len(returns_long) / 252
    annual_ret = (cum_return ** (1/years) - 1) * 100
    
    # Count number of rebalancing events (approximate)
    if freq == 'buy_and_hold':
        n_rebalances = 0
    elif freq == 'daily':
        n_rebalances = len(returns_long)
    elif freq == 'weekly':
        n_rebalances = len(returns_long) // 5
    elif freq == 'monthly':
        n_rebalances = len(returns_long) // 21
    elif freq == 'quarterly':
        n_rebalances = len(returns_long) // 63
    elif freq == 'yearly':
        n_rebalances = len(returns_long) // 252
    
    freq_results[freq] = {
        'cumulative': cum_return,
        'annual': annual_ret,
        'rebalances': n_rebalances
    }

print(f"\nResults with {tc_bps} bps transaction costs:")
print(f"{'Strategy':<20} {'Cum Return':<15} {'Annual %':<12} {'# Rebalances'}")
print("-" * 65)
for freq, res in freq_results.items():
    print(f"{freq:<20} {res['cumulative']:<15.4f} {res['annual']:<12.2f} {res['rebalances']}")

print("\nKey Insights:")
print("1. Daily rebalancing has the highest transaction costs due to frequent trading")
print("2. Buy-and-hold has the lowest costs but weights drift over time")
print("3. Monthly/quarterly rebalancing often provides a good balance")
print("4. Higher transaction costs favor less frequent rebalancing")

# =============================================================================
# Example 4: Using with calculate_portfolio_returns
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Integration with calculate_portfolio_returns")
print("=" * 80)

# Create weights DataFrame in the expected format
weights_df = pd.DataFrame({
    'asset': ['Tech_Stock', 'Bond_Fund'],
    'weight': [0.6, 0.4]
})

# Traditional daily rebalancing (default)
traditional = calculate_portfolio_returns(
    returns, weights_df,
    portfolio_name='Traditional'
)

# Buy-and-hold
buy_hold = calculate_portfolio_returns(
    returns, weights_df,
    rebalance_strategy='buy_and_hold',
    portfolio_name='Buy_Hold'
)

# Monthly with transaction costs
monthly_tc = calculate_portfolio_returns(
    returns, weights_df,
    rebalance_strategy='monthly',
    transaction_cost_bps=10,
    portfolio_name='Monthly_TC'
)

print("\nComparing different approaches:")
print(f"Traditional (daily rebal, no TC): {(1 + traditional).prod().values[0]:.4f}")
print(f"Buy-and-Hold:                     {(1 + buy_hold).prod().values[0]:.4f}")
print(f"Monthly rebal with 10bps TC:      {(1 + monthly_tc).prod().values[0]:.4f}")

# =============================================================================
# Example 5: Visualization
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Visualizing the Difference")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative returns
    ax1 = axes[0]
    cum_bh.plot(ax=ax1, label='Buy-and-Hold', linewidth=2)
    cum_monthly.plot(ax=ax1, label='Monthly Rebalancing', linewidth=2)
    cum_daily.plot(ax=ax1, label='Daily Rebalancing', linewidth=2, alpha=0.7)
    
    ax1.set_title('Cumulative Returns: Different Rebalancing Strategies', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transaction cost impact
    ax2 = axes[1]
    tc_cumulative = [results[tc] for tc in tc_levels]
    ax2.plot(tc_levels, tc_cumulative, marker='o', linewidth=2, markersize=8)
    ax2.set_title('Impact of Transaction Costs on Monthly Rebalancing', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Transaction Cost (basis points)')
    ax2.set_ylabel('Final Cumulative Return')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rebalancing_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'rebalancing_comparison.png'")
    
except Exception as e:
    print(f"\nNote: Could not create visualization: {e}")

# =============================================================================
# Summary and Recommendations
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("""
When calculating portfolio returns, it's important to be explicit about your
rebalancing assumptions:

1. TRADITIONAL APPROACH (returns.dot(weights)):
   - Implicitly assumes daily rebalancing
   - Unrealistic for most portfolios
   - Use only for theoretical analysis or when you actually rebalance daily

2. BUY-AND-HOLD STRATEGY:
   - Most realistic for individual investors
   - Lowest transaction costs (one-time purchase)
   - Weights drift with market performance
   - Use: calculate_buy_and_hold_returns()

3. PERIODIC REBALANCING:
   - Realistic for institutional portfolios
   - Balance between maintaining target allocation and minimizing costs
   - Monthly or quarterly is common in practice
   - Use: calculate_rebalanced_returns(rebalance_freq='monthly')

4. TRANSACTION COSTS:
   - Always include realistic transaction costs (typically 5-20 bps)
   - Higher costs favor less frequent rebalancing
   - Consider: brokerage fees, bid-ask spread, market impact

RECOMMENDED APPROACH:
For realistic portfolio analysis, use:
    portfolio_returns = calculate_portfolio_returns(
        returns, weights,
        rebalance_strategy='monthly',  # or 'quarterly'
        transaction_cost_bps=10        # realistic trading costs
    )

This avoids the implicit daily rebalancing assumption and provides more 
accurate performance estimates.
""")

print("=" * 80)
print("Examples completed successfully!")
print("=" * 80)
