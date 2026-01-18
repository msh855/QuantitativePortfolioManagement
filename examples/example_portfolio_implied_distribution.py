#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portfolio Implied Probability Distribution (Live Data Example)

This script mirrors the workflow in create_fanchart.py to fetch *actual* prices
and construct portfolio returns, then mirrors example_implied_distribution.py to:

1) Bootstrap a future distribution of the portfolio value
2) Use that bootstrapped range to set strikes for a synthetic option chain
3) Extract a (risk-neutral) implied density via Breeden-Litzenberger
4) Compare implied vs bootstrapped distributions

Notes
-----
- Because there are no traded options on a custom portfolio, the option chain is
  synthetic (Black-Scholes) using the portfolio's historical volatility.
- Strike range is set from bootstrap min to an upper-quantile to avoid clipping
  the implied density while limiting extreme right-tail outliers.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys

# # Ensure local imports work when running this script from any directory
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myReturns import calculate_returns, calculate_portfolio_returns
from myPortfolioManagement.myOptionPricing import create_option_chain
from myPortfolioManagement.myImpliedDistribution import (
    extract_implied_distribution,
    compare_distributions,
    find_mispricing_opportunities,
    plot_distribution_comparison,
)


# =============================================================================
# CONFIGURATION (copied from create_fanchart.py)
# =============================================================================

TICKERS = ["IGLN.L", "MSFT", "ABBV", "NVDA", "CRWD", "V", "JPM", "SOL-USD", "BTC-USD", "ETH-USD"]

WEIGHTS = {
    'IGLN.L': 0.1419,
    'MSFT': 0.1172,
    'ABBV': 0.1070,
    'NVDA': 0.0967,
    'CRWD': 0.0762,
    'V': 0.0527,
    'JPM': 0.0522,
    'SOL-USD': 0.1719,
    'BTC-USD': 0.1365,
    'ETH-USD': 0.0477,
}

START_DATE = '2016-01-01'
END_DATE = None

PORTFOLIO_NAME = 'My Custom Portfolio'

# Implied/Bootstrap parameters
T_YEARS = 5.0
RISK_FREE_RATE = 0.05
N_SIM = 10_000
TRADING_DAYS_PER_YEAR = 252

# Strike-range choice (asymmetric)
UPPER_Q = 0.999  # 99.9% for the upper tail
BUFFER = 0.05    # widen range by 5% on both ends
NUM_STRIKES = 80


def _build_weights_df(stock_info: pd.DataFrame) -> pd.DataFrame:
    if stock_info.empty:
        raise ValueError("No stock info retrieved; cannot map tickers to long names.")

    ticker_to_name = dict(zip(stock_info['yahooTicker'], stock_info['longName']))
    weights_longname = {ticker_to_name[t]: w for t, w in WEIGHTS.items()}

    total_weight = float(sum(weights_longname.values()))
    if not np.isclose(total_weight, 1.0):
        weights_longname = {k: v / total_weight for k, v in weights_longname.items()}

    return pd.DataFrame({'asset': list(weights_longname.keys()), 'weight': list(weights_longname.values())})


def _bootstrap_future_prices(returns: pd.Series, S0: float, T_years: float, n_sim: int) -> pd.DataFrame:
    periods = int(T_years * TRADING_DAYS_PER_YEAR)
    ret_values = returns.dropna().values

    rng = np.random.default_rng(123)
    sampled = rng.choice(ret_values, size=(n_sim, periods), replace=True)
    cumulative = np.prod(1.0 + sampled, axis=1)
    future_prices = S0 * cumulative

    return pd.DataFrame({'simulation': np.arange(n_sim), 'future_price': future_prices})


def _cdf_at_x_from_density(x_grid: np.ndarray, density: np.ndarray, x0: float) -> float:
    if len(x_grid) < 2:
        return float('nan')

    order = np.argsort(x_grid)
    x = np.asarray(x_grid)[order]
    f = np.asarray(density)[order]
    f = np.maximum(f, 0)

    area = simpson(f, x=x)
    if not np.isfinite(area) or area <= 0:
        return float('nan')

    # Normalize to avoid small numerical drift
    f = f / area

    # Integrate up to x0 (piecewise trapezoid is fine here)
    if x0 <= x[0]:
        return 0.0
    if x0 >= x[-1]:
        return 1.0

    idx = np.searchsorted(x, x0, side='right')
    x_left = x[:idx]
    f_left = f[:idx]
    cdf_left = np.trapz(f_left, x_left)

    # Add the partial last segment [x[idx-1], x0]
    x1, x2 = x[idx - 1], x[idx]
    f1, f2 = f[idx - 1], f[idx]
    f0 = f1 + (f2 - f1) * ((x0 - x1) / (x2 - x1))
    cdf_left += 0.5 * (f1 + f0) * (x0 - x1)

    return float(np.clip(cdf_left, 0.0, 1.0))


if __name__ == '__main__':
    print("=" * 80)
    print("Portfolio Implied Probability Distribution (Live Data)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Fetch prices and compute returns (same as create_fanchart.py)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: FETCH PRICES + BUILD PORTFOLIO RETURNS")
    print("=" * 80)

    prices = get_stock_prices(
        yahoo_tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        freq='daily',
        wide_format=True,
    )

    stock_info = get_stock_info(TICKERS)
    weights_df = _build_weights_df(stock_info)

    returns = calculate_returns(prices, log_returns=False)
    portfolio_returns = calculate_portfolio_returns(
        returns=returns,
        myweights=weights_df,
        portfolio_name=PORTFOLIO_NAME,
    )

    port_ret = portfolio_returns[PORTFOLIO_NAME].dropna()
    portfolio_price = (1.0 + port_ret).cumprod()
    S0 = float(portfolio_price.iloc[-1])

    historical_vol = float(port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    print(f"\nPrice sample: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} rows)")
    print(f"Portfolio last value (S0): {S0:.4f}")
    print(f"Portfolio annualized vol (sigma): {historical_vol:.2%}")

    # -------------------------------------------------------------------------
    # 2) Bootstrap first (so we can set strike range)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: BOOTSTRAP FUTURE PORTFOLIO VALUE")
    print("=" * 80)

    bootstrap_dist = _bootstrap_future_prices(port_ret, S0=S0, T_years=T_YEARS, n_sim=N_SIM)

    low_price = float(bootstrap_dist['future_price'].min())
    high_price = float(bootstrap_dist['future_price'].quantile(UPPER_Q))

    print(f"\nBootstrap stats:")
    print(f"  Mean: {bootstrap_dist['future_price'].mean():.4f}")
    print(f"  Std:  {bootstrap_dist['future_price'].std():.4f}")
    print(f"  Min:  {bootstrap_dist['future_price'].min():.4f}")
    print(f"  Max:  {bootstrap_dist['future_price'].max():.4f}")

    # Convert to multipliers expected by create_option_chain
    min_mult = max(0.01, (low_price / S0) * (1.0 - BUFFER))
    max_mult = max(min_mult * 1.05, (high_price / S0) * (1.0 + BUFFER))

    print("\nStrike range derived from bootstrap (min to upper-quantile):")
    print(f"  Upper quantile: {UPPER_Q:.3%}")
    print(f"  Bootstrapped range used: {low_price:.4f} to {high_price:.4f}")
    print(f"  Strike multipliers vs spot: {min_mult:.3f}x to {max_mult:.3f}x")

    # -------------------------------------------------------------------------
    # 3) Create synthetic option chain on the portfolio value
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: CREATE SYNTHETIC OPTION CHAIN")
    print("=" * 80)

    option_chain = create_option_chain(
        S=S0,
        T=T_YEARS,
        r=RISK_FREE_RATE,
        sigma=historical_vol,
        strike_range=(min_mult, max_mult),
        num_strikes=NUM_STRIKES,
    )

    print(f"\n✓ Option chain: {len(option_chain)} strikes")
    print(f"  Strike range: {option_chain['strike'].min():.4f} to {option_chain['strike'].max():.4f}")

    # -------------------------------------------------------------------------
    # 4) Extract implied distribution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: EXTRACT IMPLIED DISTRIBUTION")
    print("=" * 80)

    implied_dist = extract_implied_distribution(
        option_chain=option_chain,
        S=S0,
        r=RISK_FREE_RATE,
        T=T_YEARS,
        option_type='call',
        return_range_mass=True,
    )

    mass_in_range = float(implied_dist['probability_mass_in_range'].iloc[0])
    implied_cdf_at_s0 = _cdf_at_x_from_density(
        implied_dist['price_level'].values,
        implied_dist['probability_density'].values,
        S0,
    )
    bootstrap_cdf_at_s0 = float((bootstrap_dist['future_price'].values <= S0).mean())

    print(f"\n✓ Implied density extracted")
    print(f"  Probability mass in strike range: {mass_in_range:.2%}")
    print("\nCurrent value vs distributions (T-horizon):")
    print(f"  Current portfolio value (S0): {S0:.4f}")
    print(f"  Implied Pr(ST <= S0): {implied_cdf_at_s0:.2%}")
    print(f"  Bootstrap Pr(ST <= S0): {bootstrap_cdf_at_s0:.2%}")

    # Overlay plot: bootstrapped distribution (as density histogram) + implied density curve + S0 marker
    plt.figure(figsize=(10, 4))
    plt.hist(
        bootstrap_dist['future_price'].values,
        bins=70,
        density=True,
        alpha=0.35,
        label='Bootstrapped density (historical)'
    )
    plt.plot(
        implied_dist['price_level'],
        implied_dist['probability_density'],
        linewidth=2,
        label='Implied risk-neutral density (synthetic options)'
    )
    plt.axvline(S0, linewidth=2, linestyle='--', label='Current value (S0)')
    plt.title(f"{PORTFOLIO_NAME} - {T_YEARS:g}Y: Current vs Implied vs Bootstrapped")
    plt.xlabel('Future portfolio value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # -------------------------------------------------------------------------
    # 5) Compare distributions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: COMPARE IMPLIED VS BOOTSTRAPPED")
    print("=" * 80)

    comparison = compare_distributions(
        implied_dist=implied_dist,
        bootstrap_dist=bootstrap_dist,
        S0=S0,
        confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95],
    )
    print("\nDistribution Comparison:")
    print(comparison.to_string(index=False))

    mispricings = find_mispricing_opportunities(comparison=comparison, threshold_pct=5.0)
    print("\nPotential Mispricing Opportunities:")
    if len(mispricings) > 0 and 'opportunity' in mispricings.columns:
        print(mispricings.to_string(index=False))
    else:
        print("No significant mispricing detected")

    # -------------------------------------------------------------------------
    # 6) Plot comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: PLOT")
    print("=" * 80)

    save_path = 'implied_vs_bootstrap_portfolio.png'
    try:
        plot_distribution_comparison(
            implied_dist=implied_dist,
            bootstrap_dist=bootstrap_dist,
            S0=S0,
            title=f"{PORTFOLIO_NAME} - {T_YEARS:g}Y Implied vs Bootstrapped Distribution",
            save_path=save_path,
        )
        print(f"✓ Plot saved as '{save_path}'")
        plt.show()
    except Exception as e:
        print(f"✗ Error creating plot: {e}")
