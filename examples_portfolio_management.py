#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyPortfolioManagement - Complete Example Usage (FIXED)
======================================================

This script demonstrates all major functions in the myPortfolioManagement library.
Run sections individually or all at once.

FIXES APPLIED:
- Workaround for get_benchmark_returns() quantstats compatibility issue

Author: Ferhat
Date: December 2024
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats_lumi as qs

# Set display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

print("=" * 80)
print("MyPortfolioManagement - Complete Example Usage")
print("=" * 80)


# =============================================================================
# HELPER FUNCTION: Fix for benchmark returns
# =============================================================================
def get_benchmark_returns_fixed(ticker: str = '^GSPC', name: str = 'S&P500') -> pd.DataFrame:
    """
    Fixed version of get_benchmark_returns that handles quantstats DataFrame output.
    
    Args:
        ticker: Yahoo Finance ticker symbol (default: ^GSPC for S&P 500)
        name: Name for the return series
    
    Returns:
        DataFrame with benchmark returns
    """
    ret_data = qs.utils.download_returns(ticker)
    
    # Handle both Series and DataFrame returns
    if isinstance(ret_data, pd.DataFrame):
        ret = ret_data.squeeze()
    else:
        ret = ret_data
    
    ret.name = name
    return pd.DataFrame(ret)


# =============================================================================
# SECTION 1: DATA FETCHING (myData.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA FETCHING (myData.py)")
print("=" * 80)

from myPortfolioManagement.myData import (
    get_stock_prices,
    get_stock_info,
    get_option_exp_dates
)

# Define a diversified portfolio of ETFs
tickers = [
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq 100
    'IWM',   # Russell 2000 (Small Cap)
    'EFA',   # International Developed
    'EEM',   # Emerging Markets
    'AGG',   # US Bonds
    'TIP',   # TIPS (Inflation Protected)
    'GLD',   # Gold
    'VNQ',   # Real Estate
    'DBC',   # Commodities
]

print("\n1.1 Fetching stock prices...")
print(f"Tickers: {tickers}")

# Fetch historical prices
prices = get_stock_prices(
    yahoo_tickers=tickers,
    start_date='2018-01-01',
    end_date='2024-12-01',
    freq='daily',
    wide_format=True
)

print(f"\nPrice data shape: {prices.shape}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
print("\nFirst 5 rows of prices:")
print(prices.head())

print("\n1.2 Fetching stock information...")
stock_info = get_stock_info(tickers)
print("\nStock Information:")
print(stock_info[['yahooTicker', 'longName', 'type', 'currency', 'sector']].to_string())

print("\n1.3 Getting option expiration dates (for SPY)...")
try:
    option_dates = get_option_exp_dates('SPY')
    print(f"Available expiration dates: {len(option_dates)}")
    print(option_dates.head(10))
except Exception as e:
    print(f"Could not fetch options data: {e}")


# =============================================================================
# SECTION 2: RETURNS CALCULATION (myReturns.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: RETURNS CALCULATION (myReturns.py)")
print("=" * 80)

from myPortfolioManagement.myReturns import (
    calculate_returns,
    average_returns,
    convert_returns_freq,
    calculate_portfolio_returns
)

print("\n2.1 Calculating daily returns...")
returns_daily = calculate_returns(prices, log_returns=False)
print(f"Daily returns shape: {returns_daily.shape}")
print("\nDaily returns (first 5 rows):")
print(returns_daily.head())

print("\n2.2 Calculating log returns...")
returns_log = calculate_returns(prices, log_returns=True)
print("\nLog returns statistics:")
print(returns_log.describe().T[['mean', 'std', 'min', 'max']])

print("\n2.3 Converting to monthly returns...")
returns_monthly = calculate_returns(prices, convert_to='monthly')
print(f"Monthly returns shape: {returns_monthly.shape}")
print("\nMonthly returns (last 12 months):")
print(returns_monthly.tail(12))

print("\n2.4 Calculating expected returns (different methods)...")

# Historical mean
mu_hist = average_returns(returns_daily, method='hist', periods=252)
print("\nExpected Returns (Historical Mean):")
print(mu_hist.sort_values(ascending=False))

# Exponential Moving Average
mu_ema = average_returns(returns_daily, method='ema', span=500, periods=252)
print("\nExpected Returns (EMA):")
print(mu_ema.sort_values(ascending=False))

print("\n2.5 Getting benchmark returns (using fixed function)...")
benchmark_sp500 = get_benchmark_returns_fixed('^GSPC', 'S&P500')
print(f"\nS&P 500 benchmark shape: {benchmark_sp500.shape}")
print(benchmark_sp500.tail())


# =============================================================================
# SECTION 3: UTILITY FUNCTIONS (myUtils.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: UTILITY FUNCTIONS (myUtils.py)")
print("=" * 80)

from myPortfolioManagement.myUtils import (
    data_overview,
    balance_dates,
    check_date_index,
    cap_outliersTS
)

print("\n3.1 Data overview (checking data quality)...")
# Convert to long format for data_overview
prices_long = prices.reset_index().melt(
    id_vars='Date', 
    var_name='asset', 
    value_name='price'
)

overview = data_overview(
    prices_long,
    my_assets_col_name='asset',
    my_date_col_name='Date',
    price_col_name='price'
)
print("\nData Overview:")
print(overview)

print("\n3.2 Balancing dates between returns and benchmark...")
returns_balanced, benchmark_balanced = balance_dates(returns_daily, benchmark_sp500)
print(f"Balanced returns shape: {returns_balanced.shape}")
print(f"Balanced benchmark shape: {benchmark_balanced.shape}")

print("\n3.3 Capping outliers in time series...")
try:
    prices_capped = cap_outliersTS(
        returns_daily, 
        capping_method='iqr', 
        fold=3, 
        plot=False
    )
    print(f"Capped prices shape: {prices_capped.shape}")
except Exception as e:
    print(f"Outlier capping skipped: {e}")


# =============================================================================
# SECTION 4: PORTFOLIO OPTIMIZATION (myPortfolioOptimisation.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: PORTFOLIO OPTIMIZATION (myPortfolioOptimisation.py)")
print("=" * 80)

from myPortfolioManagement.myPortfolioOptimisation import (
    HRP,
    equal_weight_portfolio,
    inverse_vol_portfolio,
    port_GMV,
    port_max_sharpe,
    port_target_return,
    port_target_volatility,
    port_CVAR,
    generate_rp_portfolios,
    make_standard_portfolios,
    risk_contributions
)

# Use returns for optimization (fill NaN with 0 for optimization)
returns_opt = returns_daily.fillna(0)

print("\n4.1 Equal Weight Portfolio...")
weights_equal = equal_weight_portfolio(returns_opt)
print("\nEqual Weight Portfolio:")
print(weights_equal)

print("\n4.2 Inverse Volatility Portfolio...")
weights_inv_vol = inverse_vol_portfolio(returns_opt, weight_max=0.25)
print("\nInverse Volatility Portfolio:")
print(weights_inv_vol.sort_values('port_inverse_vol', ascending=False))

print("\n4.3 Hierarchical Risk Parity (HRP)...")
weights_hrp = HRP(
    model='HRP',
    returns_training=returns_opt,
    codependence='pearson',
    covariance='ledoit',
    rm='MV',
    linkage='ward',
    weight_max=0.25,
    weight_min=0.02
)
print("\nHRP Portfolio:")
print(weights_hrp.sort_values('port_weight', ascending=False))

print("\n4.4 HERC (Hierarchical Equal Risk Contribution)...")
weights_herc = HRP(
    model='HERC',
    returns_training=returns_opt,
    codependence='pearson',
    rm='CVaR',
    weight_max=0.30,
    weight_min=0.02
)
print("\nHERC Portfolio:")
print(weights_herc.sort_values('port_weight', ascending=False))

print("\n4.5 Global Minimum Variance Portfolio...")
weights_gmv = port_GMV(
    returns_training=returns_opt,
    weight_min=0.02,
    weight_max=0.30
)
print("\nGlobal Minimum Variance Portfolio:")
print(weights_gmv.sort_values('port_min_vol', ascending=False))

print("\n4.6 Maximum Sharpe Ratio Portfolio...")
weights_max_sharpe = port_max_sharpe(
    returns_training=returns_opt,
    rf=0.04,
    weight_min=0.02,
    weight_max=0.30
)
print("\nMax Sharpe Portfolio:")
print(weights_max_sharpe.sort_values('port_max_Sharpe', ascending=False))

print("\n4.7 Target Return Portfolio (targeting 10% annual return)...")
try:
    weights_target_ret = port_target_return(
        returns_training=returns_opt,
        target_return=0.10,
        rf=0.04,
        weight_min=0.02,
        weight_max=0.40
    )
    print("\nTarget Return Portfolio:")
    print(weights_target_ret.sort_values('port_target_returns', ascending=False))
except Exception as e:
    print(f"Target return optimization failed: {e}")

print("\n4.8 Target Volatility Portfolio (targeting 12% annual vol)...")
try:
    weights_target_vol = port_target_volatility(
        returns_training=returns_opt,
        target_volatility=0.12,
        rf=0.04,
        weight_min=0.02,
        weight_max=0.40
    )
    print("\nTarget Volatility Portfolio:")
    print(weights_target_vol)
except Exception as e:
    print(f"Target volatility optimization failed: {e}")

print("\n4.9 Minimum CVaR Portfolio...")
weights_cvar = port_CVAR(
    returns_training=returns_opt,
    confidence_interval=0.95,
    rf=0.04,
    weight_min=0.02,
    weight_max=0.30
)
print("\nMinimum CVaR Portfolio:")
print(weights_cvar.sort_values('port_target_CVAR', ascending=False))

print("\n4.10 Risk Parity Portfolios...")
weights_rp = generate_rp_portfolios(
    returns_training=returns_opt,
    rf=0.04,
    weight_max=0.99
)
print("\nRisk Parity Portfolios:")
print(weights_rp)

print("\n4.11 Risk Contributions Analysis...")
try:
    risk_contrib = risk_contributions(
        port_weights=weights_hrp,
        returns=returns_opt,
        risk_measure='MV',
        plot=False
    )
    print("\nRisk Contributions (HRP Portfolio):")
    print(risk_contrib)
except Exception as e:
    print(f"Risk contribution calculation: {e}")


# =============================================================================
# SECTION 5: PERFORMANCE METRICS (myPerformanceMetrics.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: PERFORMANCE METRICS (myPerformanceMetrics.py)")
print("=" * 80)

from myPortfolioManagement.myPerformanceMetrics import (
    get_main_stats,
    alpha_beta_table,
    alpha_beta_bull,
    alpha_beta_bear,
    information_ratio,
    drawdown_details,
    assets_drawdown_details,
    performance_overview,
    cagr,
    age,
    beta_Co_Moments_table
)

print("\n5.1 Main Performance Statistics...")
main_stats = get_main_stats(returns_daily, rf=0.04, smart=True)
print("\nMain Performance Stats:")
print(main_stats.round(4))

print("\n5.2 CAGR Calculation...")
cagr_values = cagr(prices)
print("\nCompound Annual Growth Rate:")
print(cagr_values.sort_values(ascending=False).round(4))

print("\n5.3 Alpha-Beta Analysis (Full Market)...")
ab_table = alpha_beta_table(
    returns=returns_daily,
    returns_benchmark=benchmark_sp500,
    rf=0.04
)
print("\nAlpha-Beta Table:")
print(ab_table)

print("\n5.4 Bull Market Alpha-Beta...")
ab_bull = alpha_beta_bull(
    returns=returns_daily,
    returns_benchmark=benchmark_sp500,
    rf=0.04
)
print("\nBull Market Alpha-Beta:")
print(ab_bull)

print("\n5.5 Bear Market Alpha-Beta...")
ab_bear = alpha_beta_bear(
    returns=returns_daily,
    returns_benchmark=benchmark_sp500,
    rf=0.04
)
print("\nBear Market Alpha-Beta:")
print(ab_bear)

print("\n5.6 Information Ratio...")
info_ratio = information_ratio(returns_daily, benchmark_sp500.squeeze())
print("\nInformation Ratio:")
print(info_ratio.round(4))

print("\n5.7 Drawdown Details (for SPY)...")
spy_prices = prices.iloc[:, 0]  # First column
dd_details = drawdown_details(spy_prices, top_drawdowns=5)
print(f"\nTop 5 Drawdowns for {prices.columns[0]}:")
print(dd_details)

print("\n5.8 All Assets Drawdown Details...")
dd_all = assets_drawdown_details(prices, top_drawdowns=3)
print("\nTop 3 Drawdowns per Asset:")
print(dd_all)

print("\n5.9 Performance Overview (FFN-style)...")
perf_overview = performance_overview(returns_daily, prices=False, short=True)
print("\nPerformance Overview:")
print(perf_overview.round(4))

print("\n5.10 Beta Co-Moments Analysis...")
try:
    beta_moments = beta_Co_Moments_table(returns_daily, benchmark_sp500.squeeze())
    print("\nBeta Co-Moments (CoVariance, CoSkewness, CoKurtosis):")
    print(beta_moments.round(4))
except Exception as e:
    print(f"Beta co-moments calculation: {e}")


# =============================================================================
# SECTION 6: LOW-LEVEL RISK METRICS (metrics.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: LOW-LEVEL RISK METRICS (metrics.py)")
print("=" * 80)

from myPortfolioManagement.metrics import (
    vol,
    beta,
    var,
    cvar,
    lpm,
    hpm,
    max_dd,
    dd,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    calmar_ratio,
    omega_ratio,
    gain_loss_ratio,
    upside_potential_ratio
)

# Use SPY returns for demonstration
spy_returns = returns_daily.iloc[:, 0].dropna().values
market_returns = benchmark_sp500.squeeze().dropna().values

# Align lengths
min_len = min(len(spy_returns), len(market_returns))
spy_returns = spy_returns[:min_len]
market_returns = market_returns[:min_len]

print("\n6.1 Basic Risk Metrics...")
print(f"Volatility (annualized): {vol(spy_returns) * np.sqrt(252):.4f}")
print(f"Beta to market: {beta(spy_returns, market_returns):.4f}")

print("\n6.2 Value at Risk...")
print(f"VaR (5%): {var(spy_returns, 0.05):.4f}")
print(f"CVaR (5%): {cvar(spy_returns, 0.05):.4f}")

print("\n6.3 Partial Moments...")
print(f"Lower Partial Moment (order 1): {lpm(spy_returns, 0, 1):.6f}")
print(f"Lower Partial Moment (order 2): {lpm(spy_returns, 0, 2):.6f}")
print(f"Higher Partial Moment (order 1): {hpm(spy_returns, 0, 1):.6f}")

print("\n6.4 Drawdown Metrics...")
print(f"Maximum Drawdown: {max_dd(spy_returns):.4f}")
print(f"Drawdown (30 days): {dd(spy_returns, 30):.4f}")

print("\n6.5 Risk-Adjusted Return Ratios...")
er = np.mean(spy_returns) * 252  # Annualized expected return
rf = 0.04  # Risk-free rate

print(f"Sharpe Ratio: {sharpe_ratio(er, spy_returns, rf/252):.4f}")
print(f"Sortino Ratio: {sortino_ratio(er, spy_returns, rf/252):.4f}")
print(f"Treynor Ratio: {treynor_ratio(er, spy_returns, market_returns, rf/252):.4f}")
print(f"Calmar Ratio: {calmar_ratio(er, spy_returns, rf/252):.4f}")
print(f"Omega Ratio: {omega_ratio(er, spy_returns, rf/252):.4f}")

print("\n6.6 Other Ratios...")
print(f"Gain/Loss Ratio: {gain_loss_ratio(spy_returns):.4f}")
print(f"Upside Potential Ratio: {upside_potential_ratio(spy_returns):.4f}")


# =============================================================================
# SECTION 7: BACKTESTING (myBacktesting.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: BACKTESTING (myBacktesting.py)")
print("=" * 80)

from myPortfolioManagement.myBacktesting import (
    bootstrap_stats,
    bootstrap_portfolio_performance,
    sim_series,
    sim_paths,
    beating_probability,
    performance
)

# Create a simple portfolio return series for backtesting
portfolio_returns = returns_daily.mean(axis=1)  # Equal weighted proxy
portfolio_returns.name = 'Portfolio'

print("\n7.1 Bootstrap Statistics...")
bootstrap_results = bootstrap_stats(
    returns=portfolio_returns,
    returns_benchmark=benchmark_sp500.squeeze(),
    rf=0.04,
    periods=252,
    n_sim=500  # Reduced for speed
)
print("\nBootstrap Statistics Distribution:")
print(bootstrap_results.describe().round(4))

print("\n7.2 Bootstrap Portfolio Performance (In-Sample vs Out-of-Sample)...")
try:
    means, distributions, dist_stats = bootstrap_portfolio_performance(
        returns=portfolio_returns,
        returns_benchmark=benchmark_sp500.squeeze(),
        periods=252,
        rf=0.04,
        out_of_sample_date='2023-01-01',
        n_sim=500
    )
    print("\nMean Performance Metrics:")
    print(means.round(4))
    print("\nDistribution Statistics:")
    print(dist_stats.round(4))
except Exception as e:
    print(f"Bootstrap performance analysis: {e}")

print("\n7.2a Visualizing Bootstrap Results...")
# Import the new plotting functions
from myPortfolioManagement.myPlots import (
    plot_bootstrap_distribution,
    plot_bootstrap_distributions,
    plot_bootstrap_comparison
)

# Single metric distribution
print("  - Plotting CAGR distribution...")
try:
    plot_bootstrap_distribution(
        bootstrap_results=bootstrap_results,
        metric='cagr',
        confidence_level=0.95,
        savefig='example_bootstrap_cagr.png',
        show=False
    )
    print("    ✓ Saved: example_bootstrap_cagr.png")
except Exception as e:
    print(f"    Could not create CAGR plot: {e}")

# Multiple metrics violin plot
print("  - Creating violin plot for all metrics...")
try:
    plot_bootstrap_distributions(
        bootstrap_results=bootstrap_results,
        plot_type='violin',
        savefig='example_bootstrap_violin.png',
        show=False
    )
    print("    ✓ Saved: example_bootstrap_violin.png")
except Exception as e:
    print(f"    Could not create violin plot: {e}")

# In-sample vs out-of-sample comparison
if 'distributions' in locals():
    print("  - Creating in-sample vs out-of-sample comparison...")
    try:
        plot_bootstrap_comparison(
            bootstrap_results=distributions,
            comparison_col='sample',
            savefig='example_bootstrap_comparison.png',
            show=False
        )
        print("    ✓ Saved: example_bootstrap_comparison.png")
    except Exception as e:
        print(f"    Could not create comparison plot: {e}")

print("\n7.3 Simulating Return Series...")
sim_returns = sim_series(
    returns=portfolio_returns,
    n_sample=1000,
    random_state=42
)
print(f"\nSimulated returns shape: {sim_returns.shape}")
print(f"Original mean: {portfolio_returns.mean():.6f}")
print(f"Simulated mean: {sim_returns.mean().mean():.6f}")

print("\n7.4 Probability of Beating Benchmark...")
try:
    beat_prob = beating_probability(
        returns=pd.DataFrame(portfolio_returns),
        returns_benchmark=benchmark_sp500,
        n_sample=500
    )
    print(f"\nProbability of beating benchmark: {beat_prob.values[0][0]:.2%}")
except Exception as e:
    print(f"Beating probability calculation: {e}")

print("\n7.5 Strategy Performance with Transaction Costs...")
# Create a simple signal (1 = long, 0 = flat)
signal = pd.Series(
    np.where(portfolio_returns.rolling(20).mean() > 0, 1, 0),
    index=portfolio_returns.index
)
strategy_returns = performance(
    signal=signal,
    returns=portfolio_returns,
    bps=2e-4  # 2 basis points per trade
)
print(f"\nStrategy returns (with 2bps transaction cost):")
print(f"Total return: {(1 + strategy_returns.dropna()).prod() - 1:.2%}")


# =============================================================================
# SECTION 8: BOOTSTRAPPING (myBootstrapping.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: BOOTSTRAPPING (myBootstrapping.py)")
print("=" * 80)

from myPortfolioManagement.myBootstrapping import (
    BootstrapIDD,
    BootstrapStationary,
    BootstrapCircular,
    BootstrapMovingBlock,
    bootstrappingTS
)

# Use single asset returns for bootstrapping
single_returns = returns_daily.iloc[:, 0].dropna()

print("\n8.1 IID Bootstrap...")
bs_iid = BootstrapIDD(
    series=single_returns,
    n_samples=100,
    seed=42
)
print(f"IID Bootstrap shape: {bs_iid.shape}")
print(f"Original std: {single_returns.std():.6f}")
print(f"Bootstrap mean std: {bs_iid.std().mean():.6f}")

print("\n8.2 Stationary Bootstrap...")
bs_stationary = BootstrapStationary(
    series=single_returns,
    block_size=20,
    n_samples=100,
    seed=42
)
print(f"Stationary Bootstrap shape: {bs_stationary.shape}")

print("\n8.3 Circular Block Bootstrap...")
bs_circular = BootstrapCircular(
    series=single_returns,
    block_size=20,
    n_samples=100,
    seed=42
)
print(f"Circular Bootstrap shape: {bs_circular.shape}")

print("\n8.4 Moving Block Bootstrap...")
bs_moving = BootstrapMovingBlock(
    series=single_returns,
    block_size=20,
    n_samples=100,
    seed=42
)
print(f"Moving Block Bootstrap shape: {bs_moving.shape}")

print("\n8.5 General Bootstrapping Function...")
bs_general = bootstrappingTS(
    series=single_returns,
    bootstrap_type='cbb',  # Circular block bootstrap
    block_size=20,
    n_samples=100,
    seed=42
)
print(f"General Bootstrap shape: {bs_general.shape}")


# =============================================================================
# SECTION 9: CLUSTERING (myClustering.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 9: CLUSTERING (myClustering.py)")
print("=" * 80)

from myPortfolioManagement.myClustering import (
    ts_clustering,
    cluster_ftca,
    detect_regimes
)

print("\n9.1 Time Series Clustering (DTW)...")
try:
    clusters, cluster_centers = ts_clustering(
        df=prices,
        number_of_clusters=3,
        algo='dtw',
        plot_bar_center=False
    )
    print("\nAsset Clusters:")
    print(clusters)
    print("\nCluster Centers shape:", cluster_centers.shape)
except Exception as e:
    print(f"Time series clustering: {e}")

print("\n9.2 Fast Threshold Clustering Algorithm (FTCA)...")
try:
    ftca_clusters = cluster_ftca(
        returns=returns_daily,
        threshold=0.50,
        col_name='asset'
    )
    print("\nFTCA Clusters:")
    print(ftca_clusters)
except Exception as e:
    print(f"FTCA clustering: {e}")

print("\n9.3 Regime Detection...")
try:
    # Detect regimes in volatility proxy
    vol_series = returns_daily.iloc[:, 0].rolling(20).std() * np.sqrt(252)
    vol_df = pd.DataFrame(vol_series.dropna(), columns=['Volatility'])
    
    regimes = detect_regimes(
        df=vol_df,
        series='Volatility',
        optimal_clusters=3,
        metric='dtw',
        plot_regimes=False
    )
    print("\nDetected Regimes:")
    print(regimes['Volatility_Regime'].value_counts())
except Exception as e:
    print(f"Regime detection: {e}")


# =============================================================================
# SECTION 10: PLOTTING (myPlots.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 10: PLOTTING (myPlots.py)")
print("=" * 80)

from myPortfolioManagement.myPlots import (
    correlation_matrix,
    monthly_heatmap,
    scatter_plot_simple
)

print("\n10.1 Correlation Matrix...")
print("Generating correlation matrix plot...")
try:
    correlation_matrix(
        returns_daily,
        corr_limit=None,
        figsize=(12, 8),
        diagonal=True
    )
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_matrix.png")
except Exception as e:
    print(f"Correlation matrix plot: {e}")

print("\n10.2 Monthly Returns Heatmap...")
print("Generating monthly heatmap...")
try:
    fig = monthly_heatmap(
        returns=portfolio_returns,
        annot_size=8,
        figsize=(12, 8),
        cbar=True,
        eoy=True,
        show=False
    )
    if fig:
        fig.savefig('monthly_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("Saved: monthly_heatmap.png")
except Exception as e:
    print(f"Monthly heatmap: {e}")

print("\n10.3 Risk-Return Scatter Plot...")
try:
    # Create risk-return dataframe
    risk_return = pd.DataFrame({
        'Return': returns_daily.mean() * 252,
        'Volatility': returns_daily.std() * np.sqrt(252)
    })
    scatter_plot_simple(risk_return, x='Volatility', y='Return')
    plt.savefig('risk_return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: risk_return_scatter.png")
except Exception as e:
    print(f"Scatter plot: {e}")


# =============================================================================
# SECTION 11: PORTFOLIO SELECTION (myPortfolioSelection.py)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 11: PORTFOLIO SELECTION (myPortfolioSelection.py)")
print("=" * 80)

from myPortfolioManagement.myPortfolioSelection import (
    possible_combinations,
    all_combinations_inner_loop
)

print("\n11.1 Generating all possible portfolio combinations...")
assets = ['SPY', 'QQQ', 'IWM', 'AGG', 'GLD']

# All combinations of 2-4 assets
combinations = possible_combinations(
    assets_to_consider=assets,
    min_assets=2,
    max_assets=4
)
print(f"\nNumber of possible portfolios (2-4 assets): {len(combinations)}")
print("\nFirst 10 combinations:")
for i, combo in enumerate(combinations[:10]):
    print(f"  {i+1}. {combo}")

print("\n11.2 Combinations with must-have assets...")
combinations_must_have = possible_combinations(
    assets_to_consider=assets,
    min_assets=3,
    max_assets=4,
    must_have=['SPY', 'AGG']  # Must include these
)
print(f"\nPortfolios that must include SPY and AGG: {len(combinations_must_have)}")
for combo in combinations_must_have:
    print(f"  {combo}")


# =============================================================================
# SECTION 12: COMPLETE WORKFLOW EXAMPLE
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 12: COMPLETE WORKFLOW - END TO END EXAMPLE")
print("=" * 80)

print("""
This section demonstrates a complete portfolio management workflow:
1. Fetch data
2. Calculate returns
3. Optimize portfolio
4. Backtest strategy
5. Analyze performance
""")

# Step 1: Data already fetched above
print("\nStep 1: Using previously fetched data...")
print(f"  Assets: {list(prices.columns)}")
print(f"  Period: {prices.index[0]} to {prices.index[-1]}")

# Step 2: Split into training and testing
split_date = '2023-01-01'
returns_train = returns_daily[returns_daily.index < split_date]
returns_test = returns_daily[returns_daily.index >= split_date]
print(f"\nStep 2: Train/Test Split at {split_date}")
print(f"  Training: {len(returns_train)} days")
print(f"  Testing: {len(returns_test)} days")

# Step 3: Optimize portfolios on training data
print("\nStep 3: Optimizing portfolios on training data...")
weights_hrp_train = HRP(
    model='HRP',
    returns_training=returns_train.fillna(0),
    weight_max=0.25,
    weight_min=0.02
)
print("  HRP weights optimized")

weights_equal_train = equal_weight_portfolio(returns_train)
print("  Equal weight portfolio created")

# Step 4: Calculate out-of-sample returns
print("\nStep 4: Calculating out-of-sample returns...")

# HRP portfolio returns
hrp_weights_dict = weights_hrp_train['port_weight'].to_dict()
portfolio_hrp_test = (returns_test * pd.Series(hrp_weights_dict)).sum(axis=1)
portfolio_hrp_test.name = 'HRP'

# Equal weight returns
equal_weights_dict = weights_equal_train['port_naive'].to_dict()
portfolio_equal_test = (returns_test * pd.Series(equal_weights_dict)).sum(axis=1)
portfolio_equal_test.name = 'Equal_Weight'

# Combine for comparison
portfolio_comparison = pd.DataFrame({
    'HRP': portfolio_hrp_test,
    'Equal_Weight': portfolio_equal_test,
    'SPY_Benchmark': returns_test.iloc[:, 0]  # SPY as benchmark
})

# Step 5: Performance analysis
print("\nStep 5: Out-of-Sample Performance Analysis...")
test_stats = get_main_stats(portfolio_comparison, rf=0.04)
print("\nOut-of-Sample Performance Metrics:")
print(test_stats.round(4))

# Cumulative returns
cum_returns = (1 + portfolio_comparison).cumprod()
print("\nCumulative Returns (End of Period):")
print(cum_returns.iloc[-1].round(4))

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Portfolio Management Analysis Complete!

Key Findings:
- Analyzed {len(tickers)} assets from {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}
- Generated {len(weights_rp.columns) + 4} different portfolio strategies
- Best performing asset (CAGR): {cagr_values.idxmax()} ({cagr_values.max():.2%})
- Worst performing asset (CAGR): {cagr_values.idxmin()} ({cagr_values.min():.2%})

Files Generated:
- correlation_matrix.png
- monthly_heatmap.png  
- risk_return_scatter.png
- example_bootstrap_cagr.png (Bootstrap CAGR distribution)
- example_bootstrap_violin.png (Bootstrap metrics comparison)
- example_bootstrap_comparison.png (In-sample vs Out-of-sample)

New Bootstrap Visualization Features:
- plot_bootstrap_distribution(): Single metric with confidence intervals
- plot_bootstrap_distributions(): Multiple metrics (violin/box/histogram)
- plot_bootstrap_comparison(): In-sample vs out-of-sample comparison

Next Steps:
1. Review portfolio weights and adjust constraints
2. Run more extensive backtests with different parameters
3. Implement rebalancing strategy
4. Add transaction costs and slippage
5. Consider regime-based allocation
""")

print("\n" + "=" * 80)
print("END OF EXAMPLE SCRIPT")
print("=" * 80)