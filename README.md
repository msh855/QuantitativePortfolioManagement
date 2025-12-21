# MyPortfolioManagement

A comprehensive Python library for quantitative portfolio management, backtesting, and performance analysis. This toolkit provides professional-grade functions for data fetching, portfolio optimization, risk management, performance metrics, and Monte Carlo simulations.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
  - [Data Fetching (myData.py)](#data-fetching-mydatazy)
  - [Returns Calculation (myReturns.py)](#returns-calculation-myreturnspy)
  - [Performance Metrics (myPerformanceMetrics.py)](#performance-metrics-myperformancemetricspy)
  - [Portfolio Optimization (myPortfolioOptimisation.py)](#portfolio-optimization-myportfoliooptimisationpy)
  - [Backtesting & Simulation (myBacktesting.py)](#backtesting--simulation-mybacktestingpy)
  - [Clustering & Analysis (myClustering.py)](#clustering--analysis-myclusteringpy)
  - [Bootstrapping (myBootstrapping.py)](#bootstrapping-mybootstrappingpy)
  - [Utilities (myUtils.py)](#utilities-myutilspy)
- [Usage Examples](#usage-examples)
- [Advanced Examples](#advanced-examples)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [License](#license)

---

## Features

✨ **Data Management**
- Automatic stock price fetching from Yahoo Finance
- Multi-asset portfolio support
- Currency adjustment for international portfolios
- Data cleaning and validation

📊 **Performance Analysis**
- Comprehensive risk and return metrics
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown and drawdown analysis
- Alpha/Beta calculation with bull/bear market decomposition
- Rolling window performance metrics

🎯 **Portfolio Optimization**
- Hierarchical Risk Parity (HRP)
- Hierarchical Equal Risk Contribution (HERC)
- Inverse Volatility weighting
- Equal Weight portfolios
- Efficient Frontier optimization
- Constrained optimization (min/max weights)

🔄 **Advanced Analytics**
- Monte Carlo simulations
- Bootstrap resampling (IID, Stationary, Circular, Moving Block)
- Fan chart forecasting
- Regime detection
- Time series clustering (K-means)
- Correlation and tail dependence analysis

📈 **Backtesting**
- Portfolio performance tracking
- Benchmark comparison
- Returns tear sheets
- Beating probability analysis
- In-sample and out-of-sample analysis

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/ferhat00/QuantitativePortfolioManagement.git
cd QuantitativePortfolioManagement
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python test_install.py
```

---

## Quick Start

### Basic Portfolio Analysis (5 minutes)

```python
import pandas as pd
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns
from myPortfolioManagement.myPerformanceMetrics import performance_overview

# Define portfolio assets
tickers = ['SPY', 'AGG', 'GLD', 'EEM']

# Fetch historical prices
prices = get_stock_prices(
    yahoo_tickers=tickers,
    start_date='2020-01-01',
    end_date='2024-12-01',
    freq='daily'
)

# Calculate returns
returns = calculate_returns(prices, log_returns=False)

# Analyze performance
performance = performance_overview(returns, prices=False)
print(performance)
```

### Portfolio Optimization (10 minutes)

```python
from myPortfolioManagement.myPortfolioOptimisation import (
    HRP, equal_weight_portfolio, inverse_vol_portfolio
)

# Hierarchical Risk Parity
weights_hrp = HRP(
    returns_training=returns,
    codependence='pearson',
    covariance='ledoit',
    linkage='ward'
)

# Equal Weight baseline
weights_equal = equal_weight_portfolio(returns)

# Inverse Volatility
weights_inv_vol = inverse_vol_portfolio(returns)

# Compare weights
print("HRP Weights:", weights_hrp)
print("Equal Weight:", weights_equal)
print("Inverse Vol:", weights_inv_vol)
```

---

## Module Overview

### Data Fetching (myData.py)

**Purpose**: Retrieve historical price data from Yahoo Finance

**Key Functions**:

#### `get_stock_prices(yahoo_tickers, start_date, end_date, freq='daily', adj_fx=False)`
Fetches historical OHLC data for multiple assets
- **Parameters**:
  - `yahoo_tickers` (list): Yahoo Finance ticker symbols
  - `start_date` (str): Start date in 'YYYY-MM-DD' format
  - `end_date` (str): End date in 'YYYY-MM-DD' format
  - `freq` (str): 'daily', 'monthly', or 'quarterly'
  - `adj_fx` (bool): Adjust for currency differences
  - `auto_adjust` (bool): Auto-adjust OHLC data

- **Returns**: DataFrame with dates as index, assets as columns

- **Example**:
  ```python
  prices = get_stock_prices(
      yahoo_tickers=['AAPL', 'MSFT', 'GOOGL'],
      start_date='2022-01-01',
      end_date='2024-12-01',
      freq='daily'
  )
  print(prices.head())
  ```

#### `get_stock_info(yahoo_tickers)`
Retrieves company metadata and information
- **Example**:
  ```python
  info = get_stock_info(['SPY', 'AGG', 'EEM'])
  print(info[['yahooTicker', 'longName', 'sector', 'currency']])
  ```

#### `get_option_exp_dates(yahoo_ticker)`
Gets available option expiration dates for derivatives analysis
- **Example**:
  ```python
  exp_dates = get_option_exp_dates('SPY')
  print(exp_dates)
  ```

---

### Returns Calculation (myReturns.py)

**Purpose**: Calculate and manipulate returns data

#### `calculate_returns(df_prices, log_returns=False, convert_to=None)`
Converts price data to returns

- **Parameters**:
  - `df_prices` (DataFrame): Price DataFrame
  - `log_returns` (bool): Use log returns if True, simple returns if False
  - `convert_to` (str): 'monthly', 'quarterly', 'annual', or None

- **Example**:
  ```python
  # Simple returns
  simple_returns = calculate_returns(prices, log_returns=False)
  
  # Log returns
  log_returns = calculate_returns(prices, log_returns=True)
  
  # Monthly returns
  monthly_returns = calculate_returns(prices, convert_to='monthly')
  ```

#### `average_returns(returns, method='hist', periods=252)`
Calculates expected returns using various methods

- **Parameters**:
  - `returns` (DataFrame/Series): Return data
  - `method` (str): 'hist' (historical), 'ema', 'capm'
  - `span` (int): EMA span for exponential moving average
  - `periods` (int): Periods per year (252 for daily, 12 for monthly)

- **Example**:
  ```python
  # Historical expected returns
  expected_returns = average_returns(returns, method='hist', periods=252)
  print(expected_returns.sort_values(ascending=False))
  
  # EMA method
  ema_returns = average_returns(returns, method='ema', span=500, periods=252)
  ```

#### `calculate_portfolio_returns(returns, myweights, portfolio_name)`
Combines asset returns and weights to create portfolio returns

- **Example**:
  ```python
  portfolio_ret = calculate_portfolio_returns(
      returns=returns,
      myweights=weights,
      portfolio_name='MyPortfolio'
  )
  ```

---

### Performance Metrics (myPerformanceMetrics.py)

**Purpose**: Calculate comprehensive risk and return metrics

#### `performance_overview(df, prices=False, short=True)`
Single-call function for all key metrics

- **Returns**: DataFrame with metrics like:
  - Cumulative Return
  - CAGR (Compound Annual Growth Rate)
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Win Rate

- **Example**:
  ```python
  metrics = performance_overview(returns)
  print(metrics)
  ```

#### `alpha_beta_table(returns, returns_benchmark, rf=0.02)`
Calculates alpha and beta vs benchmark

- **Returns**: DataFrame with alpha, beta, and correlation

- **Example**:
  ```python
  ab_table = alpha_beta_table(returns, benchmark_returns)
  print(ab_table)
  ```

#### `alpha_beta_bull(returns, returns_benchmark, rf=0.02)`
Alpha/Beta during bull markets (positive benchmark returns)

#### `alpha_beta_bear(returns, returns_benchmark, rf=0.02)`
Alpha/Beta during bear markets (negative benchmark returns)

#### `drawdown_details(prices, top_drawdowns=5)`
Identifies largest drawdowns with dates and recovery times

- **Returns**: DataFrame with drawdown peak, trough, recovery dates, and duration

- **Example**:
  ```python
  dd = drawdown_details(prices, top_drawdowns=5)
  print(dd)
  ```

#### `ranking_metrics(df, rolling_window=3, rolling_frequency='Y')`
Calculates rolling performance ranks for rotation strategies

---

### Portfolio Optimization (myPortfolioOptimisation.py)

**Purpose**: Generate optimal portfolio weights using various methods

#### `HRP(model='HRP', returns_training, covariance='hist', codependence='pearson', linkage='ward')`
Hierarchical Risk Parity and related methods

- **Parameters**:
  - `model` (str): 'HRP', 'HERC', 'HERC2', or 'NCO'
  - `returns_training` (DataFrame): Historical returns for training
  - `covariance` (str): Covariance estimation method
    - 'hist' (historical), 'ledoit', 'oas', 'ewma1', 'ewma2', etc.
  - `codependence` (str): Distance metric for clustering
    - 'pearson', 'spearman', 'distance', 'tail', 'mutual_info'
  - `linkage` (str): Clustering linkage method
    - 'single', 'complete', 'average', 'ward', 'DBHT'
  - `weight_max` (float): Maximum weight per asset
  - `weight_min` (float): Minimum weight per asset
  - `rm` (str): Risk measure ('MV' for variance, 'vol', 'equal')

- **Example**:
  ```python
  weights = HRP(
      model='HRP',
      returns_training=returns,
      covariance='ledoit',
      codependence='pearson',
      linkage='ward',
      weight_max=0.25,
      weight_min=0.02
  )
  print(weights)
  ```

#### `inverse_vol_portfolio(returns_training, weight_max=None, weight_min=None)`
Risk parity based on inverse volatility

- **Example**:
  ```python
  weights = inverse_vol_portfolio(returns, weight_max=0.25)
  ```

#### `equal_weight_portfolio(returns_training)`
Equally weighted portfolio (naive 1/N approach)

- **Example**:
  ```python
  weights = equal_weight_portfolio(returns)
  ```

#### `port_GMV(returns_training, weight_min=0.02, weight_max=0.4)`
Global Minimum Variance portfolio

#### `port_max_sharpe(returns_training, rf=0.02, weight_min=0.02, weight_max=0.4)`
Maximizes Sharpe ratio

- **Example**:
  ```python
  weights = port_max_sharpe(returns, rf=0.02, weight_max=0.4)
  ```

#### `port_target_volatility(returns_training, target_volatility=0.10)`
Target portfolio volatility

- **Example**:
  ```python
  weights = port_target_volatility(returns, target_volatility=0.10)
  ```

#### `port_target_return(returns_training, target_return=0.08)`
Target portfolio return

#### `port_CVAR(returns_training, target_CVAR=0.05, confidence_interval=0.95)`
Minimize Conditional Value at Risk

---

### Backtesting & Simulation (myBacktesting.py)

**Purpose**: Simulate portfolio performance and test strategies

#### `bootstrap_portfolio_performance(returns, returns_benchmark=None, n_sim=1000)`
Monte Carlo simulation of portfolio performance

- **Returns**: Tuple of (means, distribution, stats)

- **Example**:
  ```python
  results_means, results_dist, results_stats = bootstrap_portfolio_performance(
      returns=portfolio_returns,
      returns_benchmark=benchmark_returns,
      n_sim=10000
  )
  print(results_means)
  ```

#### `sim_series(returns, n_sample=1000, weight_period=None)`
Simulates future return series paths based on historical distribution

- **Returns**: DataFrame of simulated return paths

- **Example**:
  ```python
  simulated = sim_series(returns, n_sample=10000)
  print(simulated.mean())
  ```

#### `sim_paths(returns, out_of_sample_date='2020-01-01', n_sample=1000)`
Simulates cumulative return paths for visualization

#### `fan_chart(returns, out_of_sample_date='2020-01-01', n_sample=10000)`
Creates fan chart visualization showing distribution of outcomes

- **Example**:
  ```python
  fan_chart(returns, n_sample=10000)
  ```

#### `beating_probability(returns, returns_benchmark, n_sample=10000)`
Probability that portfolio beats benchmark based on Monte Carlo

- **Returns**: Float between 0 and 1

- **Example**:
  ```python
  prob = beating_probability(portfolio_ret, benchmark_ret, n_sample=10000)
  print(f"Probability of beating benchmark: {prob:.2%}")
  ```

#### `tear_sheet_pyfolio(returns, returns_benchmark=None, **kwargs)`
Comprehensive performance tear sheet using PyFolio library

- **Example**:
  ```python
  tear_sheet_pyfolio(portfolio_returns, benchmark_returns)
  ```

---

### Clustering & Analysis (myClustering.py)

**Purpose**: Identify asset groups and market regimes

#### `ts_clustering(df, number_of_clusters=None, algo='dtw')`
Time series clustering with DTW or Euclidean distance

- **Parameters**:
  - `df` (DataFrame): Returns or prices to cluster
  - `number_of_clusters` (int): Number of clusters (auto-calculated if None)
  - `algo` (str): 'dtw' (dynamic time warping) or 'euclidean'

- **Returns**: Tuple of (clusters DataFrame, centers DataFrame)

- **Example**:
  ```python
  clusters, centers = ts_clustering(returns, number_of_clusters=3)
  print("Clusters:")
  print(clusters)
  print("\nCluster Centers:")
  print(centers)
  ```

#### `detect_regimes(df, series, optimal_clusters=3, metric='dtw')`
Identifies market regimes for regime-switching strategies

- **Example**:
  ```python
  regimes = detect_regimes(prices, series='SPY', optimal_clusters=3)
  print(regimes)
  ```

#### `kMeansClusterSeries(df, method='ward')`
Hierarchical clustering with dendrogram visualization

- **Example**:
  ```python
  kMeansClusterSeries(returns.mean().to_frame(), title='Asset Clustering')
  ```

---

### Bootstrapping (myBootstrapping.py)

**Purpose**: Resample data to estimate uncertainty in statistics

#### `BootstrapIDD(series, n_samples=1000)`
IID (Independent, Identically Distributed) resampling

- Assumes independence between observations
- Best for cross-sectional data

- **Example**:
  ```python
  bootstrap_samples = BootstrapIDD(returns['SPY'], n_samples=1000)
  print(bootstrap_samples.describe())
  ```

#### `BootstrapStationary(series, block_size=12, n_samples=1000, optimal_block=True)`
Stationary block bootstrap preserving time series structure

- **Parameters**:
  - `block_size` (int): Size of blocks to resample
  - `optimal_block` (bool): Auto-calculate optimal block size
  - `n_samples` (int): Number of bootstrap samples

- **Example**:
  ```python
  bootstrap_samples = BootstrapStationary(
      returns['SPY'],
      block_size=12,
      n_samples=1000,
      optimal_block=True
  )
  ```

#### `BootstrapCircular(series, block_size=12, n_samples=1000)`
Circular block bootstrap wrapping series for continuous structure

#### `BootstrapMovingBlock(series, block_size=12, n_samples=1000)`
Moving block bootstrap - standard block resampling approach

---

### Utilities (myUtils.py)

**Purpose**: Helper functions for data manipulation

#### `balance_dates(returns, returns_benchmark)`
Aligns date indices of multiple datasets

- **Example**:
  ```python
  returns_aligned, benchmark_aligned = balance_dates(returns, benchmark)
  ```

#### `check_date_index(df)`
Validates that index is datetime, converts if necessary

#### `cap_outliersTS(returns, capping_method='iqr')`
Caps extreme outliers in time series

- **Parameters**:
  - `capping_method` (str): 'iqr' or 'zscore'

- **Example**:
  ```python
  returns_clean = cap_outliersTS(returns, capping_method='iqr')
  ```

#### `data_overview(df, my_assets_col_name, my_date_col_name, price_col_name)`
Summarizes data availability by asset

---

## Usage Examples

### Example 1: Complete Portfolio Analysis Workflow

```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns, average_returns
from myPortfolioManagement.myPerformanceMetrics import (
    performance_overview, alpha_beta_table, drawdown_details
)
from myPortfolioManagement.myPortfolioOptimisation import HRP, inverse_vol_portfolio

# 1. Fetch data for diverse assets
tickers = ['SPY', 'AGG', 'GLD', 'EEM', 'VNQ']
print("Fetching historical prices...")
prices = get_stock_prices(
    tickers, 
    start_date='2020-01-01', 
    end_date='2024-12-01'
)
print(f"Data shape: {prices.shape}")

# 2. Calculate returns
returns = calculate_returns(prices)
benchmark_returns = prices['SPY'].pct_change()

# 3. Get performance overview
print("\n" + "="*60)
print("PERFORMANCE OVERVIEW")
print("="*60)
perf = performance_overview(returns)
print(perf)

# 4. Optimize portfolio using different methods
print("\n" + "="*60)
print("PORTFOLIO OPTIMIZATION")
print("="*60)

weights_hrp = HRP(returns_training=returns, covariance='ledoit')
weights_inv_vol = inverse_vol_portfolio(returns)

print("\nHRP Weights:")
print(weights_hrp.sort_values('port_weight', ascending=False))

print("\nInverse Volatility Weights:")
print(weights_inv_vol.sort_values('port_inverse_vol', ascending=False))

# 5. Calculate portfolio returns
from myPortfolioManagement.myReturns import calculate_portfolio_returns

port_ret_hrp = calculate_portfolio_returns(returns, weights_hrp, 'HRP')
port_ret_inv = calculate_portfolio_returns(returns, weights_inv_vol, 'InvVol')

# 6. Compare vs benchmark
print("\n" + "="*60)
print("ALPHA/BETA ANALYSIS (HRP vs S&P 500)")
print("="*60)
ab_table = alpha_beta_table(port_ret_hrp, benchmark_returns)
print(ab_table)

# 7. Analyze drawdowns
print("\n" + "="*60)
print("LARGEST DRAWDOWNS")
print("="*60)
dd = drawdown_details(port_ret_hrp)
print(dd)
```

### Example 2: Monte Carlo Backtesting

```python
from myPortfolioManagement.myBacktesting import (
    bootstrap_portfolio_performance, 
    beating_probability, 
    fan_chart
)

print("\n" + "="*60)
print("MONTE CARLO SIMULATION")
print("="*60)

# Run simulation
print("Running 10,000 simulations...")
results_means, results_dist, results_stats = bootstrap_portfolio_performance(
    returns=port_ret_hrp,
    returns_benchmark=benchmark_returns,
    n_sim=10000
)

print("\nSimulation Results (Key Statistics):")
print(results_means)

# Calculate outperformance probability
prob = beating_probability(port_ret_hrp, benchmark_returns, n_sample=10000)
print(f"\nProbability of beating S&P 500: {prob:.2%}")

# Create fan chart (visualization)
print("\nGenerating fan chart forecast...")
fan_chart(port_ret_hrp, n_sample=5000)
```

### Example 3: Asset Clustering and Analysis

```python
from myPortfolioManagement.myClustering import ts_clustering, kMeansClusterSeries

print("\n" + "="*60)
print("ASSET CLUSTERING")
print("="*60)

# Cluster assets by time series similarity
clusters, centers = ts_clustering(returns, number_of_clusters=3)
print("\nAsset Clusters (by similarity):")
print(clusters)

print("\nCluster Centers (average behavior):")
print(centers)

# Visualize with dendrogram
print("\nGenerating dendrogram...")
kMeansClusterSeries(returns.mean().to_frame(), title='Asset Clustering')
```

### Example 4: Bootstrap Resampling for Risk Estimation

```python
from myPortfolioManagement.myBootstrapping import BootstrapStationary

print("\n" + "="*60)
print("BOOTSTRAP RESAMPLING")
print("="*60)

# Resample portfolio returns with block structure
print("Resampling portfolio returns (1000 samples)...")
bootstrap_samples = BootstrapStationary(
    series=port_ret_hrp.squeeze(),
    block_size=20,
    n_samples=1000,
    optimal_block=False
)

# Analyze distribution
print("\nBootstrap Statistics (Sharpe-like distribution):")
boot_stats = bootstrap_samples.describe()
print(boot_stats.iloc[:, :5])  # Show first 5 columns

# Confidence intervals
percentile_5 = bootstrap_samples.quantile(0.05)
percentile_95 = bootstrap_samples.quantile(0.95)
print(f"\n5th-95th Percentile Range:")
print(f"  Mean: {percentile_5.mean():.4f} to {percentile_95.mean():.4f}")
```

### Example 5: Walk-Forward Backtesting

```python
print("\n" + "="*60)
print("WALK-FORWARD OUT-OF-SAMPLE TEST")
print("="*60)

def walk_forward_optimization(returns, train_period=252, rebalance_freq=63):
    """Test portfolio with rolling window optimization"""
    results = []
    weights_history = []
    
    for i in range(train_period, len(returns), rebalance_freq):
        # Use 1-year window to optimize
        train_data = returns.iloc[i-train_period:i]
        weights = HRP(returns_training=train_data)
        
        # Apply weights to next period
        future_period = slice(i, i+rebalance_freq)
        future_ret = returns.iloc[future_period]
        port_ret = (future_ret * weights['port_weight'].values).sum(axis=1)
        
        results.append(port_ret)
        weights_history.append(weights)
    
    wf_returns = pd.concat(results)
    return wf_returns, weights_history

print("\nRunning walk-forward test (12 rebalances/year)...")
wf_returns, wf_weights = walk_forward_optimization(returns, train_period=252)

print(f"Out-of-sample Annual Return: {wf_returns.mean() * 252:.2%}")
print(f"Out-of-sample Annual Volatility: {wf_returns.std() * np.sqrt(252):.2%}")
print(f"Out-of-sample Sharpe Ratio: {(wf_returns.mean() / wf_returns.std()) * np.sqrt(252):.2f}")
```

---

## Advanced Examples

### Rolling Performance Analysis

```python
import matplotlib.pyplot as plt

def rolling_performance(returns, window=252, benchmark=None):
    """Calculate rolling performance metrics"""
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252)
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_ret = returns.rolling(window).mean() * 252
    
    return rolling_sharpe, rolling_vol, rolling_ret

sharpe_rolling, vol_rolling, ret_rolling = rolling_performance(port_ret_hrp, window=252)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

sharpe_rolling.plot(ax=axes[0], title='Rolling 1-Year Sharpe Ratio')
axes[0].set_ylabel('Sharpe Ratio')

vol_rolling.plot(ax=axes[1], title='Rolling 1-Year Volatility')
axes[1].set_ylabel('Annual Volatility')

ret_rolling.plot(ax=axes[2], title='Rolling 1-Year Return')
axes[2].set_ylabel('Annual Return')

plt.tight_layout()
plt.show()
```

### Stress Testing

```python
def stress_test(weights, returns, percentile=1):
    """Portfolio stress test using worst historical scenarios"""
    port_ret = (returns * weights['port_weight'].values).sum(axis=1)
    
    # Get worst days
    num_worst = int(len(returns) * (percentile / 100))
    worst_indices = port_ret.nsmallest(num_worst).index
    worst_days = returns.loc[worst_indices]
    
    # Calculate portfolio loss on worst days
    port_loss = (worst_days * weights['port_weight'].values).sum(axis=1)
    
    return port_loss

print("Stress Test: 1st Percentile Worst Days")
stress_losses = stress_test(weights_hrp, returns, percentile=1)
print(f"Average loss on worst days: {stress_losses.mean():.2%}")
print(f"Worst single day: {stress_losses.min():.2%}")
print(f"Best worst-day: {stress_losses.max():.2%}")
```

### Correlation and Tail Risk Analysis

```python
def tail_correlation(returns, returns_benchmark, tail_threshold=0.05):
    """Analyze correlation in tail events"""
    # Identify tail events (5th percentile)
    bench_tail = returns_benchmark < returns_benchmark.quantile(tail_threshold)
    
    # Correlation during tail events
    tail_returns = returns[bench_tail]
    tail_corr = tail_returns.corr(returns_benchmark[bench_tail])
    
    return tail_corr

tail_corr = tail_correlation(returns, benchmark_returns)
print("Correlation During Market Stress (Bottom 5%):")
print(tail_corr.sort_values(ascending=False))
```

---

## API Reference

### Core Package Structure

```
myPortfolioManagement/
├── myData.py                 # Data fetching and preparation
├── myReturns.py              # Returns calculation and conversion
├── myPerformanceMetrics.py   # Risk and performance metrics
├── myPortfolioOptimisation.py # Portfolio optimization
├── myBacktesting.py          # Simulation and backtesting
├── myClustering.py           # Asset clustering and analysis
├── myBootstrapping.py        # Resampling methods
├── myPlots.py                # Visualization utilities
├── myUtils.py                # Helper functions
├── myPortfolioRisks.py       # Risk stress testing
└── myPortfolioSelection.py   # Portfolio combination tools
```

### Import Examples

```python
# Data
from myPortfolioManagement.myData import get_stock_prices, get_stock_info

# Returns
from myPortfolioManagement.myReturns import calculate_returns, average_returns

# Metrics
from myPortfolioManagement.myPerformanceMetrics import (
    performance_overview, alpha_beta_table, drawdown_details
)

# Optimization
from myPortfolioManagement.myPortfolioOptimisation import (
    HRP, inverse_vol_portfolio, equal_weight_portfolio
)

# Backtesting
from myPortfolioManagement.myBacktesting import (
    bootstrap_portfolio_performance, beating_probability, fan_chart
)

# Clustering
from myPortfolioManagement.myClustering import ts_clustering, detect_regimes

# Bootstrapping
from myPortfolioManagement.myBootstrapping import (
    BootstrapIDD, BootstrapStationary, BootstrapCircular
)
```

### Common Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `returns` | DataFrame | Wide format returns (dates as index, assets as columns) | `prices.pct_change()` |
| `prices` | DataFrame | Price data (same format as returns) | Yahoo Finance data |
| `weights` | DataFrame | Portfolio weights (assets as index, portfolio name as column) | HRP output |
| `rf` | float | Risk-free rate (annual) | `0.02` for 2% |
| `periods` | int | Number of periods per year | `252` for daily data |
| `window` | int | Rolling window size (in periods) | `252` for annual metrics |

---

## Requirements

### Core Dependencies
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
scipy >= 1.11.0
```

### Finance & Data
```
yfinance >= 0.2.28
riskfolio-lib >= 4.0.0
pyportfolioopt >= 1.5.5
empyrical-reloaded >= 0.5.0
ffn >= 1.0.0
pyfolio >= 0.9.2
```

### Analysis & ML
```
arch >= 6.0.0
tslearn >= 0.6.0
quantstats-lumi >= 0.3.0
feature-engine >= 1.6.0
```

### Visualization
```
matplotlib >= 3.7.0
seaborn >= 0.12.0
plotly >= 5.15.0
```

### Other
```
timebudget >= 0.1.0
joblib >= 1.3.0
tqdm >= 4.65.0
jupyter >= 1.0.0
```

See `requirements.txt` for complete list with exact versions.

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pyfolio'`
**Solution**: Install the missing package
```bash
pip install pyfolio
```

### Issue: Yahoo Finance returns no data
**Solution**: Check ticker symbols and date range
```python
# Verify ticker is valid
import yfinance as yf
ticker = yf.Ticker('INVALID')
print(ticker.info)  # Will show error if invalid
```

### Issue: Portfolio optimization convergence problems
**Solution**: Try different covariance estimation methods
```python
# Ledoit-Wolf shrinkage often works well
weights = HRP(returns_training=returns, covariance='ledoit')

# Or try OAS method
weights = HRP(returns_training=returns, covariance='oas')
```

### Issue: Memory problems with large datasets
**Solution**: Use parallel processing or subset data
```python
# Process in chunks
from joblib import Parallel, delayed

# Or use sampling
returns_sample = returns.sample(frac=0.5)
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Make your changes
4. Commit: `git commit -m 'Add YourFeature'`
5. Push: `git push origin feature/YourFeature`
6. Submit a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this library in your research or projects, please cite:

```bibtex
@software{QuantitativePortfolioManagement2024,
  author = {Ferhat},
  title = {MyPortfolioManagement: A Python Library for Quantitative Portfolio Management},
  year = {2024},
  url = {https://github.com/ferhat00/QuantitativePortfolioManagement}
}
```

---

## References & Resources

### Key Papers
- Lopéz de Prado, M. (2016). Building Diversified Portfolios that Outperform. Journal of Portfolio Management
- Meucci, A. (2005). Risk and Asset Allocation. Springer
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance

### Libraries Used
- [riskfolio-lib](https://riskfolio-lib.readthedocs.io/) - Portfolio optimization
- [pyportfolioopt](https://pyportfolioopt.readthedocs.io/) - Efficient frontier
- [empyrical](https://github.com/quantopian/empyrical) - Performance metrics
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data

---

## Contact & Support

For questions, issues, or suggestions:
- **Open an Issue**: GitHub Issues tab
- **Email**: Contact repository owner
- **Discussion**: GitHub Discussions (if enabled)

---

## Changelog

### Version 0.9.2 (December 2024)
- Fixed `HRP` function to remove incorrect `assets_stats()` call
- Added `pyfolio` to requirements.txt
- Enhanced documentation with comprehensive examples
- Improved error handling in data fetching

---

**Last Updated**: December 21, 2024  
**Current Version**: 0.9.2  
**Maintainer**: Ferhat
