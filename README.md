# MyPortfolioManagement

A comprehensive Python library for quantitative portfolio management, backtesting, and performance analysis. This toolkit provides professional-grade functions for data fetching, portfolio optimization, risk management, performance metrics, and Monte Carlo simulations.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Local Installation](#local-installation)
  - [Kaggle Installation](#kaggle-installation)
  - [Google Colab Installation](#google-colab-installation)
- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
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

- **Python 3.10+** (recommended: Python 3.12)
- pip (Python package manager)
- Git

### Local Installation

#### Step 1: Clone the Repository
```bash
# Clone the repository
git clone https://github.com/msh855/QuantitativePortfolioManagement.git

# Navigate to the directory
cd QuantitativePortfolioManagement
```

#### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Step 3: Install Dependencies

**Option A: Full Installation (with OpenBB)**
```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Option B: Core Installation (without OpenBB - Recommended for most users)**

If you don't need OpenBB functionality, use this lighter installation:
```bash
# Install core dependencies
pip install pandas>=2.2.0 numpy>=2.0.0
pip install yfinance==0.2.58 quandl finvizfinance yahoofinancials
pip install riskfolio-lib>=6.0.0 pyportfolioopt
pip install quantstats-lumi empyrical-reloaded ffn pyfolio-reloaded
pip install matplotlib seaborn plotly scikit-learn scipy arch tslearn tsmoothie
pip install timebudget feature-engine joblib tqdm parallel-pandas ray

# Install the package
pip install -e .
```

#### Step 4: Verify Installation
```python
# Test the installation
python test_install.py
```

Or test manually:
```python
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns

# Fetch some data
prices = get_stock_prices(['AAPL', 'MSFT'], start_date='2024-01-01', wide_format=True)
returns = calculate_returns(prices)
print("✅ Installation successful!")
print(f"Returns shape: {returns.shape}")
```

---

### Kaggle Installation

Kaggle has pre-installed packages that can conflict with our requirements. Follow these steps carefully:

#### Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click on "Code" → "New Notebook"
3. Make sure you're using Python 3.12+

#### Step 2: Clone the Repository
```python
# In your first cell, run:
!git clone https://github.com/msh855/QuantitativePortfolioManagement.git
%cd QuantitativePortfolioManagement
```

#### Step 3: Create Updated Requirements File

Kaggle requires specific package versions. Create a Kaggle-compatible requirements file:
```python
# Create requirements file for Kaggle
%%writefile requirements-kaggle.txt
pandas>=2.2.0
numpy>=2.0.0,<2.3.0
yfinance==0.2.58
quandl
finvizfinance
yahoofinancials
riskfolio-lib>=6.0.0
pyportfolioopt>=1.5.5
quantstats-lumi>=0.3.0
empyrical-reloaded>=0.5.0
ffn>=1.0.0
pyfolio-reloaded>=0.9.0
matplotlib>=3.9.0
seaborn>=0.13.0
plotly>=5.15.0
scikit-learn>=1.6.0
scipy>=1.14.0
arch>=7.0.0
tslearn>=0.7.0
tsmoothie>=1.0.4
timebudget>=0.1.0
feature-engine>=1.9.0
joblib>=1.5.0
tqdm>=4.65.0
parallel-pandas>=0.7.0
ray==2.53.0
```

#### Step 4: Install Dependencies
```python
# Install core packages
!pip install --upgrade pip
!pip install -r requirements-kaggle.txt
```

**Note:** You will see some dependency warnings like:
```
ERROR: google-colab 1.0.0 requires ipython==7.34.0, but you have ipython 9.8.0
```

**These warnings are normal and can be safely ignored.** They're related to Kaggle's base environment and won't affect your portfolio analysis.

#### Step 5: Install OpenBB (Optional)

If you need OpenBB functionality:
```python
# Install OpenBB separately
!pip install openbb==4.5.0
```

**Note:** OpenBB is only used for currency conversion in `myUtils.py`. Most portfolio management tasks don't require it.

#### Step 6: Install the Package
```python
# Install the package
!pip install -e .
```

#### Step 7: Verify Installation
```python
# Test imports
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns
from myPortfolioManagement.myPortfolioOptimisation import HRP

# Quick test
prices = get_stock_prices(['AAPL', 'MSFT'], start_date='2024-01-01', end_date='2024-12-01', wide_format=True)
returns = calculate_returns(prices)
print("✅ Installation successful!")
print(f"Returns shape: {returns.shape}")
```

---

### Google Colab Installation

Google Colab installation is similar to Kaggle:

#### Step 1: Create New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

#### Step 2: Clone and Install
```python
# Clone repository
!git clone https://github.com/msh855/QuantitativePortfolioManagement.git
%cd QuantitativePortfolioManagement

# Create requirements file
%%writefile requirements-colab.txt
pandas>=2.2.0
numpy>=2.0.0,<2.3.0
yfinance==0.2.58
quandl
finvizfinance
yahoofinancials
riskfolio-lib>=6.0.0
pyportfolioopt>=1.5.5
quantstats-lumi>=0.3.0
empyrical-reloaded>=0.5.0
ffn>=1.0.0
pyfolio-reloaded>=0.9.0
matplotlib>=3.9.0
seaborn>=0.13.0
plotly>=5.15.0
scikit-learn>=1.6.0
scipy>=1.14.0
arch>=7.0.0
tslearn>=0.7.0
tsmoothie>=1.0.4
timebudget>=0.1.0
feature-engine>=1.9.0
joblib>=1.5.0
tqdm>=4.65.0
parallel-pandas>=0.7.0
ray==2.53.0

# Install
!pip install --upgrade pip
!pip install -r requirements-colab.txt
!pip install -e .
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
    freq='daily',
    wide_format=True
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

**Example**:
```python
prices = get_stock_prices(
    yahoo_tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2022-01-01',
    end_date='2024-12-01',
    freq='daily',
    wide_format=True
)
```

### Returns Calculation (myReturns.py)

**Purpose**: Calculate and manipulate returns data

#### `calculate_returns(df_prices, log_returns=False, convert_to=None)`
Converts price data to returns

**Example**:
```python
# Simple returns
simple_returns = calculate_returns(prices, log_returns=False)

# Log returns
log_returns = calculate_returns(prices, log_returns=True)

# Monthly returns
monthly_returns = calculate_returns(prices, convert_to='monthly')
```

### Performance Metrics (myPerformanceMetrics.py)

**Purpose**: Calculate comprehensive risk and return metrics

#### `performance_overview(df, prices=False, short=True)`
Single-call function for all key metrics

**Example**:
```python
metrics = performance_overview(returns)
print(metrics)
```

### Portfolio Optimization (myPortfolioOptimisation.py)

**Purpose**: Generate optimal portfolio weights using various methods

#### `HRP(model='HRP', returns_training, covariance='hist', codependence='pearson', linkage='ward')`
Hierarchical Risk Parity and related methods

**Example**:
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
```

### Backtesting & Simulation (myBacktesting.py)

**Purpose**: Simulate portfolio performance and test strategies

#### `bootstrap_portfolio_performance(returns, returns_benchmark=None, n_sim=1000)`
Monte Carlo simulation of portfolio performance

**Example**:
```python
results_means, results_dist, results_stats = bootstrap_portfolio_performance(
    returns=portfolio_returns,
    returns_benchmark=benchmark_returns,
    n_sim=10000
)
```

### Clustering & Analysis (myClustering.py)

**Purpose**: Identify asset groups and market regimes

#### `ts_clustering(df, number_of_clusters=None, algo='dtw')`
Time series clustering with DTW or Euclidean distance

**Example**:
```python
clusters, centers = ts_clustering(returns, number_of_clusters=3)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: NumPy Version Conflicts (Kaggle/Colab)

**Error:**
```
cesium 0.12.4 requires numpy<3.0,>=2.0, but you have numpy 1.26.4 which is incompatible
```

**Solution:**
Use `numpy>=2.0.0` in your requirements file. This is already fixed in `requirements-kaggle.txt`.

---

#### Issue 2: yfinance Version Conflict with OpenBB

**Error:**
```
openbb-yfinance 1.5.0 depends on yfinance==0.2.58
```

**Solution:**
Pin yfinance to exactly version 0.2.58:
```python
pip install yfinance==0.2.58
```

---

#### Issue 3: Dependency Warnings in Kaggle

**Error:**
```
ERROR: google-colab 1.0.0 requires ipython==7.34.0, but you have ipython 9.8.0
```

**Solution:**
**These are warnings, not errors!** Your code will work fine. These conflicts are with Kaggle's base environment packages and can be safely ignored.

---

#### Issue 4: "ModuleNotFoundError: No module named 'myPortfolioManagement'"

**Solution:**
Make sure you've run:
```python
!pip install -e .
```
from within the repository directory.

---

#### Issue 5: KeyError with Portfolio Weights

**Error:**
```python
KeyError: "None of ['asset'] are in the columns"
```

**Solution:**
The weights DataFrame is already indexed by 'asset'. Don't call `.set_index('asset')` again:

**Wrong:**
```python
weights_dict = weights_max_sharpe.set_index('asset')['port_max_Sharpe'].to_dict()
```

**Correct:**
```python
weights_dict = weights_max_sharpe['port_max_Sharpe'].to_dict()
```

---

#### Issue 6: Cannot Create File with `cat << EOF` in Kaggle

**Error:**
```
SyntaxError: invalid syntax
```

**Solution:**
Use Jupyter magic command `%%writefile` instead:
```python
# Correct way in Kaggle/Colab
%%writefile requirements.txt
pandas>=2.2.0
numpy>=2.0.0
...
```

---

#### Issue 7: OpenBB Not Essential

If you encounter persistent OpenBB installation issues, you can skip it entirely. OpenBB is only used for currency conversion in `myUtils.py` (`_load_fx()` function). Most portfolio management tasks don't require it.

**To skip OpenBB:**
Simply remove it from your requirements file or don't install it separately.

---

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/msh855/QuantitativePortfolioManagement/issues)
2. Create a new issue with:
   - Your Python version
   - Your environment (Local/Kaggle/Colab)
   - Full error message
   - Steps to reproduce

---

## Requirements

### Core Dependencies
```
pandas >= 2.0.0
numpy >= 2.0.0 (for Kaggle/Colab compatibility)
scikit-learn >= 1.3.0
scipy >= 1.14.0
```

### Finance & Data
```
yfinance == 0.2.58 (pinned for OpenBB compatibility)
riskfolio-lib >= 6.0.0
pyportfolioopt >= 1.5.5
empyrical-reloaded >= 0.5.0
ffn >= 1.0.0
pyfolio-reloaded >= 0.9.0
```

### Analysis & ML
```
arch >= 7.0.0
tslearn >= 0.7.0
quantstats-lumi >= 0.3.0
feature-engine >= 1.9.0
```

### Visualization
```
matplotlib >= 3.9.0
seaborn >= 0.13.0
plotly >= 5.15.0
```

### Optional
```
ray >= 2.53.0
openbb >= 4.5.0 (optional - only for FX conversion)
```

See `requirements.txt` for complete list with exact versions.

---

## Usage Examples

See the `examples_portfolio_management.py` file for comprehensive examples covering all major functions.

Quick examples:

### Example 1: Complete Portfolio Analysis
```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import calculate_returns
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myPortfolioOptimisation import HRP, inverse_vol_portfolio

# 1. Fetch data
tickers = ['SPY', 'AGG', 'GLD', 'EEM', 'VNQ']
prices = get_stock_prices(tickers, start_date='2020-01-01', end_date='2024-12-01', wide_format=True)

# 2. Calculate returns
returns = calculate_returns(prices)

# 3. Performance overview
perf = performance_overview(returns)
print(perf)

# 4. Optimize portfolio
weights_hrp = HRP(returns_training=returns, covariance='ledoit')
weights_inv_vol = inverse_vol_portfolio(returns)

print("\nHRP Weights:")
print(weights_hrp)
```

### Example 2: Monte Carlo Simulation
```python
from myPortfolioManagement.myBacktesting import bootstrap_portfolio_performance, fan_chart

# Run simulation
results_means, results_dist, results_stats = bootstrap_portfolio_performance(
    returns=portfolio_returns,
    returns_benchmark=benchmark_returns,
    n_sim=10000
)

print("Simulation Results:")
print(results_means)

# Create fan chart
fan_chart(portfolio_returns, n_sample=5000)
```

### Example 3: Asset Clustering
```python
from myPortfolioManagement.myClustering import ts_clustering

clusters, centers = ts_clustering(returns, number_of_clusters=3)
print("Asset Clusters:")
print(clusters)
```

---

## Advanced Features

### Walk-Forward Backtesting
```python
def walk_forward_optimization(returns, train_period=252, rebalance_freq=63):
    results = []
    weights_history = []
    
    for i in range(train_period, len(returns), rebalance_freq):
        train_data = returns.iloc[i-train_period:i]
        weights = HRP(returns_training=train_data)
        
        future_period = slice(i, i+rebalance_freq)
        future_ret = returns.iloc[future_period]
        port_ret = (future_ret * weights['port_weight'].values).sum(axis=1)
        
        results.append(port_ret)
        weights_history.append(weights)
    
    return pd.concat(results), weights_history

wf_returns, wf_weights = walk_forward_optimization(returns, train_period=252)
print(f"Out-of-sample Sharpe: {(wf_returns.mean() / wf_returns.std()) * np.sqrt(252):.2f}")
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

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

---

## Citation

If you use this library in your research or projects, please cite:
```bibtex
@software{QuantitativePortfolioManagement2024,
  author = {Moustafa C and Ferhat C},
  title = {MyPortfolioManagement: A Python Library for Quantitative Portfolio Management},
  year = {2024},
  url = {https://github.com/msh855/QuantitativePortfolioManagement}
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
- **GitHub Issues**: [Open an issue](https://github.com/msh855/QuantitativePortfolioManagement/issues)
- **Email**: Contact repository owner

---

## Changelog

### Version 1.0.0 (December 2024)
- Initial release
- Added Kaggle/Colab compatibility
- Updated numpy to 2.x support
- Fixed yfinance version conflicts
- Enhanced documentation with explicit installation guides

---

**Last Updated**: December 22, 2024  
**Current Version**: 1.0.0  
**Python Compatibility**: 3.10+  
**Maintainers**: Moustafa C, Ferhat C
