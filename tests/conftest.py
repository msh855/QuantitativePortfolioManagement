"""
Shared test fixtures and configuration for myPortfolioManagement tests.

This module provides reusable fixtures for testing:
- Sample price data (synthetic and real)
- Sample returns data
- Portfolio weights
- Benchmark returns
- GPU availability detection
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# GPU Detection
# =============================================================================


def is_gpu_available():
    """Check if CuPy/GPU is available for testing."""
    try:
        import cupy as cp

        # Try to allocate a small array to verify GPU works
        test_array = cp.array([1, 2, 3])
        _ = cp.asnumpy(test_array)
        return True
    except Exception:
        return False


GPU_AVAILABLE = is_gpu_available()


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Standard unit test")
    config.addinivalue_line("markers", "integration: Multi-module integration test")
    config.addinivalue_line("markers", "gpu: Test requires GPU (CuPy)")
    config.addinivalue_line("markers", "slow: Long-running test (>5s)")
    config.addinivalue_line("markers", "data_fetch: Test requires internet/API access")
    config.addinivalue_line("markers", "parametrize: Parametrized test")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if GPU is not available."""
    if not GPU_AVAILABLE:
        skip_gpu = pytest.mark.skip(reason="GPU (CuPy) not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# Date Fixtures
# =============================================================================


@pytest.fixture
def sample_dates():
    """Generate a sequence of business dates for testing."""
    return pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")


@pytest.fixture
def short_dates():
    """Generate a short sequence of dates for quick tests."""
    return pd.date_range(start="2023-01-01", end="2023-03-31", freq="B")


@pytest.fixture
def monthly_dates():
    """Generate monthly date sequence for frequency conversion tests."""
    return pd.date_range(start="2020-01-01", end="2023-12-31", freq="ME")


# =============================================================================
# Price Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prices(sample_dates):
    """
    Generate synthetic stock prices with realistic characteristics.

    Creates prices for 5 assets with:
    - Different drift rates (expected returns)
    - Different volatilities
    - Realistic log-normal price dynamics
    """
    np.random.seed(42)
    n_days = len(sample_dates)
    n_assets = 5
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Asset characteristics (annualized)
    drifts = np.array([0.12, 0.10, 0.08, 0.15, 0.05])  # Expected returns
    vols = np.array([0.25, 0.22, 0.28, 0.35, 0.30])  # Volatilities

    # Generate log returns
    dt = 1 / 252  # Daily time step
    log_returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        log_returns[:, i] = (drifts[i] - 0.5 * vols[i] ** 2) * dt + vols[i] * np.sqrt(dt) * np.random.randn(n_days)

    # Convert to prices (starting at 100)
    prices = np.exp(np.cumsum(log_returns, axis=0)) * 100

    return pd.DataFrame(prices, index=sample_dates, columns=tickers)


@pytest.fixture
def single_asset_prices(sample_dates):
    """Generate prices for a single asset."""
    np.random.seed(42)
    n_days = len(sample_dates)

    drift = 0.10
    vol = 0.20
    dt = 1 / 252

    log_returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * np.random.randn(n_days)
    prices = np.exp(np.cumsum(log_returns)) * 100

    return pd.DataFrame({"SPY": prices}, index=sample_dates)


@pytest.fixture
def prices_with_nans(sample_prices):
    """Generate price data with some NaN values for edge case testing."""
    prices = sample_prices.copy()
    # Add some random NaNs (about 2% of data)
    np.random.seed(123)
    mask = np.random.random(prices.shape) < 0.02
    prices.values[mask] = np.nan
    return prices


@pytest.fixture
def prices_with_zeros(sample_prices):
    """Generate price data with some zero values for edge case testing."""
    prices = sample_prices.copy()
    # Add some random zeros (about 1% of data)
    np.random.seed(456)
    mask = np.random.random(prices.shape) < 0.01
    prices.values[mask] = 0.0
    return prices


@pytest.fixture
def constant_prices(sample_dates):
    """Generate constant (zero volatility) prices for edge case testing."""
    tickers = ["FLAT1", "FLAT2", "FLAT3"]
    prices = pd.DataFrame(np.ones((len(sample_dates), 3)) * 100, index=sample_dates, columns=tickers)
    return prices


@pytest.fixture
def extreme_prices(sample_dates):
    """Generate prices with extreme movements for stress testing."""
    np.random.seed(789)
    n_days = len(sample_dates)

    # Create a price series with occasional extreme jumps
    base_returns = np.random.randn(n_days) * 0.01  # 1% daily vol

    # Add extreme events (crashes and rallies)
    crash_days = [50, 200, 500]
    rally_days = [100, 300, 600]

    for day in crash_days:
        if day < n_days:
            base_returns[day] = -0.20  # 20% crash

    for day in rally_days:
        if day < n_days:
            base_returns[day] = 0.15  # 15% rally

    prices = np.exp(np.cumsum(base_returns)) * 100
    return pd.DataFrame({"EXTREME": prices}, index=sample_dates)


# =============================================================================
# Returns Data Fixtures
# =============================================================================


@pytest.fixture
def sample_returns(sample_prices):
    """Calculate simple returns from sample prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_log_returns(sample_prices):
    """Calculate log returns from sample prices."""
    return np.log(sample_prices / sample_prices.shift(1)).dropna()


@pytest.fixture
def short_returns(short_dates):
    """Generate short returns series for quick tests."""
    np.random.seed(42)
    n_days = len(short_dates)
    tickers = ["A", "B", "C"]

    returns = pd.DataFrame(np.random.randn(n_days, 3) * 0.02, index=short_dates, columns=tickers)  # 2% daily vol
    return returns


@pytest.fixture
def correlated_returns(sample_dates):
    """Generate returns with known correlation structure."""
    np.random.seed(42)
    n_days = len(sample_dates)

    # Define correlation matrix
    corr_matrix = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])

    # Generate correlated returns via Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.randn(n_days, 3) * 0.02
    correlated = uncorrelated @ L.T

    return pd.DataFrame(correlated, index=sample_dates, columns=["HIGH_CORR_A", "HIGH_CORR_B", "LOW_CORR"])


@pytest.fixture
def negative_returns(sample_dates):
    """Generate consistently negative returns for testing."""
    np.random.seed(42)
    n_days = len(sample_dates)

    # Negative drift with some noise
    returns = -0.001 + np.random.randn(n_days) * 0.01

    return pd.DataFrame({"LOSER": returns}, index=sample_dates)


@pytest.fixture
def positive_returns(sample_dates):
    """Generate consistently positive returns for testing."""
    np.random.seed(42)
    n_days = len(sample_dates)

    # Positive drift with some noise
    returns = 0.002 + np.random.randn(n_days) * 0.01

    return pd.DataFrame({"WINNER": returns}, index=sample_dates)


@pytest.fixture
def zero_mean_returns(sample_dates):
    """Generate returns with approximately zero mean."""
    np.random.seed(42)
    n_days = len(sample_dates)

    returns = np.random.randn(n_days) * 0.02
    # Center exactly at zero
    returns = returns - returns.mean()

    return pd.DataFrame({"ZERO_MEAN": returns}, index=sample_dates)


# =============================================================================
# Portfolio Weights Fixtures
# =============================================================================


@pytest.fixture
def equal_weights():
    """Generate equal weights for 5 assets."""
    return pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])


@pytest.fixture
def concentrated_weights():
    """Generate concentrated (non-equal) weights."""
    return pd.Series([0.4, 0.3, 0.15, 0.1, 0.05], index=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])


@pytest.fixture
def single_asset_weight():
    """Generate weights for single-asset portfolio."""
    return pd.Series([1.0], index=["AAPL"])


@pytest.fixture
def weights_dict():
    """Generate weights as dictionary (alternative format)."""
    return {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25}


@pytest.fixture
def multiple_portfolios_weights():
    """Generate weights for multiple portfolios."""
    return pd.DataFrame(
        {
            "Conservative": [0.6, 0.3, 0.1, 0.0, 0.0],
            "Balanced": [0.3, 0.3, 0.2, 0.1, 0.1],
            "Aggressive": [0.1, 0.1, 0.2, 0.3, 0.3],
        },
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    )


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def market_returns(sample_dates):
    """Generate market benchmark returns (S&P 500-like)."""
    np.random.seed(100)
    n_days = len(sample_dates)

    # Market characteristics
    drift = 0.08  # 8% annual return
    vol = 0.18  # 18% annual vol
    dt = 1 / 252

    returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * np.random.randn(n_days)

    return pd.DataFrame({"SPY": returns}, index=sample_dates)


@pytest.fixture
def bond_returns(sample_dates):
    """Generate bond benchmark returns (AGG-like)."""
    np.random.seed(101)
    n_days = len(sample_dates)

    # Bond characteristics
    drift = 0.03  # 3% annual return
    vol = 0.05  # 5% annual vol
    dt = 1 / 252

    returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * np.random.randn(n_days)

    return pd.DataFrame({"AGG": returns}, index=sample_dates)


# =============================================================================
# Option Pricing Fixtures
# =============================================================================


@pytest.fixture
def option_params():
    """Standard option pricing parameters."""
    return {
        "S": 100,  # Spot price
        "K": 100,  # Strike price (ATM)
        "T": 0.25,  # Time to expiration (3 months)
        "r": 0.05,  # Risk-free rate (5%)
        "sigma": 0.20,  # Volatility (20%)
    }


@pytest.fixture
def option_chain_params():
    """Parameters for generating option chains."""
    return {"S": 100, "T": 0.25, "r": 0.05, "sigma": 0.20, "strikes": np.arange(80, 121, 5)}  # Strikes from 80 to 120


@pytest.fixture
def itm_call_params():
    """In-the-money call option parameters."""
    return {"S": 110, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.20}


@pytest.fixture
def otm_call_params():
    """Out-of-the-money call option parameters."""
    return {"S": 90, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.20}


# =============================================================================
# Covariance Matrix Fixtures
# =============================================================================


@pytest.fixture
def simple_cov_matrix():
    """Generate a simple positive-definite covariance matrix."""
    return pd.DataFrame(
        np.array([[0.04, 0.02, 0.01], [0.02, 0.05, 0.015], [0.01, 0.015, 0.03]]),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


@pytest.fixture
def identity_cov_matrix():
    """Generate identity covariance matrix (uncorrelated assets)."""
    n = 5
    tickers = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.eye(n) * 0.04, index=tickers, columns=tickers)  # 20% volatility


@pytest.fixture
def high_corr_cov_matrix():
    """Generate highly correlated covariance matrix."""
    corr = 0.9
    vol = 0.20
    n = 3
    tickers = ["X", "Y", "Z"]

    cov = np.full((n, n), corr * vol * vol)
    np.fill_diagonal(cov, vol * vol)

    return pd.DataFrame(cov, index=tickers, columns=tickers)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def risk_free_rate():
    """Standard risk-free rate for testing."""
    return 0.02  # 2%


@pytest.fixture
def periods_annual():
    """Annualization periods for different frequencies."""
    return {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def relative_tolerance():
    """Relative tolerance for approximate equality tests."""
    return 0.01  # 1%


# =============================================================================
# GPU Testing Fixtures
# =============================================================================


@pytest.fixture
def gpu_available():
    """Check if GPU is available for the test."""
    return GPU_AVAILABLE


@pytest.fixture
def large_returns_for_gpu(sample_dates):
    """Generate large returns dataset for GPU performance testing."""
    np.random.seed(42)
    n_days = len(sample_dates)
    n_assets = 50  # Larger portfolio for GPU testing

    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    returns = pd.DataFrame(np.random.randn(n_days, n_assets) * 0.02, index=sample_dates, columns=tickers)
    return returns


@pytest.fixture
def bootstrap_params():
    """Standard bootstrap parameters."""
    return {"n_samples": 1000, "seed": 42, "block_size": 20}


# =============================================================================
# Edge Case Fixtures
# =============================================================================


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_row_returns():
    """Single row of returns for edge case testing."""
    return pd.DataFrame({"A": [0.01], "B": [-0.02], "C": [0.005]}, index=[pd.Timestamp("2023-01-01")])


@pytest.fixture
def two_row_returns():
    """Two rows of returns for minimum data edge case."""
    return pd.DataFrame(
        {"A": [0.01, -0.02], "B": [-0.02, 0.03]}, index=pd.date_range("2023-01-01", periods=2, freq="D")
    )


@pytest.fixture
def infinite_values_returns(sample_returns):
    """Returns with infinite values for edge case testing."""
    returns = sample_returns.copy()
    returns.iloc[10, 0] = np.inf
    returns.iloc[20, 1] = -np.inf
    return returns


@pytest.fixture
def all_nan_returns(sample_dates):
    """All-NaN returns for edge case testing."""
    return pd.DataFrame(np.nan, index=sample_dates[:100], columns=["NAN_ASSET"])


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def full_portfolio_setup(sample_prices, sample_returns, equal_weights, market_returns):
    """Complete portfolio setup for integration tests."""
    return {
        "prices": sample_prices,
        "returns": sample_returns,
        "weights": equal_weights,
        "benchmark": market_returns,
        "rf": 0.02,
        "periods": 252,
    }


@pytest.fixture
def optimization_setup(sample_returns, simple_cov_matrix):
    """Setup for optimization tests."""
    # Filter returns to match covariance matrix assets
    returns = sample_returns[["AAPL", "MSFT", "GOOGL"]].copy()
    returns.columns = ["A", "B", "C"]

    return {
        "returns": returns,
        "cov_matrix": simple_cov_matrix,
        "constraints": {"min_weight": 0.05, "max_weight": 0.50},
    }


# =============================================================================
# Real Data Fixtures (for integration tests with API)
# =============================================================================


@pytest.fixture
def real_tickers():
    """Real ticker symbols for API tests."""
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def real_date_range():
    """Date range for real data tests."""
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    return {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")}
