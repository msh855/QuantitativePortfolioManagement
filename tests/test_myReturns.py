"""
Unit tests for myPortfolioManagement.myReturns module.

Tests cover:
- Simple and log returns calculation (calculate_returns)
- Frequency conversion (convert_returns_freq)
- Average returns (average_returns)
- Benchmark returns (get_benchmark_returns)
"""

import numpy as np
import pandas as pd
import pytest

from myPortfolioManagement.myReturns import (
    average_returns,
    calculate_returns,
    convert_returns_freq,
    get_benchmark_returns,
)

# =============================================================================
# Test Simple Returns Calculation
# =============================================================================


@pytest.mark.unit
class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_simple_returns_basic(self, sample_prices):
        """Test basic simple returns calculation."""
        returns = calculate_returns(sample_prices, log_returns=False)

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1
        assert returns.columns.tolist() == sample_prices.columns.tolist()

    def test_log_returns_basic(self, sample_prices):
        """Test basic log returns calculation."""
        returns = calculate_returns(sample_prices, log_returns=True)

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1
        assert returns.columns.tolist() == sample_prices.columns.tolist()

    def test_simple_vs_log_returns_approximation(self, sample_prices):
        """Test that log returns approximately equal simple returns for small changes."""
        simple = calculate_returns(sample_prices, log_returns=False)
        log = calculate_returns(sample_prices, log_returns=True)

        # For small returns, simple ≈ log returns
        mask = simple.abs() < 0.05
        simple_small = simple[mask]
        log_small = log[mask]

        diff = (simple_small - log_small).abs()
        assert (diff.dropna() < 0.01).all().all()

    def test_returns_datetime_index_preserved(self, sample_prices):
        """Test that DatetimeIndex is preserved in returns."""
        returns = calculate_returns(sample_prices)
        assert isinstance(returns.index, pd.DatetimeIndex)

    def test_returns_no_nan_introduced(self, sample_prices):
        """Test that returns calculation doesn't introduce extra NaNs."""
        returns = calculate_returns(sample_prices)
        assert returns.isna().sum().sum() == 0

    def test_positive_price_change_positive_return(self):
        """Test that positive price change gives positive return."""
        prices = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.date_range("2023-01-01", periods=2))

        returns = calculate_returns(prices, log_returns=False)
        assert returns.iloc[0, 0] == pytest.approx(0.1, rel=1e-6)

    def test_negative_price_change_negative_return(self):
        """Test that negative price change gives negative return."""
        prices = pd.DataFrame({"A": [100.0, 90.0]}, index=pd.date_range("2023-01-01", periods=2))

        returns = calculate_returns(prices, log_returns=False)
        assert returns.iloc[0, 0] == pytest.approx(-0.1, rel=1e-6)

    def test_zero_return_for_unchanged_price(self):
        """Test that unchanged price gives zero return."""
        prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=pd.date_range("2023-01-01", periods=3))

        returns = calculate_returns(prices, log_returns=False)
        assert (returns == 0).all().all()

    @pytest.mark.parametrize("log_returns", [True, False])
    def test_returns_calculation_both_methods(self, sample_prices, log_returns):
        """Parametrized test for both return calculation methods."""
        returns = calculate_returns(sample_prices, log_returns=log_returns)

        assert isinstance(returns, pd.DataFrame)
        assert not returns.empty
        assert returns.abs().max().max() < 1.0

    def test_single_column_returns(self, single_asset_prices):
        """Test returns calculation for single asset."""
        returns = calculate_returns(single_asset_prices)
        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) == 1

    def test_log_returns_additive_property(self, sample_prices):
        """Test that log returns are additive over time."""
        log_ret = calculate_returns(sample_prices, log_returns=True)

        total_log_return = log_ret.sum()
        expected = np.log(sample_prices.iloc[-1] / sample_prices.iloc[0])

        np.testing.assert_array_almost_equal(total_log_return.values, expected.values, decimal=10)


# =============================================================================
# Test Frequency Conversion
# =============================================================================


@pytest.mark.unit
class TestConvertReturnsFreq:
    """Tests for convert_returns_freq function."""

    def test_daily_to_weekly(self, sample_returns):
        """Test converting daily returns to weekly."""
        weekly = convert_returns_freq(sample_returns, convert_to="weekly")

        assert isinstance(weekly, pd.DataFrame)
        assert len(weekly) < len(sample_returns)

    def test_daily_to_monthly(self, sample_returns):
        """Test converting daily returns to monthly."""
        monthly = convert_returns_freq(sample_returns, convert_to="monthly")

        assert isinstance(monthly, pd.DataFrame)
        assert len(monthly) < len(sample_returns) / 10

    def test_daily_to_quarterly(self, sample_returns):
        """Test converting daily returns to quarterly."""
        quarterly = convert_returns_freq(sample_returns, convert_to="quarterly")

        assert isinstance(quarterly, pd.DataFrame)
        monthly = convert_returns_freq(sample_returns, convert_to="monthly")
        assert len(quarterly) < len(monthly)

    def test_daily_to_yearly(self, sample_returns):
        """Test converting daily returns to yearly."""
        yearly = convert_returns_freq(sample_returns, convert_to="yearly")

        assert isinstance(yearly, pd.DataFrame)
        years = sample_returns.index.year.nunique()
        assert len(yearly) <= years

    def test_frequency_conversion_preserves_columns(self, sample_returns):
        """Test that frequency conversion preserves column names."""
        monthly = convert_returns_freq(sample_returns, convert_to="monthly")
        assert monthly.columns.tolist() == sample_returns.columns.tolist()


# =============================================================================
# Test Average Returns
# =============================================================================


@pytest.mark.unit
class TestAverageReturns:
    """Tests for average_returns function."""

    def test_historical_average(self, sample_returns):
        """Test historical average return calculation."""
        avg = average_returns(sample_returns, method="hist")

        assert isinstance(avg, (float, pd.Series))
        if isinstance(avg, pd.Series):
            assert len(avg) == len(sample_returns.columns)

    def test_ema_average(self, sample_returns):
        """Test exponential moving average return calculation."""
        avg = average_returns(sample_returns, method="ema", span=60)
        assert isinstance(avg, (float, pd.Series))

    def test_positive_returns_positive_average(self, positive_returns):
        """Test that consistently positive returns have positive average."""
        avg = average_returns(positive_returns, method="hist")

        if isinstance(avg, pd.Series):
            avg = avg.iloc[0]
        assert avg > 0

    def test_negative_returns_negative_average(self, negative_returns):
        """Test that consistently negative returns have negative average."""
        avg = average_returns(negative_returns, method="hist")

        if isinstance(avg, pd.Series):
            avg = avg.iloc[0]
        assert avg < 0


# =============================================================================
# Test Benchmark Returns
# =============================================================================


@pytest.mark.unit
class TestGetBenchmarkReturns:
    """Tests for get_benchmark_returns function."""

    @pytest.mark.data_fetch
    def test_sp500_benchmark(self):
        """Test fetching S&P 500 benchmark returns."""
        try:
            benchmark = get_benchmark_returns("S&P500")
            assert isinstance(benchmark, pd.DataFrame)
            if not benchmark.empty:
                assert isinstance(benchmark.index, pd.DatetimeIndex)
        except Exception:
            pytest.skip("Benchmark data API unavailable")

    @pytest.mark.data_fetch
    def test_nasdaq_benchmark(self):
        """Test fetching NASDAQ benchmark returns."""
        try:
            benchmark = get_benchmark_returns("Nasdaq")
            assert isinstance(benchmark, pd.DataFrame)
        except Exception:
            pytest.skip("Benchmark data API unavailable")

    def test_invalid_benchmark_raises_error(self):
        """Test that invalid benchmark name raises error."""
        with pytest.raises((ValueError, KeyError)):
            get_benchmark_returns("InvalidBenchmark123")


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.unit
class TestReturnsEdgeCases:
    """Tests for edge cases in returns calculation."""

    def test_two_row_returns(self):
        """Test returns with minimum data (2 rows gives 1 return)."""
        prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 55.0]}, index=pd.date_range("2023-01-01", periods=2))

        returns = calculate_returns(prices)
        assert len(returns) == 1

    def test_nan_handling_in_returns(self, prices_with_nans):
        """Test that NaNs are handled in returns calculation."""
        returns = calculate_returns(prices_with_nans)
        assert isinstance(returns, pd.DataFrame)
        assert returns.isna().any().any()


# =============================================================================
# Test Mathematical Properties
# =============================================================================


@pytest.mark.unit
class TestReturnsMathematicalProperties:
    """Tests for mathematical properties of returns."""

    def test_log_returns_sum_equals_total_return(self, sample_prices):
        """Test that sum of log returns equals total log return."""
        log_ret = calculate_returns(sample_prices, log_returns=True)

        total_log = np.log(sample_prices.iloc[-1] / sample_prices.iloc[0])
        sum_log = log_ret.sum()

        np.testing.assert_array_almost_equal(total_log.values, sum_log.values, decimal=10)

    def test_simple_returns_product_equals_total(self, sample_prices):
        """Test that product of (1+r) equals price ratio."""
        simple_ret = calculate_returns(sample_prices, log_returns=False)

        for col in sample_prices.columns:
            product = (1 + simple_ret[col]).prod()
            expected = sample_prices[col].iloc[-1] / sample_prices[col].iloc[0]
            assert product == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Test Annualization
# =============================================================================


@pytest.mark.unit
class TestAnnualization:
    """Tests for return annualization."""

    def test_annualized_returns_reasonable(self, sample_returns, periods_annual):
        """Test that annualized returns are reasonable."""
        daily_mean = sample_returns.mean()
        annual_return = daily_mean * periods_annual["daily"]

        assert (annual_return > -1.0).all()
        assert (annual_return < 2.0).all()


# =============================================================================
# Test Data Alignment
# =============================================================================


@pytest.mark.unit
class TestDataAlignment:
    """Tests for data alignment in returns calculations."""

    def test_returns_benchmark_alignment(self, sample_returns, market_returns):
        """Test alignment between returns and benchmark."""
        common_dates = sample_returns.index.intersection(market_returns.index)

        aligned_returns = sample_returns.loc[common_dates]
        aligned_benchmark = market_returns.loc[common_dates]

        assert len(aligned_returns) == len(aligned_benchmark)
        assert aligned_returns.index.equals(aligned_benchmark.index)


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("freq", ["weekly", "monthly", "quarterly", "yearly"])
def test_frequency_conversion_all_frequencies(sample_returns, freq):
    """Test frequency conversion for all supported frequencies."""
    converted = convert_returns_freq(sample_returns, convert_to=freq)

    assert isinstance(converted, pd.DataFrame)
    assert len(converted) < len(sample_returns)
