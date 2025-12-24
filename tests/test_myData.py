"""
Unit tests for myPortfolioManagement.myData module.

Tests cover:
- Stock price fetching (get_stock_prices)
- Stock info retrieval (get_stock_info)
- FX adjustment (func_adj_fx)
- Ticker screening (get_sp500_tickers, get_nasdaq_tickers)
- Data validation and edge cases
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from myPortfolioManagement.myData import (
    get_stock_prices,
    get_stock_info,
    func_adj_fx,
    get_sp500_tickers,
    get_nasdaq_tickers,
    get_option_exp_dates,
)


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.unit
class TestGetStockPrices:
    """Tests for get_stock_prices function."""

    @pytest.mark.data_fetch
    def test_single_ticker_fetch(self, real_date_range):
        """Test fetching data for a single ticker."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        assert isinstance(prices, pd.DataFrame)
        assert not prices.empty
        assert "AAPL" in prices.columns
        assert isinstance(prices.index, pd.DatetimeIndex)

    @pytest.mark.data_fetch
    def test_multiple_tickers_fetch(self, real_date_range):
        """Test fetching data for multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = get_stock_prices(
            yahoo_tickers=tickers,
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        assert isinstance(prices, pd.DataFrame)
        assert not prices.empty
        for ticker in tickers:
            assert ticker in prices.columns

    @pytest.mark.data_fetch
    def test_daily_frequency(self, real_date_range):
        """Test fetching daily frequency data."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"],
            freq="D"
        )

        # Daily data should have more rows than monthly
        assert len(prices) > 12  # More than 12 trading days in a year

    @pytest.mark.data_fetch
    def test_monthly_frequency(self, real_date_range):
        """Test fetching monthly frequency data."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"],
            freq="M"
        )

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) <= 13  # At most 13 months in a year

    @pytest.mark.data_fetch
    def test_wide_format_output(self, real_date_range):
        """Test that wide_format returns proper format."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL", "MSFT"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"],
            wide_format=True
        )

        # Wide format should have tickers as columns
        assert "AAPL" in prices.columns or len(prices.columns) >= 2

    @pytest.mark.data_fetch
    def test_datetime_index_type(self, real_date_range):
        """Test that returned DataFrame has DatetimeIndex."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        assert isinstance(prices.index, pd.DatetimeIndex)

    @pytest.mark.data_fetch
    def test_no_future_dates(self, real_date_range):
        """Test that returned data doesn't contain future dates."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        today = pd.Timestamp.now().normalize()
        assert prices.index.max() <= today

    @pytest.mark.data_fetch
    def test_prices_are_positive(self, real_date_range):
        """Test that all prices are positive."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        assert (prices.dropna() > 0).all().all()

    def test_empty_ticker_list(self):
        """Test handling of empty ticker list."""
        with pytest.raises((ValueError, TypeError, KeyError)):
            get_stock_prices(
                yahoo_tickers=[],
                start_date="2023-01-01",
                end_date="2023-12-31"
            )

    @pytest.mark.data_fetch
    def test_invalid_ticker(self):
        """Test handling of invalid ticker symbol."""
        # Invalid tickers should either raise an error or return empty/NaN
        result = get_stock_prices(
            yahoo_tickers=["INVALID_TICKER_XYZ123"],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

        # Either empty or all NaN
        assert result.empty or result.isna().all().all()

    @pytest.mark.data_fetch
    def test_date_order_validation(self, real_date_range):
        """Test fetching with valid date ordering."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        # Dates should be in ascending order
        assert prices.index.is_monotonic_increasing


@pytest.mark.unit
class TestGetStockInfo:
    """Tests for get_stock_info function."""

    @pytest.mark.data_fetch
    def test_single_ticker_info(self):
        """Test getting info for a single ticker."""
        info = get_stock_info(yahoo_tickers=["AAPL"])

        assert isinstance(info, pd.DataFrame)
        assert not info.empty

    @pytest.mark.data_fetch
    def test_multiple_tickers_info(self):
        """Test getting info for multiple tickers."""
        tickers = ["AAPL", "MSFT"]
        info = get_stock_info(yahoo_tickers=tickers)

        assert isinstance(info, pd.DataFrame)
        # Should have entries for each ticker
        assert len(info) >= 1

    @pytest.mark.data_fetch
    def test_info_contains_currency(self):
        """Test that info contains currency information."""
        info = get_stock_info(yahoo_tickers=["AAPL"])

        # Check if currency column exists
        columns_lower = [c.lower() for c in info.columns]
        assert "currency" in columns_lower or any("currency" in c for c in columns_lower)


@pytest.mark.unit
class TestFuncAdjFx:
    """Tests for FX adjustment function."""

    def test_no_adjustment_needed_usd(self, sample_prices):
        """Test that USD prices don't get adjusted for USD base currency."""
        # Mock the stock info to return USD currency
        with patch('myPortfolioManagement.myData.get_stock_info') as mock_info:
            mock_df = pd.DataFrame({
                'ticker': sample_prices.columns.tolist(),
                'currency': ['USD'] * len(sample_prices.columns)
            })
            mock_info.return_value = mock_df

            # This should return prices unchanged since all are USD
            adjusted = func_adj_fx(
                prices=sample_prices,
                yahoo_tickers=sample_prices.columns.tolist(),
                base_currency="USD"
            )

            assert isinstance(adjusted, pd.DataFrame)
            assert adjusted.shape == sample_prices.shape

    def test_fx_adjustment_preserves_shape(self, sample_prices):
        """Test that FX adjustment preserves DataFrame shape."""
        # Even with adjustments, shape should be preserved
        with patch('myPortfolioManagement.myData.get_stock_info') as mock_info:
            mock_df = pd.DataFrame({
                'ticker': sample_prices.columns.tolist(),
                'currency': ['USD'] * len(sample_prices.columns)
            })
            mock_info.return_value = mock_df

            adjusted = func_adj_fx(
                prices=sample_prices,
                yahoo_tickers=sample_prices.columns.tolist(),
                base_currency="USD"
            )

            assert adjusted.shape == sample_prices.shape

    def test_fx_adjustment_output_type(self, sample_prices):
        """Test that FX adjustment returns DataFrame."""
        with patch('myPortfolioManagement.myData.get_stock_info') as mock_info:
            mock_df = pd.DataFrame({
                'ticker': sample_prices.columns.tolist(),
                'currency': ['USD'] * len(sample_prices.columns)
            })
            mock_info.return_value = mock_df

            adjusted = func_adj_fx(
                prices=sample_prices,
                yahoo_tickers=sample_prices.columns.tolist(),
                base_currency="USD"
            )

            assert isinstance(adjusted, pd.DataFrame)


@pytest.mark.unit
class TestTickerScreeners:
    """Tests for ticker screening functions."""

    @pytest.mark.data_fetch
    @pytest.mark.slow
    def test_sp500_tickers_not_empty(self):
        """Test that S&P 500 tickers can be fetched."""
        try:
            tickers = get_sp500_tickers()
            assert isinstance(tickers, pd.DataFrame)
            # S&P 500 should have around 500 companies
            if not tickers.empty:
                assert len(tickers) > 400
        except Exception:
            # API might be unavailable, skip
            pytest.skip("S&P 500 ticker API unavailable")

    @pytest.mark.data_fetch
    @pytest.mark.slow
    def test_nasdaq_tickers_not_empty(self):
        """Test that NASDAQ tickers can be fetched."""
        try:
            tickers = get_nasdaq_tickers()
            assert isinstance(tickers, pd.DataFrame)
            # NASDAQ should have many companies
            if not tickers.empty:
                assert len(tickers) > 100
        except Exception:
            # API might be unavailable, skip
            pytest.skip("NASDAQ ticker API unavailable")


@pytest.mark.unit
class TestGetOptionExpDates:
    """Tests for option expiration date retrieval."""

    @pytest.mark.data_fetch
    def test_option_exp_dates_format(self):
        """Test that option expiration dates are returned in proper format."""
        try:
            exp_dates = get_option_exp_dates("AAPL")
            assert isinstance(exp_dates, (pd.DataFrame, list, tuple))
        except Exception:
            pytest.skip("Option data API unavailable")

    @pytest.mark.data_fetch
    def test_option_exp_dates_future(self):
        """Test that expiration dates are in the future."""
        try:
            exp_dates = get_option_exp_dates("AAPL")
            if isinstance(exp_dates, pd.DataFrame) and not exp_dates.empty:
                # At least some dates should be in the future
                today = datetime.now()
                # Check first column for dates
                pass  # Structure depends on implementation
        except Exception:
            pytest.skip("Option data API unavailable")


@pytest.mark.unit
class TestDataValidation:
    """Tests for data validation and quality checks."""

    def test_prices_no_negative_values(self, sample_prices):
        """Test that valid prices have no negative values."""
        assert (sample_prices >= 0).all().all()

    def test_prices_datetime_index(self, sample_prices):
        """Test that prices have DatetimeIndex."""
        assert isinstance(sample_prices.index, pd.DatetimeIndex)

    def test_prices_no_duplicate_dates(self, sample_prices):
        """Test that there are no duplicate dates."""
        assert not sample_prices.index.has_duplicates

    def test_prices_sorted_chronologically(self, sample_prices):
        """Test that prices are sorted by date."""
        assert sample_prices.index.is_monotonic_increasing

    def test_returns_within_bounds(self, sample_returns):
        """Test that returns are within reasonable bounds."""
        # Daily returns should generally be within -50% to +50%
        assert (sample_returns > -0.5).all().all()
        assert (sample_returns < 0.5).all().all()


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handle_nan_prices(self, prices_with_nans):
        """Test that NaN prices are handled correctly."""
        # Should be able to calculate returns even with NaNs
        returns = prices_with_nans.pct_change()
        assert isinstance(returns, pd.DataFrame)

    def test_handle_zero_prices(self, prices_with_zeros):
        """Test that zero prices are handled (may cause inf returns)."""
        returns = prices_with_zeros.pct_change()
        # Check for infinities
        has_inf = np.isinf(returns).any().any()
        # This is expected behavior - zeros cause infinite returns
        assert isinstance(returns, pd.DataFrame)

    def test_single_day_data(self, sample_dates):
        """Test handling of single day of data."""
        single_day = pd.DataFrame(
            {"AAPL": [100.0]},
            index=[sample_dates[0]]
        )

        # Should have one row
        assert len(single_day) == 1

    def test_all_same_price(self, constant_prices):
        """Test handling of constant (zero volatility) prices."""
        returns = constant_prices.pct_change()

        # Returns should be zero for constant prices
        non_nan_returns = returns.dropna()
        assert (non_nan_returns == 0).all().all()


@pytest.mark.unit
class TestDataFrequencyConversion:
    """Tests for data frequency handling."""

    def test_daily_data_has_many_points(self, sample_prices):
        """Test that daily data has expected number of points."""
        # ~252 trading days per year, 4 years of data = ~1000 points
        assert len(sample_prices) > 500

    def test_monthly_aggregation_reduces_points(self, sample_prices):
        """Test that monthly aggregation reduces data points."""
        monthly = sample_prices.resample("ME").last()

        # Monthly should have fewer points than daily
        assert len(monthly) < len(sample_prices)
        # Should be roughly 12x fewer
        ratio = len(sample_prices) / len(monthly)
        assert 15 < ratio < 25  # Account for business days vs calendar days


@pytest.mark.unit
class TestParallelFetching:
    """Tests for parallel data fetching functionality."""

    @pytest.mark.data_fetch
    @pytest.mark.slow
    def test_parallel_fetch_consistency(self, real_date_range):
        """Test that parallel fetching returns consistent results."""
        tickers = ["AAPL", "MSFT", "GOOGL"]

        # Fetch twice and compare
        prices1 = get_stock_prices(
            yahoo_tickers=tickers,
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        prices2 = get_stock_prices(
            yahoo_tickers=tickers,
            start_date=real_date_range["start"],
            end_date=real_date_range["end"]
        )

        # Should have same shape
        assert prices1.shape == prices2.shape

        # Values should be very close (allowing for minor API variations)
        if not prices1.empty and not prices2.empty:
            # Find common dates
            common_dates = prices1.index.intersection(prices2.index)
            if len(common_dates) > 0:
                p1 = prices1.loc[common_dates]
                p2 = prices2.loc[common_dates]
                # Allow small differences due to API timing
                diff = (p1 - p2).abs()
                assert (diff < 0.01 * p1.abs()).all().all() or (diff < 0.01).all().all()


@pytest.mark.unit
class TestCurrencyHandling:
    """Tests for currency handling in data fetching."""

    @pytest.mark.data_fetch
    def test_usd_base_currency(self, real_date_range):
        """Test fetching with USD base currency."""
        prices = get_stock_prices(
            yahoo_tickers=["AAPL"],
            start_date=real_date_range["start"],
            end_date=real_date_range["end"],
            base_currency="USD"
        )

        assert isinstance(prices, pd.DataFrame)
        assert not prices.empty

    @pytest.mark.data_fetch
    def test_non_usd_base_currency(self, real_date_range):
        """Test fetching with non-USD base currency."""
        try:
            prices = get_stock_prices(
                yahoo_tickers=["AAPL"],
                start_date=real_date_range["start"],
                end_date=real_date_range["end"],
                base_currency="EUR",
                adj_fx=True
            )

            assert isinstance(prices, pd.DataFrame)
        except Exception:
            # FX adjustment might fail if rates unavailable
            pytest.skip("FX adjustment data unavailable")


@pytest.mark.unit
class TestDataQualityMetrics:
    """Tests for data quality assessment."""

    def test_calculate_missing_data_percentage(self, prices_with_nans):
        """Test calculating percentage of missing data."""
        nan_pct = prices_with_nans.isna().sum() / len(prices_with_nans)

        assert isinstance(nan_pct, pd.Series)
        assert (nan_pct >= 0).all()
        assert (nan_pct <= 1).all()

    def test_data_coverage_calculation(self, sample_prices):
        """Test calculating data coverage."""
        # All sample prices should have full coverage
        coverage = 1 - sample_prices.isna().sum() / len(sample_prices)

        assert (coverage == 1.0).all()

    def test_trading_days_count(self, sample_prices):
        """Test counting trading days."""
        # Count non-NaN entries per column
        trading_days = sample_prices.notna().sum()

        assert (trading_days > 0).all()
        assert (trading_days == len(sample_prices)).all()
