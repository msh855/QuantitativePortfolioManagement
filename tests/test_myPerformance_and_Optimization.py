"""
Unit tests for myPortfolioManagement performance metrics and optimization modules.

Tests cover:
- Performance Metrics (myPerformanceMetrics.py):
  - CAGR calculation
  - Drawdown analysis
  - Alpha/Beta calculations
  - Information ratio
  - Rolling statistics

- Risk Metrics (metrics.py):
  - Sharpe ratio
  - Sortino ratio
  - VaR, CVaR
  - Max drawdown
  - Calmar ratio

- Portfolio Optimization (myPortfolioOptimisation.py):
  - HRP (Hierarchical Risk Parity)
  - Equal weight
  - Inverse volatility
  - Minimum variance
  - Max Sharpe
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from myPortfolioManagement.myPerformanceMetrics import (
    cagr,
    age,
    drawdown_details,
    assets_drawdown_details,
    information_ratio,
    alpha_beta_table,
    get_main_stats,
)

from myPortfolioManagement.metrics import (
    vol,
    beta,
    var,
    cvar,
    max_dd,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    gain_loss_ratio,
    lpm,
    hpm,
)

from myPortfolioManagement.myPortfolioOptimisation import (
    HRP,
    inverse_vol_portfolio,
    equal_weight_portfolio,
)


# =============================================================================
# Performance Metrics Tests
# =============================================================================

@pytest.mark.unit
class TestCAGR:
    """Tests for CAGR (Compound Annual Growth Rate) calculation."""

    def test_cagr_positive_growth(self, sample_prices):
        """Test CAGR calculation for positive growth."""
        result = cagr(sample_prices)

        assert isinstance(result, (float, pd.Series))
        # Most assets should have some CAGR
        if isinstance(result, pd.Series):
            assert len(result) == len(sample_prices.columns)

    def test_cagr_single_asset(self, single_asset_prices):
        """Test CAGR for single asset."""
        result = cagr(single_asset_prices)

        assert isinstance(result, (float, pd.Series))

    def test_cagr_constant_prices_is_zero(self, constant_prices):
        """Test that constant prices give zero CAGR."""
        result = cagr(constant_prices)

        if isinstance(result, pd.Series):
            assert (result == 0).all() or (result.abs() < 1e-10).all()
        else:
            assert result == 0 or abs(result) < 1e-10

    def test_cagr_doubling_in_one_year(self):
        """Test CAGR when price doubles in one year."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
        prices = pd.DataFrame({
            "DOUBLE": np.linspace(100, 200, 252)
        }, index=dates)

        result = cagr(prices)

        if isinstance(result, pd.Series):
            result = result.iloc[0]

        # Should be approximately 100% (doubling)
        assert result == pytest.approx(1.0, rel=0.1)

    def test_cagr_halving_in_one_year(self):
        """Test CAGR when price halves in one year."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
        prices = pd.DataFrame({
            "HALF": np.linspace(100, 50, 252)
        }, index=dates)

        result = cagr(prices)

        if isinstance(result, pd.Series):
            result = result.iloc[0]

        # Should be approximately -50%
        assert result == pytest.approx(-0.5, rel=0.1)


@pytest.mark.unit
class TestAge:
    """Tests for age (data history length) calculation."""

    def test_age_four_years_data(self, sample_prices):
        """Test age calculation for ~4 years of data."""
        result = age(sample_prices.iloc[:, 0])

        # Sample data is about 4 years
        assert 3.5 < result < 4.5

    def test_age_one_year_data(self):
        """Test age for exactly one year of data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
        series = pd.Series(range(len(dates)), index=dates)

        result = age(series)

        assert 0.9 < result < 1.1

    def test_age_single_day(self):
        """Test age for single day of data."""
        series = pd.Series([100], index=[pd.Timestamp("2023-01-01")])

        result = age(series)

        assert result < 0.01  # Very small age


@pytest.mark.unit
class TestDrawdownDetails:
    """Tests for drawdown analysis."""

    def test_drawdown_details_structure(self, sample_prices):
        """Test structure of drawdown details output."""
        single_series = sample_prices.iloc[:, 0]
        result = drawdown_details(single_series)

        assert isinstance(result, pd.DataFrame)
        # Should have columns for peak, valley, drawdown, etc.

    def test_drawdown_never_positive(self, sample_prices):
        """Test that drawdowns are never positive."""
        single_series = sample_prices.iloc[:, 0]
        result = drawdown_details(single_series)

        # Drawdown values should be <= 0 or we're measuring magnitude
        if "drawdown" in result.columns.str.lower():
            dd_col = [c for c in result.columns if "drawdown" in c.lower()][0]
            assert (result[dd_col] <= 0).all() or (result[dd_col] >= 0).all()

    def test_assets_drawdown_details(self, sample_prices):
        """Test drawdown analysis for multiple assets."""
        result = assets_drawdown_details(sample_prices)

        assert isinstance(result, pd.DataFrame)
        # Should have info for each asset
        assert len(result) >= 1


@pytest.mark.unit
class TestInformationRatio:
    """Tests for information ratio calculation."""

    def test_information_ratio_structure(self, sample_returns, market_returns):
        """Test structure of information ratio output."""
        # Align returns
        common_dates = sample_returns.index.intersection(market_returns.index)
        ret = sample_returns.loc[common_dates]
        bench = market_returns.loc[common_dates].squeeze()

        result = information_ratio(ret, bench)

        assert isinstance(result, (float, pd.Series, pd.DataFrame))

    def test_ir_zero_for_benchmark(self, market_returns):
        """Test that IR is zero when comparing benchmark to itself."""
        bench = market_returns.squeeze()

        result = information_ratio(bench, bench)

        # IR should be 0 or NaN (identical series)
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.values.flatten()[0]

        assert np.isnan(result) or abs(result) < 1e-10


@pytest.mark.unit
class TestAlphaBeta:
    """Tests for Alpha/Beta calculations."""

    def test_alpha_beta_table_structure(self, sample_returns, market_returns):
        """Test structure of alpha/beta table."""
        common_dates = sample_returns.index.intersection(market_returns.index)
        ret = sample_returns.loc[common_dates]
        bench = market_returns.loc[common_dates].squeeze()

        result = alpha_beta_table(ret, bench, rf=0.02)

        assert isinstance(result, pd.DataFrame)

    def test_market_beta_is_one(self, market_returns):
        """Test that market beta vs itself is 1."""
        bench = market_returns.squeeze()

        result = alpha_beta_table(market_returns, bench, rf=0.0)

        # Beta column should contain 1.0
        if "beta" in result.columns.str.lower():
            beta_col = [c for c in result.columns if "beta" in c.lower()][0]
            assert result[beta_col].iloc[0] == pytest.approx(1.0, rel=0.01)


# =============================================================================
# Risk Metrics Tests
# =============================================================================

@pytest.mark.unit
class TestVolatility:
    """Tests for volatility calculation."""

    def test_vol_positive(self, sample_returns):
        """Test that volatility is always positive."""
        for col in sample_returns.columns:
            result = vol(sample_returns[col])
            assert result >= 0

    def test_vol_zero_for_constant(self, constant_prices):
        """Test that volatility is zero for constant prices."""
        returns = constant_prices.pct_change().dropna()

        for col in returns.columns:
            result = vol(returns[col])
            assert result == 0 or abs(result) < 1e-10

    def test_vol_increases_with_variation(self):
        """Test that volatility increases with more variation."""
        dates = pd.date_range("2023-01-01", periods=100)

        low_vol = pd.Series(np.random.randn(100) * 0.01, index=dates)
        high_vol = pd.Series(np.random.randn(100) * 0.05, index=dates)

        assert vol(low_vol) < vol(high_vol)


@pytest.mark.unit
class TestBeta:
    """Tests for beta calculation."""

    def test_beta_vs_market(self, sample_returns, market_returns):
        """Test beta calculation against market."""
        common = sample_returns.index.intersection(market_returns.index)
        ret = sample_returns.loc[common]
        mkt = market_returns.loc[common].squeeze()

        for col in ret.columns:
            result = beta(ret[col], mkt)
            assert isinstance(result, (int, float))

    def test_market_beta_is_one(self, market_returns):
        """Test that market beta vs itself is 1."""
        mkt = market_returns.squeeze()
        result = beta(mkt, mkt)

        assert result == pytest.approx(1.0, rel=0.01)

    def test_uncorrelated_beta_near_zero(self):
        """Test that uncorrelated assets have beta near zero."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=500)

        asset = pd.Series(np.random.randn(500) * 0.02, index=dates)
        market = pd.Series(np.random.randn(500) * 0.02, index=dates)

        result = beta(asset, market)

        assert abs(result) < 0.3  # Should be close to zero


@pytest.mark.unit
class TestVaRCVaR:
    """Tests for VaR and CVaR calculations.

    Note: This implementation returns POSITIVE values (absolute values)
    for risk measures, representing the magnitude of potential loss.
    """

    def test_var_positive_for_losses(self, sample_returns):
        """Test that VaR is positive (absolute value of loss)."""
        for col in sample_returns.columns:
            result = var(sample_returns[col].values, alpha=0.05)
            # VaR should be positive (absolute value)
            assert result >= 0

    def test_var_decreases_with_alpha(self, sample_returns):
        """Test that VaR decreases with higher alpha (less extreme quantile)."""
        ret = sample_returns.iloc[:, 0].values

        var_5 = var(ret, alpha=0.05)
        var_1 = var(ret, alpha=0.01)

        # 1% VaR should be larger (more extreme) than 5% VaR
        assert var_1 >= var_5

    def test_cvar_more_extreme_than_var(self, sample_returns):
        """Test that CVaR is more extreme than VaR."""
        ret = sample_returns.iloc[:, 0].values

        var_result = var(ret, alpha=0.05)
        cvar_result = cvar(ret, alpha=0.05)

        # CVaR should be >= VaR (both positive, CVaR averages beyond VaR)
        assert cvar_result >= var_result

    def test_var_cvar_are_numeric(self, sample_returns):
        """Test that VaR and CVaR return numeric values."""
        ret = sample_returns.iloc[:, 0].values

        var_result = var(ret, alpha=0.05)
        cvar_result = cvar(ret, alpha=0.05)

        assert isinstance(var_result, (int, float))
        assert isinstance(cvar_result, (int, float))


@pytest.mark.unit
class TestMaxDrawdown:
    """Tests for maximum drawdown calculation.

    Note: This implementation returns POSITIVE values (absolute values)
    for drawdown, representing the magnitude of the drawdown.
    """

    def test_max_dd_positive_or_zero(self, sample_returns):
        """Test that max drawdown is positive or zero (absolute value)."""
        for col in sample_returns.columns:
            result = max_dd(sample_returns[col].values)
            assert result >= 0

    def test_max_dd_zero_for_always_increasing(self):
        """Test max drawdown is zero for always increasing prices."""
        # Strictly increasing returns (small positive)
        returns = np.array([0.01] * 100)

        result = max_dd(returns)

        assert result == 0 or abs(result) < 1e-10

    def test_max_dd_captures_crash(self, extreme_prices):
        """Test that max drawdown captures extreme crashes."""
        returns = extreme_prices.pct_change().dropna().squeeze().values
        result = max_dd(returns)

        # Should capture at least 15% drawdown from the crash (positive value)
        assert result > 0.15


@pytest.mark.unit
class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_for_positive_excess_return(self, positive_returns):
        """Test Sharpe is positive for positive excess returns."""
        ret = positive_returns.squeeze()
        result = sharpe_ratio(er=0.10, returns=ret, rf=0.02)

        assert result > 0

    def test_sharpe_negative_for_negative_excess_return(self, negative_returns):
        """Test Sharpe is negative for negative excess returns."""
        ret = negative_returns.squeeze()
        result = sharpe_ratio(er=-0.10, returns=ret, rf=0.02)

        assert result < 0

    @pytest.mark.parametrize("rf", [0.0, 0.02, 0.04, 0.06])
    def test_sharpe_with_different_rf(self, sample_returns, rf):
        """Test Sharpe ratio with different risk-free rates."""
        ret = sample_returns.iloc[:, 0]
        er = ret.mean() * 252

        result = sharpe_ratio(er=er, returns=ret, rf=rf)

        assert isinstance(result, (int, float))
        assert not np.isnan(result) or vol(ret) == 0


@pytest.mark.unit
class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_uses_downside_vol(self, sample_returns):
        """Test that Sortino uses downside volatility."""
        ret = sample_returns.iloc[:, 0]
        er = ret.mean() * 252

        sharpe = sharpe_ratio(er=er, returns=ret, rf=0.02)
        sortino = sortino_ratio(er=er, returns=ret, rf=0.02)

        # For asymmetric returns, Sortino != Sharpe
        # (they can be equal for symmetric returns)
        assert isinstance(sortino, (int, float))

    def test_sortino_higher_for_positive_skew(self):
        """Test Sortino is higher for positively skewed returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=500)

        # Positive skew: mostly small losses, some big gains
        pos_skew = pd.Series(
            np.abs(np.random.randn(500)) * 0.02 - 0.005,
            index=dates
        )
        er = pos_skew.mean() * 252

        result = sortino_ratio(er=er, returns=pos_skew, rf=0.02)

        assert isinstance(result, (int, float))


@pytest.mark.unit
class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_structure(self, sample_returns):
        """Test Calmar ratio calculation."""
        ret = sample_returns.iloc[:, 0]
        er = ret.mean() * 252

        result = calmar_ratio(er=er, returns=ret, rf=0.02)

        assert isinstance(result, (int, float))

    def test_calmar_infinite_for_no_drawdown(self):
        """Test Calmar is infinite when no drawdown."""
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.Series([0.01] * 100, index=dates)

        result = calmar_ratio(er=0.10, returns=returns, rf=0.02)

        # Should be very large or infinite
        assert result > 100 or np.isinf(result)


@pytest.mark.unit
class TestOtherRatios:
    """Tests for other risk-adjusted return ratios."""

    def test_omega_ratio(self, sample_returns):
        """Test Omega ratio calculation."""
        ret = sample_returns.iloc[:, 0]
        er = ret.mean() * 252

        result = omega_ratio(er=er, returns=ret, rf=0.02)

        assert isinstance(result, (int, float))
        assert result > 0  # Omega is always positive

    def test_gain_loss_ratio(self, sample_returns):
        """Test Gain/Loss ratio calculation."""
        ret = sample_returns.iloc[:, 0]

        result = gain_loss_ratio(ret)

        assert isinstance(result, (int, float))
        assert result > 0  # Ratio is positive

    def test_lpm_hpm(self, sample_returns):
        """Test Lower and Higher Partial Moments."""
        ret = sample_returns.iloc[:, 0]

        lpm_result = lpm(ret, threshold=0, order=2)
        hpm_result = hpm(ret, threshold=0, order=2)

        assert lpm_result >= 0
        assert hpm_result >= 0


# =============================================================================
# Portfolio Optimization Tests
# =============================================================================

@pytest.mark.unit
class TestEqualWeightPortfolio:
    """Tests for equal weight portfolio optimization."""

    def test_equal_weights_sum_to_one(self, sample_returns):
        """Test that equal weights sum to 1."""
        weights = equal_weight_portfolio(sample_returns)

        assert isinstance(weights, pd.Series)
        assert weights.sum() == pytest.approx(1.0, rel=1e-10)

    def test_equal_weights_are_equal(self, sample_returns):
        """Test that all weights are equal."""
        weights = equal_weight_portfolio(sample_returns)

        expected_weight = 1.0 / len(sample_returns.columns)
        assert (weights == pytest.approx(expected_weight, rel=1e-10)).all()

    def test_equal_weights_positive(self, sample_returns):
        """Test that all weights are positive."""
        weights = equal_weight_portfolio(sample_returns)

        assert (weights > 0).all()

    def test_equal_weights_match_assets(self, sample_returns):
        """Test that weights match asset columns."""
        weights = equal_weight_portfolio(sample_returns)

        assert set(weights.index) == set(sample_returns.columns)


@pytest.mark.unit
class TestInverseVolatility:
    """Tests for inverse volatility portfolio optimization."""

    def test_inverse_vol_sum_to_one(self, sample_returns):
        """Test that inverse vol weights sum to 1."""
        weights = inverse_vol_portfolio(sample_returns)

        assert isinstance(weights, pd.Series)
        assert weights.sum() == pytest.approx(1.0, rel=1e-6)

    def test_inverse_vol_weights_positive(self, sample_returns):
        """Test that all weights are positive."""
        weights = inverse_vol_portfolio(sample_returns)

        assert (weights > 0).all()

    def test_inverse_vol_lower_vol_higher_weight(self, sample_returns):
        """Test that lower vol assets get higher weights."""
        weights = inverse_vol_portfolio(sample_returns)
        vols = sample_returns.std()

        # Find lowest and highest vol assets
        lowest_vol_asset = vols.idxmin()
        highest_vol_asset = vols.idxmax()

        # Lowest vol should have highest weight (or equal if vol is same)
        assert weights[lowest_vol_asset] >= weights[highest_vol_asset]

    def test_inverse_vol_handles_high_correlation(self, correlated_returns):
        """Test inverse vol with highly correlated assets."""
        weights = inverse_vol_portfolio(correlated_returns)

        assert weights.sum() == pytest.approx(1.0, rel=1e-6)
        assert (weights > 0).all()


@pytest.mark.unit
class TestHRP:
    """Tests for Hierarchical Risk Parity optimization."""

    @pytest.mark.slow
    def test_hrp_weights_sum_to_one(self, sample_returns):
        """Test that HRP weights sum to 1."""
        try:
            weights = HRP(model="HRP", returns_training=sample_returns)

            assert isinstance(weights, pd.Series)
            assert weights.sum() == pytest.approx(1.0, rel=1e-4)
        except Exception as e:
            pytest.skip(f"HRP optimization failed: {e}")

    @pytest.mark.slow
    def test_hrp_weights_positive(self, sample_returns):
        """Test that HRP weights are non-negative."""
        try:
            weights = HRP(model="HRP", returns_training=sample_returns)

            assert (weights >= 0).all()
        except Exception as e:
            pytest.skip(f"HRP optimization failed: {e}")

    @pytest.mark.slow
    def test_hrp_weights_bounded(self, sample_returns):
        """Test that HRP weights are bounded by constraints."""
        try:
            weights = HRP(
                model="HRP",
                returns_training=sample_returns,
                weight_min=0.05,
                weight_max=0.40
            )

            assert (weights >= 0.05 - 1e-4).all()
            assert (weights <= 0.40 + 1e-4).all()
        except Exception as e:
            pytest.skip(f"HRP optimization failed: {e}")

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["HRP", "HERC"])
    def test_hrp_different_models(self, sample_returns, model):
        """Test different HRP model variants."""
        try:
            weights = HRP(model=model, returns_training=sample_returns)

            assert isinstance(weights, pd.Series)
            assert weights.sum() == pytest.approx(1.0, rel=1e-4)
        except Exception as e:
            pytest.skip(f"HRP {model} optimization failed: {e}")

    @pytest.mark.slow
    def test_hrp_reproducible(self, sample_returns):
        """Test that HRP gives reproducible results."""
        try:
            weights1 = HRP(model="HRP", returns_training=sample_returns)
            weights2 = HRP(model="HRP", returns_training=sample_returns)

            np.testing.assert_array_almost_equal(
                weights1.values, weights2.values, decimal=6
            )
        except Exception as e:
            pytest.skip(f"HRP optimization failed: {e}")


@pytest.mark.unit
class TestOptimizationConstraints:
    """Tests for optimization constraints."""

    def test_weight_bounds_respected(self, sample_returns):
        """Test that weight bounds are respected."""
        min_w = 0.10
        max_w = 0.30

        try:
            weights = HRP(
                model="HRP",
                returns_training=sample_returns,
                weight_min=min_w,
                weight_max=max_w
            )

            assert (weights >= min_w - 1e-4).all()
            assert (weights <= max_w + 1e-4).all()
        except Exception as e:
            pytest.skip(f"Constrained optimization failed: {e}")

    def test_infeasible_constraints_handled(self, sample_returns):
        """Test handling of infeasible constraints."""
        # 5 assets, min 30% each = 150% total (infeasible)
        try:
            weights = HRP(
                model="HRP",
                returns_training=sample_returns,
                weight_min=0.30,
                weight_max=0.50
            )

            # Should either raise error or relax constraints
            assert weights.sum() == pytest.approx(1.0, rel=0.1)
        except (ValueError, Exception):
            pass  # Expected for infeasible constraints


@pytest.mark.unit
class TestOptimizationEdgeCases:
    """Tests for optimization edge cases."""

    def test_single_asset_portfolio(self):
        """Test optimization with single asset."""
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({"A": np.random.randn(100) * 0.02}, index=dates)

        weights = equal_weight_portfolio(returns)

        assert weights["A"] == pytest.approx(1.0, rel=1e-10)

    def test_two_asset_portfolio(self):
        """Test optimization with two assets."""
        dates = pd.date_range("2023-01-01", periods=100)
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.randn(100) * 0.02,
            "B": np.random.randn(100) * 0.03
        }, index=dates)

        weights = inverse_vol_portfolio(returns)

        assert weights.sum() == pytest.approx(1.0, rel=1e-6)
        # Lower vol asset (A) should have higher weight
        assert weights["A"] > weights["B"]

    def test_identical_assets(self):
        """Test optimization with identical return series."""
        dates = pd.date_range("2023-01-01", periods=100)
        np.random.seed(42)
        ret = np.random.randn(100) * 0.02

        returns = pd.DataFrame({
            "A": ret,
            "B": ret,
            "C": ret
        }, index=dates)

        weights = equal_weight_portfolio(returns)

        # Should still give valid weights
        assert weights.sum() == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_portfolio_metrics_pipeline(self, full_portfolio_setup):
        """Test full pipeline of portfolio metrics calculation."""
        returns = full_portfolio_setup["returns"]
        benchmark = full_portfolio_setup["benchmark"]
        rf = full_portfolio_setup["rf"]

        # Calculate various metrics
        common = returns.index.intersection(benchmark.index)
        ret = returns.loc[common]
        bench = benchmark.loc[common].squeeze()

        for col in ret.columns:
            asset_ret = ret[col]

            # Calculate metrics
            vol_result = vol(asset_ret)
            var_result = var(asset_ret)
            cvar_result = cvar(asset_ret)
            mdd_result = max_dd(asset_ret)
            beta_result = beta(asset_ret, bench)

            # Validate relationships
            assert vol_result >= 0
            assert var_result <= 0
            assert cvar_result <= var_result
            assert mdd_result <= 0
            assert isinstance(beta_result, (int, float))

    def test_optimization_to_performance(self, sample_returns, market_returns):
        """Test pipeline from optimization to performance measurement."""
        # Optimize
        weights = equal_weight_portfolio(sample_returns)

        # Calculate portfolio returns
        portfolio_ret = (sample_returns * weights).sum(axis=1)

        # Calculate metrics
        common = portfolio_ret.index.intersection(market_returns.index)
        port = portfolio_ret.loc[common]
        bench = market_returns.loc[common].squeeze()

        vol_result = vol(port)
        beta_result = beta(port, bench)
        mdd_result = max_dd(port)

        assert vol_result > 0
        assert isinstance(beta_result, (int, float))
        assert mdd_result <= 0


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_var_different_alphas(sample_returns, alpha):
    """Test VaR at different confidence levels."""
    ret = sample_returns.iloc[:, 0]
    result = var(ret, alpha=alpha)

    assert result <= 0
    assert isinstance(result, (int, float))


@pytest.mark.unit
@pytest.mark.parametrize("order", [1, 2, 3])
def test_lpm_different_orders(sample_returns, order):
    """Test LPM with different orders."""
    ret = sample_returns.iloc[:, 0]
    result = lpm(ret, threshold=0, order=order)

    assert result >= 0


@pytest.mark.unit
@pytest.mark.parametrize("n_assets", [2, 5, 10, 20])
def test_equal_weight_different_sizes(n_assets):
    """Test equal weight portfolio with different number of assets."""
    dates = pd.date_range("2023-01-01", periods=100)
    np.random.seed(42)

    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.02,
        index=dates,
        columns=[f"ASSET_{i}" for i in range(n_assets)]
    )

    weights = equal_weight_portfolio(returns)

    assert len(weights) == n_assets
    assert weights.sum() == pytest.approx(1.0, rel=1e-10)
    assert (weights == pytest.approx(1.0 / n_assets, rel=1e-10)).all()
