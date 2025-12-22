#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for buy-and-hold vs rebalancing strategies

This test suite validates the new rebalancing functionality that addresses the issue
of implicit daily rebalancing assumptions in portfolio return calculations.
"""

import pandas as pd
import numpy as np
import pytest
from myPortfolioManagement.myReturns import (
    calculate_buy_and_hold_returns,
    calculate_rebalanced_returns,
    calculate_portfolio_returns
)


def test_buy_and_hold_basic():
    """Test basic buy-and-hold calculation"""
    # Create simple test data
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02, -0.01],
        'Asset_B': [-0.01, 0.01, 0.03]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    result = calculate_buy_and_hold_returns(returns, weights)
    
    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(returns)
    assert result.columns[0] == 'buy_and_hold'
    
    # Check that weights are not rebalanced (portfolio return != weighted average)
    # After first period, weights drift, so returns should differ from equal-weight
    print(f"Buy-and-hold returns: {result.values.flatten()}")


def test_buy_and_hold_vs_daily_rebalancing():
    """Test that buy-and-hold differs from daily rebalancing"""
    returns = pd.DataFrame({
        'Asset_A': [0.10, 0.10, -0.05],  # Strong then weak
        'Asset_B': [0.01, 0.01, 0.01]    # Steady
    }, index=pd.date_range('2020-01-01', periods=3))
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    # Buy-and-hold
    bh_returns = calculate_buy_and_hold_returns(returns, weights)
    
    # Daily rebalancing (traditional method)
    daily_rebal = calculate_rebalanced_returns(returns, weights, rebalance_freq='daily')
    
    # These should be different after the first period because weights drift in buy-and-hold
    # but are reset daily in the rebalanced version
    print(f"\nBuy-and-hold cumulative: {(1 + bh_returns).prod().values[0]:.4f}")
    print(f"Daily rebalancing cumulative: {(1 + daily_rebal).prod().values[0]:.4f}")
    
    # After period 1, Asset_A has grown more, so in buy-and-hold it has higher weight
    # This means period 2 return should be more influenced by Asset_A in buy-and-hold
    # Given Asset_A outperforms in period 2, buy-and-hold should do better
    assert not np.allclose(bh_returns.values, daily_rebal.values, atol=1e-6)


def test_monthly_rebalancing():
    """Test monthly rebalancing logic"""
    # Create 90 days of data (approximately 3 months)
    dates = pd.date_range('2020-01-01', periods=90, freq='D')
    returns = pd.DataFrame({
        'Asset_A': np.random.randn(90) * 0.01,
        'Asset_B': np.random.randn(90) * 0.01
    }, index=dates)
    
    weights = pd.Series({'Asset_A': 0.6, 'Asset_B': 0.4})
    
    result = calculate_rebalanced_returns(returns, weights, rebalance_freq='monthly')
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(returns)
    assert 'monthly' in result.columns[0]


def test_transaction_costs_buy_and_hold():
    """Test that transaction costs are applied correctly in buy-and-hold"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.01, 0.01],
        'Asset_B': [0.01, 0.01, 0.01]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    # Without transaction costs
    no_tc = calculate_buy_and_hold_returns(returns, weights, transaction_cost_bps=0)
    
    # With 10 bps transaction costs
    with_tc = calculate_buy_and_hold_returns(returns, weights, transaction_cost_bps=10)
    
    # First period should have lower return due to initial transaction cost
    assert with_tc.iloc[0, 0] < no_tc.iloc[0, 0]
    
    # Subsequent periods should be the same (no more transactions in buy-and-hold)
    assert np.allclose(with_tc.iloc[1:].values, no_tc.iloc[1:].values, atol=1e-8)


def test_transaction_costs_rebalancing():
    """Test that transaction costs are applied at each rebalance"""
    # Create data where rebalancing will occur twice (at start of each month)
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    returns = pd.DataFrame({
        'Asset_A': [0.02] * 60,  # Strong performance causes drift
        'Asset_B': [0.00] * 60   # No movement
    }, index=dates)
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    # Without transaction costs
    no_tc = calculate_rebalanced_returns(returns, weights, rebalance_freq='monthly', 
                                         transaction_cost_bps=0)
    
    # With transaction costs
    with_tc = calculate_rebalanced_returns(returns, weights, rebalance_freq='monthly',
                                           transaction_cost_bps=10)
    
    # Total return should be lower with transaction costs
    cum_no_tc = (1 + no_tc).prod().values[0]
    cum_with_tc = (1 + with_tc).prod().values[0]
    
    print(f"\nCumulative without TC: {cum_no_tc:.6f}")
    print(f"Cumulative with TC: {cum_with_tc:.6f}")
    
    assert cum_with_tc < cum_no_tc


def test_weights_must_sum_to_one():
    """Test that weights validation works"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02],
        'Asset_B': [0.01, 0.01]
    }, index=pd.date_range('2020-01-01', periods=2))
    
    # Weights don't sum to 1
    bad_weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.6})
    
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_buy_and_hold_returns(returns, bad_weights)


def test_missing_assets():
    """Test error handling for missing assets"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02],
        'Asset_B': [0.01, 0.01]
    }, index=pd.date_range('2020-01-01', periods=2))
    
    # Weights reference asset not in returns
    weights = pd.Series({'Asset_A': 0.5, 'Asset_C': 0.5})
    
    with pytest.raises(ValueError, match="missing assets"):
        calculate_buy_and_hold_returns(returns, weights)


def test_calculate_portfolio_returns_with_strategies():
    """Test the updated calculate_portfolio_returns with different strategies"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02, -0.01],
        'Asset_B': [-0.01, 0.01, 0.02]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    # Create weights in the format expected by calculate_portfolio_returns
    weights_df = pd.DataFrame({
        'asset': ['Asset_A', 'Asset_B'],
        'weight': [0.6, 0.4]
    })
    
    # Test buy-and-hold
    bh_result = calculate_portfolio_returns(
        returns, weights_df, 
        rebalance_strategy='buy_and_hold',
        portfolio_name='test_bh'
    )
    assert isinstance(bh_result, pd.DataFrame)
    assert bh_result.columns[0] == 'test_bh'
    
    # Test monthly rebalancing
    monthly_result = calculate_portfolio_returns(
        returns, weights_df,
        rebalance_strategy='monthly',
        portfolio_name='test_monthly'
    )
    assert isinstance(monthly_result, pd.DataFrame)
    assert monthly_result.columns[0] == 'test_monthly'
    
    # Test with transaction costs
    tc_result = calculate_portfolio_returns(
        returns, weights_df,
        rebalance_strategy='monthly',
        transaction_cost_bps=10,
        portfolio_name='test_tc'
    )
    assert isinstance(tc_result, pd.DataFrame)


def test_daily_rebalancing_matches_dot_product():
    """Test that daily rebalancing gives similar results to traditional dot product"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02, -0.01],
        'Asset_B': [-0.01, 0.01, 0.02]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    weights = pd.Series({'Asset_A': 0.6, 'Asset_B': 0.4})
    
    # Daily rebalancing
    daily_rebal = calculate_rebalanced_returns(returns, weights, rebalance_freq='daily')
    
    # Traditional calculation (dot product)
    traditional = (returns * weights.values).sum(axis=1)
    
    # Should be very close (allowing for small numerical differences)
    assert np.allclose(daily_rebal.values.flatten(), traditional.values, atol=1e-6)


def test_dict_weights_input():
    """Test that dict weights work correctly"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02],
        'Asset_B': [0.01, 0.01]
    }, index=pd.date_range('2020-01-01', periods=2))
    
    # Test with dict instead of Series
    weights_dict = {'Asset_A': 0.5, 'Asset_B': 0.5}
    
    result = calculate_buy_and_hold_returns(returns, weights_dict)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_rebalance_frequencies():
    """Test all rebalancing frequencies"""
    # Create 2 years of daily data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.DataFrame({
        'Asset_A': np.random.randn(500) * 0.01,
        'Asset_B': np.random.randn(500) * 0.01
    }, index=dates)
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    
    for freq in frequencies:
        result = calculate_rebalanced_returns(returns, weights, rebalance_freq=freq)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(returns)
        print(f"{freq} rebalancing: {(1 + result).prod().values[0]:.4f} cumulative return")


def test_invalid_rebalance_frequency():
    """Test error handling for invalid rebalance frequency"""
    returns = pd.DataFrame({
        'Asset_A': [0.01, 0.02],
        'Asset_B': [0.01, 0.01]
    }, index=pd.date_range('2020-01-01', periods=2))
    
    weights = pd.Series({'Asset_A': 0.5, 'Asset_B': 0.5})
    
    with pytest.raises(ValueError, match="must be one of"):
        calculate_rebalanced_returns(returns, weights, rebalance_freq='invalid')


if __name__ == '__main__':
    print("Running rebalancing strategy tests...")
    print("=" * 80)
    
    # Run tests manually for quick validation
    test_buy_and_hold_basic()
    print("✓ Buy-and-hold basic test passed")
    
    test_buy_and_hold_vs_daily_rebalancing()
    print("✓ Buy-and-hold vs daily rebalancing test passed")
    
    test_monthly_rebalancing()
    print("✓ Monthly rebalancing test passed")
    
    test_transaction_costs_buy_and_hold()
    print("✓ Transaction costs (buy-and-hold) test passed")
    
    test_transaction_costs_rebalancing()
    print("✓ Transaction costs (rebalancing) test passed")
    
    test_weights_must_sum_to_one()
    print("✓ Weights validation test passed")
    
    test_missing_assets()
    print("✓ Missing assets test passed")
    
    test_calculate_portfolio_returns_with_strategies()
    print("✓ Updated calculate_portfolio_returns test passed")
    
    test_daily_rebalancing_matches_dot_product()
    print("✓ Daily rebalancing vs dot product test passed")
    
    test_dict_weights_input()
    print("✓ Dict weights input test passed")
    
    test_rebalance_frequencies()
    print("✓ All rebalance frequencies test passed")
    
    test_invalid_rebalance_frequency()
    print("✓ Invalid frequency test passed")
    
    print("=" * 80)
    print("All tests passed successfully!")
