"""
Test Suite for Option Pricing and Implied Distribution Modules

This module tests the functionality of myOptionPricing.py and myImpliedDistribution.py

Author: MyPortfolioManagement Team
Date: December 2024
"""

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from myPortfolioManagement.myOptionPricing import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility,
    option_delta,
    option_gamma,
    option_vega,
    create_option_chain
)
from myPortfolioManagement.myImpliedDistribution import (
    breeden_litzenberger_density,
    extract_implied_distribution,
    compare_distributions,
    find_mispricing_opportunities
)


class TestBlackScholesModel:
    """Test Black-Scholes option pricing."""
    
    def test_call_price_at_the_money(self):
        """Test call option price at-the-money."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        call_price = black_scholes_call(S, K, T, r, sigma)
        
        # At-the-money call should be worth something
        assert call_price > 0
        assert call_price < S  # Can't be worth more than stock
        
        # Approximate expected value (should be around 10.45)
        assert 9 < call_price < 12
    
    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)"""
        S = 100
        K = 95
        T = 1.0
        r = 0.05
        sigma = 0.25
        
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        # Put-call parity should hold
        assert np.abs(lhs - rhs) < 0.01
    
    def test_option_bounds(self):
        """Test option price bounds."""
        S = 100
        K = 90
        T = 0.5
        r = 0.03
        sigma = 0.3
        
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        # Call must be >= intrinsic value
        assert call_price >= max(S - K, 0)
        
        # Put must be >= intrinsic value
        assert put_price >= max(K - S, 0)
        
        # Call can't exceed stock price
        assert call_price <= S
        
        # Put can't exceed strike price
        assert put_price <= K
    
    def test_zero_time_to_expiration(self):
        """Test option values at expiration."""
        S = 105
        K = 100
        T = 0
        r = 0.05
        sigma = 0.2
        
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        # At expiration, option equals intrinsic value
        assert np.abs(call_price - max(S - K, 0)) < 0.01
        assert np.abs(put_price - max(K - S, 0)) < 0.01
    
    def test_deep_in_the_money_call(self):
        """Test deep ITM call behaves like stock."""
        S = 150
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        call_price = black_scholes_call(S, K, T, r, sigma)
        
        # Deep ITM call should be close to intrinsic value
        intrinsic = S - K * np.exp(-r * T)
        assert call_price > intrinsic - 1
        assert call_price > 45  # Should be substantial


class TestImpliedVolatility:
    """Test implied volatility calculations."""
    
    def test_iv_recovery(self):
        """Test that we can recover the volatility used to price an option."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma_true = 0.25
        
        # Price the option
        call_price = black_scholes_call(S, K, T, r, sigma_true)
        
        # Recover implied volatility
        sigma_implied = implied_volatility(call_price, S, K, T, r, option_type='call')
        
        # Should recover the original volatility
        assert np.abs(sigma_implied - sigma_true) < 0.001
    
    def test_iv_put_option(self):
        """Test implied volatility for put options."""
        S = 100
        K = 95
        T = 0.5
        r = 0.04
        sigma_true = 0.3
        
        put_price = black_scholes_put(S, K, T, r, sigma_true)
        sigma_implied = implied_volatility(put_price, S, K, T, r, option_type='put')
        
        assert np.abs(sigma_implied - sigma_true) < 0.001
    
    def test_iv_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        
        # Negative option price
        try:
            implied_volatility(-1, S, K, T, r)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Zero time to expiration
        try:
            implied_volatility(10, S, K, 0, r)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestGreeks:
    """Test option Greeks calculations."""
    
    def test_call_delta_range(self):
        """Test that call delta is between 0 and 1."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = option_delta(S, K, T, r, sigma, option_type='call')
        
        assert 0 <= delta <= 1
        # ATM call delta should be around 0.5-0.6
        assert 0.4 < delta < 0.7
    
    def test_put_delta_range(self):
        """Test that put delta is between -1 and 0."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = option_delta(S, K, T, r, sigma, option_type='put')
        
        assert -1 <= delta <= 0
        # ATM put delta should be around -0.5 to -0.4
        assert -0.6 < delta < -0.3
    
    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        test_cases = [
            (100, 100, 1.0),  # ATM
            (100, 90, 1.0),   # OTM
            (100, 110, 1.0),  # ITM
        ]
        
        r = 0.05
        sigma = 0.25
        
        for S, K, T in test_cases:
            gamma = option_gamma(S, K, T, r, sigma)
            assert gamma >= 0
    
    def test_vega_positive(self):
        """Test that vega is always positive."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        vega = option_vega(S, K, T, r, sigma)
        
        assert vega > 0


class TestOptionChain:
    """Test synthetic option chain creation."""
    
    def test_create_option_chain_structure(self):
        """Test that created option chain has correct structure."""
        S = 100
        T = 1.0
        r = 0.05
        sigma = 0.25
        
        chain = create_option_chain(S, T, r, sigma, num_strikes=10)
        
        # Check structure
        assert isinstance(chain, pd.DataFrame)
        assert len(chain) == 10
        assert all(col in chain.columns for col in ['strike', 'call_price', 'put_price'])
    
    def test_option_chain_prices_valid(self):
        """Test that option chain prices are valid."""
        S = 100
        T = 1.0
        r = 0.05
        sigma = 0.25
        
        chain = create_option_chain(S, T, r, sigma, strike_range=(0.8, 1.2))
        
        # All prices should be non-negative
        assert (chain['call_price'] >= 0).all()
        assert (chain['put_price'] >= 0).all()
        
        # Strikes should be sorted
        assert (chain['strike'].diff().dropna() > 0).all()


class TestBreedenLitzenberger:
    """Test Breeden-Litzenberger density extraction."""
    
    def test_density_extraction_lognormal(self):
        """Test density extraction on lognormal prices."""
        # Create synthetic option chain with known distribution
        S = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        strikes = np.linspace(70, 130, 30)
        call_prices = np.array([black_scholes_call(S, K, T, r, sigma) for K in strikes])
        
        strikes_dense, density = breeden_litzenberger_density(strikes, call_prices, r, T)
        
        # Density should be non-negative
        assert (density >= 0).all()
        
        # Density should integrate to approximately 1
        total_prob = simpson(density, x=strikes_dense)
        assert 0.8 < total_prob < 1.2  # Allow some numerical error
    
    def test_density_extraction_requires_minimum_strikes(self):
        """Test that density extraction needs at least 3 strikes."""
        strikes = np.array([100, 110])
        call_prices = np.array([10, 5])
        
        try:
            breeden_litzenberger_density(strikes, call_prices, 0.05, 1.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestImpliedDistribution:
    """Test implied distribution extraction from option chain."""
    
    def test_extract_implied_distribution(self):
        """Test extracting implied distribution from option chain."""
        S = 100
        T = 2.0
        r = 0.05
        sigma = 0.25
        
        # Create synthetic option chain
        chain = create_option_chain(S, T, r, sigma, num_strikes=25)
        
        # Extract distribution
        implied_dist = extract_implied_distribution(chain, S, r, T, option_type='call')
        
        # Check structure
        assert isinstance(implied_dist, pd.DataFrame)
        assert 'price_level' in implied_dist.columns
        assert 'probability_density' in implied_dist.columns
        
        # Check properties
        assert len(implied_dist) > 0
        assert (implied_dist['probability_density'] >= 0).all()


class TestDistributionComparison:
    """Test distribution comparison functionality."""
    
    def test_compare_distributions_structure(self):
        """Test that comparison returns correct structure."""
        # Create mock distributions
        implied_dist = pd.DataFrame({
            'price_level': np.linspace(80, 120, 100),
            'probability_density': np.random.rand(100)
        })
        
        bootstrap_dist = pd.DataFrame({
            'future_price': np.random.normal(100, 15, 1000)
        })
        
        comparison = compare_distributions(implied_dist, bootstrap_dist, S0=100)
        
        # Check structure
        assert isinstance(comparison, pd.DataFrame)
        assert 'metric' in comparison.columns
        assert 'implied' in comparison.columns
        assert 'bootstrap' in comparison.columns
        assert 'difference' in comparison.columns
        
        # Should have mean, std, and quantiles
        assert 'mean' in comparison['metric'].values
        assert 'std' in comparison['metric'].values


class TestMispricingDetection:
    """Test mispricing opportunity detection."""
    
    def test_find_mispricing_opportunities(self):
        """Test mispricing detection."""
        # Create mock comparison with significant differences
        comparison = pd.DataFrame({
            'metric': ['mean', 'std', 'quantile_50'],
            'implied': [110, 20, 108],
            'bootstrap': [100, 15, 100],
            'pct_difference': [10, 33.3, 8],
            'current_price': [100, 100, 100]
        })
        
        mispricings = find_mispricing_opportunities(comparison, threshold_pct=10.0)
        
        # Should detect mispricing in std
        assert len(mispricings) >= 1
        assert 'opportunity' in mispricings.columns


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("Running Option Pricing and Implied Distribution Tests")
    print("=" * 80)
    
    test_classes = [
        TestBlackScholesModel,
        TestImpliedVolatility,
        TestGreeks,
        TestOptionChain,
        TestBreedenLitzenberger,
        TestImpliedDistribution,
        TestDistributionComparison,
        TestMispricingDetection
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 80)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n✅ All tests passed!")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
