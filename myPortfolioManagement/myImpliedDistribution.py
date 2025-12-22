"""
Implied Probability Distribution Module

This module provides functionality for extracting implied probability distributions
from option prices using the Breeden-Litzenberger formula and comparing them with
bootstrapped distributions.

Author: MyPortfolioManagement Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Optional
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import simpson
from scipy.stats import norm

from myPortfolioManagement.myOptionPricing import (
    black_scholes_call,
    black_scholes_put,
    create_option_chain
)


def breeden_litzenberger_density(strikes: np.ndarray, call_prices: np.ndarray,
                                 r: float, T: float,
                                 smoothing_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract risk-neutral probability density from call option prices using the
    Breeden-Litzenberger formula.
    
    The formula states that the second derivative of the call price with respect to
    strike gives the risk-neutral density:
    
    f(K) = exp(rT) * d²C/dK²
    
    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices (must be sorted)
    call_prices : np.ndarray
        Array of call option prices corresponding to strikes
    r : float
        Risk-free rate (annualized)
    T : float
        Time to expiration (in years)
    smoothing_factor : float, optional
        Smoothing parameter for spline interpolation (0-1), default 0.1
        Higher values = more smoothing
    
    Returns
    -------
    tuple of np.ndarray
        (strikes_dense, density) where strikes_dense are interpolated strikes
        and density is the risk-neutral probability density
        
    References
    ----------
    Breeden, D. T., & Litzenberger, R. H. (1978). Prices of state-contingent claims
    implicit in option prices. Journal of Business, 621-651.
    """
    if len(strikes) < 3:
        raise ValueError("Need at least 3 strike prices for density estimation")
    
    if not np.all(np.diff(strikes) > 0):
        raise ValueError("Strikes must be sorted in ascending order")
    
    # Ensure call prices are monotonically decreasing (arbitrage-free constraint)
    # Use isotonic regression or simple smoothing
    call_prices = np.maximum.accumulate(call_prices[::-1])[::-1]
    
    # Create smooth interpolation of call prices
    # Use cubic spline with smoothing to estimate second derivative
    spline = UnivariateSpline(strikes, call_prices, s=smoothing_factor * len(strikes), k=3)
    
    # Generate dense grid of strikes for smooth density
    strikes_dense = np.linspace(strikes[0], strikes[-1], 200)
    
    # Calculate second derivative analytically from spline
    second_derivative = spline.derivative(n=2)(strikes_dense)
    
    # Apply Breeden-Litzenberger formula
    density = np.exp(r * T) * second_derivative
    
    # Ensure non-negative density (can be slightly negative due to numerical issues)
    density = np.maximum(density, 0)
    
    # Normalize to ensure it's a proper probability density
    area = simpson(density, x=strikes_dense)
    if area > 0:
        density = density / area
    
    return strikes_dense, density


def extract_implied_distribution(option_chain: pd.DataFrame, S: float, r: float, T: float,
                                 option_type: str = 'call',
                                 method: str = 'breeden_litzenberger') -> pd.DataFrame:
    """
    Extract implied probability distribution from an option chain.
    
    Parameters
    ----------
    option_chain : pd.DataFrame
        DataFrame with columns 'strike' and either 'call_price' or 'put_price'
    S : float
        Current stock price
    r : float
        Risk-free rate (annualized)
    T : float
        Time to expiration (in years)
    option_type : str, optional
        'call' or 'put', default 'call'
    method : str, optional
        Method for extraction, currently only 'breeden_litzenberger' supported
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: price_level, probability_density
    """
    if method != 'breeden_litzenberger':
        raise ValueError(f"Method '{method}' not supported. Use 'breeden_litzenberger'.")
    
    # Select price column based on option type
    price_col = 'call_price' if option_type.lower() == 'call' else 'put_price'
    
    if price_col not in option_chain.columns:
        raise ValueError(f"Column '{price_col}' not found in option_chain")
    
    # Sort by strike
    option_chain = option_chain.sort_values('strike').reset_index(drop=True)
    
    strikes = option_chain['strike'].values
    prices = option_chain[price_col].values
    
    # Extract density using Breeden-Litzenberger
    strikes_dense, density = breeden_litzenberger_density(strikes, prices, r, T)
    
    result = pd.DataFrame({
        'price_level': strikes_dense,
        'probability_density': density
    })
    
    return result


def bootstrap_future_distribution(returns: pd.Series, S0: float, T: float,
                                  n_sim: int = 10000, bootstrap_method: str = 'iid',
                                  block_size: int = 20) -> pd.DataFrame:
    """
    Create bootstrapped distribution of future stock prices.
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    S0 : float
        Current stock price
    T : float
        Time horizon (in years)
    n_sim : int, optional
        Number of simulations, default 10000
    bootstrap_method : str, optional
        Bootstrap method: 'iid', 'block', 'stationary', default 'iid'
    block_size : int, optional
        Block size for block bootstrap methods, default 20
    
    Returns
    -------
    pd.DataFrame
        DataFrame with bootstrapped future prices
    """
    from myPortfolioManagement.myBootstrapping import bootstrappingTS
    
    # Convert time to number of periods
    periods_per_year = 252  # Assuming daily returns
    n_periods = int(T * periods_per_year)
    
    # Map bootstrap method names
    method_map = {
        'iid': 'nbb',
        'block': 'mbb',
        'stationary': 'sb',
        'circular': 'cbb'
    }
    
    bootstrap_type = method_map.get(bootstrap_method, 'nbb')
    
    # Bootstrap returns
    bootstrapped_returns = bootstrappingTS(
        series=returns,
        block_size=block_size,
        n_samples=n_sim,
        bootstrap_type=bootstrap_type
    )
    
    # Take first n_periods of each simulation
    if len(bootstrapped_returns) > n_periods:
        bootstrapped_returns = bootstrapped_returns.iloc[:n_periods, :]
    
    # Calculate cumulative returns and future prices
    cumulative_returns = (1 + bootstrapped_returns).cumprod()
    future_prices = S0 * cumulative_returns.iloc[-1, :]
    
    return pd.DataFrame({
        'simulation': range(len(future_prices)),
        'future_price': future_prices.values
    })


def compare_distributions(implied_dist: pd.DataFrame,
                         bootstrap_dist: pd.DataFrame,
                         S0: float,
                         confidence_levels: list = [0.05, 0.25, 0.5, 0.75, 0.95]) -> pd.DataFrame:
    """
    Compare implied and bootstrapped probability distributions.
    
    Parameters
    ----------
    implied_dist : pd.DataFrame
        Implied distribution with columns 'price_level' and 'probability_density'
    bootstrap_dist : pd.DataFrame
        Bootstrapped prices with column 'future_price'
    S0 : float
        Current stock price
    confidence_levels : list, optional
        Quantile levels to compare, default [0.05, 0.25, 0.5, 0.75, 0.95]
    
    Returns
    -------
    pd.DataFrame
        Comparison statistics including quantiles, means, and standard deviations
    """
    # Calculate statistics from implied distribution
    implied_prices = implied_dist['price_level'].values
    implied_probs = implied_dist['probability_density'].values
    
    # Normalize probability if needed
    prob_sum = simpson(implied_probs, x=implied_prices)
    if prob_sum > 0:
        implied_probs = implied_probs / prob_sum
    
    # Calculate implied mean and std
    implied_mean = simpson(implied_prices * implied_probs, x=implied_prices)
    implied_var = simpson((implied_prices - implied_mean)**2 * implied_probs, x=implied_prices)
    implied_std = np.sqrt(implied_var)
    
    # Calculate implied quantiles using cumulative distribution
    cumulative_prob = np.cumsum(implied_probs) * (implied_prices[1] - implied_prices[0])
    cumulative_prob = cumulative_prob / cumulative_prob[-1]  # Normalize
    
    implied_quantiles = {}
    for q in confidence_levels:
        idx = np.searchsorted(cumulative_prob, q)
        if idx < len(implied_prices):
            implied_quantiles[q] = implied_prices[idx]
        else:
            implied_quantiles[q] = implied_prices[-1]
    
    # Calculate statistics from bootstrap distribution
    bootstrap_prices = bootstrap_dist['future_price'].values
    bootstrap_mean = np.mean(bootstrap_prices)
    bootstrap_std = np.std(bootstrap_prices)
    
    bootstrap_quantiles = {q: np.quantile(bootstrap_prices, q) for q in confidence_levels}
    
    # Create comparison dataframe
    comparison_data = {
        'metric': ['mean', 'std'] + [f'quantile_{int(q*100)}' for q in confidence_levels],
        'implied': [implied_mean, implied_std] + [implied_quantiles[q] for q in confidence_levels],
        'bootstrap': [bootstrap_mean, bootstrap_std] + [bootstrap_quantiles[q] for q in confidence_levels]
    }
    
    comparison = pd.DataFrame(comparison_data)
    comparison['difference'] = comparison['implied'] - comparison['bootstrap']
    comparison['pct_difference'] = (comparison['difference'] / comparison['bootstrap']) * 100
    comparison['current_price'] = S0
    
    return comparison


def plot_distribution_comparison(implied_dist: pd.DataFrame,
                                bootstrap_dist: pd.DataFrame,
                                S0: float,
                                title: str = "Implied vs Bootstrapped Distribution",
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of implied and bootstrapped distributions.
    
    Parameters
    ----------
    implied_dist : pd.DataFrame
        Implied distribution with columns 'price_level' and 'probability_density'
    bootstrap_dist : pd.DataFrame
        Bootstrapped prices with column 'future_price'
    S0 : float
        Current stock price
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure, if None figure is not saved
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Probability densities
    ax1 = axes[0]
    
    # Implied distribution
    ax1.plot(implied_dist['price_level'], implied_dist['probability_density'],
             label='Implied (from options)', linewidth=2, color='blue')
    
    # Bootstrap distribution (as histogram)
    bootstrap_prices = bootstrap_dist['future_price'].values
    ax1.hist(bootstrap_prices, bins=50, density=True, alpha=0.5,
             label='Bootstrapped (historical)', color='green', edgecolor='black')
    
    # Mark current price
    ax1.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${S0:.2f}')
    
    ax1.set_xlabel('Future Price Level', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'{title} - Probability Densities', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distributions
    ax2 = axes[1]
    
    # Implied cumulative distribution
    implied_prices = implied_dist['price_level'].values
    implied_probs = implied_dist['probability_density'].values
    dx = implied_prices[1] - implied_prices[0]
    implied_cdf = np.cumsum(implied_probs) * dx
    implied_cdf = implied_cdf / implied_cdf[-1]  # Normalize
    
    ax2.plot(implied_prices, implied_cdf, label='Implied CDF', linewidth=2, color='blue')
    
    # Bootstrap cumulative distribution
    sorted_bootstrap = np.sort(bootstrap_prices)
    bootstrap_cdf = np.arange(1, len(sorted_bootstrap) + 1) / len(sorted_bootstrap)
    ax2.plot(sorted_bootstrap, bootstrap_cdf, label='Bootstrap CDF', linewidth=2, color='green')
    
    # Mark current price
    ax2.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${S0:.2f}')
    
    # Mark key percentiles
    for pct in [0.05, 0.5, 0.95]:
        ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Future Price Level', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title(f'{title} - Cumulative Distributions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def find_mispricing_opportunities(comparison: pd.DataFrame,
                                 threshold_pct: float = 10.0) -> pd.DataFrame:
    """
    Identify potential mispricing opportunities from distribution comparison.
    
    Parameters
    ----------
    comparison : pd.DataFrame
        Output from compare_distributions function
    threshold_pct : float, optional
        Percentage difference threshold for flagging mispricing, default 10.0
    
    Returns
    -------
    pd.DataFrame
        DataFrame with potential mispricing opportunities and recommendations
    """
    mispricings = comparison[
        np.abs(comparison['pct_difference']) > threshold_pct
    ].copy()
    
    if len(mispricings) == 0:
        return pd.DataFrame({
            'metric': ['No significant mispricing detected'],
            'opportunity': ['Markets appear fairly priced']
        })
    
    # Add interpretation
    mispricings['opportunity'] = mispricings.apply(
        lambda row: (
            f"{'Bullish' if row['pct_difference'] > 0 else 'Bearish'} signal: "
            f"Implied {row['metric']} is "
            f"{abs(row['pct_difference']):.1f}% {'higher' if row['pct_difference'] > 0 else 'lower'} "
            f"than bootstrap estimate"
        ),
        axis=1
    )
    
    return mispricings[['metric', 'implied', 'bootstrap', 'pct_difference', 'opportunity']]
