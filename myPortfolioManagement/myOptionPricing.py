"""
Option Pricing Module

This module provides functionality for pricing options using the Black-Scholes model,
calculating implied volatility, and computing option Greeks.

Author: MyPortfolioManagement Team
Date: December 2024
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from typing import Union, Tuple


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float
        Call option price
        
    References
    ----------
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.
    """
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float
        Put option price
        
    References
    ----------
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.
    """
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def option_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option vega (sensitivity to volatility).
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float
        Vega of the option
    """
    if T <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega


def implied_volatility(option_price: float, S: float, K: float, T: float, r: float,
                       option_type: str = 'call', max_iter: int = 100,
                       tol: float = 1e-6) -> float:
    """
    Calculate implied volatility using Newton-Raphson method with Brent's method fallback.
    
    Parameters
    ----------
    option_price : float
        Market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    option_type : str, optional
        'call' or 'put', default 'call'
    max_iter : int, optional
        Maximum number of iterations for Newton-Raphson, default 100
    tol : float, optional
        Convergence tolerance, default 1e-6
    
    Returns
    -------
    float
        Implied volatility
        
    Raises
    ------
    ValueError
        If implied volatility cannot be calculated or option is too far out of the money
    """
    if T <= 0:
        raise ValueError("Time to expiration must be positive")
    
    if option_price <= 0:
        raise ValueError("Option price must be positive")
    
    # Check intrinsic value
    if option_type.lower() == 'call':
        intrinsic = max(S - K, 0)
        pricing_func = black_scholes_call
    else:
        intrinsic = max(K - S, 0)
        pricing_func = black_scholes_put
    
    if option_price < intrinsic:
        raise ValueError("Option price is below intrinsic value")
    
    # Try Newton-Raphson first
    try:
        sigma = 0.5  # Initial guess
        price_diff = float('inf')  # Initialize to ensure it's defined
        for i in range(max_iter):
            price = pricing_func(S, K, T, r, sigma)
            vega = option_vega(S, K, T, r, sigma)
            
            if abs(vega) < 1e-10:
                break
            
            price_diff = price - option_price
            if abs(price_diff) < tol:
                return sigma
            
            sigma = sigma - price_diff / vega
            
            # Keep sigma in reasonable bounds
            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 10:
                sigma = 10
        
        # If Newton-Raphson succeeded but didn't converge perfectly, return best guess
        if abs(price_diff) < tol * 10:
            return sigma
            
    except Exception:
        pass
    
    # Fallback to Brent's method
    try:
        def objective(vol):
            return pricing_func(S, K, T, r, vol) - option_price
        
        sigma = brentq(objective, 0.001, 10.0, xtol=tol)
        return sigma
    except Exception as e:
        raise ValueError(f"Unable to calculate implied volatility: {str(e)}")


def option_delta(S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str = 'call') -> float:
    """
    Calculate option delta (sensitivity to underlying price).
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str, optional
        'call' or 'put', default 'call'
    
    Returns
    -------
    float
        Delta of the option
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def option_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option gamma (sensitivity of delta to underlying price).
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float
        Gamma of the option
    """
    if T <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def create_option_chain(S: float, T: float, r: float, sigma: float,
                       strike_range: Tuple[float, float] = None,
                       num_strikes: int = 20) -> pd.DataFrame:
    """
    Create a synthetic option chain with call and put prices.
    
    This function is useful for assets that don't have traded options,
    allowing analysis of implied distributions based on estimated volatility.
    
    Parameters
    ----------
    S : float
        Current stock price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    strike_range : tuple of float, optional
        (min_strike, max_strike) as proportion of spot price, default (0.7, 1.3)
    num_strikes : int, optional
        Number of strike prices to generate, default 20
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: strike, call_price, put_price, call_iv, put_iv
    """
    if strike_range is None:
        strike_range = (0.7, 1.3)
    
    min_strike = S * strike_range[0]
    max_strike = S * strike_range[1]
    
    strikes = np.linspace(min_strike, max_strike, num_strikes)
    
    option_data = []
    for K in strikes:
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        option_data.append({
            'strike': K,
            'call_price': call_price,
            'put_price': put_price,
            'call_iv': sigma,
            'put_iv': sigma
        })
    
    return pd.DataFrame(option_data)
