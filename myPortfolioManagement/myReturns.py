#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 07:09:13 2022

@author: safishajjouz
"""

import pandas as pd
import numpy as np
from typing import Union

from pypfopt.expected_returns import returns_from_prices, mean_historical_return, capm_return, ema_historical_return
from empyrical.stats import aggregate_returns
import ffn

import quantstats_lumi as qs


# TODO
# This part here has a function the 'get_stock_returns` that I dropped
# from myPortfolioManagement.myData import get_stock_returns


def calculate_buy_and_hold_returns(returns: pd.DataFrame,
                                   initial_weights: Union[pd.Series, dict],
                                   portfolio_name: str = None,
                                   transaction_cost_bps: float = 0.0) -> pd.DataFrame:
    """
    Calculate portfolio returns using a buy-and-hold strategy where weights drift 
    naturally with asset performance (no rebalancing).
    
    In buy-and-hold, you invest according to initial weights and then hold those positions.
    The weights will naturally drift as assets perform differently over time.
    Transaction costs are only applied at the initial purchase.
    
    Args:
        returns (pd.DataFrame): Wide DataFrame with asset returns (columns = assets, index = dates)
        initial_weights (pd.Series or dict): Initial portfolio weights (must sum to 1.0)
        portfolio_name (str, optional): Name for the portfolio returns series. Defaults to "buy_and_hold"
        transaction_cost_bps (float, optional): Transaction cost in basis points (1 bps = 0.0001). 
                                                Applied only at initial investment. Defaults to 0.0
    
    Returns:
        pd.DataFrame: Portfolio returns series with portfolio_name as column name
        
    Example:
        >>> returns = pd.DataFrame({'AAPL': [0.01, 0.02], 'MSFT': [-0.01, 0.03]})
        >>> weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        >>> port_returns = calculate_buy_and_hold_returns(returns, weights, transaction_cost_bps=10)
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Pandas DataFrame")
    
    # Convert weights to Series if dict
    if isinstance(initial_weights, dict):
        initial_weights = pd.Series(initial_weights)
    
    # Validate weights
    if not np.isclose(initial_weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {initial_weights.sum()}")
    
    # Ensure returns has all assets in weights
    missing_assets = set(initial_weights.index) - set(returns.columns)
    if missing_assets:
        raise ValueError(f"Returns missing assets: {missing_assets}")
    
    # Filter returns to only include assets in weights
    returns_subset = returns[initial_weights.index].copy()
    
    # For buy-and-hold, we calculate the weighted return for each period
    # but the weights drift based on cumulative performance
    
    # Initialize with starting weights
    portfolio_returns = []
    current_value = initial_weights.copy()  # Start with weights as initial investment fractions
    
    for i, date in enumerate(returns_subset.index):
        period_returns = returns_subset.loc[date].fillna(0)
        
        # Calculate portfolio return for this period
        # Return = sum of (weight_i * return_i) where weights are current portfolio positions
        port_ret = (current_value * period_returns).sum() / current_value.sum()
        
        # Apply transaction cost on first period only (initial purchase)
        if i == 0 and transaction_cost_bps > 0:
            tc = transaction_cost_bps / 10000.0
            port_ret = port_ret - tc
        
        portfolio_returns.append(port_ret)
        
        # Update values based on returns (weights drift naturally)
        current_value = current_value * (1 + period_returns)
    
    port_returns = pd.Series(portfolio_returns, index=returns_subset.index)
    
    # Name the series
    if portfolio_name is None:
        portfolio_name = "buy_and_hold"
    port_returns.name = portfolio_name
    
    return pd.DataFrame(port_returns)


def calculate_rebalanced_returns(returns: pd.DataFrame,
                                 target_weights: Union[pd.Series, dict],
                                 rebalance_freq: str = 'monthly',
                                 portfolio_name: str = None,
                                 transaction_cost_bps: float = 0.0) -> pd.DataFrame:
    """
    Calculate portfolio returns with periodic rebalancing to target weights.
    
    This function simulates a portfolio that is rebalanced to target weights at 
    regular intervals (daily, weekly, monthly, quarterly, or yearly). Transaction 
    costs are applied whenever rebalancing occurs.
    
    NOTE: Daily rebalancing with zero transaction costs is equivalent to the 
    traditional returns.dot(weights) calculation, but is rarely realistic in practice.
    
    Args:
        returns (pd.DataFrame): Wide DataFrame with asset returns (columns = assets, index = dates)
        target_weights (pd.Series or dict): Target portfolio weights (must sum to 1.0)
        rebalance_freq (str, optional): Rebalancing frequency. Options:
                                       'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
                                       Defaults to 'monthly'
        portfolio_name (str, optional): Name for the portfolio returns series. 
                                       Defaults to "rebalanced_{freq}"
        transaction_cost_bps (float, optional): Transaction cost in basis points per trade.
                                               Applied to the absolute weight change at each rebalance.
                                               Defaults to 0.0
    
    Returns:
        pd.DataFrame: Portfolio returns series with portfolio_name as column name
        
    Example:
        >>> returns = pd.DataFrame({'AAPL': [0.01, 0.02, -0.01], 'MSFT': [-0.01, 0.03, 0.01]},
        ...                        index=pd.date_range('2020-01-01', periods=3))
        >>> weights = pd.Series({'AAPL': 0.6, 'MSFT': 0.4})
        >>> port_returns = calculate_rebalanced_returns(returns, weights, 
        ...                                             rebalance_freq='monthly',
        ...                                             transaction_cost_bps=10)
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Pandas DataFrame")
    
    # Convert weights to Series if dict
    if isinstance(target_weights, dict):
        target_weights = pd.Series(target_weights)
    
    # Validate weights
    if not np.isclose(target_weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {target_weights.sum()}")
    
    # Ensure returns has all assets in weights
    missing_assets = set(target_weights.index) - set(returns.columns)
    if missing_assets:
        raise ValueError(f"Returns missing assets: {missing_assets}")
    
    # Validate rebalance frequency
    valid_freqs = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    if rebalance_freq.lower() not in valid_freqs:
        raise ValueError(f"rebalance_freq must be one of {valid_freqs}, got '{rebalance_freq}'")
    
    # Filter returns to only include assets in weights
    returns_subset = returns[target_weights.index].copy()
    
    # Determine rebalancing dates based on frequency
    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'MS',  # Month Start
        'quarterly': 'QS',  # Quarter Start
        'yearly': 'YS'  # Year Start
    }
    
    freq_code = freq_map[rebalance_freq.lower()]
    
    # Create rebalancing schedule - find dates closest to each period start
    if rebalance_freq.lower() == 'daily':
        rebalance_dates = returns_subset.index
    else:
        # Generate period starts
        period_starts = pd.date_range(
            start=returns_subset.index[0],
            end=returns_subset.index[-1],
            freq=freq_code
        )
        # Find actual trading dates closest to each period start
        rebalance_dates = []
        for period_start in period_starts:
            # Find the first available date on or after the period start
            available_dates = returns_subset.index[returns_subset.index >= period_start]
            if len(available_dates) > 0:
                rebalance_dates.append(available_dates[0])
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    
    # Initialize tracking variables
    portfolio_returns = []
    current_weights = target_weights.copy()
    
    # Iterate through each period
    for i, date in enumerate(returns_subset.index):
        # Get returns for this period
        period_returns = returns_subset.loc[date].fillna(0)
        
        # Calculate portfolio return for this period (before rebalancing)
        port_return = (period_returns * current_weights).sum()
        
        # Check if we need to rebalance on this date
        if date in rebalance_dates:
            # Calculate transaction costs based on weight changes needed
            # Current weights after market movement but before rebalancing
            post_return_weights = current_weights * (1 + period_returns)
            post_return_weights = post_return_weights / post_return_weights.sum()
            
            # Weight changes needed to rebalance
            weight_changes = (target_weights - post_return_weights).abs()
            total_turnover = weight_changes.sum()
            
            # Apply transaction costs (cost is proportional to turnover)
            if transaction_cost_bps > 0:
                tc = (total_turnover * transaction_cost_bps) / 10000.0
                port_return = port_return - tc
            
            # Reset to target weights
            current_weights = target_weights.copy()
        else:
            # Update weights based on returns (weights drift)
            current_weights = current_weights * (1 + period_returns)
            current_weights = current_weights / current_weights.sum()
        
        portfolio_returns.append(port_return)
    
    # Create return series
    port_returns = pd.Series(portfolio_returns, index=returns_subset.index)
    
    # Name the series
    if portfolio_name is None:
        portfolio_name = f"rebalanced_{rebalance_freq}"
    port_returns.name = portfolio_name
    
    return pd.DataFrame(port_returns)


# calculate portfolio returns
def calculate_portfolio_returns(returns: pd.DataFrame,
                                myweights: pd.DataFrame,
                                portfolio_name: str = None,
                                rebalance_strategy: str = 'daily',
                                transaction_cost_bps: float = 0.0) -> pd.DataFrame:
    """
    Calculate portfolio returns with specified rebalancing strategy.
    
    IMPORTANT: This function's default behavior assumes DAILY REBALANCING, which means
    the portfolio is rebalanced to target weights every single day. This is equivalent 
    to the traditional returns.dot(weights) calculation but is rarely realistic in practice.
    
    For more realistic portfolio simulations:
    - Use 'buy_and_hold' for a buy-and-hold strategy (no rebalancing)
    - Use 'monthly', 'quarterly', or other frequencies for periodic rebalancing
    - Add transaction_cost_bps > 0 to account for trading costs
    
    Args:
        returns (pd.DataFrame): Wide DataFrame with asset returns (columns = assets, index = dates)
        myweights (pd.DataFrame): DataFrame with asset names in first column and weights in second
        portfolio_name (str, optional): Name for the portfolio. Defaults to None
        rebalance_strategy (str, optional): Rebalancing strategy. Options:
            - 'daily': Rebalance every day (default, traditional calculation)
            - 'weekly': Rebalance weekly
            - 'monthly': Rebalance monthly  
            - 'quarterly': Rebalance quarterly
            - 'yearly': Rebalance yearly
            - 'buy_and_hold': No rebalancing, weights drift naturally
            Defaults to 'daily'
        transaction_cost_bps (float, optional): Transaction cost in basis points.
                                               Defaults to 0.0
    
    Returns:
        pd.DataFrame: Portfolio returns with specified strategy
        
    Example:
        >>> # Traditional daily rebalancing (implicit assumption)
        >>> port_ret = calculate_portfolio_returns(returns, weights)
        >>> 
        >>> # More realistic: monthly rebalancing with transaction costs
        >>> port_ret = calculate_portfolio_returns(returns, weights, 
        ...                                        rebalance_strategy='monthly',
        ...                                        transaction_cost_bps=10)
        >>> 
        >>> # Buy and hold strategy
        >>> port_ret = calculate_portfolio_returns(returns, weights,
        ...                                        rebalance_strategy='buy_and_hold')
    """
    # sanity checks 
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    # Parse weights from myweights DataFrame
    weights_temp = myweights.transpose()
    weights_temp.columns = weights_temp.iloc[0, :]
    returns_filtered = returns[weights_temp.columns]
    weights = np.array(weights_temp.iloc[1, :])
    
    # Convert to Series for new functions
    weight_series = pd.Series(weights, index=weights_temp.columns)
    
    # Route to appropriate function based on strategy
    if rebalance_strategy == 'buy_and_hold':
        port_returns = calculate_buy_and_hold_returns(
            returns_filtered, 
            weight_series,
            portfolio_name=portfolio_name,
            transaction_cost_bps=transaction_cost_bps
        )
    elif rebalance_strategy in ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
        port_returns = calculate_rebalanced_returns(
            returns_filtered,
            weight_series,
            rebalance_freq=rebalance_strategy,
            portfolio_name=portfolio_name,
            transaction_cost_bps=transaction_cost_bps
        )
    else:
        raise ValueError(f"Invalid rebalance_strategy: {rebalance_strategy}. "
                        f"Must be one of: 'buy_and_hold', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'")
    
    # Convert to numeric and return
    port_returns = pd.to_numeric(port_returns.iloc[:, 0], errors='coerce')
    
    # Name the column of portfolio returns 
    if portfolio_name is None:
        port_returns = pd.Series(port_returns, name="myPortfolio")
    else:
        port_returns = pd.Series(port_returns, name=portfolio_name)
    
    ret = pd.DataFrame(port_returns)
    
    return ret


def create_portfolios(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        returns (TYPE): DESCRIPTION.
        weights (TYPE): DESCRIPTION.

    Returns:
        df_portfolio_returns (TYPE): DESCRIPTION.

    """

    # calculate return series 
    df_portfolio_returns = pd.DataFrame([])

    for portfolio_name in list(weights.columns):
        temp = calculate_portfolio_returns(returns,
                                           weights.index.to_list(),
                                           weights[portfolio_name].to_list(),
                                           portfolio_name=portfolio_name)
        df_portfolio_returns = pd.concat([df_portfolio_returns, temp], axis=1)

    return df_portfolio_returns


def convert_returns_freq(ret: pd.DataFrame, convert_to: str) -> pd.DataFrame:
    """
    Args:
        ret (pd.DataFrame): DESCRIPTION.
        convert_to (str): DESCRIPTION.

    Raises:
        : DESCRIPTION.

    Returns:
        ret (TYPE): DESCRIPTION.
    """

    if convert_to is not None:
        first_date = ret.index[0]
        last_date = ret.index[-1]

        ret = aggregate_returns(ret, convert_to)

        if convert_to == 'weekly':
            new_dates = pd.date_range(first_date, periods=len(ret), freq="w")
            new_dates = list(new_dates.strftime("%Y-%m-%d"))
            new_dates[-1] = last_date.strftime("%Y-%m-%d")
            ret['Date'] = new_dates
            ret = ret.set_index('Date')
            ret.index = pd.to_datetime(ret.index, infer_datetime_format=True)

        elif convert_to == 'monthly':
            new_dates = pd.date_range(first_date, periods=len(ret), freq="m")
            new_dates = list(new_dates.strftime("%Y-%m-%d"))
            new_dates[-1] = last_date.strftime("%Y-%m-%d")
            ret['Date'] = new_dates
            ret = ret.set_index('Date')
            ret.index = pd.to_datetime(ret.index, infer_datetime_format=True)

        elif convert_to == 'quarterly':
            new_dates = pd.date_range(first_date, periods=len(ret), freq="q")
            new_dates = list(new_dates.strftime("%Y-%m-%d"))
            new_dates[-1] = last_date.strftime("%Y-%m-%d")
            ret['Date'] = new_dates
            ret = ret.set_index('Date')
            ret.index = pd.to_datetime(ret.index, infer_datetime_format=True)


        elif convert_to == 'yearly':
            new_dates = pd.date_range(first_date, periods=len(ret), freq="y")
            new_dates = list(new_dates.strftime("%Y-%m-%d"))
            new_dates[-1] = last_date.strftime("%Y-%m-%d")
            ret['Date'] = new_dates
            ret = ret.set_index('Date')
            ret.index = pd.to_datetime(ret.index, infer_datetime_format=True)
        else:
            raise 'This frequency conversion is not supported'

    return ret


def calculate_returns(df_prices: pd.DataFrame, log_returns: bool = False,
                      convert_to: str = None,
                      annualize_factor: int = None,
                      add_portfolio: bool = False, **kwargs) -> pd.DataFrame:
    """
    function to compute returns. Can be simple returns or log returns. It 
    also allows aggregation of retunrs from daily to ['weekly', 'monthly', 
                                                      quarterly, 'yearly']

    The user can also calculate returns on a rolling basis.

    Args:
        df_prices (pd.DataFrame): DESCRIPTION.
        log_returns (bool, optional): DESCRIPTION. Defaults to False.
        convert_to (str, optional): DESCRIPTION. Defaults to None.
        add_portfolio (bool, optional): DESCRIPTION. Defaults to False.
        **kwargs (TYPE): DESCRIPTION.

    Raises:
        : DESCRIPTION.

    Returns:
        ret (TYPE): DESCRIPTION.
    """

    ret = returns_from_prices(df_prices, log_returns=log_returns)

    if add_portfolio:
        ret_port = calculate_portfolio_returns(ret, **kwargs)
        ret = pd.concat([ret, ret_port], axis=1)

    if convert_to in ['weekly', 'monthly', 'quarterly', 'yearly']:
        ret = convert_returns_freq(ret, convert_to)

    if annualize_factor:
        # memo: 252 for daily, 52 for weekly, 12 for monthly
        # output from fnn is in 100 so I devide with 100 
        ret = ffn.core.annualize(ret, annualize_factor, one_year=365) / 100

    return ret


def average_returns(returns: pd.DataFrame or pd.Series,
                    method: str = 'hist',
                    benchmark_returns: pd.Series or pd.DataFrame = None,
                    span=500,
                    periods=252, rf=0.02, log_returns=False) -> float or pd.Series:
    """
    

    Args:
        returns (pd.DataFrame or pd.Series): DESCRIPTION.
        method (str, optional): DESCRIPTION. Defaults to 'hist'.
        benchmark_returns (pd.Series or pd.DataFrame, optional): DESCRIPTION. Defaults to None.
        span (TYPE, optional): DESCRIPTION. Defaults to 500.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        log_returns (TYPE, optional): DESCRIPTION. Defaults to False.

    Raises:
        ValueError: DESCRIPTION.

    Returns:
        mu (TYPE): DESCRIPTION.

    """

    if method == 'hist':
        mu = mean_historical_return(prices=returns,
                                    returns_data=True,
                                    compounding=True,
                                    frequency=periods,
                                    log_returns=log_returns)

    if method == 'capm':
        if isinstance(benchmark_returns, type(None)):
            raise ValueError('benchmark_returns is missing')

        mu = capm_return(returns,
                         market_prices=benchmark_returns,
                         returns_data=True,
                         risk_free_rate=rf,
                         compounding=True,
                         frequency=periods,
                         log_returns=log_returns)

    if method == 'ema':
        mu = ema_historical_return(returns,
                                   returns_data=True,
                                   compounding=True,
                                   span=span,
                                   frequency=periods,
                                   log_returns=log_returns)

    return mu


def get_benchmark_porfolios(rebalance=None):
    # All-Weather-Porfolio based on weights
    tickers = {'VTI': 0.30,
               'VGLT': 0.40,
               'VGIT': 0.15,
               'GLD': 0.075,
               'DBC': 0.075}  # DBC replaces DJP (which is delisted)

    try:
        ret_all_weather_dalio = qs.utils.make_index(ticker_weights=tickers,
                                                    rebalance=rebalance,
                                                    period='max',
                                                    returns=None,
                                                    match_dates=False)

        ret_all_weather_dalio = pd.Series(ret_all_weather_dalio,
                                          name='All_Weather_Dalio')
    except (ValueError, Exception) as e:
        print(f"Warning: Could not create All-Weather portfolio: {e}")
        ret_all_weather_dalio = pd.Series(dtype=float, name='All_Weather_Dalio')

    # 60/40 based on BlackRock
    try:
        ret_60_40 = pd.Series(qs.utils.download_returns('BAGPX'),
                              name='Port_60/40_BlackRock')
    except Exception as e:
        print(f"Warning: Could not download 60/40 portfolio: {e}")
        ret_60_40 = pd.Series(dtype=float, name='Port_60/40_BlackRock')

    retun_bench_port = pd.concat([ret_all_weather_dalio, ret_60_40], axis=1)

    return retun_bench_port


def get_benchmark_returns(choose_bench: str or list = None) -> pd.DataFrame:
    '''
    Download benchmark returns. Only downloads the benchmarks you request.
    
    Args:
        choose_bench: str, list of str, or None
            - str: download a single benchmark (e.g., 'S&P500')
            - list: download multiple benchmarks (e.g., ['S&P500', 'Nasdaq'])
            - None: download all available benchmarks
    
    Options for benchmarking: 'Nasdaq', 'S&P500', 'World_Index', 'Cash', 
                               'Emerging_Markets', 'US_Real_Estate',
                               'US_mid_Cap', 'US_small_Cap', 'World_Non_US',
                               'US_TIPS', 'US_Bonds', 'Bloomberg_Commodity_Index'
    
    Returns:
        pd.DataFrame: DataFrame with benchmark returns
    '''
    # Mapping of benchmark names to Yahoo tickers
    benchmark_map = {
        'Nasdaq': '^IXIC',
        'S&P500': '^GSPC',
        'World_Index': 'VT',
        'Cash': 'BIL',
        'Emerging_Markets': 'EEM',
        'US_Real_Estate': 'VNQ',
        'US_mid_Cap': 'MDY',
        'US_small_Cap': 'SLY',
        'World_Non_US': 'EFA',
        'US_TIPS': 'TIP',
        'US_Bonds': 'AGG',
        'Bloomberg_Commodity_Index': 'DBC'  # DBC replaces DJP (delisted)
    }
    
    # Determine which benchmarks to download
    if choose_bench is None:
        # Download all benchmarks
        benchmarks_to_download = list(benchmark_map.keys())
    elif isinstance(choose_bench, str):
        # Single benchmark
        benchmarks_to_download = [choose_bench]
    elif isinstance(choose_bench, list):
        # List of benchmarks
        benchmarks_to_download = choose_bench
    else:
        raise ValueError("choose_bench must be a string, list of strings, or None")
    
    # Validate benchmark names
    invalid = [b for b in benchmarks_to_download if b not in benchmark_map]
    if invalid:
        raise ValueError(f"Invalid benchmark(s): {invalid}. "
                        f"Available options: {list(benchmark_map.keys())}")
    
    # Download only requested benchmarks
    ret_bench = []
    for name in benchmarks_to_download:
        ticker = benchmark_map[name]
        try:
            ret_data = qs.utils.download_returns(ticker)
            # Handle both Series and DataFrame returns from quantstats
            if isinstance(ret_data, pd.DataFrame):
                ret = ret_data.squeeze()
            else:
                ret = ret_data
            ret.name = name
            ret_bench.append(ret)
        except Exception as e:
            print(f"Warning: Could not download {name} ({ticker}): {e}")
    
    if not ret_bench:
        raise ValueError("No benchmark data could be downloaded")
    
    ret_bench = pd.concat(ret_bench, axis=1)

    return ret_bench


#
# def get_multi_asset_returns() -> pd.DataFrame:
#     # - EEM  # iShares Emerging Markets - Emering Markets
#     # - VNQ  # Vangaurd Real Estate  - Real Estate
#     # - MDY  # SPDR S&P MIDCAP 400 ETF Trust (MDY) - mid cap
#     # - SLY  # SPDR S&P 600 Small Cap ETF (SLY) - small cap
#     # - SPY  # S&P 500   - large cap
#     # - EFA  # International Stocks iShares MSCI EAFE ETF (EFA)
#     # - TIP  # iShares TIPS Bond ETF (TIP) - TIPS
#     # - AGG  # iShares Core U.S. Aggregate Bond ETF (AGG) - Bonds
#     # - DJP  # iPath Bloomberg Commodity Index Total Return(SM) ETN (DJP) - Commodities
#     # - BIL  # SPDR Bloomberg Barclays 1-3 Month T-Bill ETF (BIL)         - Cash
#
#     multi_asset_tickers = ['EEM',
#                            'VNQ', 'MDY', 'SLY',
#                            'SPY', 'EFA',
#                            'TIP',
#                            'AGG',
#                            'DJP', 'BIL']
#
#     df_multi_asset = get_stock_returns(multi_asset_tickers)
#
#     df_multi_asset = df_multi_asset.dropna()
#
#     # naive or equal weight portfolio allocation
#     from myPortfolioManagement.myPortfolioOptimisation import equal_weight_portfolio
#     df_naive = equal_weight_portfolio(df_multi_asset)
#
#     port_returns = calculate_portfolio_returns(df_multi_asset,
#                                                myassets_list=multi_asset_tickers,
#                                                myweights_list=df_naive.port_naive.to_list(),
#                                                portfolio_name='port_multi_asset')
#     return port_returns
