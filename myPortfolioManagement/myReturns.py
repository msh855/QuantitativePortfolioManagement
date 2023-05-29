#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 07:09:13 2022

@author: safishajjouz
"""

import pandas as pd
import numpy as np

from pypfopt.expected_returns import returns_from_prices, mean_historical_return, capm_return, ema_historical_return
from empyrical.stats import aggregate_returns
import ffn

import quantstats as qs


# TODO
# This part here has a function the 'get_stock_returns` that I dropped
# from myPortfolioManagement.myData import get_stock_returns


# calculate portfolio returns
def calculate_portfolio_returns(returns: pd.DataFrame,
                                myweights: pd.DataFrame,
                                portfolio_name: str = None) -> pd.DataFrame:
    """
    returns: a wide DataFrame series with assets returns 
    df_wegiths: a long Datafrane with Fund names in the first colum 
                and weight in the other 

    Args:
        returns (pd.DataFrame): DESCRIPTION.
        myassets_list (list): DESCRIPTION.
        myweights_list (list): DESCRIPTION.
        portfolio_name (str, optional): DESCRIPTION. Defaults to None.

    Raises:
        ValueError: DESCRIPTION.
        TypeError: DESCRIPTION.

    Returns:
        port_returns (TYPE): DESCRIPTION.

    """

    # sanity checks 
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    # if returns.index.inferred_type != "datetime64":
    #     raise TypeError('Date not an Index')

    # # slice returns dataframe
    # returns = returns[myassets_list]
    # weights = np.array(myweights_list)

    weights_temp = myweights.transpose()
    weights_temp.columns = weights_temp.iloc[0, :]
    returns = returns[weights_temp.columns]
    weights = np.array(weights_temp.iloc[1, :])

    # set zero values with NA
    port_returns = returns.fillna(0).dot(weights)

    # name the column of portfolio returns 
    if portfolio_name is None:
        port_returns = pd.Series(port_returns, name="myPortfolio")
    else:
        port_returns = pd.Series(port_returns, name=portfolio_name)

    port_returns = pd.DataFrame(port_returns)

    return port_returns


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
               'GLD': 0.08,
               'DJP': 0.07}

    ret_all_weather_dalio = qs.utils.make_index(ticker_weights=tickers,
                                                rebalance=rebalance,
                                                period='max',
                                                returns=None,
                                                match_dates=False)

    ret_all_weather_dalio = pd.Series(ret_all_weather_dalio,
                                      name='All_Weather_Dalio')

    # 60/40 based on BlackRock
    ret_60_40 = pd.Series(qs.utils.download_returns('BAGPX'),
                          name='Port_60/40_BlackRock')

    retun_bench_port = pd.concat([ret_all_weather_dalio, ret_60_40], axis=1)

    return retun_bench_port


def get_benchmark_returns(choose_bench: str = 'S&P500') -> pd.DataFrame:
    '''
    Options for benchmarking: 'Nasdaq', 'S&P500', 'World_Index', 'Cash', 
                               'Emerging_Markets','All_Weather_US'
              'US_Real_Estate', 'US_mid_Cap', 'US_small_Cap', 'World_Non_US',
              'US_TIPS', 'US_Bonds', 'Bloomberg_Commodity_Index', 
              'Port_60/40_BlackRock'
        
    
    '''
    # options 
    tickers = ['^IXIC', '^GSPC', 'VT', 'BIL', 'EEM', 'VNQ',
               'MDY', 'SLY', 'EFA', 'TIP', 'AGG', 'DJP']

    names = ['Nasdaq', 'S&P500', 'World_Index', 'Cash', 'Emerging_Markets',
             'US_Real_Estate', 'US_mid_Cap', 'US_small_Cap', 'World_Non_US',
             'US_TIPS', 'US_Bonds', 'Bloomberg_Commodity_Index']

    ret_bench = list()
    for tick, names in zip(tickers, names):
        ret = pd.Series(qs.utils.download_returns(tick), name=names)
        ret_bench.append(ret)

    ret_bench = pd.concat(ret_bench, axis=1)

    retun_bench_port = get_benchmark_porfolios()

    ret_bench = pd.concat([ret_bench, retun_bench_port], axis=1)

    if choose_bench:
        ret_bench = ret_bench[[choose_bench]]

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
