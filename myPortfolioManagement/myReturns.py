#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 07:09:13 2022

@author: safishajjouz
"""

import pandas as pd 
import numpy as np 

from pypfopt.expected_returns import returns_from_prices
from empyrical.stats import aggregate_returns 
import ffn 
import statistics
from pypfopt import expected_returns
import quantstats as qs



#from myPortfolioManagement.myReturns import download_returns
# from myPortfolioManagement import myPortfolioOptimisation


#calculate portfolio returns 
def calculate_portfolio_returns(returns: pd.DataFrame, 
                                myassets_list:list,
                                myweights_list:list,
                                portfolio_name:str = None) -> pd.DataFrame:
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
     
    # slice returns dataframe 
    returns = returns[myassets_list]
    weights = np.array(myweights_list)
    
    # set zero values with NA
    port_returns = returns.fillna(0).dot(weights)  
    
    # name the column of portfolio returns 
    if portfolio_name is None: 
        port_returns = pd.Series(port_returns, name = "myPortfolio")
    else:
        port_returns = pd.Series(port_returns, name = portfolio_name)
        
    port_returns = pd.DataFrame(port_returns)
    
    return port_returns

def create_portfolios(returns:pd.DataFrame, weights:pd.DataFrame) -> pd.DataFrame:
  
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
                                    portfolio_name = portfolio_name)
        df_portfolio_returns = pd.concat([df_portfolio_returns, temp], axis=1)
        
    return df_portfolio_returns


def convert_returns_freq(ret: pd.DataFrame, convert_to:str) ->pd.DataFrame:
    """
    

    Args:
        ret (pd.DataFrame): DESCRIPTION.
        convert_to (str): DESCRIPTION.

    Raises:
        : DESCRIPTION.

    Returns:
        ret (TYPE): DESCRIPTION.

    """
    
    if convert_to !=None:
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
            raise  'This frequency conversion is not supported'
   
    return ret 


def calculate_returns(df_prices: pd.DataFrame, log_returns:bool = False, 
                           convert_to:str = None, 
                           rolling_window:int =None, 
                           annualize_factor = None, 
                           add_portfolio:bool = False, **kwargs) -> pd.DataFrame:
    """
    function to compute returns. Can be simple returns or log returns. It 
    also allows aggregation of retunrs from daily to ['weekly', 'monthly', 
                                                      quarterly, 'yearly']
    
    The useer can also calculate returns on a rolling basis. 

    Args:
        df_prices (pd.DataFrame): DESCRIPTION.
        log_returns (bool, optional): DESCRIPTION. Defaults to False.
        convert_to (str, optional): DESCRIPTION. Defaults to None.
        rolling_window (int, optional): DESCRIPTION. Defaults to None.
        add_portfolio (bool, optional): DESCRIPTION. Defaults to False.
        **kwargs (TYPE): DESCRIPTION.

    Raises:
        : DESCRIPTION.

    Returns:
        ret (TYPE): DESCRIPTION.

    """
   
    ret =  returns_from_prices(df_prices, log_returns= log_returns)
   
    if add_portfolio:
      ret_port = calculate_portfolio_returns(ret, **kwargs) 
      ret = pd.concat([ret, ret_port], axis = 1)
    
    if convert_to in ['weekly', 'monthly', 'quarterly', 'yearly']:
        ret = convert_returns_freq(ret, convert_to)
    
    if annualize_factor:
        # memo: 252 for daily, 52 for weekly, 12 for monthly
        # output from fnn is in 100 so I devide with 100 
        ret =  ffn.core.annualize(ret, annualize_factor, one_year=365)/100 
        
      
    if rolling_window !=None:
        ret = pd.concat([ffn.core.rollapply(ret[col], rolling_window, 
                                            statistics.mean) 
                         for col in list(ret.columns)], axis = 1)
    
    return ret 

def average_returns(returns:pd.DataFrame or pd.Series, 
                      method:str = 'hist', 
                      benchmark_returns:pd.Series or pd.DataFrame=None,
                      span=500, 
                      periods=252, rf = 0.02, log_returns = False) -> float or pd.Series:
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
        mu = expected_returns.mean_historical_return(prices = returns, 
                                                      returns_data=True, 
                                                      compounding=True, 
                                                      frequency=periods, 
                                                      log_returns=log_returns)
   
    if method == 'capm':
        if isinstance(benchmark_returns, type(None)):
            raise ValueError('benchmark_returns is missing')
            
        mu = expected_returns.capm_return(returns, 
                                          market_prices=benchmark_returns, 
                                          returns_data=True, 
                                          risk_free_rate=rf, 
                                          compounding=True, 
                                          frequency=periods, 
                                          log_returns=log_returns)
    
    if method == 'ema': 
        mu = expected_returns.ema_historical_return(returns, 
                                                returns_data=True, 
                                                compounding=True, 
                                                span=span, 
                                                frequency=periods, 
                                                log_returns=log_returns)
   
    
    return mu

    
def get_benchmark_porfolios(rebalance = None):
    
    # All-Weather-Porfolio
    tickers = {'VTI':  0.30,
               'VGLT': 0.40,
               'VGIT': 0.15,
               'IAU':  0.08,
               'DJP': 0.07}
    
    ret_all_weather_US = qs.utils.make_index(ticker_weights = tickers, 
                                                 rebalance=rebalance, 
                                                 period='max', 
                                                 returns=None, 
                                                 match_dates=False)
    
    ret_all_weather_US = pd.Series(ret_all_weather_US, 
                                  name= 'All_Weather_Dalio')
    
    
    # # All-Weather-Porfolio UK version 
    # tickers_UK = {'VUSA.L': 0.10,
    #               'VEUR.L': 0.10,
    #               'VMID.L':0.10,
    #               'VJPN.L':0.10,
    #               'EMIM.L':0.10,
    #               'IWDP.L':0.10,
    #               'VGOV.L':0.10,
    #               'ITPS.L':0.10,
    #               'IAU' :0.10,
    #               'BIL' :0.10} 

    # ret_all_weather_UK = qs.utils.make_index(ticker_weights = tickers_UK, 
    #                                              rebalance=rebalance, 
    #                                              period='max', 
    #                                              returns=None, 
    #                                              match_dates=False)
    
    # ret_all_weather_UK = pd.Series(ret_all_weather_US, name = 'All_Weather_UK')
    
   
    # 60/40 porfolio 
    ret_60_40 = pd.Series(qs.utils.download_returns('BAGPX'), 
                          name= 'Port_60/40_BlackRock')
    
    
    retun_bench_port = pd.concat([ret_all_weather_US, ret_60_40], axis = 1)
    
    
    
    return retun_bench_port 

def get_benchmark_returns(choose_bench:str = 'S&P500') -> pd.DataFrame:
    
    '''
    Options for benchmarking: 'Nasdaq', 'S&P500', 'World_Index', 'Cash', 
                               'Emerging_Markets','All_Weather_US'
              'US_Real_Estate', 'US_mid_Cap', 'US_small_Cap', 'World_Non_US',
              'US_TIPS', 'US_Bonds', 'Bloomberg_Commodity_Index', 
              'Port_60/40_BlackRock'
        
    
    '''
    # options 
    tickers = [ '^IXIC', '^GSPC' , 'VT', 'BIL', 'EEM', 'VNQ',
               'MDY', 'SLY', 'EFA', 'TIP', 'AGG', 'DJP']
    
    names  = ['Nasdaq', 'S&P500', 'World_Index', 'Cash', 'Emerging_Markets',
              'US_Real_Estate', 'US_mid_Cap', 'US_small_Cap', 'World_Non_US',
              'US_TIPS', 'US_Bonds', 'Bloomberg_Commodity_Index']
    
    ret_bench  = list()
    for tick, names in zip(tickers, names):
        ret = pd.Series(qs.utils.download_returns(tick), name= names)
        ret_bench.append(ret)
        
    ret_bench = pd.concat(ret_bench, axis = 1)
    
    retun_bench_port =  get_benchmark_porfolios()
    
    ret_bench = pd.concat([ret_bench, retun_bench_port], axis = 1)
    
    if choose_bench:
        ret_bench = ret_bench[[choose_bench]]

    return ret_bench
 

def download_returns(yahoo_tickers:list) -> pd.DataFrame:
      
    ret_bench  = list()
    for tick, names in zip(yahoo_tickers, yahoo_tickers):
        ret = pd.Series(qs.utils.download_returns(tick), name= names)
        ret_bench.append(ret)
        
    ret_bench = pd.concat(ret_bench, axis = 1)
    
    return ret_bench


def  get_multi_asset_returns() -> pd.DataFrame:
     
    # - EEM  # iShares Emerging Markets - Emering Markets  
    # - VNQ  # Vangaurd Real Estate  - Real Estate          
    # - MDY  # SPDR S&P MIDCAP 400 ETF Trust (MDY) - mid cap 
    # - SLY  # SPDR S&P 600 Small Cap ETF (SLY) - small cap 
    # - SPY  # S&P 500   - large cap 
    # - EFA  # International Stocks iShares MSCI EAFE ETF (EFA) 
    # - TIP  # iShares TIPS Bond ETF (TIP) - TIPS
    # - AGG  # iShares Core U.S. Aggregate Bond ETF (AGG) - Bonds 
    # - DJP  # iPath Bloomberg Commodity Index Total Return(SM) ETN (DJP) - Commodities 
    # - BIL  # SPDR Bloomberg Barclays 1-3 Month T-Bill ETF (BIL)         - Cash 


    multi_asset_tickers = ['EEM', 
                           'VNQ', 'MDY', 'SLY', 
                           'SPY', 'EFA', 
                           'TIP', 
                           'AGG', 
                           'DJP', 'BIL']
    
    df_multi_asset = download_returns(multi_asset_tickers)

    df_multi_asset = df_multi_asset.dropna()
    
    # naive or equal weight portfolio allocation 
    df_naive = myPortfolioOptimisation.equal_weight_portfolio(df_multi_asset)
     
    
    port_returns = calculate_portfolio_returns(df_multi_asset, 
                                    myassets_list = multi_asset_tickers,
                                    myweights_list = df_naive.port_naive.to_list(),
                                    portfolio_name = 'port_multi_asset')
    return port_returns
    