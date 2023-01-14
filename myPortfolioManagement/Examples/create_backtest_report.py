#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:24:05 2022

@author: safishajjouz
"""

cd /Users/safishajjouz/GitHub/myPythonPackages

from myPortfolioManagement.myData import * 
from myPortfolioManagement.myPortfolioOptimisation import *
from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myPortfolioSelection import *
from myPortfolioManagement.myBacktesting import *
from myPortfolioManagement.myReturns import *
from myPortfolioManagement.myReports import metrics 

# returns 
port = get_multi_asset_returns()
bench1 = download_returns(['^IXIC'])
bench2 = download_returns(['^GSPC'])

stock = download_returns(['SMT.L'])
stock = stock[stock.index>='2005']


Nasdaq = get_stock_prices(['^IXIC'])
S_P = get_stock_prices(['^GSPC'])
SP = SP500[['adjclose']]

fan_chart(returns = bench2,  
               weight_period = ['2008-01-01', '2020-01-01'], 
               out_of_sample_date = '2020-01-01', 
               n_sample = 10000,
               starting_value = 1,
               chart_title = 'S&P Bootstrapping returns: 1950 -2020' )


port = get_multi_asset_returns()
bench = download_returns(['^IXIC'])
bench2 = download_returns(['^GSPC'])
stock = download_returns(['SMT.L'])
stock = stock[stock.index>='2005']

backtest_report(returns=bench2)

multi_asset_tickers = ['EEM', 'VNQ', 'MDY', 'SLY', 
                       'SPY', 'EFA', 
                       'TIP', 
                       'AGG', 
                       'DJP', 'BIL']

df_multi_asset = download_returns(multi_asset_tickers)


tear_sheet_pyfolio(returns = bench)


# performance stats 
import quantstats as qs
qs.reports.full(port['port_multi_asset'], '^IXIC')

import ffn 
price_index = ffn.core.to_price_index(port, start=100) 
perf = price_index.calc_stats()

perf[0].display_monthly_returns()

df_monthly_returns = qs.stats.monthly_returns(port)

df_monthly_returns.style.background_gradient(cmap='Blues')



information_ratio(df_multi_asset, bench).transpose()

beta_Co_Moments_table(port, bench.squeeze()).sort_values('BetaCoVariance')

probabilistic_sharpe_ratio(port, returns_bench=bench, sr_benchmark=False)




def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """

    probab_sr = ss.norm.cdf(((observed_sr - benchmark_sr) * (number_of_returns - 1) ** (1 / 2)) / \
                            (1 - skewness_of_returns * observed_sr +
                             (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) ** (1 / 2))

    return probab_sr



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 08:16:09 2022

@author: safishajjouz
"""



file = '/Users/safishajjouz/GitHub/myPortfolioManagement/BBG_CommodityIndex.xlsx'

df_com_prices = pd.read_excel(file, sheet_name='USD_daily', skiprows = 6)
df_com_prices = df_com_prices.drop(['PX_VOLUME'], axis =1)
df_com_prices = df_com_prices.set_index('Date')
df_com_prices = df_com_prices.reindex(index=df_com_prices.index[::-1])
ret_comm = calculate_returns(df_com_prices)


temp1 = bootstrap_portfolio_performance(returns = ret_comm['PX_LAST'], 
                                returns_benchmark = None,
                                periods = 252,
                                rf = 0.02, 
                                out_of_sample_date = '2020-01-01',  
                                n_sim = 100)






ret_yearly = convert_returns_freq(ret_sp[['^GSPC']], convert_to = 'monthly')
ret_bench_yearly = convert_returns_freq(ret_sp[['VGLT']], convert_to = 'yearly')

ret_yearly.rolling(5).mean()

d = {'col1':ret_yearly.rolling(5).mean()}
df_roll = pd.DataFrame(data = d)

beating_probability(returns = ret_yearly.rolling(5).mean(),
                    returns_benchmark = ret_bench_yearly.rolling(5).mean(), 
                    n_sample=10000)


inf = download_fred_data(fred_sumbol = ['CPALTT01USM657N'], freq = ['m', 'q', 'a'][0], 
                       my_fred_API = 'cc628b51e21828ae6b98c06f4eef6714')

inf
ret , inf = balance_dates(ret_yearly, inf['CPALTT01USM657N'].dropna())

ret_comm = convert_returns_freq(ret_comm[['PX_LAST']], convert_to = 'monthly')


beta_Co_Moments_table(ret_comm*100, inf['CPALTT01USM657N']).sort_values('BetaCoVariance')


import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
 
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
 
start, end = datetime.datetime(2009, 12, 30), datetime.datetime(2016, 12, 31)
 
tickers = ["TLT", "HYG", "^GSPC", "^STOXX50E", "^N225"]
 
universe = pd.DataFrame([web.DataReader(t, 'yahoo', start, end).loc[:, 'Adj Close'] for t in tickers], index=tickers).T.fillna(method='ffill')
universe = universe/universe.iloc[0,:]-1

equally_weights = pd.DataFrame(1./len(universe.columns), index=universe.index, columns=universe.columns)
monkey_weights = pd.DataFrame(np.random.uniform(low=0, high=1, size=universe.shape),index=universe.index, columns=universe.columns)
monkey_weights = monkey_weights/pd.DataFrame(monkey_weights.sum(1).repeat(len(tickers)).values.reshape(universe.shape), index=universe.index, columns=universe.columns)
 
returns = (1+universe).pct_change(periods=1).fillna(0)
 
benchmark_returns = (returns*equally_weights).sum(1)
benchmark = (1+benchmark_returns).cumprod()-1
 
portfolio_returns = (returns*monkey_weights).sum(1)
portfolio = (1+portfolio_returns).cumprod()-1

benchmark_bootstrapping = (1+pd.DataFrame([random.sample(list(benchmark_returns.values), 261) for i in range(1000)]).T.shift(1).fillna(0)).cumprod()-1
portfolio_bootstrapping = (1+pd.DataFrame([random.sample(list(portfolio_returns.values), 261) for i in range(1000)]).T.shift(1).fillna(0)).cumprod()-1

benchmark_bootstrapping_prc = benchmark_bootstrapping.quantile([0.05, 0.50, 0.95], axis=1).T
 
portfolio_bootstrapping_prc = portfolio_bootstrapping.quantile([0.05, 0.50, 0.95], axis=1).T


portfolio_bootstrapping_prc.plot()
benchmark_bootstrapping_prc.plot()
