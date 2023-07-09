#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:24:05 2022

@author: safishajjouz
"""

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import get_multi_asset_returns
from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myBacktesting import fan_chart, backtest_report, tear_sheet_pyfolio
from myPortfolioManagement.myPerformanceMetrics import probabilistic_sharpe_ratio, information_ratio
from myPortfolioManagement.myReports import metrics
import quantstats as qs
from myPortfolioManagement.myPlots import monthly_heatmap
import ffn

# returns 
port = get_multi_asset_returns()

metrics(port)
monthly_heatmap(port)


bench1 = get_stock_returns(['^IXIC'])
bench2 = get_stock_returns(['^GSPC'])

stock = get_stock_returns(['SMT.L'])
stock = stock[stock.index >= '2005']

Nasdaq = get_stock_prices(['^IXIC'])
S_P = get_stock_prices(['^GSPC'])

fan_chart(returns=bench2,
          weight_period=['2008-01-01', '2020-01-01'],
          out_of_sample_date='2020-01-01',
          n_sample=10000,
          starting_value=1,
          chart_title='S&P Bootstrapping returns: 1950 -2020')

port = get_multi_asset_returns()
bench = get_stock_returns(['^IXIC'])
bench2 = get_stock_returns(['^GSPC'])
stock = get_stock_returns(['SMT.L'])
stock = stock[stock.index >= '2005']

bench2.index = bench2.index.tz_convert(None)

backtest_report(returns=bench2)

multi_asset_tickers = ['EEM', 'VNQ', 'MDY', 'SLY',
                       'SPY', 'EFA',
                       'TIP',
                       'AGG',
                       'DJP', 'BIL']

df_multi_asset = get_stock_returns(multi_asset_tickers)

tear_sheet_pyfolio(returns=bench)

# performance stats
mystock = port['port_multi_asset']
mystock.index = mystock.index.tz_convert(None)
qs.reports.full(mystock, '^IXIC')

price_index = ffn.core.to_price_index(port, start=100)
perf = price_index.calc_stats()

perf[0].display_monthly_returns()

df_monthly_returns = qs.stats.monthly_returns(port)

df_monthly_returns.style.background_gradient(cmap='Blues')

information_ratio(df_multi_asset, bench).transpose()

beta_Co_Moments_table(port, bench.squeeze()).sort_values('BetaCoVariance')

probabilistic_sharpe_ratio(port, returns_bench=bench, sr_benchmark=False)

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 08:16:09 2022

@author: safishajjouz
"""

file = '/Users/safishajjouz/GitHub/myPortfolioManagement/BBG_CommodityIndex.xlsx'

df_com_prices = pd.read_excel(file, sheet_name='USD_daily', skiprows=6)
df_com_prices = df_com_prices.drop(['PX_VOLUME'], axis=1)
df_com_prices = df_com_prices.set_index('Date')
df_com_prices = df_com_prices.reindex(index=df_com_prices.index[::-1])
ret_comm = calculate_returns(df_com_prices)

temp1 = bootstrap_portfolio_performance(returns=ret_comm['PX_LAST'],
                                        returns_benchmark=None,
                                        periods=252,
                                        rf=0.02,
                                        out_of_sample_date='2020-01-01',
                                        n_sim=100)

ret_yearly = convert_returns_freq(ret_sp[['^GSPC']], convert_to='monthly')
ret_bench_yearly = convert_returns_freq(ret_sp[['VGLT']], convert_to='yearly')

ret_yearly.rolling(5).mean()

d = {'col1': ret_yearly.rolling(5).mean()}
df_roll = pd.DataFrame(data=d)

beating_probability(returns=ret_yearly.rolling(5).mean(),
                    returns_benchmark=ret_bench_yearly.rolling(5).mean(),
                    n_sample=10000)

inf = download_fred_data(fred_sumbol=['CPALTT01USM657N'], freq=['m', 'q', 'a'][0],
                         my_fred_API='cc628b51e21828ae6b98c06f4eef6714')

inf
ret, inf = balance_dates(ret_yearly, inf['CPALTT01USM657N'].dropna())

ret_comm = convert_returns_freq(ret_comm[['PX_LAST']], convert_to='monthly')

beta_Co_Moments_table(ret_comm * 100, inf['CPALTT01USM657N']).sort_values('BetaCoVariance')

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

universe = pd.DataFrame([web.DataReader(t, 'yahoo', start, end).loc[:, 'Adj Close'] for t in tickers],
                        index=tickers).T.fillna(method='ffill')
universe = universe / universe.iloc[0, :] - 1

equally_weights = pd.DataFrame(1. / len(universe.columns), index=universe.index, columns=universe.columns)
monkey_weights = pd.DataFrame(np.random.uniform(low=0, high=1, size=universe.shape), index=universe.index,
                              columns=universe.columns)
monkey_weights = monkey_weights / pd.DataFrame(
    monkey_weights.sum(1).repeat(len(tickers)).values.reshape(universe.shape), index=universe.index,
    columns=universe.columns)

returns = (1 + universe).pct_change(periods=1).fillna(0)

benchmark_returns = (returns * equally_weights).sum(1)
benchmark = (1 + benchmark_returns).cumprod() - 1

portfolio_returns = (returns * monkey_weights).sum(1)
portfolio = (1 + portfolio_returns).cumprod() - 1

benchmark_bootstrapping = (1 + pd.DataFrame(
    [random.sample(list(benchmark_returns.values), 261) for i in range(1000)]).T.shift(1).fillna(0)).cumprod() - 1
portfolio_bootstrapping = (1 + pd.DataFrame(
    [random.sample(list(portfolio_returns.values), 261) for i in range(1000)]).T.shift(1).fillna(0)).cumprod() - 1

benchmark_bootstrapping_prc = benchmark_bootstrapping.quantile([0.05, 0.50, 0.95], axis=1).T

portfolio_bootstrapping_prc = portfolio_bootstrapping.quantile([0.05, 0.50, 0.95], axis=1).T

portfolio_bootstrapping_prc.plot()
benchmark_bootstrapping_prc.plot()
