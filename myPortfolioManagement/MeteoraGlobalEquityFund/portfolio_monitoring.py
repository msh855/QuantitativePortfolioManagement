import warnings

warnings.filterwarnings('ignore')

import numpy as np
import ffn
import pandas as pd
import os
from myPortfolioManagement.myData import get_stock_prices, get_stock_info, get_nasdaq_tickers, get_sp500_tickers
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.Trade212_Account.get_account_info import get_pie_details, get_pies, \
    get_portfolio_info
import quantstats as qs
from myPortfolioManagement.myClustering import cluster_ftca
from myPortfolioManagement.myTimeSeries import decomposeTS
from myPortfolioManagement.myPlots import scatter_plot_simple
from myPortfolioManagement.myDataCleaning import cap_outliersTS

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
df_pie_details = get_pie_details(id='1547833')  # global Meteora
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]  #

# all stocks
df_trade212_all_stocks = get_portfolio_info()

df_trade212_all_stocks['value_initial'] = df_trade212_all_stocks['quantity'] * df_trade212_all_stocks['averagePrice']
df_trade212_all_stocks.loc[df_trade212_all_stocks['Country'] == 'UK', 'value_initial'] = (df_trade212_all_stocks.loc[
                                                                                              df_trade212_all_stocks[
                                                                                                  'Country'] == 'UK',
                                                                                              'value_initial'] / 1000)

# import tickers
# ======================================================================================================================
working_directory = ('/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/'
                     'MeteoraGlobalEquityFund/Data')

file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)

# initially designed portfolio
df_weights_int = df_port_info[['yahooTicker', 'adj_weight_GBP']]

# download prices per date of purchase
# ====================================
df_port_main_info = df_trade212_all_stocks.merge(df_port_info)
df_port_main_info = df_port_main_info.merge(df_pie_details)

ticker_col_name = df_port_main_info.columns[0]
tickers = df_port_main_info[ticker_col_name]
tickers_int_port = list(df_weights_int['yahooTicker'])

# get weights
df_st = get_stock_info(tickers)
df_st_info_all = df_st.merge(df_port_main_info[[ticker_col_name, 'expectedShare']], on=ticker_col_name)
df_weights = df_st_info_all[['longName', 'expectedShare']]

# get weights initial port
df_st_initial = get_stock_info(tickers_int_port)
df_st_info_all = df_st_initial.merge(df_weights_int[[ticker_col_name, 'adj_weight_GBP']], on=ticker_col_name)
df_weights_int = df_st_info_all[['longName', 'adj_weight_GBP']]

# get prices
df_prices = get_stock_prices(yahoo_tickers=tickers, adj_fx=True)

df_prices_in = get_stock_prices(yahoo_tickers=tickers_int_port, adj_fx=True)

ret_softbank = df_prices_in[['SoftBank Group Corp.']].pct_change().dropna()
price_soft = cap_outliersTS(ret_softbank)
df_prices_in = df_prices_in.drop(['SoftBank Group Corp.'], axis=1)
df_prices_in = df_prices_in.join(price_soft)

# Backtest
# ======================================================================================================================
ret = df_prices.pct_change().dropna()
df_clusters = cluster_ftca(ret, col_name='longName')
df_clusters = df_clusters.merge(df_st_info_all[['industry', 'sector', 'longName', 'expectedShare']])
df_clusters[['sector', 'expectedShare']].groupby(['sector']).sum().sort_values('expectedShare').plot.barh()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_weights)
ret_port_in = calculate_portfolio_returns(returns=df_prices_in.pct_change().dropna(), myweights=df_weights_int,
                                          portfolio_name='myPortfolio_int')

# ======================================================================================================================
# return attribution
# ======================================================================================================================

df_prices_long = df_prices.melt(ignore_index=False, var_name='longName', value_name='Prices')
prt_value = calculate_portfolio_returns(df_prices.dropna(), df_weights, portfolio_name='PortValue')

ret_long = ret.melt(ignore_index=False, var_name='longName', value_name='ret')
ret_long = ret_long.reset_index().merge(df_prices_long.reset_index(), on=['Date', 'longName'])

ret_long = ret_long.merge(df_weights, on='longName')
ret_long = ret_long.merge(ret_port.reset_index(), on='Date')
ret_long = ret_long.merge(prt_value.reset_index(), on='Date')

ret_long['cont1'] = ret_long['expectedShare'] * ret_long['ret']

ret_long['cont2'] = ret_long.groupby('longName').apply(
    lambda x: x['Prices'].shift() / x['Price_port'].shift()).reset_index(drop=True)

ret_long['cont_total'] = ret_long['cont1'] * ret_long['cont2']
ret_long = ret_long.sort_values(by=['longName', 'Date'])
ret_long.dropna(inplace=True)

ret_wide = ret_long.pivot(index='Date', columns='longName', values='cont_total')
ret_wide['total_port_ret'] = ret_wide.sum(axis=1)
ret_wide = ret_wide.join(ret_port)
ret_wide[['myPortfolio', 'total_port_ret']]

# calculate benchmark
# ======================================================================================================================
ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]
ret_all = pd.concat([ret_port_in, ret_port, ret_bench], axis=1).dropna()
# ret_all = ret_port.join(ret_bench).dropna()

price_index = ffn.core.to_price_index(ret_all, start=1)
price_index = ffn.core.rebase(price_index, value=1)
price_index.plot()

port_stats = get_main_stats(ret_all).transpose()

get_main_stats(ret_wide[['myPortfolio', 'total_port_ret']])

# backtest
qs.reports.basic(ret_all['myPortfolio'], benchmark=ret_all[bench_name], rf=0.05)

qs.stats.monthly_returns(returns=ret_all, eoy=False).plot.bar()

# ======================================================================================================================
# get the number of stocks outperformed index
# ======================================================================================================================

# get info about the date of purchase
df_info = df_port_main_info[['yahooTicker', 'Company', 'Date_of_Purchase']]
df_info.groupby('Date_of_Purchase').count()[['Company']].plot.bar()

# merge my tickers with S&P
mytickers = list(tickers) + ['^GSPC', '^IXIC']
df_pr = get_stock_prices(yahoo_tickers=tickers_int_port + ['^GSPC', '^IXIC'], adj_fx=True)

min_date = pd.to_datetime(df_info['Date_of_Purchase']).min()
df_pr = df_pr[df_pr.index >= min_date]
df_pr = df_pr[:-1]

# get total returns
ret = df_pr.pct_change()
total_returns = pd.Series(qs.stats.comp(ret), name='total_ret')

# get performance since date of purchase of S&P and Nasdaq
total_ret_sp = total_returns['S&P 500']
total_ret_nsdaq = total_returns['NASDAQ Composite']

# performance of my stocks since the date of purchase
tot_ret_mystocks = total_returns.drop(['S&P 500', 'NASDAQ Composite'])

# find the stock that beat index
mystocks_beat_sp = tot_ret_mystocks[tot_ret_mystocks > total_ret_sp]
mystocks_beat_nsdq = tot_ret_mystocks[tot_ret_mystocks > total_ret_nsdaq]

# calcuate the fraction of them
perc_of_stocks_beating_sp = round(mystocks_beat_sp.count() / tot_ret_mystocks.count(), 2)
perc_of_stocks_beating_nsdq = round(mystocks_beat_nsdq.count() / tot_ret_mystocks.count(), 2)

# below average performers
tot_ret_mystocks[(tot_ret_mystocks > 0) & (tot_ret_mystocks <= total_ret_sp)]
tot_ret_mystocks[(tot_ret_mystocks > 0) & (tot_ret_mystocks <= total_ret_sp)].count() / len(mytickers)

# losers
tot_ret_mystocks[tot_ret_mystocks < 0]
tot_ret_mystocks[tot_ret_mystocks < 0].count() / len(mytickers)

# find the percentage of stocks within S&P that performed better than the index
df_sp_companies = get_sp500_tickers()
df_sp_companies = df_sp_companies.drop_duplicates('Company')

# get prices for S&P companies
sp_tickers = list(df_sp_companies['Ticker'])
df_sp_comp_prices = get_stock_prices(yahoo_tickers=sp_tickers, adj_fx=True)
df_sp_comp_prices = df_sp_comp_prices[df_sp_comp_prices.index >= min_date]
df_sp_comp_prices = df_sp_comp_prices[:-1]

# calculate the number of companies beat S&P
ret_sp = df_sp_comp_prices.pct_change()
total_returns_sp_comp = pd.Series(qs.stats.comp(ret_sp), name='total_ret')

sp_com_beat_sp = total_returns_sp_comp[total_returns_sp_comp > total_ret_sp]
sp_com_beat_sp.count() / len(ret_sp.columns)

common_elements = list(set(mytickers) & set(sp_tickers))
len(common_elements)

# overall portfolio
pd.Series(qs.stats.comp(ret_all[ret_all.index >= min_date]), name='total_ret')
pd.Series(qs.stats.sharpe(ret_all[ret_all.index >= min_date], rf=0.05, smart=True), name='total_ret')

# ======================================================================================================================
# get the number of stocks outperformed index
# ======================================================================================================================
df_attr = pd.DataFrame(tot_ret_mystocks)
df_attr.sort_values(by='total_ret', inplace=True)
colors = ['r' if m < 0 else 'g' for m in df_attr['total_ret']]
df_attr.plot.barh(y='total_ret', color=colors)

df_info_temp = get_stock_info(tickers)
df_info_atr = df_info_temp[['longName', 'yahooTicker']].set_index('longName')
df_info_atr = df_info_atr.join(df_attr)

# merge
df_info_atr = df_info_atr.reset_index().merge(df_port_main_info[['yahooTicker', 'expectedShare']], on='yahooTicker')
df_info_atr['ret_cont'] = df_info_atr['total_ret'] * df_info_atr['expectedShare']
df_info_atr['ret_cont_prct'] = (df_info_atr['ret_cont'] / df_info_atr['ret_cont'].sum()) * 100

# final plot
df_info_atr_plot = df_info_atr[['longName', 'ret_cont_prct']].set_index('longName')
df_info_atr_plot.sort_values(by='ret_cont_prct', inplace=True)
colors = ['r' if m < 0 else 'g' for m in df_info_atr_plot.ret_cont_prct]
df_info_atr_plot.plot.barh(y='ret_cont_prct', color=colors)

df_info_atr_plot.tail(10).sum()

scatter_plot_simple(df_info_atr.set_index('yahooTicker'), x='expectedShare', y='ret_cont_prct')
scatter_plot_simple(df_info_atr.set_index('yahooTicker'), x='expectedShare', y='total_ret')
