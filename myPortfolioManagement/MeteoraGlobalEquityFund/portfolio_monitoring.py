import warnings

import ffn

import pandas as pd
import os
from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.Trade212_Account.get_account_info import get_pie_details, get_pies, \
    get_portfolio_info
import quantstats as qs
from myPortfolioManagement.myClustering import cluster_ftca
from myPortfolioManagement.myTimeSeries import decomposeTS

warnings.filterwarnings('ignore')

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
df_pie_details = get_pie_details(id='1547833')  # global Meteora
df_trade212_all_stocks = get_portfolio_info()
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]

# import tickers
# ======================================================================================================================
working_directory = ('/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/'
                     'MeteoraGlobalEquityFund/Data')

file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)

# download prices per date of purchase
# ====================================
df_port_main_info = df_port_info.merge(df_trade212_all_stocks[['Date_of_Purchase', 'tickers_212']])
df_port_main_info = df_port_main_info.merge(df_pie_details)

ticker_col_name = df_port_main_info.columns[0]
tickers = df_port_main_info[ticker_col_name]

# get weights
df_st = get_stock_info(tickers)
df_st_info_all = df_st.merge(df_port_main_info[[ticker_col_name, 'expectedShare']], on=ticker_col_name)
df_weights = df_st_info_all[['longName', 'expectedShare']]

# get prices
df_prices = get_stock_prices(yahoo_tickers=tickers, adj_fx=True)

# Backtest
# ======================================================================================================================
ret = df_prices.pct_change().dropna()
df_clusters = cluster_ftca(ret, col_name='longName')
df_clusters = df_clusters.merge(df_st_info_all[['industry', 'sector', 'longName', 'expectedShare']])
df_clusters[['cluster', 'expectedShare']].groupby(['cluster']).sum().sort_values('expectedShare').plot.bar()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_weights)

# calculate benchmark
# ===================
ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

price_index = ffn.core.to_price_index(ret_all, start=1)
price_index = ffn.core.rebase(price_index, value=1)
price_index.plot()

port_stats = get_main_stats(ret_all).transpose()

# backtest
qs.reports.basic(ret_all['myPortfolio'], benchmark=ret_all[bench_name], rf=0.05)

qs.stats.monthly_returns(returns=ret_all, eoy=False).plot.bar()
