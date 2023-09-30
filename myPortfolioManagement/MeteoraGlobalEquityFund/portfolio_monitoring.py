import pandas as pd
import os
from myPortfolioManagement.myPortfolioOptimisation import inverse_vol_portfolio
import warnings
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs

# import tickers
# ==============
working_directory = os.getcwd()
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
path = os.path.join(working_directory, file_path)
df_tickers = pd.read_excel(path)
index_name = df_tickers.columns[0]

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "1995-01-01"
base_currency = 'GBP'

# download prices
df_prices = get_stock_prices(yahoo_tickers=tickers, wide_format=True)
ret = df_prices.pct_change()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_tickers[['YahooTicker', 'adj_weight_GBP']])
ret_port = ret_port[ret_port.index >= '2023-07-13']

# calculate benchmark
# ===================
ret_bench = get_stock_prices(yahoo_tickers=['VWRL.L'], wide_format=True).pct_change()
bench_name = 'Vanguard Global'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

# backtest
qs.reports.html(ret_all['myPortfolio'], benchmark=ret_all[bench_name], output='/Users/safishajjouz/Downloads',
                download_filename='port_perf.html', eoy=True)

performance_overview(ret_all, prices=False, short=False).transpose()
