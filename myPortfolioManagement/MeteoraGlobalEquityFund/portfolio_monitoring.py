import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os

from myPortfolioManagement.myData import get_stock_prices, get_stock_prices_fx_adj
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs

#
working_directory = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data'
os.chdir(working_directory)

# import tickers
# ==============
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
file1 = os.path.join(working_directory, 'df_prices.csv')

# get main info
# ======================================================================================================================
df_port_main_info = pd.read_csv(file1)
df_port_main_info = df_port_main_info[['YahooTicker', 'Currency']].drop_duplicates()
df_port_main_info['Currency'] = [x.upper() for x in df_port_main_info['Currency']]
file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)
df_port_main_info = df_port_main_info.merge(df_port_info)

ticker_col_name = df_port_main_info.columns[0]

# identify Non GBP stocks and download FX
# ======================================================================================================================
base_currency = 'GBP'
start_date = "1995-01-01"
df_temp2 = df_port_main_info[[ticker_col_name, 'Currency']]
tickers = list(df_port_main_info.YahooTicker)

df_prices = get_stock_prices_fx_adj(yahoo_tickers=tickers, start_date=start_date, df_currency=df_temp2,
                                    base_currency='GBP',
                                    wide_format=True)

# correct SoftBank Values due to Yahoo Data
wrong_value1 = df_prices[(df_prices.index == '2016-10-28')].iloc[:, 0][0]
wrong_value2 = df_prices[(df_prices.index == '2020-04-27')].iloc[:, 0][0]
df_prices['9984.T'] = np.where(df_prices['9984.T'] == wrong_value1, wrong_value1 / 100, df_prices['9984.T'])
df_prices['9984.T'] = np.where(df_prices['9984.T'] == wrong_value2, wrong_value2 / 100, df_prices['9984.T'])
df_prices[['9984.T']].dropna().plot()

# Backtest
# ======================================================================================================================
ret = df_prices.pct_change().dropna()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_port_main_info[['YahooTicker', 'adj_weight_GBP']])
ret_port = ret_port[ret_port.index >= '2023-07-12']

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
