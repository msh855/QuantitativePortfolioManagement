import numpy as np
import pandas as pd
import os
from myPortfolioManagement.myPortfolioOptimisation import inverse_vol_portfolio
import warnings
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs
from myPortfolioManagement.Macroeconomics.getdata import get_FX_spots
from openbb_terminal.sdk import openbb

# import tickers
# ==============
working_directory = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data'
os.chdir(working_directory)

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
df_foreign_stocks = df_port_main_info[df_port_main_info['Currency'] != 'GBP'][['YahooTicker', 'Currency', 'Company']]
currencies = df_foreign_stocks['Currency'].drop_duplicates()
base_currency = 'GBP'
df_fx = get_FX_spots(currencies, base_currency=base_currency)

start_date = "1995-01-01"
df_temp2 = df_port_main_info[[ticker_col_name, 'Currency']]

tickers = list(df_port_main_info.YahooTicker)
data_list = []
for ticker in tickers:
    data = openbb.stocks.load(ticker, start_date=start_date)
    data[ticker_col_name] = ticker
    data = data.reset_index()
    data = data.merge(df_temp2)
    if data['Currency'].drop_duplicates()[0] != base_currency:
        fx_temp = openbb.forex.load(to_symbol=base_currency, from_symbol=data['Currency'].drop_duplicates()[0],
                                    start_date=start_date)
        fx_temp = fx_temp[['Adj Close']]
        fx_temp.columns = ['Spot']
        fx_temp['FX'] = data['Currency'].drop_duplicates()[0] + base_currency
        data = data.set_index('date').join(fx_temp)
    else:
        data['Spot'] = 1
        data['FX'] = base_currency + base_currency
        data = data.set_index('date')

    data_list.append(data)

df_prices = pd.concat(data_list)
df_prices = df_prices[['Adj Close', ticker_col_name, 'Currency', 'Spot', 'FX']]
df_prices['adj_close_' + base_currency] = df_prices['Adj Close'] * df_prices['Spot']

df_prices = df_prices.pivot_table(index='date',
                                  columns=ticker_col_name,
                                  values='adj_close_' + base_currency)

wrong_value1 = df_prices[(df_prices.index=='2016-10-28')].iloc[:,0][0]
wrong_value2 = df_prices[(df_prices.index=='2020-04-27')].iloc[:,0][0]

df_prices['9984.T'] = np.where(df_prices['9984.T']==wrong_value1, wrong_value1 /100, df_prices['9984.T'] )
df_prices['9984.T'] = np.where(df_prices['9984.T']==wrong_value2, wrong_value2 /100, df_prices['9984.T'] )

df_prices[['9984.T']].dropna().plot()

# download prices
# ======================================================================================================================
tickers = list(df_port_main_info.YahooTicker)
companies = list(df_port_main_info['Company'])
index_name = df_port_main_info.columns[0]

ret = df_prices.pct_change().dropna()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_port_main_info[['YahooTicker', 'adj_weight_GBP']])
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
