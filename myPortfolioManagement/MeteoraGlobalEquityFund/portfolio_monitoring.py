import warnings

warnings.filterwarnings('ignore')

from openbb_terminal.sdk import openbb
import ffn

import pandas as pd
import os
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.MeteoraGlobalEquityFund.Trade212_Account.get_account_info import get_pie_details, get_pies, \
    get_portfolio_info
import quantstats as qs

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
# [
df_pie_details = get_pie_details(id='1547833')  # global Meteora
df_trade212_all_stocks = get_portfolio_info()
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]

# import tickers
# ======================================================================================================================
working_directory = ('/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/'
                     'MeteoraGlobalEquityFund/Data')

# get currency info
# ======================================================================================================================
file1 = os.path.join(working_directory, 'df_prices.csv')
df_currency_info = pd.read_csv(file1)
df_currency_info = df_currency_info[['YahooTicker', 'Currency']].drop_duplicates()
df_currency_info['Currency'] = [x.upper() for x in df_currency_info['Currency']]

file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)
df_port_main_info = df_currency_info.merge(df_port_info)

# download prices per date of purchase
# ====================================
df_port_main_info = df_port_main_info.merge(df_trade212_all_stocks[['Date_of_Purchase', 'tickers_212']])
df_port_main_info = df_port_main_info.merge(df_pie_details)

ticker_col_name = df_port_main_info.columns[0]

# identify Non GBP stocks and download FX
# ======================================================================================================================
base_currency = 'GBP'
prices_list = []
for item in zip(df_port_main_info[ticker_col_name], df_port_main_info['Date_of_Purchase'],
                df_port_main_info['Currency']):

    data = get_stock_prices(yahoo_tickers=[item[0]], start_date=str(item[1]), fix_data=True)

    if item[2] != base_currency:
        fx_temp = openbb.forex.load(to_symbol=base_currency, from_symbol=item[2],
                                    start_date=str(item[1]))
        fx_temp.index.name = 'Date'
        fx_temp = fx_temp[['Adj Close']]
        fx_temp.columns = ['Spot']

        data = data.join(fx_temp)
    else:
        data['Spot'] = 1

    prices_list.append(data)

df_prices = pd.concat(prices_list)
df_prices['price_fx_adj'] = df_prices['adjclose'] * df_prices['Spot']
df_prices = df_prices.pivot(columns='yahooTicker', values='price_fx_adj')

# Backtest
# ======================================================================================================================
ret = df_prices.pct_change().dropna()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_port_main_info[['YahooTicker', 'expectedShare']])

# calculate benchmark
# ===================
ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

price_index = ffn.core.to_price_index(ret_all, start=1)
price_index = ffn.core.rebase(price_index, value=1)
price_index.plot()

get_main_stats(price_index).transpose()

# backtest
qs.reports.html(ret_all['myPortfolio'], benchmark=ret_all[bench_name], output='/Users/safishajjouz/Downloads',
                download_filename='port_perf.html', eoy=True)

qs.reports.basic(ret_all['myPortfolio'], benchmark=ret_all[bench_name], rf=0.05)
