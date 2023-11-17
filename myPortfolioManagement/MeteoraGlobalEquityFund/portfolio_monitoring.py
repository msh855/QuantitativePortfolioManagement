import warnings
from openbb_terminal.sdk import openbb
import ffn

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myPerformanceMetrics import performance_overview, get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs
import requests

# Load API
myAPI = "4666063ZUKaNsJbECbaOGawwQgMdesMbKGwG"
account_id = '4666063'
headers = {"Authorization": myAPI}

url_my_portfolio = "https://live.trading212.com/api/v0/equity/portfolio"
url_fetch_all_pies = "https://live.trading212.com/api/v0/equity/pies"

response = requests.get(url_my_portfolio, headers=headers)
data = response.json()
df_212 = pd.DataFrame(data)
df_212['Date_of_Purchase'] = pd.to_datetime(df_212['initialFillDate']).dt.date
df_212 = df_212.drop(['frontend', 'maxBuy', 'maxSell', 'pieQuantity', 'initialFillDate'], axis=1)

inv_212 = [x.split('_') for x in df_212['ticker']]
tickers_raw = []
for i in range(0, len(inv_212)):
    tick = [x.split('_') for x in df_212['ticker']][i][0]
    tickers_raw.append(tick)

df_212['tickers_212'] = tickers_raw

# load data
# ======================================================================================================================
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

# download prices per date of purchase
# ====================================
df_port_main_info = df_port_main_info.merge(df_212[['Date_of_Purchase', 'tickers_212']])
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

prices = pd.concat(prices_list)
prices['price_fx_adj'] = prices['adjclose'] * prices['Spot']

# investment performance
# ======================
prices_temp = prices.pivot(columns='yahooTicker', values='adjclose')
prices_temp2 = prices.pivot(columns='yahooTicker', values='price_fx_adj')

df_per_no_fx = performance_overview(prices_temp, prices=True, short=True)
df_per_fx_adj = performance_overview(prices_temp2, prices=True, short=True)

df_per_fx_adj['total_return'].name = 'total_return_fxadj'
pd.concat([df_per_fx_adj['total_return'], df_per_no_fx['total_return']], axis=1)
df_per_fx_adj['total_return'].sort_values().plot.bar()

prices = prices.rename(columns={'yahooTicker': ticker_col_name})

df_prices = prices.pivot_table(index='Date',
                               columns=ticker_col_name,
                               values='price_fx_adj')

if str(df_prices.index[0]) <= '2021-01-01':
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

# calculate benchmark
# ===================
ret_bench = get_stock_prices(yahoo_tickers=['VWRL.L'], wide_format=True).pct_change()
bench_name = 'Vanguard Global'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

# backtest
qs.reports.html(ret_all['myPortfolio'], benchmark=ret_all[bench_name], output='/Users/safishajjouz/Downloads',
                download_filename='port_perf.html', eoy=True)

price_index = ffn.core.to_price_index(ret_all, start=1)
price_index = ffn.core.rebase(price_index, value=1)
price_index.plot()

df_main_sta = get_main_stats(df_prices, smart=True).sort_values(by=['total_return'], ascending=False)

df_main_sta[['total_return']].plot.bar()

performance_overview(price_index, prices=True, short=True).transpose().dropna()
