from myPortfolioManagement.myData import get_stock_prices
import pandas as pd
from myPortfolioManagement.Trade212_Account.get_account_info import get_pie_details, get_pies, \
    get_portfolio_info
import os
from openbb import obb

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
df_pie_details = get_pie_details(id='1789551')  # global Meteora
df_trade212_all_stocks = get_portfolio_info()
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]
tickers212 = df_pie_weights.tickers_212

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
df_port_main_info = df_port_main_info[df_port_main_info['tickers_212'].isin(tickers212)]

# download prices per date of purchase
# ====================================
df_port_main_info = df_port_main_info.merge(df_trade212_all_stocks[['Date_of_Purchase', 'tickers_212']])
df_port_main_info = df_port_main_info.merge(df_pie_details)

tickers = list(df_port_main_info['YahooTicker']) + ['AAPL', 'TSLA', 'NESN.SW', 'NIO', '9984.T']

# load currencies
prices = get_stock_prices(tickers, wide_format=True)
prices_fx_adj = get_stock_prices(tickers, adj_fx = True)


obb.equity.fundamental.overview(symbol="SMT", provider='yfinance').to_df()
