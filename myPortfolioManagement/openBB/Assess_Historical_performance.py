from myPortfolioManagement.myData import get_stock_prices, get_stock_mini_info
import os
import pandas as pd
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import ffn

file_path = '/Users/safishajjouz/GitHub/myFinances'
file_name = 'portfoliols_eval_all.xlsx'
file_to_load = os.path.join(file_path, file_name)
portfolio_list = []
for i in range(4):
    prt_temp = pd.read_excel(file_to_load, sheet_name=i)
    prt_temp['porfolio'] = 'portfolio' + str(i+1)
    portfolio_list.append(prt_temp)

# Initial portfolio
start_trading_date = '2020-11-24'

portfolios = pd.concat(portfolio_list)

df_port = portfolio_list[0]
tickers = list(df_port['yahooTicker'])
df_info = get_stock_mini_info(tickers)
df_info = df_port.merge(df_info[['yahooTicker', 'longName']])

# load currencies
prices = get_stock_prices(tickers, start_date = start_trading_date,  wide_format=True)
ret = prices.pct_change().dropna()

df_weights = df_info[['longName', 'weight']]

ret_port = calculate_portfolio_returns(ret, df_weights, portfolio_name='portfolio1')
ret_port.cumsum().plot()



prices_nomalised = ffn.rebase(prices.dropna(), value=100)
prices_nomalised.plot()