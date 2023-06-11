import pandas as pd
import os
from myPortfolioManagement.myPortfolioOptimisation import inverse_vol_portfolio
import warnings
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myData import get_stock_prices_from_openBB

from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs

warnings.filterwarnings("ignore")

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

df_prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date="2019-01-01", base_currency='GBP',
                                    index_name='YahooTicker')

prices_USD = df_prices.pivot(columns='YahooTicker', values='Adj_Close_USD')
prices_base = df_prices.pivot(columns='YahooTicker', values='Adj_Close_GBP')

prices_USD_temp = prices_USD.copy()
prices_base_temp = prices_base.copy()

prices_USD_temp['Currency'] = 'USD'
prices_base_temp['Currency'] = base_currency

prices = pd.concat([prices_USD_temp, prices_base_temp])
df_main_stats_USD = get_main_stats(prices[prices['Currency'] == 'USD'].drop(['Currency'], axis=1))
df_main_stats_base = get_main_stats(prices[prices['Currency'] == base_currency].drop(['Currency'], axis=1))

df_main_stats_USD.columns = [x + '_USD' for x in df_main_stats_USD.columns]
df_main_stats_base.columns = [x + '_GBP' for x in df_main_stats_base.columns]

df_main_stats = df_main_stats_USD.join(df_main_stats_base)

df_non_USD_stocks = df_prices[df_prices.Currency != 'USD']
non_USD_stocks = df_non_USD_stocks['YahooTicker'].unique()

df_main_stats_non_usd_stocks = df_main_stats[df_main_stats.index.isin(non_USD_stocks)]
df_main_stats_non_usd_stocks.filter(regex='cagr').plot.bar()

ret_USD = prices_USD.pct_change()
ret_LC = prices_base.pct_change()

df_inv_vol_USD = inverse_vol_portfolio(ret_USD, my_assets_col_name='YahooTicker')
df_inv_vol_USD.columns = ['Weights_Inv_USD_ret']
df_inv_vol_LC = inverse_vol_portfolio(ret_LC, my_assets_col_name='YahooTicker')
df_inv_vol_LC.columns = ['Weights_Inv_LC_ret']

df_inv_vol = df_inv_vol_USD.join(df_inv_vol_LC)
df_inv_vol.sort_values(by=['Weights_Inv_USD_ret']).plot.bar()

temp_portf_USD = calculate_portfolio_returns(ret_USD, df_inv_vol_USD.reset_index(), portfolio_name='portf_ret_USD')
temp_portf_LC = calculate_portfolio_returns(ret_LC, df_inv_vol_LC.reset_index(), portfolio_name='portf_ret_LC')

df_portfolios = pd.concat([temp_portf_USD, temp_portf_LC], axis=1)
portf_prices = prices_from_returns(df_portfolios)
portf_prices.plot()

out_put = os.path.join(working_directory, 'myPortfolioManagement/MeteoraGlobalEquityFund/Output')

qs.reports.html(temp_portf_LC.iloc[:,0], temp_portf_USD.iloc[:,0],
                output=out_put,
                download_filename='fx_impact' + '.html')
