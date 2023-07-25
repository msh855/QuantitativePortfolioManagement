import numpy
import pandas as pd
import os
from openbb_terminal.sdk import openbb
import warnings
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPortfolioOptimisation import inverse_vol_portfolio, equal_weight_portfolio
from pypfopt.expected_returns import prices_from_returns
import quantstats as qs
from myPortfolioManagement.myData import get_stock_prices_from_openBB
import matplotlib
from openbb_terminal.sdk import TerminalStyle

theme = TerminalStyle("light", "light", "light")
warnings.filterwarnings("ignore")

# import tickers
# ==============
working_directory = os.getcwd()
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
path = os.path.join(working_directory, file_path)
mil_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Github\QuantitativePortfolioManagement\myPortfolioManagement\MeteoraGlobalEquityFund\Data\stock_screening.xlsx'
df_tickers = pd.read_excel(path)
index_name = df_tickers.columns[0]

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "2016-01-01"

df_prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date, base_currency='GBP',
                                         index_name=index_name)
prices = df_prices.pivot(columns=index_name, values='Adj_Close_GBP')

# Optimase Portfolio
# ===================
returns_training = prices.pct_change()
df_weights = df_tickers[[index_name, 'adj_weight', 'adj_weight_GBP']]

# optimal weight
# ===============
inv_vol_weights = inverse_vol_portfolio(returns_training=returns_training)  # Inverse Vol
equal_weights = equal_weight_portfolio(returns_training)  # Equally Waighted

df_port_weights = pd.concat([inv_vol_weights, equal_weights], axis=1)
df_port_weights.index.name = index_name
df_port_weights = pd.concat([df_port_weights, df_weights.set_index(index_name)], axis=1)

df_port_weights.sum(axis=0)

# portfolio returns
# =================
df_portfolios = []
for col in df_port_weights.columns:
    temp_portf = calculate_portfolio_returns(returns_training, df_port_weights[[col]].reset_index(), portfolio_name=col)
    df_portfolios.append(temp_portf)

df_portfolios = pd.concat(df_portfolios, axis=1)

# benchmark
bench_ticker = ['SCHX', '^GSPC']
df_prices_bench = get_stock_prices_from_openBB(yahoo_tickers=bench_ticker, start_date=start_date, base_currency='GBP',
                                               index_name=index_name)
prices_bench = df_prices_bench.pivot(columns=index_name, values='Adj_Close_GBP')
ret_sp = prices_bench.pct_change()

# total returns
index_name = 'date'
ret = ret_sp.reset_index(index_name).merge(df_portfolios.reset_index(index_name))
ret = ret.set_index(index_name)

ret_backtest = ret[ret.index >= '2012-01-01']
portf_prices = prices_from_returns(ret_backtest)
portf_prices.plot()

port_names = ret.columns[2:7]
out_put = os.path.join(working_directory, 'myPortfolioManagement/MeteoraGlobalEquityFund/Output')

for pot_name in port_names:
    qs.reports.html(ret_backtest[pot_name], ret_backtest[bench_ticker[1]],
                    output=out_put,
                    download_filename=pot_name + '.html')


qs.reports.html(ret_backtest['adj_weight_GBP'], ret_backtest['port_inverse_vol'],
                    output=out_put,
                    download_filename='adj_weight_vs_inv_vol' + '.html')

# check
portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

index_valuation = (portf_prices.tail(1) - portf_prices.max()) / portf_prices.max()
index_valuation = index_valuation.transpose()
index_valuation.columns = ['Undervalued']

index_valuation.sort_values(by='Undervalued').plot.barh()

inv_vol_weights.to_clipboard()