import numpy
import pandas as pd
import os
from openbb_terminal.sdk import openbb
import warnings
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio, equal_weight_portfolio
from pypfopt.expected_returns import prices_from_returns
import quantstats as qs
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
start_date = "1995-01-01"

df_prices = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data['Ticker'] = ticker
    data['Company'] = company
    df_prices.append(data)

df_prices = pd.concat(df_prices)

prices = df_prices[['Adj Close', 'Ticker']]
prices = prices.pivot(columns='Ticker', values='Adj Close')

# Optimase Portfolio
# ===================
returns = prices.pct_change()
returns_training = returns
df_weights = df_tickers[[index_name, 'weight', 'adj_weight']]

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
bench_ret = []

for tick in bench_ticker:
    price_sp = openbb.stocks.load(tick, start_date=start_date)
    ret_sp = price_sp['Adj Close'].pct_change()
    ret_sp.name = tick
    ret_sp = pd.DataFrame(ret_sp)
    bench_ret.append(ret_sp)

ret_sp = pd.concat(bench_ret, axis=1)

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

# check
portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

index_valuation = (portf_prices.tail(1) - portf_prices.max()) / portf_prices.max()
index_valuation = index_valuation.transpose()
index_valuation.columns = ['Undervalued']

index_valuation.sort_values(by='Undervalued').plot.barh()

df_incex = []
for tick in ['SCHX', 'TSLA']:
    price_sp = openbb.stocks.load(tick, start_date=start_date)
    price_sp['Ticker'] = tick
    price_sp = pd.DataFrame(price_sp)
    df_incex.append(price_sp)

df_inceces = pd.concat(df_incex)
df_inceces = df_inceces.pivot(columns='Ticker', values='Adj Close')
df_inceces.dropna(inplace=True)

df_inceces.to_clipboard()

import numpy as np
def get_hurst_exponent(time_series:pd.Series, max_lag=20):
    """Returns the Hurst Exponent of the time series
       A value above 0.40 denotes some long-term persistance
    """
    time_series = time_series.to_numpy()
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


df_inceces.apply(get_hurst_exponent, result_type='reduce')

get_hurst_exponent(df_inceces)

