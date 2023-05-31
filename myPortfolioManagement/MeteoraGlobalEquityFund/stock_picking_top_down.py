import pandas as pd
from openbb_terminal.sdk import openbb
from scipy.stats import zscore
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio, equal_weight_portfolio
from sklearn import preprocessing as pre
from myPortfolioManagement.myPerformanceMetrics import cagr
from pypfopt.expected_returns import prices_from_returns
import numpy as np
import quantstats as qs

import matplotlib
from openbb_terminal.sdk import TerminalStyle
theme = TerminalStyle("light", "light", "light")

matplotlib.use("TkAgg")

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

# load tickers
path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Github\QuantitativePortfolioManagement\myPortfolioManagement\MeteoraGlobalEquityFund\Data\stock_screening.xlsx'
df_tickers = pd.read_excel(path)
# df_tickers.dropna(inplace=True)

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "2005-01-01"

df_prices = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data['Ticker'] = ticker
    data['Company'] = company
    df_prices.append(data)

df_prices = pd.concat(df_prices)

prices_temp = df_prices[['Adj Close', 'Ticker']]
prices_temp = prices_temp.pivot(columns='Ticker', values='Adj Close')

# decompose
# =========

df_decompose_list = []

for tick in tickers:
    temp_decomp = openbb.qa.decompose(data=prices_temp[tick].dropna(), multiplicative=True)
    df_decompose = pd.concat([temp_decomp[1], temp_decomp[2], prices_temp[tick].dropna()], axis=1).dropna()
    df_decompose.columns = ['trend_cycle', 'trend_trend', 'price']
    df_decompose['fitted'] = df_decompose['trend_cycle'] + df_decompose['trend_trend']
    df_decompose['residual'] = df_decompose['price'] - df_decompose['fitted']
    df_decompose['cycle_importance'] = abs(df_decompose['trend_cycle']) / (
            abs(df_decompose['trend_cycle']) + abs(df_decompose['trend_trend']))
    df_decompose['trend_importance'] = abs(df_decompose['trend_trend']) / (
            abs(df_decompose['trend_cycle']) + abs(df_decompose['trend_trend']))
    df_decompose['YahooTicker'] = tick
    df_decompose_list.append(df_decompose)

df_decompose = pd.concat(df_decompose_list)

df_decompose['Undervalued'] = np.where(df_decompose['trend_cycle'] < 0, 'Yes', 'No')
df_decompose['valuation'] = (abs(df_decompose['price']) - abs(df_decompose['trend_trend'])) / abs(
    df_decompose['trend_trend'])

for tick in tickers:
    temp_dec = df_decompose[df_decompose.index >= '2022-01-01']
    temp_dec = temp_dec[temp_dec['YahooTicker'] == tick]
    temp_dec[['trend_importance']].plot(title=tick)

# Count Likely to be overvalued vs Undervalued
df_count = df_decompose.groupby(['YahooTicker', 'Undervalued']).count()
df_count['count'] = df_count['trend_cycle']

df_count = df_count.reset_index().pivot(index='YahooTicker', columns='Undervalued', values='count')
df_count['Undervalued_Chances'] = np.where(df_count['No'] < df_count['Yes'], 'Undervalued',
                                           'Overvalued')


fair_values = []
for tick in tickers:
    yearly_price = df_decompose['trend_trend'][df_decompose['YahooTicker'] == tick].resample('Y').mean()
    grouped = pd.Series(yearly_price.pct_change().mean())
    grouped = pd.DataFrame(grouped)
    grouped.columns = ['Average_Growth_Fair_Price']
    grouped['average_growth'] = df_decompose['price'][df_decompose['YahooTicker'] == tick].resample(
        'Y').mean().pct_change().mean()
    grouped['cagr'] = cagr(df_decompose['price'][df_decompose['YahooTicker'] == tick])
    grouped['YahooTicker'] = tick
    fair_values.append(grouped)

df_fair_values = pd.concat(fair_values)
df_fair_values = df_fair_values.set_index('YahooTicker')
df_fair_values['Second_Valuation_Critirion'] = np.where(
    df_fair_values['cagr'] > df_fair_values['Average_Growth_Fair_Price'], 'Overvalued', 'Undervalued')

df_fair_values.join(df_count[['Undervalued_Chances']]).to_clipboard()


# current upside
df_upside = df_decompose[['trend_cycle', 'trend_trend', 'price', 'YahooTicker', 'valuation']].groupby(
    'YahooTicker').tail(1)
df_upside = df_upside.set_index('YahooTicker')
df_upside['Upside_norm'] = zscore(df_upside['valuation']) * -1  # change sign so positive more upside

x = df_upside['Upside_norm'].to_numpy().reshape(-1, 1)
df_upside['Upside_norm'] = pre.MinMaxScaler().fit_transform(x)
df_upside.sort_values('Upside_norm')

# performance of stocks
# ======================
df_perfm = performance_overview(prices_temp, prices=True, short=False)
df_perfm.index.name = 'YahooTicker'
df_overview = df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker').join(df_perfm)
df_overview['Age'] = (((df_overview['end'] - df_overview['start'])).dt.days)/365

df_overview.to_clipboard()

# portfolio selection
# ====================
keep = ['Company', 'cagr', 'max_drawdown', 'calmar', 'daily_vol',
        'twelve_month_win_perc', 'Age']

df_rank = df_overview[keep]
df_rank['daily_vol'] = df_rank['daily_vol']*-1  # multiplied with -1 to penalize when aggregate

# join
df_rank = df_rank.join(df_upside[['Upside_norm']])


# define ranking
df_rank = df_rank.reset_index().set_index(['YahooTicker', 'Company'])

weights_for_ranking = [0.20, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10]
df_rank_norm = df_rank.astype(float).apply(zscore)
df_rank_norm['ranking'] = df_rank_norm.dot(weights_for_ranking)
df_rank_norm = df_rank_norm.sort_values(by='ranking', ascending=False)

x = df_rank_norm['ranking'].to_numpy().reshape(-1, 1)

# normalize all values to be between 0 and 1
df_rank_norm['ranking_norm'] = pre.MinMaxScaler().fit_transform(x)
df_rank_norm['weight'] = df_rank_norm['ranking_norm'] / df_rank_norm['ranking_norm'].sum()
df_rank_norm.sort_values('weight', ascending=False)
df_rank_norm.to_clipboard()


# Optimase Portfolio
# ===================
returns = prices_temp.pct_change()
returns_training = returns.dropna()
df_weights = df_rank_norm.reset_index()[['YahooTicker', 'weight']]


# optimal weight
# ===============
inv_vol_weights = inverse_vol_portfolio(returns_training=returns_training)  # Inverse Vol
equal_weights = equal_weight_portfolio(returns_training)  # Equally Waighted
min_vol_weights = port_GMV(returns_training=returns_training)

df_port_weights = pd.concat([inv_vol_weights, equal_weights, min_vol_weights], axis=1)
df_port_weights.index.name = df_weights.columns[0]
df_port_weights = pd.concat([df_port_weights, df_weights.set_index('YahooTicker')], axis=1)

df_port_weights.to_clipboard()

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

ret_sp = pd.concat(bench_ret, axis = 1)

# total returns
index_name = 'date'
ret = ret_sp.reset_index(index_name).merge(df_portfolios.reset_index(index_name))
ret = ret.set_index(index_name)



qs.reports.html(ret['port_inverse_vol'], ret[bench_ticker[1]],
                output='S:\\Investment Solutions Group\\Quant_research\\Moustafa\\Github\\QuantitativePortfolioManagement\\myPortfolioManagement\\MeteoraGlobalEquityFund\\Output',
                download_filename='test2_inverse_vol.html')




# check
portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

index_valuation = (portf_prices.tail(1) - portf_prices.max()) / portf_prices.max()
index_valuation = index_valuation.transpose()
index_valuation.columns = ['Undervalued']

index_valuation.sort_values(by='Undervalued').plot.barh()

# Performance
df_performance = performance_overview(returns_training)
df_performance.index.name = df_overview.index.name
df_portf_perfm = performance_overview(prices_from_returns(ret), prices=True, short=False)
df_portf_perfm.index.name = df_overview.index.name
df_portf_perfm.to_clipboard()
