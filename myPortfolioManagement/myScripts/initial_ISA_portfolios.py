import ffn
import matplotlib.pyplot as plt
import numpy as np
from myPortfolioManagement.myData import get_stock_prices, get_stock_info
import os
import pandas as pd
from myPortfolioManagement.myReturns import calculate_portfolio_returns, get_benchmark_porfolios
from myPortfolioManagement.myUtils import clean_stock_prices
from myPortfolioManagement.myPerformanceMetrics import get_main_stats, performance_overview
from myPortfolioManagement.myBootstrapping import bootstrappingTS
from myPortfolioManagement.myBacktesting import prep_dist
from myPortfolioManagement.myTimeSeries import Trends, decomposeTS, forecast_trend, look_back
from maData.getdata import get_US_yields

file_path = '/Users/safishajjouz/GitHub/myFinances'
file_name = 'portfoliols_eval_all.xlsx'
file_to_load = os.path.join(file_path, file_name)
portfolio_list = []
num_portf = 5
start_trading_date = '2020-11-24'

for i in range(num_portf):
    prt_temp = pd.read_excel(file_to_load, sheet_name=i)
    prt_temp['porfolio'] = 'portfolio' + str(i + 1)
    portfolio_list.append(prt_temp)

# ======================================================================================================================
# Initial portfolio performance
# ======================================================================================================================

ret_port_list = []
for i in range(num_portf):
    df_port = portfolio_list[i]
    tickers = list(df_port['yahooTicker'])
    df_info = get_stock_info(tickers)
    df_info = df_port.merge(df_info[['yahooTicker', 'longName']])
    df_weights = df_info[['longName', 'weight']]

    # load currencies
    prices = get_stock_prices(tickers, start_date=start_trading_date, wide_format=True)
    ret = clean_stock_prices(prices, fold=10)

    ret_port = calculate_portfolio_returns(ret, df_weights, portfolio_name='portfolio' + str(i + 1))
    ret_port['portfolio' + str(i + 1)].cumsum().plot()
    ret_port_list.append(ret_port)

df_port_ret = pd.concat(ret_port_list, axis=1)
# df_port_ret = df_port_ret.interpolate(method='linear', limit_direction='both', axis=0)
df_port_ret.cumsum().plot()

# get benchmarks
tickers_bench = ['VWRP.L', 'EQQQ.L', 'VUAG.L']
prices_bench = get_stock_prices(tickers_bench, start_date=start_trading_date, wide_format=True)
ret_bench = clean_stock_prices(prices_bench, fold=10)
ret_bench.columns = ['Nasdaq', 'FTSE World', 'S&P 500']
ret_bench.cumsum().plot()

# other benchmarks
ret_other_bench = get_benchmark_porfolios()

# combine
ret_all = df_port_ret.join(ret_bench)
ret_all = ret_all.join(ret_other_bench)
ret_all.cumsum().plot()

# stats
df_main_stats = get_main_stats(ret_all)

metric = 'total_ret'
df_main_stats[[metric, 'adjusted_sortino']].sort_values(by=metric).plot.barh()

# ======================================================================================================================
# fund performance
# ======================================================================================================================
df_funds = pd.concat(portfolio_list)
funds = df_funds['yahooTicker'].unique()
df_fund_prices = get_stock_prices(funds, wide_format=True, start_date=start_trading_date)
ret_funds = clean_stock_prices(df_fund_prices)

df_fund_main_stats = get_main_stats(ret_funds, add_age=True)

metric = 'total_ret'
df_fund_main_stats[[metric, 'adjusted_sortino']].sort_values(by=metric).plot.barh(rot=0, fontsize=8)

# top perf
top = 5
df_fund_main_stats.sort_values(by='total_ret', ascending=False).head(top)
df_fund_main_stats.sort_values(by='sharpe', ascending=False).head(top)
df_fund_main_stats.sort_values(by='adjusted_sortino', ascending=False).head(top)
df_fund_main_stats.sort_values(by='calmar', ascending=False).head(top)
df_fund_main_stats.sort_values(by='cagr', ascending=False)[['cagr']].head(top)

# ======================================================================================================================
# Assess expected Returns
# ======================================================================================================================
df_funds = pd.concat(portfolio_list)
date_tr = '2020-07-01'

df_funds[['Name', 'yahooTicker']].drop_duplicates()
df_prices = get_stock_prices(['0P0000X9F5.L'], start_date='2012-01-01', wide_format=True)
ret_clipped = clean_stock_prices(df_prices, fold=10)
prices_clipped = ffn.to_price_index(ret_clipped, 100)
ffn.rebase(df_prices, 100).join(pd.Series(prices_clipped.iloc[:, 0], name='prices_clipped')).plot()

df_prices = prices_clipped

# training
prices_tr = df_prices[(df_prices.index <= date_tr)]

# out of sample
prices_out_of_sample = df_prices[df_prices.index > date_tr]

# boostrap
ret_tr = prices_tr.iloc[:, 0].pct_change().dropna()
df_three_years = look_back(ret_tr, years=3)
df_three_years.cumsum().plot()

ret_boots = bootstrappingTS(df_three_years, bootstrap_type='sb', optimal_block=True, n_samples=10000)
total_ret_oos = get_main_stats(prices_out_of_sample.pct_change())['total_ret'][0]

stats_boost = get_main_stats(ret_boots)
stats_hist = get_main_stats(pd.DataFrame(df_three_years))

# outcome
stats_boost['total_ret'].plot.density(label='Expect Total Returns')
plt.axvline(x=total_ret_oos, color='red', label='Realized (ex-post)')
plt.axvline(x=stats_hist['total_ret'].values[0], color='black', label='historical (ex-ante)')
plt.legend()
plt.title('Ex-post Evaluation: ' + stats_hist.index[0])

# check yields to detect balance of risks
df_yields = get_US_yields(freq='d')
df_yields['Spread'] = df_yields['2Y'] - df_yields['3M']
df_yields_tr = df_yields[df_yields.index <= date_tr]
#df_yields_tr = look_back(df_yields_tr, 3)
df_yields_out = df_yields[df_yields.index > date_tr]

# boostrap yields
boost_tep = 'mbb'
blc_years = 5*365
yields_boots3y = bootstrappingTS(df_yields_tr['3Y'], bootstrap_type=boost_tep, block_size = blc_years, n_samples=10000)
yields_boots1y = bootstrappingTS(df_yields_tr['1Y'], bootstrap_type=boost_tep, block_size = blc_years, n_samples=10000)
yields_boots10 = bootstrappingTS(df_yields_tr['10Y'], bootstrap_type=boost_tep, block_size = blc_years, n_samples=10000)
yields_bootsYC = bootstrappingTS(df_yields_tr['Spread'], bootstrap_type=boost_tep, block_size = blc_years, n_samples=10000)


for col, yeld in zip(['3Y', '1Y', '10Y', 'Spread'], [yields_boots3y, yields_boots1y, yields_boots10, yields_bootsYC]):
    plt.figure()
    yeld.mean().plot.density()
    df_yields[col].plot.density()
    plt.axvline(x=df_yields_tr[col].mean(), color='black', label='historical (ex-ante)')
    plt.axvline(x=df_yields_out[col].mean(), color='red', label='Realized (ex-post)')
    plt.legend()
    plt.title('Ex-post Evaluation: ' + col)
    plt.show()

# ======================================================================================================================
# Forecasting
# ======================================================================================================================
price_d = decomposeTS(df_prices)
price_d = price_d.HPfilter(freq='daily')

# estimate trends
trends = Trends(prices_tr)
price_dec = decomposeTS(prices_tr)
price_dec = price_dec.HPfilter(freq='daily')
data = price_dec.filter(like='trend')

forcast_period = list(range(1, prices_out_of_sample.shape[0] + 1))
pred = forecast_trend(data, steps=forcast_period)

data_pred = pd.concat([data, pred])
data_pred.index = df_prices.index
data_pred.columns = ['trend_pred']

df_all = price_d.join(data_pred)

df_all.iloc[:, [0, 2, 3]].plot()
plt.axvline(x='2020-07-01', color='red', label='Realized')
