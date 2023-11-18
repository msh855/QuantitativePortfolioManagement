import ffn.core
import pandas as pd
from myPortfolioManagement.myData import get_stock_prices, get_sp500_tickers
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myBacktesting import bootstrap_stats
import quantstats as qs

# ======================================================================================================================
sp500 = get_sp500_tickers()

sp500['sp_weight'] = sp500['Market Cap'] / sp500['Market Cap'].sum()
df10 = sp500.sort_values(by=['sp_weight'], ascending=False).head(10)
df10 = df10[df10.duplicated('Company') == False]
google_weight = df10[df10['Ticker'] == 'GOOG']['sp_weight']
google_mark_cap = df10[df10['Ticker'] == 'GOOG']['Market Cap']
df10.loc[df10['Ticker'] == 'GOOG', ['sp_weight']] = google_weight * 2

df10.loc[df10['Ticker'] == 'GOOG', ['Market Cap']] = google_mark_cap * 2
df10 = df10[df10['Ticker'] != 'BRK-B']
df10 = df10.sort_values(by=['sp_weight'], ascending=False)
df10['adj_weight'] = df10['Market Cap'] / df10['Market Cap'].sum()

# prices
# get stock prices
# ======================================================================================================================
tickers = df10.Ticker
df_prices = get_stock_prices(tickers, wide_format=True)
df_main_stats = get_main_stats(df_prices).sort_values('cagr', ascending=False)


df_main_stats['adjusted_sortino'].plot.bar()
df_main_stats['cagr'].plot.bar()
df10[['Ticker', 'adj_weight']].set_index('Ticker').plot.bar(y='adj_weight')

# bootstapping
# ======================================================================================================================
ret = df_prices.pct_change()
stats_boost_results = []
for tik in tickers:
    stats_temp = bootstrap_stats(ret[tik], n_sim=50)
    stats_boost_results.append(stats_temp)

import cProfile
import pstats

with cProfile.Profile() as pr:
    stats_temp = bootstrap_stats(ret[tik], n_sim=50)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
# Now you have two options, either print the data or save it as a file
stats.print_stats()  # Print The Stats


import matplotlib.pyplot as plt
import seaborn as sns

df = stats_temp
selection = df.columns

fig, axes = plt.subplots(1, len(selection))
for i, col in enumerate(selection):
    ax = sns.boxplot(y=df[col], ax=axes.flatten()[i])
    ax.set_ylim(df[col].min(), df[col].max())
    ax.set_ylabel(col)
plt.show()

# backtest
# ======================================================================================================================
ret_port = calculate_portfolio_returns(returns=df_prices.pct_change().dropna(),
                                       myweights=df10[['Ticker', 'adj_weight']])

ret_bench = get_stock_prices(yahoo_tickers=['VWRL.L'], wide_format=True).pct_change()
bench_name = 'Vanguard Global'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

qs.reports.html(ret_all['myPortfolio'], benchmark=ret_all[bench_name], output='/Users/safishajjouz/Downloads',
                download_filename='port_perf.html', eoy=True)
