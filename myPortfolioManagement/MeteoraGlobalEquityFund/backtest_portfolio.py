import pandas as pd
from openbb_terminal.sdk import openbb
from openbb_terminal.sdk import TerminalStyle
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from scipy.stats import zscore
from sklearn import preprocessing as pre
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio, equal_weight_portfolio
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myReturns import calculate_portfolio_returns
import quantstats as qs

theme = TerminalStyle("light", "light", "light")

warnings.filterwarnings("ignore")
plt.style.use('seaborn')

# load tickers
df_tickers = pd.read_excel(
    'S:\Investment Solutions Group\Quant_research\Moustafa\Github\QuantitativePortfolioManagement\myPortfolioManagement\Examples\myWatchlist\Data\stock_screening.xlsx')
df_tickers.dropna(inplace=True)
index_name = 'YahooTicker'

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "2005-01-01"

df_prices = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data[index_name] = ticker
    data['Company'] = company
    df_prices.append(data)

df_prices = pd.concat(df_prices)

prices = df_prices[['Adj Close', index_name]]
prices = prices.pivot(columns=index_name, values='Adj Close')

# Optimase Portfolio
# ===================
returns = prices.pct_change()

df_perfm = performance_overview(prices, prices=True, short=False)
df_perfm.index.name = 'YahooTicker'
df_overview = df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker').join(df_perfm)
df_overview['Age'] = (((df_overview['end'] - df_overview['start'])).dt.days) / 365
df_overview.to_clipboard()

# portfolio selection
# ====================
keep = ['Company', 'cagr', 'max_drawdown', 'calmar', 'daily_sortino', 'monthly_sortino',
        'yearly_sortino', 'twelve_month_win_perc', 'Age']

df_rank = df_overview[keep]
df_rank = df_rank.reset_index().set_index(['YahooTicker', 'Company'])

weights_for_ranking = [0.30, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05, 0.10]
df_rank_norm = df_rank.astype(float).apply(zscore)
df_rank_norm['ranking'] = df_rank_norm.dot(weights_for_ranking)
df_rank_norm = df_rank_norm.sort_values(by='ranking', ascending=False)

x = df_rank_norm['ranking'].to_numpy().reshape(-1, 1)

# normalize all values to be between 0 and 1
df_rank_norm['ranking_norm'] = pre.MinMaxScaler().fit_transform(x)
df_rank_norm['weight'] = df_rank_norm['ranking_norm'] / df_rank_norm['ranking_norm'].sum()

df_weights = df_rank_norm[['weight']]
df_weights = df_weights.reset_index().drop(['Company'], axis=1)

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

# optimal weight
# ===============
returns_training = returns
inv_vol_weights = inverse_vol_portfolio(returns_training=returns_training)  # Inverse Vol
equal_weights = equal_weight_portfolio(returns_training)  # Equally Waighted
min_vol_weights = port_GMV(returns_training=returns_training)

df_port_weights = pd.concat([inv_vol_weights, equal_weights, min_vol_weights], axis=1)
df_port_weights.index.name = df_weights.columns[0]
df_port_weights = pd.concat([df_port_weights, df_weights.set_index('YahooTicker'),
                             df_tickers[['YahooTicker', 'adjusted_weights']].set_index('YahooTicker')], axis=1)

df_port_weights.join(df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker')).to_clipboard()

# portfolio returns
# =================
df_portfolios = []
for col in df_port_weights.columns:
    temp_portf = calculate_portfolio_returns(returns_training, df_port_weights[[col]].reset_index(), portfolio_name=col)
    df_portfolios.append(temp_portf)

df_portfolios = pd.concat(df_portfolios, axis=1)

# total returns
index_name = 'date'
ret = ret_sp.reset_index(index_name).merge(df_portfolios.reset_index(index_name))
ret = ret.set_index(index_name)


qs.reports.html(ret['adjusted_weights'], ret[bench_ticker[1]],
                output='C:\\Users\\MChatzouz\\PycharmProjects\\PortfolioManagement',
                download_filename='test2.html')

# check
portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

corr_matrix = returns.corr()
corr_matrix.to_clipboard()

returns.corr()[returns.corr() < 1].max().sort_values()

import matplotlib

matplotlib.use('TkAgg')
from phik.report import plot_correlation_matrix

plot_correlation_matrix(corr_matrix.to_numpy(), x_labels=corr_matrix.columns, y_labels=corr_matrix.columns,
                        vmin=-5, vmax=5, title='outlier significance',
                        identity_layout=False, fontsize_factor=1.2)
plt.show()
