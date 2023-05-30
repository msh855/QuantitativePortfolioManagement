import pandas as pd
from openbb_terminal.sdk import openbb
from scipy.stats import zscore
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio,  equal_weight_portfolio
from sklearn import preprocessing as pre
from pypfopt.expected_returns import prices_from_returns

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

# load tickers
df_tickers = pd.read_excel('/Users/safishajjouz/PycharmProjects/myWatchlist/Data/stock_screening.xlsx')
df_tickers.dropna(inplace=True)

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

prices = df_prices[['Adj Close', 'Ticker']]
prices = prices.pivot(columns='Ticker', values='Adj Close')

# performance of stocks
# ======================

df_perfm = performance_overview(prices, prices=True, short=False)
df_perfm.index.name = 'YahooTicker'
df_overview = df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker').join(df_perfm)
df_overview['Age'] = (((df_overview['end'] - df_overview['start'])).dt.days)/365
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

# Optimase Portfolio
# ===================
returns = prices.pct_change()
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


# export
temp_df = df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker')
temp_df.join(df_port_weights).to_clipboard()

df_port_weights.plot.bar()

# portfolio returns
# =================
df_portfolios = []
for col in df_port_weights.columns:
    temp_portf = calculate_portfolio_returns(returns_training, df_port_weights[[col]].reset_index(), portfolio_name=col)
    df_portfolios.append(temp_portf)

df_portfolios = pd.concat(df_portfolios, axis=1)

# benchmark
price_sp = openbb.stocks.load('^GSPC', start_date=start_date)
ret_sp = price_sp['Adj Close'].pct_change()
ret_sp.name = 'S&P'
ret_sp = pd.DataFrame(ret_sp)

# total returns
index_name = 'date'
ret = ret_sp.reset_index(index_name).merge(df_portfolios.reset_index(index_name))
ret = ret.set_index(index_name)


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