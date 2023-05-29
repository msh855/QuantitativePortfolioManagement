import pandas as pd
from openbb_terminal.sdk import openbb
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from pypfopt.expected_returns import prices_from_returns

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

openbb.keys.finnhub(key="btd8eef48v6t4umjg6r0")
openbb.keys.polygon(key='qnyrMp6qOcr76VtuH9e5wihMPt5mg6Pi')

# load tickers
df_tickers = pd.read_excel('/Users/safishajjouz/PycharmProjects/myWatchlist/Data/yahoo_tickers.xlsx')
df_tickers.dropna(inplace=True)

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])


def get_stats_of_similar_companies(stocks_ticker: str = 'ALB', start_date="1990-01-01"):
    similar_stocks1 = openbb.stocks.ca.similar(symbol=stocks_ticker, source="TSNE")
    similar_stocks2 = openbb.stocks.ca.similar(symbol=stocks_ticker, source='Polygon')
    similar_stocks3 = openbb.stocks.ca.similar(symbol=stocks_ticker, source='Finnhub')
    similar_stocks4 = openbb.stocks.ca.similar(symbol=stocks_ticker, source='Finviz')

    similar_stocks = similar_stocks1 + similar_stocks2 + similar_stocks3 + similar_stocks4
    similar_stocks = [*set(similar_stocks)]
    similar_stocks = list(filter(None, similar_stocks))

    df_similar_stocks = openbb.stocks.ca.screener(similar=similar_stocks, data_type="overview")
    df_similar_stocks.columns = ['Ticker'] + list(df_similar_stocks.columns[1:len(df_similar_stocks.columns)])

    df_similar_stocks['Market Cap'] = df_similar_stocks['Market Cap'] / 1000000000

    tickers = df_similar_stocks.Ticker
    companies = df_similar_stocks.Company

    df_prices = []
    for ticker, company in zip(tickers, companies):
        data = openbb.stocks.load(ticker, start_date=start_date)
        data['Ticker'] = ticker
        data['Company'] = company
        df_prices.append(data)

    df_prices = pd.concat(df_prices)

    prices = df_prices[['Adj Close', 'Ticker']]
    prices = prices.pivot(columns='Ticker', values='Adj Close')

    returns = prices.pct_change()

    df_prices_norm = prices_from_returns(returns)
    df_prices_norm.fillna(method='ffill', inplace=True)
    df_perfm = performance_overview(df_prices_norm, prices=True)
    df_perfm.index.name = 'Ticker'
    df_perfm = df_perfm.join(df_similar_stocks.set_index('Ticker'))

    return df_perfm, returns, df_prices_norm, df_prices


# similar stocks
stocks_ticker = 'SMT.L'
df_perfm, returns, df_prices_norm, df_prices = get_stats_of_similar_companies(stocks_ticker=stocks_ticker)
df_perfm.to_clipboard()

# checks
returns.resample('Y').mean().corr()
df_perfm.to_clipboard()
df_prices_norm.plot()
