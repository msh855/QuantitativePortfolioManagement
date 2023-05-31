import pandas as pd
from openbb_terminal.sdk import openbb
from openbb_terminal.sdk import TerminalStyle
import warnings
from matplotlib import pyplot as plt

theme = TerminalStyle("light", "light", "light")

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

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
