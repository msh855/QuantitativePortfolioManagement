import pandas as pd
from myPortfolioManagement.myData import get_stock_prices, get_sp500_tickers
from myPortfolioManagement.myPerformanceMetrics import get_main_stats

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
tickers = df10.Ticker
df_prices = get_stock_prices(tickers, wide_format=True)

get_main_stats(df_prices).sort_values('cagr')['cagr'].plot.bar()
