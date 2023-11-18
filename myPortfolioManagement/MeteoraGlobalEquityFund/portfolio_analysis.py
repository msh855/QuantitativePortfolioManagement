from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myPerformanceMetrics import get_main_stats, reward_metric
import requests
import pandas as pd
import os

# Trade212 Account
# ======================================================================================================================
myAPI = "4666063ZUKaNsJbECbaOGawwQgMdesMbKGwG"
account_id = '4666063'
headers = {"Authorization": myAPI}

url_my_portfolio = "https://live.trading212.com/api/v0/equity/portfolio"
url_fetch_all_pies = "https://live.trading212.com/api/v0/equity/pies"

response = requests.get(url_my_portfolio, headers=headers)
data = response.json()
df_212 = pd.DataFrame(data)
df_212['Date_of_Purchase'] = pd.to_datetime(df_212['initialFillDate']).dt.date
df_212 = df_212.drop(['frontend', 'maxBuy', 'maxSell', 'pieQuantity', 'initialFillDate'], axis=1)

inv_212 = [x.split('_') for x in df_212['ticker']]
tickers_raw = []
for i in range(0, len(inv_212)):
    tick = [x.split('_') for x in df_212['ticker']][i][0]
    tickers_raw.append(tick)

df_212['tickers_212'] = tickers_raw


# load data
# ======================================================================================================================
working_directory = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data'
os.chdir(working_directory)

# import tickers
# ==============
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
file1 = os.path.join(working_directory, 'df_prices.csv')

# get main info
# ======================================================================================================================
df_port_main_info = pd.read_csv(file1)
df_port_main_info = df_port_main_info[['YahooTicker', 'Currency']].drop_duplicates()
df_port_main_info['Currency'] = [x.upper() for x in df_port_main_info['Currency']]

file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)
df_port_main_info = df_port_main_info.merge(df_port_info)


# overview
# ======================================================================================================================
tickers = df_port_main_info['YahooTicker']
df_prices_nonadj = get_stock_prices(tickers)

df_reward_metrics = reward_metric(df_prices_nonadj.reset_index(),
                                  rolling_window=6,
                                  rolling_frequency='M',
                                  my_assets_col_name='yahooTicker',
                                  my_date_col_name='Date',
                                  price_col_name='adjclose')

prices = df_prices_nonadj.pivot(columns='yahooTicker', values='adjclose')
df_main_stats = get_main_stats(prices).sort_values('Age(sample)', ascending=False)
df_main_stats[['cagr']].sort_values('cagr').plot.bar()

get_stock_info(yahoo_tickers=tickers, data_type='all')