import pandas as pd
import requests
import os

myAPI = "4666063ZUKaNsJbECbaOGawwQgMdesMbKGwG"
account_id = '4666063'
headers = {"Authorization": myAPI}

# all available exchanges
url_main = 'https://live.trading212.com/api/v0/equity/'
str_list_general = ['metadata/exchanges', 'metadata/instruments']
str_account_info = ['account/info', 'account/cash']

str_list = ['portfolio', 'pies', 'orders', 'history/orders', 'history/dividends', 'history/transactions']
data_list = []

for i in range(0, len(str_list)):
    url_temp = os.path.join(url_main, str_list[i])
    response = requests.get(url_temp, headers=headers)
    data = response.json()
    try:
        df = pd.DataFrame(data)
        data_list.append(df)
    except:
        data_list.append(data)

df_212 = data_list[0]
df_212['Date_of_Purchase'] = pd.to_datetime(df_212['initialFillDate']).dt.date
df_212 = df_212.drop(['frontend', 'maxBuy', 'maxSell', 'pieQuantity', 'initialFillDate'], axis=1)

inv_212 = [x.split('_') for x in df_212['ticker']]
tickers_raw = []
for i in range(0, len(inv_212)):
    tick = [x.split('_') for x in df_212['ticker']][i][0]
    tickers_raw.append(tick)

df_212['tickers_212'] = tickers_raw

data_orders_list = []
for i in range(0, data_list[3]['items'].shape[0]):
    data_ord_temp = pd.DataFrame(data_list[3]['items'].iloc[i])
    data_orders_list.append(data_ord_temp)

data_orders = pd.concat(data_orders_list)

query = {"cursor": "0", "ticker": "string", "limit": "20"}

ticker = "/AAPL_US_EQ"
url_my_portfolio_specific_position = url_main + 'portfolio' + ticker
response = requests.get(url_my_portfolio_specific_position, headers=headers)
data = response.json()

myPie_ID = "/1547833"
url_my_portfolio_specific_pie = url_main + 'pies' + myPie_ID
response = requests.get(url_my_portfolio_specific_position, headers=headers)
data = response.json()

# Load API
myAPI = "4666063ZUKaNsJbECbaOGawwQgMdesMbKGwG"
account_id = '4666063'
headers = {"Authorization": myAPI}

url_my_portfolio = "https://live.trading212.com/api/v0/equity/portfolio"
url_fetch_all_pies = "https://live.trading212.com/api/v0/equity/pies"

response = requests.get(url_my_portfolio, headers=headers)
data = response.json()


