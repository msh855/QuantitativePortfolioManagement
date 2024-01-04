import numpy as np
import pandas as pd
import requests

myAPI = '4666063ZGrEDEcHLjIquabEpUJOtrpVPvMNv'

def get_pie_details(id: str = "1547833", myAPI: str = myAPI) -> pd.DataFrame:
    url = "https://live.trading212.com/api/v0/equity/pies/" + id
    headers = {"Authorization": myAPI}
    response = requests.get(url, headers=headers)
    data = response.json()

    # main info of the pie
    df_pie = pd.DataFrame(data['instruments'])

    # clean to get details
    tickers = df_pie.ticker
    df_results_list = []
    for i, tik in enumerate(tickers):
        df_temp = df_pie['result'][df_pie['ticker'] == tik][i]
        df_results = pd.DataFrame.from_dict(df_temp, orient='index', columns=[tik]).transpose()
        df_results_list.append(df_results)

    df_pie_portf = pd.concat(df_results_list)

    inv_212 = [x.split('_') for x in df_pie_portf.index]
    tickers_raw = []
    country_list = []
    for i in range(0, len(inv_212)):
        tick = [x.split('_') for x in df_pie_portf.index][i][0]
        tick_count = [x.split('_') for x in df_pie_portf.index][i][1]
        tickers_raw.append(tick)
        country_list.append(tick_count)

    df_pie_portf['tickers_212'] = tickers_raw
    df_pie_portf['Country'] = country_list

    df_pie_portf['Country'] = np.where(df_pie_portf['Country'] == 'EQ', 'UK', 'US')
    df_pie_portf = df_pie_portf.rename(columns={'result': 'unrealised_profit', 'resultCoef': 'returns'})
    df_pie_portf.index.name = 'ticker'
    df_pie_portf = df_pie_portf.reset_index()

    df_pie_portf = df_pie_portf.merge(df_pie)

    df_pie_portf['pie_name'] = data['settings']['name']
    df_pie_portf['pie_id'] = data['settings']['id']

    df_pie_portf.drop(['issues'], axis=1, inplace=True)

    return df_pie_portf


def get_portfolio_info(myAPI: str = myAPI):
    # Trade212 Account overall portfolio
    headers = {"Authorization": myAPI}
    url_my_portfolio = "https://live.trading212.com/api/v0/equity/portfolio"
    response = requests.get(url_my_portfolio, headers=headers)
    data = response.json()
    df_212 = pd.DataFrame(data)
    df_212['Date_of_Purchase'] = [str(x)[0:10] for x in df_212['initialFillDate']]
    df_212['Date_of_Purchase'] = pd.to_datetime(df_212['Date_of_Purchase']).dt.date
    df_212 = df_212.drop(['frontend', 'maxBuy', 'maxSell', 'pieQuantity', 'initialFillDate'], axis=1)

    inv_212 = [x.split('_') for x in df_212['ticker']]
    tickers_raw = []
    country_list = []
    for i in range(0, len(inv_212)):
        tick = [x.split('_') for x in df_212['ticker']][i][0]
        tick_country = [x.split('_') for x in df_212['ticker']][i][1]
        tickers_raw.append(tick)
        country_list.append(tick_country)

    df_212['tickers_212'] = tickers_raw
    df_212['Country'] = country_list
    df_212['Country'] = np.where(df_212['Country'] == 'EQ', 'UK', 'US')

    df_212['fxPpl'] = df_212['fxPpl'].fillna(1)

    return df_212


def get_pies(myAPI: str = myAPI):
    headers = {"Authorization": myAPI}
    url_fetch_all_pies = "https://live.trading212.com/api/v0/equity/pies"
    response = requests.get(url_fetch_all_pies, headers=headers)
    data = response.json()
    data = pd.DataFrame(data)
    data = data.drop(['progress', 'status'], axis=1)
    pie_list = []
    pie_list2 = []
    for i, tik in enumerate(data.id):
        temp = pd.DataFrame.from_dict(data['result'][i], orient='index', columns=[tik]).transpose()
        temp_div = pd.DataFrame.from_dict(data['dividendDetails'][i], orient='index', columns=[tik]).transpose()
        pie_list.append(temp)
        pie_list2.append(temp_div)

    df_pies1 = pd.concat(pie_list)
    df_pies2 = pd.concat(pie_list2)

    df_pies = df_pies1.join(df_pies2)
    df_pies.index.name = data.columns[0]
    data = data.drop(['dividendDetails', 'result'], axis=1)
    data = data.merge(df_pies.reset_index())

    data = data.rename(columns={'result': 'unrealised_profit', 'resultCoef': 'returns', 'gained': 'div_paid'})

    return data
