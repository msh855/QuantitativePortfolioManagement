import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, create_spreads_from_policy_rates, \
    get_policy_rates, get_historical_spots, get_yields_countries

import monthly_returns_heatmap as mrh
import os

# plotting
import matplotlib
import matplotlib.dates as mdates

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.use('Qt5Agg')

######
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
start_date = '1970-01-01'

# trends in yields
df_yiedls = get_yields_countries()
df_yiedls = df_yiedls[df_yiedls.index >= start_date]
df_yiedls = df_yiedls.fillna(method= 'ffill')
df_yileds_trend = df_yiedls.fillna(method='ffill').rolling(30 * 12).mean().pct_change()

df_spread_list = []

for ccy in currencies:

    # ccy = currencies[0]
    country = ccy[0:3]
    name = 'slope_' + country

    # get signal for slope factor
    df_policy_rates = get_policy_rates(country=['US', country])
    df_policy_rates = df_policy_rates.fillna(method='ffill')

    # calculate slopes
    df_US_spread = create_spreads_from_policy_rates(country='US', df_policy_rates=df_policy_rates,
                                                yields='2Y')
    df_US_spread = df_US_spread.fillna(method='ffill')

    df_Foreign_spread = create_spreads_from_policy_rates(country=country, df_policy_rates=df_policy_rates,
                                                     yields='2Y')

    df_Foreign_spread = df_Foreign_spread.fillna(method='ffill')

    df_spreads = df_US_spread.reset_index().merge(df_Foreign_spread.reset_index())
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads[name] - df_spreads['slope_US']

    df_2y_spreads = df_yiedls[['2Y_US', '2Y_' + country]]
    df_2y_spreads['2y_Bond_spreads'] = df_2y_spreads['2Y_' + country] - df_2y_spreads['2Y_US']


    df_spreads_copy = df_spreads.copy()
    df_spreads_copy.columns = ['slope_US', 'slope_Foreign', 'rel_slope']
    df_spreads_copy = df_spreads_copy.join(df_2y_spreads[['2y_Bond_spreads']])
    df_spreads_copy['Currency'] = ccy
    df_spread_list.append(df_spreads_copy)



pd.concat(df_spread_list)