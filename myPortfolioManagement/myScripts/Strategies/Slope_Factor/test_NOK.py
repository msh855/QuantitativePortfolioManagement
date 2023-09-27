import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, create_spreads_from_policy_rates, \
    get_policy_rates, get_historical_spots, get_yields_countries, get_bbq_ccy

import monthly_returns_heatmap as mrh
import os

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})
plt.style.use('seaborn-dark-palette')
matplotlib.use('Qt5Agg')

######
# trends in yields
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
start_date = '2000-01-01'
country = 'NOK'
name = 'slope_' + country

df_yiedls = get_yields_countries()
df_yiedls = df_yiedls[df_yiedls.index >= start_date]
df_yiedls = df_yiedls[['2Y_NOK', '2Y_US']]

spot_mb = get_historical_spots(ccy='NOKUSD').dropna()

# get signal for slope factor
df_policy_rates = get_policy_rates(country=['US', country])

# calculate slopes
df_US_spread = create_spreads_from_policy_rates(country='US', df_policy_rates=df_policy_rates,
                                                yields='2Y')
df_Foreign_spread = create_spreads_from_policy_rates(country=country, df_policy_rates=df_policy_rates,
                                                     yields='2Y')
df_spreads = df_US_spread.reset_index().merge(df_Foreign_spread.reset_index())
df_spreads = df_spreads.set_index('Date')
df_spreads['rel_slope'] = df_spreads[name] - df_spreads['slope_US']

df_spreads_copy = df_spreads.copy()
df_spreads_copy.columns = ['slope_US', 'slope_Foreign', 'rel_slope']

df_spreads_copy = df_spreads_copy.fillna(method='ffill')
df_spreads_copy.dropna(inplace=True)

df_all = df_spreads_copy.join(spot_mb)
df_all = df_all.join(df_yiedls)
df_all['Yield_Spreads'] = df_all['2Y_US'] - df_all['2Y_NOK']
df_all = df_all.fillna(method='ffill')

df_all[['Yield_Spreads','NOKUSD']].rolling(30).mean().dropna().plot(secondary_y = 'NOKUSD')

df_all[['Yield_Spreads','NOKUSD']].rolling(30).mean().dropna().corr()