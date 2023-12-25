import pandas as pd
import numpy as np
import plotly.express as px
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, \
    get_policy_rates, get_bond_yield, get_historical_spots, get_macrobond_data, prices_from_returns
import monthly_returns_heatmap as mrh

# plotting
import matplotlib.dates as mdates

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

import matplotlib

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})

'''
Anything  else equal:  
    1.Positive (negative) Spread between the policy rate and the 2 year (or any other horizon) yield implies 
    that interest rates is expected to be falling (rising) 
    -> this tends to weaken (strengthen) Dollar vs foreign currency 
'''
ccy = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD'][1]
country = ccy[0:3]
yields = int_name = '2Y'
spot_mb = get_historical_spots(ccy=ccy).dropna()
spot_mb = spot_mb[spot_mb.index >= '1985-01-01']

# policy rate
# =============
df_policy_rates = get_policy_rates(country=['US', country])

# load USD
# =========================
df_US_bonds = get_bond_yield(country='US')

# calculate US slope
# ==================
df_US_spread = df_US_bonds.copy()
df_US_spread = df_US_spread.reset_index().merge(df_policy_rates['US'].reset_index())
df_US_spread['slope_US'] = df_US_spread['US'] - df_US_spread[yields]
df_US_spread = df_US_spread[['Date', 'slope_US']]

# Make smoothing average
# ==================================================================================================================
df_spreads = df_US_spread.copy()
df_spreads = df_spreads.set_index('Date')
df_ema = make_sma(df=df_spreads[df_spreads.index >= '1985-01-01'], spans=[25, 50, 60, 90, 120, 200])
col_names = df_ema.columns.tolist()

# Collect Data
# ==================================================================================================================
df_all = df_spreads.reset_index().merge(df_ema.reset_index())
df_all = df_all.merge(spot_mb.reset_index())
df_all = df_all.set_index('Date')

# create strategy
# ===================================================================================================================

# signals
df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)

# check other windows
for sm in col_names:
    df_all['factor_' + sm] = np.where(df_all['slope_US'] > df_all[sm], 1, -1)

# strategy performance
# =====================================================================================================================
df_performance = pd.DataFrame()
factors = ['Fed_Factor'] + ['factor_' + x for x in col_names]

for factor in factors:
    df_performance[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()

# average
df_performance['EMA_Composite'] = df_performance.mean(axis=1)

# plot Fed Factor and compose
df_performance[['Fed_Factor', 'EMA_Composite']].plot(title=ccy)

# get stats
df_stats = list()
for i in range(0, df_performance.shape[1]):
    df_stats.append(mainStats(df_performance.iloc[:, i].diff()))

df_stats = pd.concat(df_stats, axis=1)
df_stats.columns = df_performance.columns
df_stats = df_stats[['Fed_Factor', 'EMA_Composite']]
df_stats = df_stats.drop(['min', 'max', 'skew', 'kurtosis', 'MDD/vol'])
df_stats.plot.bar(title=ccy)

# monthly performance
# ====================================================================================================================
returns = performance(df_all["Fed_Factor"], total_returns=np.log(df_all[ccy]).diff())
mrh.plot(returns, eoy=True)
mrh.get(returns)

# Signals plot
df_all[['Fed_Factor']].plot()

# results = pd.melt(df_pnl_all.drop(drop_factors, axis=1).reset_index(), id_vars=['Date', 'Currency'])
#
# # plot
# fig = px.line(results, x='Date', y='value', color='variable',
#               facet_col='Currency', facet_col_wrap=5)
# fig.update_yaxes(matches=None)
# fig.show(renderer="browser")
#
# temp = performance(df_all['Composite'], total_returns=np.log(df_all[ccy]).diff()).cumsum()
# mainStats(temp.diff())
