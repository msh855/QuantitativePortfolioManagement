import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, \
    get_policy_rates, get_historical_spots, get_bond_yield, get_yields_countries
from medh.momentum import BaseMomentum
from scipy.stats.mstats import zscore

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


###


def get_slopes(country1='US', country2='CHF'):
    name = 'slope_' + country
    df_policy_rates = get_policy_rates(country=[country1, country2])
    df_Foreign_bonds = get_bond_yield(country=country)

    df_Foreign_bonds = df_Foreign_bonds.reset_index().merge(df_policy_rates[[country2]].reset_index())
    df_Foreign_bonds = df_Foreign_bonds.set_index('Date')
    df_Foreign_slope = pd.DataFrame({name: df_Foreign_bonds.iloc[:, 0] - df_Foreign_bonds['2Y']})
    df_Foreign_slope[name + '_policy'] = df_Foreign_bonds.iloc[:, -1] - df_Foreign_bonds['2Y']

    # slope Foreign
    df_US_bonds = get_bond_yield(country='US')
    df_US_slope = pd.DataFrame({'slope_US': df_US_bonds.iloc[:, 0] - df_US_bonds['2Y']})
    df_US_slope['slope_US' + '_policy'] = pd.DataFrame({name: df_US_bonds.iloc[:, -1] - df_US_bonds['2Y']})

    # rel slope
    df_spreads = df_US_slope.reset_index().merge(df_Foreign_slope.reset_index())
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads[name] - df_spreads['slope_US']
    df_spreads['rel_slope_policy'] = df_spreads[name + '_policy'] - df_spreads['slope_US' + '_policy']
    df_spreads = df_spreads[['rel_slope', 'rel_slope_policy']]
    df_spreads.dropna(inplace=True)

    return df_spreads


def vol_adj(df, vol_window=60):
    raw_vol = BaseMomentum.std_vol(df, window=vol_window)
    # Vol normalise returns
    adj_return = df / raw_vol * np.sqrt(252)
    return adj_return, raw_vol


# ========================================


####
ccy = 'EURUSD'
country = ccy[0:3]
start_date = '2000-01-01'
spot_mb = get_historical_spots(ccy=ccy).dropna()
spot_mb = spot_mb[spot_mb.index >= start_date]

df_spreads = get_slopes(country1='US', country2=country)

#
df_spreads_vol_adj, raw = vol_adj(df_spreads, vol_window=30)
df_spreads_vol_adj.plot()

data = df_spreads.apply(zscore)
data.hist()
df_ema = make_sma(df=data[data.index >= start_date], spans=[25, 50, 60, 90, 120, 200])
col_names = df_ema.columns.tolist()
df_ema.hist()

# Collect Data
# ==================================================================================================================
df_all = data.reset_index().merge(df_ema.reset_index())
df_all = df_all.merge(spot_mb.reset_index())
df_all = df_all.set_index('Date')
df_all.dropna(inplace=True)

df_all.plot(secondary_y=ccy)
df_all[['rel_slope', 'rel_slope_policy']].plot()

df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)
df_all['rel_slope_Factor_z'] = np.where(abs(df_all['rel_slope_Factor']) > 1, 0, 1)
df_all['rel_slope_Factor_policy'] = np.where(df_all['rel_slope_policy'] > df_all['rel_slope_policy_sm50'], -1, 1)

# create strategy
# ============================================================
factors = ['rel_slope_Factor', 'rel_slope_Factor_policy', 'rel_slope_Factor_z']
df_all['Composite_all'] = df_all[factors].mean(axis=1)

df_performance = pd.DataFrame()
for factor in factors + ['Composite_all']:
    df_performance[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()
df_performance.plot()

df_performance.dropna(inplace=True)
df_performance = df_performance / df_performance.iloc[0]

# rebase
(df_performance['rel_slope_Factor'] / df_performance['rel_slope_Factor'].iloc[0]).plot()

#################### 12-month 2 year ##########################################################


####
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
start_date = '2000-01-01'

# trends in yields
df_yiedls = get_yields_countries()
df_yiedls = df_yiedls[df_yiedls.index >= start_date]

# df_yiedls_vol_adj = vol_adj(df_yiedls.fillna(method='ffill'), vol_window=30)[0]
# df_yiedls.plot()
# df_yiedls_vol_adj.plot()
# df_yiedls.fillna(method='ffill').rolling(30 * 12).mean().plot()
# df_yileds_trend = df_yiedls.fillna(method='ffill').rolling(30 * 12).mean()

df_yileds_trend = df_yiedls.fillna(method='ffill').rolling(30 * 12).mean().pct_change()

df_yiedls['2Y_EUR'].plot()

df_yiedls.pct_change().plot()

df_performance = pd.DataFrame()
df_MC_regime_list = []
for ccy in currencies:
    country = ccy[0:3]
    df_MC_regime = df_yileds_trend['2Y_US'] - df_yileds_trend['2Y' + '_' + country]
    df_MC_regime.name = 'MC_regime'
    df_MC_regime = pd.DataFrame(df_MC_regime)
    df_MC_regime['Signal'] = np.where(df_MC_regime['MC_regime'] > 0, -1, 1)  # this assumes XUSD cross convention

    spot_mb = get_historical_spots(ccy=ccy).dropna()
    spot_mb.columns = ['Spot']
    df_MC_regime = df_MC_regime.reset_index().merge(spot_mb.reset_index())
    df_MC_regime = df_MC_regime.set_index('Date')
    df_MC_regime.dropna(inplace=True)

    df_MC_regime['ret'] = np.log(df_MC_regime['Spot']).diff()
    df_MC_regime['ccy'] = ccy
    df_MC_regime.dropna(inplace=True)
    df_MC_regime_list.append(df_MC_regime)

    df_performance['MC_factor' + country] = performance(df_MC_regime['Signal'],
                                                        total_returns=df_MC_regime['ret']).cumsum()

df_stats = pd.DataFrame()
for i in range(0, df_performance.shape[1]):
    cctry = df_performance.iloc[:, i].name[9:12]
    df_stats[cctry] = mainStats(df_performance.iloc[:, i].diff()).transpose()['IR']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
df_stats.transpose().sort_values(by='Stat', ascending=False).plot.bar(ax=axes[0],
                                                                      title='Monetary Policy Trend Factor: Average IR since 2000')
df_performance.plot(ax=axes[1])

ret_def = pd.concat(df_MC_regime_list).groupby(['ccy', 'Signal']).mean()[['ret']] * 100
df_performance[['MC_factorCHF']].plot()
pd.concat(df_MC_regime_list)

ret_def.plot.bar()

df_performance.mean(axis=1).plot(title='Portfolio Level')
