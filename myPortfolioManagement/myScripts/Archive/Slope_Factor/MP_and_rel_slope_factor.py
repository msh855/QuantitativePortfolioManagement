import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, create_spreads_from_policy_rates, \
    get_policy_rates, get_historical_spots, get_yields_countries

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

currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD'][1]
start_date = '1970-01-01'

#### Monetary Policy Trend Factor
#################################

# trends in yields
df_yiedls = get_yields_countries()
df_yiedls = df_yiedls[df_yiedls.index >= start_date]
df_yileds_trend = df_yiedls.fillna(method='ffill').rolling(30 * 12).mean().pct_change()

df_performance = pd.DataFrame()
returns = pd.DataFrame()
df_signals_list = []

for ccy in currencies:
    # ccy = currencies[0]
    country = ccy[0:3]
    name = 'slope_' + country

    df_MC_regime = df_yileds_trend['2Y_US'] - df_yileds_trend['2Y' + '_' + country]
    df_MC_regime.name = 'MC_regime'
    df_MC_regime = pd.DataFrame(df_MC_regime)
    df_MC_regime['Signal_MP_Trend'] = np.where(abs(df_MC_regime['MC_regime']) > 0, -1,
                                               1)  # this assumes XUSD cross convention

    spot_mb = get_historical_spots(ccy=ccy).dropna()
    spot_mb.columns = ['Spot']

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

    # Make smoothing average
    # ==================================================================================================================
    df_ema = make_sma(df=df_spreads[df_spreads.index >= start_date], spans=[50])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    #### Vol spreads

    vol_spread = df_US_spread.dropna().rolling(10).std().join(df_Foreign_spread.dropna().rolling(10).std())
    vol_spread.columns = vol_spread.columns + '_vol'
    vol_spread['vol_spread'] = vol_spread.iloc[:, 0] - vol_spread.iloc[:, 1]

    vol_spread['vol_spread_sm'] = vol_spread['vol_spread'].ewm(span=10).mean()

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.merge(df_MC_regime[['Signal_MP_Trend']].reset_index())
    df_all = df_all.set_index('Date')
    df_all = df_all.join(vol_spread)
    df_all.dropna(inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Signal'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    # df_all['Signal_vol_cross'] = np.where(df_all['vol_spread'] > vol_spread['vol_spread_sm'], 1, -1)
    df_all['Signal_vol'] = np.where(df_all['vol_spread_sm'] > 0, 1, -1)

    df_all['Composite'] = (df_all['rel_slope_Signal'] + df_all['Signal_MP_Trend']) / 2

    df_all['Composite_vol'] = (df_all['rel_slope_Signal'] + df_all['Signal_MP_Trend'] + df_all['Signal_vol']) / 3

    df_all['Composite_vol_MP'] = (df_all['Signal_MP_Trend'] + df_all['Signal_vol']) / 2

    signals_list = ['rel_slope_Signal', 'Signal_MP_Trend', 'Signal_vol', 'Composite', 'Composite_vol',
                    'Composite_vol_MP']

    df_signal = df_all[signals_list].copy()
    df_signal['Currency'] = ccy

    df_signals_list.append(df_signal)

    for factor in signals_list:
        df_performance[factor + '_' + country] = performance(df_all[factor],
                                                             total_returns=np.log(df_all['Spot']).diff()).cumsum()
        returns[factor + '_' + country] = performance(df_all[factor],
                                                      total_returns=np.log(df_all['Spot']).diff())

df_signals = pd.concat(df_signals_list)
df_signals[['rel_slope_Signal', 'Signal_MP_Trend', 'Composite', 'Currency']].groupby(by='Currency').tail(1)

# per currency signal manual
# ==========================
date_slice = '2023-06-01'
keep_signals = ['rel_slope_Signal', 'Signal_MP_Trend', 'Composite', 'Currency']
ccy_temp = 'JPYUSD'
temp_signals = df_signals[df_signals.Currency == ccy_temp]
temp_signals = temp_signals[temp_signals.index >= date_slice][keep_signals]
spot_ccy = get_historical_spots(ccy=ccy_temp).dropna()
temp_signals = temp_signals.join(spot_ccy)

performance(temp_signals['Signal_MP_Trend'], temp_signals[ccy_temp]).cumsum().plot(title=ccy_temp)

filter_list = [x + '_' for x in signals_list]

df_temp_list = []
df_temp_ret_list = []
for fil in filter_list:
    df_temp = df_performance.filter(regex=fil).fillna(method='ffill').mean(axis=1)
    df_temp_ret = returns.filter(regex=fil).fillna(method='ffill').mean(axis=1)
    df_temp_list.append(df_temp)
    df_temp_ret_list.append(df_temp_ret)

## portfolio level performance
# df1 = df_performance.filter(regex='rel_slope_Signal_').fillna(method='ffill').mean(axis=1)
# df2 = df_performance.filter(regex='Signal_MP_Trend_').fillna(method='ffill').mean(axis=1)
# df3 = df_performance.filter(regex='Composite_').fillna(method='ffill').mean(axis=1)
# df_perf_portf = pd.concat([df1, df2, df3], axis=1)
df_perf_portf = pd.concat(df_temp_list, axis=1)
df_perf_portf.columns = signals_list

df_perf_portf.plot()

# ret_port1 = returns.filter(regex='rel_slope_Signal_').fillna(method='ffill').mean(axis=1)
# ret_port2 = returns.filter(regex='Signal_MP_Trend_').fillna(method='ffill').mean(axis=1)
# ret_port3 = returns.filter(regex='Composite_').fillna(method='ffill').mean(axis=1)
# ret_port = pd.concat([ret_port1, ret_port2, ret_port3], axis=1)
ret_port = pd.concat(df_temp_ret_list, axis=1)
ret_port.columns = signals_list

df_performance = df_performance.join(df_perf_portf)
returns = returns.join(ret_port)

df_performance[['rel_slope_Signal_NOK']].plot()

df_performance[['rel_slope_Signal_NOK']][df_performance.index >= '2007-05-25'].plot()

#############
df_performance_filtered = df_performance.filter(regex='Composite_')

keep_comp = df_performance_filtered.columns.drop(list(df_performance_filtered.filter(regex='Composite_vol')))

df_performance_filtered = df_performance_filtered[keep_comp]
df_performance_filtered_slope = df_performance.filter(regex='rel_slope_')

df_performance_filtered = df_performance_filtered.fillna(method='ffill')

df_stats = pd.DataFrame()
for i in range(0, df_performance_filtered.shape[1]):
    cctry = df_performance_filtered.iloc[:, i].name[10:13]
    df_stats[cctry] = mainStats(df_performance_filtered.iloc[:, i].diff()).transpose()['IR']

df_stats_all = []

for name in ['rel_slope_Signal', 'Signal_MP_Trend', 'Signal_vol', 'Composite']:
    df_performance_filtered = df_performance.filter(regex=name)
    df_performance_filtered = df_performance_filtered.fillna(method='ffill')
    for i in range(0, df_performance_filtered.shape[1]):
        cctry = df_performance_filtered.iloc[:, i].name.split(name)[1]
        temp = mainStats(df_performance_filtered.iloc[:, i].diff())
        temp.columns = [name + cctry]
        df_stats_all.append(temp)

df_stats_all = pd.concat(df_stats_all, axis=1)
df_stats_all.to_csv('S:\Investment Solutions Group\Quant_research\Moustafa\Slope_and_MP_Factor_results\stats_all.csv')

## plot performance portfolio
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Slope_and_MP_Factor_results'
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
df_stats.transpose().sort_values(by='Stat', ascending=False).plot.bar(ax=axes[0],
                                                                      title='Slope Factor + Monetary Policy Factor Combined')
df_performance_filtered.plot(ax=axes[1])

df_performance_filtered.mean(axis=1).plot(ax=axes[2], title='Portfolio Level')
plt.savefig(os.path.join(fig_path, 'general_results'))

df_performance_filtered

###############################

for ccy in currencies:
    currency_path = os.path.join(fig_path, ccy)
    # os.mkdir(currency_path)

    save_pnl = os.path.join(fig_path, currency_path, ccy + '_PnL')
    save_Signals = os.path.join(fig_path, currency_path, ccy + '_Signals')
    monthly_stats = os.path.join(fig_path, currency_path, ccy + '_monthly_stats')
    save_risk = os.path.join(fig_path, currency_path, ccy + '_risk')

    # os.mkdir(monthly_stats)
    # os.mkdir(save_risk)

    country = ccy[0:3]
    name_to_filter = '_' + country
    df_ccy = df_performance.filter(regex=name_to_filter)
    df_ccy.plot()
    plt.savefig(save_pnl)

    # signals
    if ccy != 'por':
        df_signals[df_signals.Currency == ccy].drop(['Currency'], axis=1).plot(subplots=True, title=ccy)
        plt.savefig(save_Signals)

    # monthly returns
    ret = returns.filter(regex=name_to_filter)
    ret = ret[ret.columns.drop(list(ret.filter(regex='Composite_vol')))]

    ret0 = ret.iloc[:, 0]
    mrh.plot(ret0, eoy=True, title=ret0.name)
    plt.savefig(os.path.join(monthly_stats, 'Strategy1'))

    ret1 = ret.iloc[:, 1]
    mrh.plot(ret1, eoy=True, title=ret1.name)
    plt.savefig(os.path.join(monthly_stats, 'Strategy2'))

    ret2 = ret.iloc[:, 2]
    mrh.plot(ret2, eoy=True, title=ret2.name)
    plt.savefig(os.path.join(monthly_stats, 'Strategy3'))

    # risky strategy
    density_plot = ret.resample('Y').mean().plot.density(title='Risk of the strategy (Yearly Returns)')
    density_plot.axvline(x=0, color='black', linewidth=2)
    plt.savefig(os.path.join(save_risk, 'Risk_yearly'))

    # risky strategy
    density_plot = ret.resample('M').mean().plot.density(title='Risk of the strategy (Monthly Returns)')
    density_plot.axvline(x=0, color='black', linewidth=1, )
    plt.savefig(os.path.join(save_risk, 'Risk_monthly'))

ret = df_performance.filter(regex='EUR').pct_change()
mrh.plot(ret, eoy=True, title=ret.name)

# eurate0021 - ECB meetings
# usrate2056 - Fed meetings

from myScripts.functions.download_and_prepare_data import get_macrobond_data

df_meetings = get_macrobond_data(['eurate0021', 'usrate2056']).dropna()
df_meetings['ECB_meeting_mod'] = df_meetings['eurate0021']
df_meetings['Fed_meeting_mod'] = df_meetings['usrate2056']

import datetime

dates_previous = df_meetings[df_meetings['ECB_meeting_mod'] == 1].index.date - datetime.timedelta(
    days=1)  # previous date
dates_next = df_meetings[df_meetings['ECB_meeting_mod'] == 1].index.date + datetime.timedelta(days=1)  # next date

dates_previous = pd.DatetimeIndex(dates_previous)
dates_next = pd.DatetimeIndex(dates_next)

df_signal_ccy = df_signals[df_signals.Currency == 'EURUSD']
df_signal_ccy = df_signal_ccy.join(df_meetings)

df_signal_ccy['rel_slope_Signal_ex_meetings'] = df_signal_ccy['rel_slope_Signal']
df_signal_ccy['rel_slope_Signal_ex_ECB_meetings'] = np.where(df_signal_ccy['eurate0021'] == 1, 0,
                                                             df_signal_ccy['rel_slope_Signal'])

df_signal_ccy['rel_slope_Signal_ex_FED_meetings'] = np.where(df_signal_ccy['usrate2056'] == 1, 0,
                                                             df_signal_ccy['rel_slope_Signal'])

df_signal_ccy['rel_slope_Signal_ex_FED_ECB_meetings'] = np.where(
    (df_signal_ccy['eurate0021'] == 1) | (df_signal_ccy['usrate2056']) == 1, 0,
    df_signal_ccy['rel_slope_Signal'])

df_spot = get_historical_spots('EURUSD')
df_signal_ccy = df_signal_ccy.join(df_spot)

df_performance_ex_meetings = pd.DataFrame()
for factor in ['rel_slope_Signal', 'rel_slope_Signal_ex_ECB_meetings', 'rel_slope_Signal_ex_FED_meetings',
               'rel_slope_Signal_ex_FED_ECB_meetings']:
    df_performance_ex_meetings[factor] = performance(df_signal_ccy[factor],
                                                     total_returns=np.log(df_signal_ccy['EURUSD']).diff()).cumsum()

df_performance_ex_meetings.plot(title='EUR_REL_SLOPE_FACTOR')
# df_meetings[(df_meetings.index >= '2023-01-01') & (df_meetings.index <= '2023-06-08')].plot()

df_signal_ccy.to_clipboard()
