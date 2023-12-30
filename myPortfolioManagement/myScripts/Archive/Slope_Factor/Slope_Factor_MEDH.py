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

currencies = ['NOKUSD'] # ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
start_date = '2000-01-01'

#### Monetary Policy Trend Factor
#################################
# trends in yields
df_yiedls = get_yields_countries()
df_yiedls = df_yiedls[df_yiedls.index >= start_date]
df_yileds_trend = df_yiedls.fillna(method='ffill').rolling(30 * 12).mean().pct_change()

df_yileds_trend[['2Y_US']].plot()

df_MEDH_Signals = pd.read_csv('S:\\Investment Solutions Group\\Quant_research\\Moustafa\\singals_factors.csv')
df_MEDH_Signals = df_MEDH_Signals.set_index(df_MEDH_Signals.columns[0])
df_MEDH_Signals.index.name = 'Date'
df_MEDH_Signals.index = pd.DatetimeIndex(df_MEDH_Signals.index)

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

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.merge(df_MC_regime[['Signal_MP_Trend']].reset_index())

    df_MDH_temp = df_MEDH_Signals[df_MEDH_Signals.Currency == country]
    df_MDH_temp = df_MDH_temp.drop(['Currency'], axis=1)
    df_all = df_all.merge(df_MDH_temp.reset_index())
    df_all = df_all.set_index('Date')
    df_all = df_all.dropna()

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Signal'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    df_all['Composite'] = (df_all['rel_slope_Signal'] + df_all['Signal_MP_Trend']) / 2

    df_all['prod'] = df_all['OR_fact'] * 1 / 3 + (
            df_all['value'] * 0.3 + df_all['momentum'] * 0.2 + df_all['carry'] * 0.3 + df_all[
        'equity'] * 0.2) * 2 / 3

    df_all['prod_composite'] = df_all['OR_fact'] * 1 / 3 + (
            df_all['value'] * 0.3 + df_all['momentum'] * 0.2 + df_all['carry'] * 0.3 + df_all[
        'equity'] * 0.2) * 1 / 3 + df_all['Composite'] * 1 / 3

    df_all['prod_slope_factor'] = df_all['OR_fact'] * 0 + (
            df_all['value'] * 0.2 + df_all['momentum'] * 0.2 + df_all['carry'] * 0.2 + df_all[
        'equity'] * 0.2) * 2 / 3 + df_all['rel_slope_Signal'] * 1 / 3

    signals_list = ['prod', 'prod_composite', 'prod_slope_factor']

    df_signal = df_all[signals_list].copy()
    df_signal['Currency'] = ccy

    df_signals_list.append(df_signal)

    for factor in signals_list:
        df_performance[factor + '_' + country] = performance(df_all[factor],
                                                             total_returns=np.log(df_all['Spot']).diff()).cumsum()
        returns[factor + '_' + country] = performance(df_all[factor],
                                                      total_returns=np.log(df_all['Spot']).diff())

df_signals = pd.concat(df_signals_list)
df_signals.groupby(by='Currency').tail(1)

import matplotlib.pyplot as plt
save_pic = 'S:\Investment Solutions Group\Quant_research\Moustafa\Slope_MEDH\\various_experiments'

for ccy in currencies:
    country = ccy[0:3]
    keep_signals = [x + '_' + country for x in signals_list]
    df_performance[keep_signals].fillna(method='ffill').plot(title=ccy)
    df_performance.columns = ['prod_'+country,  'prod_Alphasiv_SF_enhanced',  'Alpha_siv_prod_SF']
    plt.savefig(os.path.join(save_pic, ccy))


df_performance.plot()

filter_list = [x + '_' for x in signals_list]

df_temp_list = []
df_temp_ret_list = []
for fil in filter_list:
    df_temp = df_performance.filter(regex=fil).fillna(method='ffill').mean(axis=1)
    df_temp_ret = returns.filter(regex=fil).fillna(method='ffill').mean(axis=1)
    df_temp_list.append(df_temp)
    df_temp_ret_list.append(df_temp_ret)

## portfolio level performance
df_perf_portf = pd.concat(df_temp_list, axis=1)
df_perf_portf.columns = signals_list
df_perf_portf.plot()
plt.savefig(os.path.join(save_pic, 'port_level'))

#############
df_stats_port = []
for col in df_perf_portf.columns:
    df_stats_port.append(mainStats(df_perf_portf[col].fillna(method='ffill').diff()).transpose())

df_stats_port = pd.concat(df_stats_port)
df_stats_port = df_stats_port.transpose()
df_stats_port.columns = df_perf_portf.columns

df_stats_port.transpose()['IR'].plot.bar(rot=20, title='Performance: 2015-03-17 - 2023-06-09')
plt.savefig(os.path.join(save_pic, 'port_IR'))
df_stats_port.transpose()['Calmar'].plot.bar(rot=20, title='Performance: 2015-03-17 - 2023-06-09')
plt.savefig(os.path.join(save_pic, 'port_Calmar'))

ret_port = pd.concat(df_temp_ret_list, axis=1)
ret_port.columns = signals_list

ret_port.columns = ['Prod', 'Alpha_Siv_plus_Slope_MP_Factor', 'Alpha_Siv_plus_Slope_factor']
for i in range(0,3):
    ret2 = ret_port.iloc[:, i]
    mrh.plot(ret2, eoy=True, title=ret2.name)
    plt.savefig(os.path.join(save_pic, ret2.name))

# df_performance = df_performance.join(df_perf_portf)
returns = returns.join(ret_port)

#############
df_stats = pd.DataFrame()
for i in range(0, df_performance.shape[1]):
    cctry = df_performance.iloc[:, i].name[10:13]
    df_stats[cctry] = mainStats(df_performance.fillna(method='ffill').iloc[:, i].diff()).transpose()['IR']

df_stats_all_list = []
for ccy in currencies:
    country = ccy[0:3]
    keep_signals = [x + '_' + country for x in signals_list]

    temp_df = df_performance[keep_signals]
    nseries = len(temp_df.columns)

    df_stats_ccy_list = []

    for i in range(0, nseries):
        fact_name = temp_df.iloc[:, i].name
        df_stats_temp = mainStats(temp_df.iloc[:, i])
        df_stats_temp.columns = [fact_name]
        df_stats_ccy_list.append(df_stats_temp)

    df_stats_ccy = pd.concat(df_stats_ccy_list, axis=1)

    df_stats_all_list.append(df_stats_ccy)

df_stats_all_currencies = pd.concat(df_stats_all_list, axis=1)

for ccy in currencies:
    country = ccy[0:3]
    df_stats_all_currencies[df_stats_all_currencies.index.isin(['IR', 'MDD/vol'])].filter(
        regex=country).transpose().plot.barh(title=country)
