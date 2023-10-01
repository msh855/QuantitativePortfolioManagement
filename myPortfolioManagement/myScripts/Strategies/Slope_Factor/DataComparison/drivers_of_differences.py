import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, get_bbq_ccy, get_policy_rates, get_historical_spots, \
    create_spreads_from_policy_rates, get_bbq_data, df_bbq_2y_bonds
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

# currencies
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']

# bbq data: policy rates
bbq_tickers_policy_rates = ['FDTR Index', 'EURR002W Index', 'SZBRLOMB Index', 'BOJDPBAL Index', 'UKBRBASE Index',
                            'FMSTTGOV Index', 'NZOCRS Index', 'RBATCTR Index', 'SWBRDEP Index', 'NOBRDEPA Index']

df_bbq_policy_rates = get_bbq_data(bbq_tickers_policy_rates, start_date=start_date)

mapping = {'FDTR Index': 'US',
           'SZBRLOMB Index': 'CHF',
           'EURR002W Index': 'EUR',
           'BOJDPBAL Index': 'JPY',
           'UKBRBASE Index': 'GBP',
           'FMSTTGOV Index': 'CAD',
           'NZOCRS Index': 'NZD',
           'RBATCTR Index': 'AUD',
           'SWBRDEP Index': 'SEK',
           'NOBRDEPA Index': 'NOK'}

df_bbq_policy_rates = df_bbq_policy_rates.rename(columns=mapping)
df_bbq_policy_rates.fillna(method="ffill", inplace=True)

# 2 year Bonds bbq
# ================
bbq_tickers_bonds = ['USGG2YR Index', 'GTDEM2Y Govt', 'GTCHF2Y Govt', 'GTJPY20Y Govt', 'GTGBP2Y Govt', 'GCAN2YR Index',
                     'GNZGB2 Index',
                     'GTAUD2Y Govt', 'GTSEK2Y Govt', 'GTNOK2Y Govt']

df_bbq_2y_bonds = get_bbq_data(bbq_tickers_bonds, start_date=start_date)

mapping_bonds = {'USGG2YR Index': 'US',
                 'GTCHF2Y Govt': 'CHF',
                 'GTDEM2Y Govt': 'EUR',
                 'GTJPY20Y Govt': 'JPY',
                 'GTGBP2Y Govt': 'GBP',
                 'GCAN2YR Index': 'CAD',
                 'GNZGB2 Index': 'NZD',
                 'GTAUD2Y Govt': 'AUD',
                 'GTSEK2Y Govt': 'SEK',
                 'GTNOK2Y Govt': 'NOK'}
df_bbq_2y_bonds = df_bbq_2y_bonds.rename(columns=mapping_bonds)
df_bbq_2y_bonds.columns = ['2Y_' + x for x in df_bbq_2y_bonds.columns]

# append
df_stats_results = []
start_date = '2000-01-01'

for ccy in currencies:
    country = ccy[0:3]
    yields = int_name = '2Y'

    spot_mb = get_bbq_ccy(ccy=ccy, start_date=start_date).dropna()
    #spot_mb = get_historical_spots(ccy=ccy).dropna()
    spot_mb = spot_mb[spot_mb.index >= start_date]

    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    # policy rate
    # =============
    # df_policy_rates = get_policy_rates(country=['US', country])
    df_policy_rates = df_bbq_policy_rates[['US', country]]

    # calculate slopes
    # ==================
    # df_US_spread = create_spreads_from_policy_rates(country='US', df_policy_rates=df_policy_rates,
    #                                                 yields='2Y')
    #
    # df_Foreign_spread = create_spreads_from_policy_rates(country=country, df_policy_rates=df_policy_rates,
    #                                                      yields='2Y')

    # calculate slopes
    # ==================
    poliy = df_bbq_policy_rates[['US']]
    yields = df_bbq_2y_bonds[['2Y_US']]
    df_US_spread = poliy.reset_index().merge(yields.reset_index())
    df_US_spread['slope_US'] = df_US_spread['US'] - df_US_spread['2Y_US']
    df_US_spread = df_US_spread[['slope_US', 'Date']]
    df_US_spread = df_US_spread.set_index('Date')

    poliyf = df_bbq_policy_rates[[country]]
    yieldsf = df_bbq_2y_bonds[['2Y_' + country]]
    df_Foreign_spread = poliyf.reset_index().merge(yieldsf.reset_index())
    df_Foreign_spread['slope_' + country] = df_Foreign_spread[country] - df_Foreign_spread['2Y_' + country]
    df_Foreign_spread = df_Foreign_spread[['slope_' + country, 'Date']]
    df_Foreign_spread = df_Foreign_spread.set_index('Date')

    # merge spreads
    df_spreads = df_US_spread.reset_index().merge(df_Foreign_spread.reset_index())
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads[name] - df_spreads['slope_US']

    # Make smoothing average
    # ==================================================================================================================
    df_ema = make_sma(df=df_spreads[df_spreads.index >= start_date], spans=[25, 50, 60, 90, 120, 200])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.set_index('Date')
    df_all.dropna(inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)
    df_all['Foreign_Factor'] = np.where(df_all['slope_' + country] > df_all['slope_' + country + '_sm50'], -1, 1)
    df_all['Composite_Fed_Foreign_Factor'] = (0.7 * df_all['Fed_Factor'] + 0.3 * df_all['Foreign_Factor'])

    factors = ['rel_slope_Factor', 'Fed_Factor', 'Foreign_Factor', 'Composite_Fed_Foreign_Factor']

    df_all['Composite_all'] = df_all[factors].mean(axis=1)

    # strategy performance
    # =====================================================================================================================
    df_performance = pd.DataFrame()

    for factor in factors + ['Composite_all']:
        df_performance[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()

    # get stats
    # ======================================================================================================================
    df_stats_list = list()
    for i in range(0, df_performance.shape[1]):
        df_stats_list.append(mainStats(df_performance.iloc[:, i].diff()))

    df_stats = pd.concat(df_stats_list, axis=1)
    df_stats.columns = df_performance.columns
    df_stats = df_stats.drop(['min', 'max', 'skew', 'kurtosis', 'MDD/vol'])

    # collect
    df_stats['Currency'] = ccy
    df_stats_results.append(df_stats)

# benchmark
df_stats_all_macrobond = pd.concat(df_stats_results, axis=0)

# collect results
df_stats_all_diff_bbq = pd.concat(df_stats_results, axis=0)

# collect results
df_stats_all_diff_rates = pd.concat(df_stats_results, axis=0)

# collect s2 year bonds
df_stats_all_diff_bonds = pd.concat(df_stats_results, axis=0)

# all bbq
df_stats_all_bbq = pd.concat(df_stats_results, axis=0)

df_stats_all_macrobond['Source'] = 'All MacroBond'
df_stats_all_diff_bbq['Source'] = 'Only Spots'
df_stats_all_diff_rates['Source'] = 'Only Rates'
df_stats_all_diff_bonds['Source'] = 'Only Bonds'
df_stats_all_bbq['Source'] = 'All bbq'

df_all_comparison = pd.concat(
    [df_stats_all_macrobond, df_stats_all_diff_bbq, df_stats_all_diff_rates, df_stats_all_diff_bonds, df_stats_all_bbq])

#
df_perf_average = df_all_comparison[df_all_comparison.index.isin(['IR'])]

df_perf_average.pivot(index='Currency', columns='Source', values='rel_slope_Factor').plot.bar()

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Yield_Strategy_presentation_drivers_of_diff'

vars = ['rel_slope_Factor', 'Fed_Factor', 'Composite_all']

for var in vars:
    temp = df_perf_average.pivot(index='Currency', columns='Source', values=var)
    ax = temp.plot(kind='bar', figsize=(20, 10),
                                                           title=' Sharpe Ratio: Source of Diff for Factor ' + var,
                                                           legend=True, xlabel=None)
    plt.savefig(os.path.join(fig_path, var + '_signal'))

#
#
#
# df_all_comparison.to_csv('S:\Investment Solutions Group\Quant_research\Moustafa\compare_macrobond_bbq.csv')
