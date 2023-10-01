import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, get_bbq_data, get_historical_spots, \
    get_bbq_ccy
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

currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']

start_date = '2016-01-01'

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

df_bbq_2y_bonds.to_clipboard()

# append
df_performance_results = []
df_stats_results = []
returns_results = []
returns_fed_factor_results = []
df_all_results = []

for ccy in currencies:
    country = ccy[0:3]
    yields = int_name = '2Y'

    spot_mb = get_bbq_ccy(ccy=ccy, start_date=start_date).dropna()
    spot_mb = spot_mb[spot_mb.index >= '1985-01-01']

    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    # save
    spot_mb_copy = spot_mb.copy()
    spot_mb_copy.columns = ['Spot']
    spot_mb_copy['Data'] = 'Spot'
    spot_mb_copy['Currency'] = ccy

    # policy rate
    # =============
    df_policy_rates = df_bbq_policy_rates[['US', country]]

    # calculate slopes
    # ==================
    poliy = df_bbq_policy_rates[['US']]
    yields = df_bbq_2y_bonds[['2Y_US']]
    df_US_spread = poliy.reset_index().merge(yields.reset_index())
    df_US_spread['slope_US'] = df_US_spread['US'] - df_US_spread['2Y_US']

    poliyf = df_bbq_policy_rates[[country]]
    yieldsf = df_bbq_2y_bonds[['2Y_' + country]]
    df_Foreign_spread = poliyf.reset_index().merge(yieldsf.reset_index())
    df_Foreign_spread['slope_' + country] = df_Foreign_spread[country] - df_Foreign_spread['2Y_' + country]

    df_spreads = df_US_spread.merge(df_Foreign_spread)
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads['slope_' + country] - df_spreads['slope_US']

    # Make smoothing average
    # ==================================================================================================================
    df_ema = make_sma(df=df_spreads[df_spreads.index >= '1985-01-01'], spans=[25, 50, 60, 90, 120, 200])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.set_index('Date')

    if country == 'NZD':
        df_all.fillna(method='ffill', inplace=True)

    df_all.dropna(inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    # df_all['relative_path_Factor'] = np.where(df_all['rel_slope_smoothed_ind_sm'] > 0, -1, 1)
    # df_all['rel_slope_and_path_Factor'] = (0.7 * df_all['rel_slope_Factor'] + 0.3 * df_all['relative_path_Factor'])

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

    # find best factor according to calmar ratio
    temp = df_stats.tail(1).melt()
    temp = temp.set_index('variable').sort_values(by=['value'])
    best_factor = temp.tail(1).index[0]

    # monthly performance
    # ===================
    returns = performance(df_all[best_factor], total_returns=np.log(df_all[ccy]).diff())
    returns_fed_factor = performance(df_all['Fed_Factor'], total_returns=np.log(df_all[ccy]).diff())

    returns = pd.DataFrame(returns)
    returns.columns = ['ret']
    returns['best_factor'] = best_factor

    returns_fed_factor = pd.DataFrame(returns_fed_factor)
    returns_fed_factor.columns = ['ret_Fed_Factor']

    # collect
    df_performance['Currency'] = ccy
    returns['Currency'] = ccy
    df_stats['Currency'] = ccy
    returns_fed_factor['Currency'] = ccy
    df_all['Currency'] = ccy

    # append
    df_performance_results.append(df_performance)
    df_stats_results.append(df_stats)
    returns_results.append(returns)
    returns_fed_factor_results.append(returns_fed_factor)

    df_all.columns = [x.replace(country, "Foreign") for x in df_all.columns]
    df_all_results.append(df_all)

# collect results
df_performance_all = pd.concat(df_performance_results, axis=0)
df_stats_all = pd.concat(df_stats_results, axis=0)
returns_results_all = pd.concat(returns_results, axis=0)
returns_fed_factor_all = pd.concat(returns_fed_factor_results, axis=0)
df_all_all = pd.concat(df_all_results, axis=0)

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Yield_Strategy_presentation_bbq_2016'

for ccy in currencies:
    df_performance = df_performance_all[df_performance_all['Currency'] == ccy]
    df_stats = df_stats_all[df_stats_all['Currency'] == ccy]
    df_all = df_all_all[df_all_all['Currency'] == ccy]
    returns = returns_results_all[returns_results_all['Currency'] == ccy]
    df_eoy = mrh.get(returns['ret'], eoy=True)
    best_factor = returns.iloc[:, 1].unique()[0]

    currency_path = os.path.join(fig_path, ccy)
    os.mkdir(currency_path)

    save_pnl = os.path.join(fig_path, currency_path, ccy + '_PnL')
    save_stats = os.path.join(fig_path, currency_path, ccy + '_Stats')
    monthly_stats = os.path.join(fig_path, currency_path, ccy + '_monthly_stats')

    # define subplot layout
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    df_performance.plot(ax=axes[0], title=ccy + '_BBQ')
    df_all[[best_factor]].plot(ax=axes[1], title=ccy + ': Signal ' + best_factor, legend=False)
    plt.savefig(save_pnl)

    # Performance stats
    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_stats.plot.bar(ax=axes[0], title=ccy)
    df_eoy[['eoy']].plot.bar(ax=axes[1], title=ccy + ':' + best_factor + ' EOY Returns_BBQ', legend=False, figsize=(20, 10))
    plt.savefig(save_stats)

    mrh.plot(returns, eoy=True, title=ccy + ':' + best_factor + '_BBQ')
    plt.savefig(monthly_stats)


df_perf_average = df_stats_all.reset_index().melt(id_vars=['Currency', 'index'])
df_perf_average = df_perf_average[df_perf_average['index'].isin(['IR', 'Calmar'])]

vars = ['rel_slope_Factor', 'Fed_Factor', 'Foreign_Factor', 'Composite_Fed_Foreign_Factor', 'Composite_all']

# note this is based on the sample size of the relative slope

for var in vars:
    tem = df_perf_average[df_perf_average['variable'] == var]
    temp = tem.pivot(index='Currency', columns='index', values='value')
    temp.columns.name = 'Metric'
    ax = temp.sort_values(by=['IR'], ascending=False).plot(kind='bar', figsize=(20, 10),
                                                           title='Performance of Factor BBQ: ' + var,
                                                           legend=True, xlabel=None)
    plt.savefig(os.path.join(fig_path, 'performance_plots', var + '_signal'))
