import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, \
    get_policy_rates, get_historical_spots, create_spreads_from_policy_rates
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

########################################################################################################################
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']

# append
df_performance_results = []
df_stats_results = []
returns_results = []
returns_fed_factor_results = []
df_all_results = []

spots_data = []
policy_rates_data = []
US_spreads_data = []
foreign_spreads_data = []
spreads_clean = []
sma_data = []
df_all_data = []

start_date = '2023-05-10'

for ccy in currencies:
    country = ccy[0:3]
    yields = int_name = '2Y'
    spot_mb = get_historical_spots(ccy=ccy).dropna()
    spot_mb = spot_mb[spot_mb.index >= start_date]
    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    # save
    spot_mb_copy = spot_mb.copy()
    spot_mb_copy.columns = ['Spot']
    spot_mb_copy['Data'] = 'Spot'
    spot_mb_copy['Currency'] = ccy
    spots_data.append(spot_mb_copy)

    # policy rate
    # =============
    df_policy_rates = get_policy_rates(country=['US', country])

    # save
    df_policy_rates_copy = df_policy_rates.copy()
    df_policy_rates_copy.columns = ['US_rate', 'Foreign']
    df_policy_rates_copy['Data'] = 'Policy Rates'
    df_policy_rates_copy['Currency'] = ccy
    policy_rates_data.append(df_policy_rates_copy)

    # calculate slopes
    # ==================
    df_US_spread = create_spreads_from_policy_rates(country='US', df_policy_rates=df_policy_rates,
                                                    yields='2Y')
    # save
    df_US_spread_copy = df_US_spread.copy()
    df_US_spread_copy['Data'] = 'US Slope'
    df_US_spread_copy['Currency'] = ccy
    US_spreads_data.append(df_US_spread_copy)

    df_Foreign_spread = create_spreads_from_policy_rates(country=country, df_policy_rates=df_policy_rates,
                                                         yields='2Y')
    # save
    df_Foreign_spread_copy = df_Foreign_spread.copy()
    df_Foreign_spread_copy.columns = ['slope_Foreign']
    df_Foreign_spread_copy['Data'] = 'Foreign Slope'
    df_Foreign_spread_copy['Currency'] = ccy
    foreign_spreads_data.append(df_Foreign_spread_copy)

    df_spreads = df_US_spread.reset_index().merge(df_Foreign_spread.reset_index())
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads[name] - df_spreads['slope_US']

    df_spreads_copy = df_spreads.copy()
    df_spreads_copy.columns = ['slope_US', 'slope_Foreign', 'rel_slope']

    # Make smoothing average
    # ==================================================================================================================
    df_ema = make_sma(df=df_spreads[df_spreads.index >= start_date], spans=[25, 50, 60, 90, 120, 200])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    # save
    df_spreads_copy['Data'] = 'relative Slope'
    df_spreads_copy['Currency'] = ccy
    spreads_clean.append(df_spreads_copy)

    # save
    df_ema_copy = df_ema.copy()
    df_ema_copy['Data'] = 'EMA'
    df_ema_copy['Currency'] = ccy
    sma_data.append(df_ema_copy)

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

    if country == 'NZD':
        df_all.fillna(method='ffill', inplace=True)

    # save
    df_all_copy = df_all.copy()
    df_all_copy['Data'] = 'Clean Data'
    df_all_copy['Currency'] = ccy
    df_all_data.append(df_all_copy)

    # strategy performance
    # =====================================================================================================================
    df_performance = pd.DataFrame()

    for factor in factors + ['Composite_all']:
        df_performance[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()

    # df_performance['Composite'] = df_performance[['rel_slope_Factor', 'Fed_Factor', 'Foreign_Factor']].mean(axis=1)
    #

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

# collect data to save
sma_data = []
df_all_data = []

# collect results
df_performance_all = pd.concat(df_performance_results, axis=0)
df_stats_all = pd.concat(df_stats_results, axis=0)
returns_results_all = pd.concat(returns_results, axis=0)
returns_fed_factor_all = pd.concat(returns_fed_factor_results, axis=0)
df_all_all = pd.concat(df_all_results, axis=0)

df_all_all[['rel_slope', 'Currency']][df_all_all.index >= '2023-05-10'].groupby('Currency').mean().mean()

# save for Ana/Luke

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Slope_Factor_Stategy_Update_results'

for ccy in currencies:
    df_performance = df_performance_all[df_performance_all['Currency'] == ccy]
    df_stats = df_stats_all[df_stats_all['Currency'] == ccy]
    df_all = df_all_all[df_all_all['Currency'] == ccy]
    returns = returns_results_all[returns_results_all['Currency'] == ccy]
    df_eoy = mrh.get(returns['ret'], eoy=True)
    best_factor = returns.iloc[:, 1].unique()[0]

    currency_path = os.path.join(fig_path, ccy)
    #os.mkdir(currency_path)

    save_pnl = os.path.join(fig_path, currency_path, ccy + '_PnL')
    save_stats = os.path.join(fig_path, currency_path, ccy + '_Stats')
    monthly_stats = os.path.join(fig_path, currency_path, ccy + '_monthly_stats')

    # define subplot layout
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    df_performance.plot(ax=axes[0], title=ccy)
    df_all[[best_factor]].plot(ax=axes[1], title=ccy + ': Signal ' + best_factor, legend=False)
    plt.savefig(save_pnl)

    # Performance stats
    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_stats.plot.bar(ax=axes[0], title=ccy)
    df_eoy[['eoy']].plot.bar(ax=axes[1], title=ccy + ':' + best_factor + ' EOY Returns', legend=False, figsize=(20, 10))
    plt.savefig(save_stats)

    mrh.plot(returns, eoy=True, title=ccy + ':' + best_factor)
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
                                                           title='Performance of Factor: ' + var,
                                                           legend=True, xlabel=None)
    plt.savefig(os.path.join(fig_path, 'performance_plots', var + '_signal'))
