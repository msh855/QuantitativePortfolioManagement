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

currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
# append
df_performance_results = []
df_stats_results = []
returns_results = []
returns_fed_factor_results = []
df_all_results = []

for ccy in currencies:
    country = ccy[0:3]
    yields = int_name = '2Y'
    spot_mb = get_historical_spots(ccy=ccy).dropna()
    spot_mb = spot_mb[spot_mb.index >= '1985-01-01']
    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    # policy rate
    # =============
    df_policy_rates = get_policy_rates(country=['US', country])

    # calculate slopes
    # ==================
    df_US_spread = create_spreads_from_policy_rates(country='US', df_policy_rates=df_policy_rates,
                                                    yields='2Y')

    df_spreads = df_US_spread

    # Make smoothing average
    # ==================================================================================================================
    df_ema = make_sma(df=df_spreads[df_spreads.index >= '1985-01-01'], spans=[25, 50, 60, 90, 120, 200])
    col_names = df_ema.columns.tolist()
    df_ema.dropna(inplace=True)

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.set_index('Date')
    df_all.dropna(inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)

    # check other windows
    for sm in col_names:
        df_all['factor_' + sm] = np.where(df_all['slope_US'] > df_all[sm], 1, -1)

    factors = ['Fed_Factor', 'factor_slope_US_sm25', 'factor_slope_US_sm50',
               'factor_slope_US_sm60', 'factor_slope_US_sm90', 'factor_slope_US_sm120',
               'factor_slope_US_sm200', 'factor_slope_US_sm_prod']

    df_all['EMA_Composite'] = df_all[factors].mean(axis=1)

    # strategy performance
    # =====================================================================================================================
    df_performance = pd.DataFrame()

    for factor in ['Fed_Factor', 'EMA_Composite']:
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

    # # find best factor according to calmar ratio
    # temp = df_stats.tail(1).melt()
    # temp = temp.set_index('variable').sort_values(by=['value'])
    # best_factor = temp.tail(1).index[0]

    # monthly performance
    # ===================
    returns = performance(df_all['EMA_Composite'], total_returns=np.log(df_all[ccy]).diff())
    returns_fed_factor = performance(df_all['Fed_Factor'], total_returns=np.log(df_all[ccy]).diff())

    returns = pd.DataFrame(returns)
    returns.columns = ['ret']
    returns['best_factor'] = 'EMA_Composite'

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

    # df_all.columns = [x.replace(country, "Foreign") for x in df_all.columns]

    df_all_results.append(df_all)

# collect results
df_performance_all = pd.concat(df_performance_results, axis=0)
df_stats_all = pd.concat(df_stats_results, axis=0)
returns_results_all = pd.concat(returns_results, axis=0)
returns_fed_factor_all = pd.concat(returns_fed_factor_results, axis=0)
df_all_all = pd.concat(df_all_results, axis=0)

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Yield_Strategy_presentation\Fed_Factor_Only_proxy_rate'

for ccy in currencies:
    df_performance = df_performance_all[df_performance_all['Currency'] == ccy]
    df_stats = df_stats_all[df_stats_all['Currency'] == ccy]
    df_all = df_all_all[df_all_all['Currency'] == ccy]
    returns = returns_results_all[returns_results_all['Currency'] == ccy]
    returns_fed_factor = returns_fed_factor_all[returns_fed_factor_all['Currency'] == ccy]
    df_eoy = mrh.get(returns['ret'], eoy=True)
    best_factor = returns.iloc[:, 1].unique()[0]

    currency_path = os.path.join(fig_path, ccy)
    os.mkdir(currency_path)

    save_pnl = os.path.join(fig_path, currency_path, ccy + '_PnL')
    save_stats = os.path.join(fig_path, currency_path, ccy + '_Stats')
    monthly_stats = os.path.join(fig_path, currency_path, ccy + '_monthly_stats')
    monthly_stats_fed = os.path.join(fig_path, currency_path, ccy + '_monthly_stats_fed')

    # define subplot layout
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
    df_performance.plot(ax=axes[0], title=ccy)
    df_all[['EMA_Composite']].plot(ax=axes[1], title=ccy + ': Signal ' + 'EMA_Composite', legend=False)
    df_all[['Fed_Factor']].plot(ax=axes[2], title=ccy + ': Signal ' + 'Fed_Factor', legend=False)
    plt.savefig(save_pnl)

    # Performance stats
    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_stats.plot.bar(ax=axes[0], title=ccy)
    df_eoy[['eoy']].plot.bar(ax=axes[1], title=ccy + ':' + best_factor + ' EOY Returns', legend=False, figsize=(20, 10))
    plt.savefig(save_stats)

    mrh.plot(returns, eoy=True, title=ccy + ':' + best_factor)
    plt.savefig(monthly_stats)


    mrh.plot(returns_fed_factor, eoy=True, title=ccy + ':' + 'Fed_Factor')
    plt.savefig(monthly_stats_fed)

