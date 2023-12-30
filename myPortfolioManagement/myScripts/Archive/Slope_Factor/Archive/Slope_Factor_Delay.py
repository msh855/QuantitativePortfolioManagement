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

start_date = '2018-01-01'

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
    df_ema = make_sma(df=df_spreads[df_spreads.index >= start_date], spans=[5, 25, 50, 60, 90, 120, 200])
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
    df_all['rel_slope_Factor_sm'] = np.where(df_all['rel_slope_sm5'] > df_all['rel_slope_sm50'], -1, 1)
    df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    # df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)
    # df_all['Foreign_Factor'] = np.where(df_all['slope_' + country] > df_all['slope_' + country + '_sm50'], -1, 1)
    # df_all['Composite_Fed_Foreign_Factor'] = (0.7 * df_all['Fed_Factor'] + 0.3 * df_all['Foreign_Factor'])

    factors = ['rel_slope_Factor', 'rel_slope_Factor_sm']

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

    for factor in factors:
        df_performance[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()
        df_performance[factor + '_delay'] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff(),
                                                        delay=2).cumsum()

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

    # collect
    df_performance['Currency'] = ccy

    df_stats['Currency'] = ccy

    df_all['Currency'] = ccy

    # append
    df_performance_results.append(df_performance)
    df_stats_results.append(df_stats)

    df_all.columns = [x.replace(country, "Foreign") for x in df_all.columns]
    df_all_results.append(df_all)

# collect data to save
sma_data = []
df_all_data = []

# collect results
df_performance_all = pd.concat(df_performance_results, axis=0)

# save for Ana/Luke

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Slope_Factor_Stategy_Delay'

for ccy in currencies:
    df_performance = df_performance_all[df_performance_all['Currency'] == ccy]
    currency_path = os.path.join(fig_path, ccy)
    os.mkdir(currency_path)

    save_pnl = os.path.join(fig_path, currency_path, ccy + '_PnL')

    # define subplot layout
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    df_performance.plot(ax=axes[0], title=ccy)
    plt.savefig(save_pnl)
