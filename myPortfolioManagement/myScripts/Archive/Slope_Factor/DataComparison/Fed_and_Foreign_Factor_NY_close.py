import pandas as pd
from myScripts.functions.download_and_prepare_data import make_sma
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
import os
# plotting
import matplotlib
import matplotlib.pyplot as plt
import monthly_returns_heatmap as mrh

import matplotlib.dates as mdates

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})


def clean_data(df):
    # df = df.drop([0, 1], axis=0)
    # df = df.set_index('ticker')
    df.set_index('Date')
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df = df.set_index('Date')
    df = df.sort_index()
    return df


start_date = '1998-01-01'

bonds = pd.read_csv('/myScripts/bonds_NY.csv')
ccies = pd.read_csv('/myScripts/currencies_NY.csv')
policy_rates = pd.read_csv('/myScripts/policy_rates_NY.csv')

mapping_bonds = {'USGG2YR BGN Index': 'US',
                 'GSWISS02 BGN Index': 'CHF',
                 'GDBR2 BGN Index': 'EUR',
                 'GJGB2 BGN Index': 'JPY',
                 'GUKG2 BGN Index': 'GBP',
                 'GCAN2YR BGN Index': 'CAD',
                 'GNZGB2 BGN Index': 'NZD',
                 'GACGB2 BGN Index': 'AUD',
                 'GSGB2YR BGN Index': 'SEK',
                 'GNOR2YR BGN Index': 'NOK'}

bonds = bonds.rename(columns=mapping_bonds)
bonds = clean_data(bonds)
bonds.columns = ['2Y_' + x for x in bonds.columns]

# currencies data
ccies.columns = [x.replace(' BGN Curncy', '') for x in ccies.columns]
ccies = clean_data(ccies)
df_ccy = ccies.copy()

for ccy in ccies.columns:
    if 'USD' in ccy[0:3]:
        df_ccy[ccy] = 1 / df_ccy[ccy]

df_ccy.columns = [x.replace('USD', '') for x in df_ccy.columns]

# policy rates
policy_rates = clean_data(policy_rates)
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
policy_rates = policy_rates.rename(columns=mapping)

currencies = df_ccy.columns
df_performance_results = []
df_all_results = []
returns_results_all = []

# slice data
bonds = bonds[bonds.index >= start_date]
policy_rates = policy_rates[policy_rates.index >= start_date]
df_ccy = df_ccy[df_ccy.index >= start_date]

for ccy in currencies:
    country = ccy[0:3]
    yields = int_name = '2Y'

    spot_mb = df_ccy.copy()
    spot_mb = spot_mb[[ccy]]
    spot_mb = spot_mb[spot_mb.index >= '1985-01-01']

    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    # calculate slopes
    # ==================
    poliy = policy_rates[['US']]
    poliy.fillna(method='ffill', inplace=True)
    yields = bonds[['2Y_US']]
    df_US_spread = poliy.reset_index().merge(yields.reset_index())
    df_US_spread['slope_US'] = df_US_spread['US'] - df_US_spread['2Y_US']

    poliyf = policy_rates[[country]]
    poliyf.fillna(method='ffill', inplace=True)
    yieldsf = bonds[['2Y_' + country]]
    df_Foreign_spread = poliyf.reset_index().merge(yieldsf.reset_index())
    df_Foreign_spread['slope_' + country] = df_Foreign_spread[country] - df_Foreign_spread['2Y_' + country]

    df_spreads = df_US_spread.merge(df_Foreign_spread)
    df_spreads = df_spreads.set_index('Date')
    df_spreads['rel_slope'] = df_spreads['slope_' + country] - df_spreads['slope_US']

    # Make smoothing average
    # ==================================================================================================================
    keep = ['rel_slope', 'slope_US', 'slope_' + country]
    df_spreads = df_spreads[keep]
    df_ema = make_sma(df=df_spreads[df_spreads.index >= '1985-01-01'], spans=[25, 50, 60, 90, 120, 200])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())

    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.set_index('Date')
    df_all.dropna(inplace=True)
    df_all.fillna(method='ffill', inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)

    # df_all['relative_path_Factor'] = np.where(df_all['rel_slope_smoothed_ind_sm'] > 0, -1, 1)
    # df_all['rel_slope_and_path_Factor'] = (0.7 * df_all['rel_slope_Factor'] + 0.3 * df_all['relative_path_Factor'])

    df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)
    df_all['Foreign_Factor'] = np.where(df_all['slope_' + country] > df_all['slope_' + country + '_sm50'], -1, 1)
    df_all['Composite_Fed_Foreign_Factor'] = (0.7 * df_all['Fed_Factor'] + 0.3 * df_all['Foreign_Factor'])

    #

    # df_all['Composite_all'] = df_all[factors].mean(axis=1)

    # strategy performance
    # =====================================================================================================================
    df_performance = pd.DataFrame()
    factors = ['rel_slope_Factor', 'Fed_Factor', 'Foreign_Factor', 'Composite_Fed_Foreign_Factor']

    for factor in factors:
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
    returns = pd.DataFrame(returns)
    returns.columns = ['ret']
    returns['best_factor'] = best_factor

    # collect
    df_performance['Currency'] = ccy
    df_all['Currency'] = ccy
    returns['Currency'] = ccy

    df_performance_results.append(df_performance)
    returns_results_all.append(returns)
    df_all_results.append(df_all)

# export results
fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Yield_Strategy_presentation_NY_Close'
df_performance_all = pd.concat(df_performance_results, axis=0)
returns_results_all = pd.concat(returns_results_all, axis=0)

for ccy in currencies:
    df_performance = df_performance_all[df_performance_all['Currency'] == ccy]
    # df_stats = df_stats_all[df_stats_all['Currency'] == ccy]
    # df_all = df_all_all[df_all_all['Currency'] == ccy]
    returns = returns_results_all[returns_results_all['Currency'] == ccy]
    # df_eoy = mrh.get(returns['ret'], eoy=True)
    # best_factor = returns.iloc[:, 1].unique()[0]

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

    # # Performance stats
    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # df_stats.plot.bar(ax=axes[0], title=ccy)
    # df_eoy[['eoy']].plot.bar(ax=axes[1], title=ccy + ':' + best_factor + ' EOY Returns_BBQ', legend=False, figsize=(20, 10))
    # plt.savefig(save_stats)

    mrh.plot(returns, eoy=True, title=ccy + ':' + best_factor + '_BBQ')
    plt.savefig(monthly_stats)
