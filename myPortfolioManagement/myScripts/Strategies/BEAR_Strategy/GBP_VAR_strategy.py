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
import seaborn as sns

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})
# ======================================================================================================================

currencies = ['EURUSD']
start_date = '2000-01-01'

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
                 'GTJPY2Y Govt': 'JPY',
                 'GTGBP2Y Govt': 'GBP',
                 'GCAN2YR Index': 'CAD',
                 'GNZGB2 Index': 'NZD',
                 'GTAUD2Y Govt': 'AUD',
                 'GTSEK2Y Govt': 'SEK',
                 'GTNOK2Y Govt': 'NOK'}

df_bbq_2y_bonds = df_bbq_2y_bonds.rename(columns=mapping_bonds)
df_bbq_2y_bonds.columns = ['2Y_' + x for x in df_bbq_2y_bonds.columns]


for ccy in ['EURUSD']:
    country = ccy[0:3]
    yields = int_name = '2Y'

    spot_mb = get_bbq_ccy(ccy=ccy, start_date=start_date).dropna()
    spot_mb = spot_mb[spot_mb.index >= '1985-01-01']

    name = 'slope_' + country
    spot_name = list(spot_mb.columns)[0]

    df_yileds_trend = df_bbq_2y_bonds.fillna(method='ffill').rolling(30 * 12).mean().pct_change()
    df_MC_regime = df_yileds_trend['2Y_US'] - df_yileds_trend['2Y' + '_' + country]
    df_MC_regime.name = 'MC_regime'
    df_MC_regime = pd.DataFrame(df_MC_regime)
    df_MC_regime['Signal_MP_Trend'] = np.where(abs(df_MC_regime['MC_regime']) > 0, -1, 1)

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
    df_ema = make_sma(df=df_spreads[df_spreads.index >= start_date], spans=[50, 60])
    col_names = df_ema.columns.tolist()

    # df_ema['rel_slope_smoothed_ind_sm'] = df_ema['slope_' + country + '_sm50'] - df_ema['slope_US_sm50']
    df_ema.dropna(inplace=True)

    # Collect Data
    # ==================================================================================================================
    df_all = df_spreads.reset_index().merge(df_ema.reset_index())
    df_all = df_all.merge(spot_mb.reset_index())
    df_all = df_all.set_index('Date')
    df_all = df_all.join(df_MC_regime)

    if country == 'NZD':
        df_all.fillna(method='ffill', inplace=True)

    df_all.dropna(inplace=True)

    # create strategy
    # ===================================================================================================================
    df_all['rel_slope_Factor'] = np.where(df_all['rel_slope'] > df_all['rel_slope_sm50'], -1, 1)
    df_all['Fed_Factor'] = np.where(df_all['slope_US'] > df_all['slope_US_sm50'], 1, -1)
    df_all['Foreign_Factor'] = np.where(df_all['slope_' + country] > df_all['slope_' + country + '_sm50'], -1, 1)


    factors = ['rel_slope_Factor', 'Fed_Factor', 'Foreign_Factor', 'Signal_MP_Trend']

    # strategy performance
    # =====================================================================================================================
    df_perf_slope = pd.DataFrame(index=df_all.index)
    for factor in factors:
        df_perf_slope[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff()).cumsum()

    df_ret = pd.DataFrame(index=df_all.index)
    for factor in factors:
        df_ret[factor] = performance(df_all[factor], total_returns=np.log(df_all[ccy]).diff())

# ======================================================================================================================
# BEAR Signal
# ======================================================================================================================

# Download raw data
# ======================================================================================================================
bbq_tickers_policy_rates = ['GDBR10 Index', 'GT10 Govt', 'SX5E Index', 'SPX Index', 'EURUSD Curncy']

BVAR_raw_data = get_bbq_data(bbq_tickers_policy_rates, start_date=start_date)
BVAR_raw_data.dropna(inplace=True)
BVAR_raw_data.columns = ['10Y_GER', '10Y_US', 'EUR_Stocks', 'US_Stocks', 'EURUSD']

# Clean Data
# ======================================================================================================================
BVAR_raw_data['10Y_GER_daily_change'] = BVAR_raw_data['10Y_GER'].diff()
BVAR_raw_data['Spread_yields'] = BVAR_raw_data['10Y_GER'] - BVAR_raw_data['10Y_US']
BVAR_raw_data['Spread_yields_daily_change'] = BVAR_raw_data['Spread_yields'].diff()
BVAR_raw_data['EUR_Stocks_daily_log_dif'] = np.log(BVAR_raw_data['EUR_Stocks']).diff()
BVAR_raw_data['US_Stocks_daily_log_dif'] = np.log(BVAR_raw_data['US_Stocks']).diff()
BVAR_raw_data['EURUSD_daily_log_dif'] = np.log(BVAR_raw_data['EURUSD']).diff()

# Signals
# ======================================================================================================================
keep = ['10Y_GER_daily_change', 'Spread_yields_daily_change', 'EUR_Stocks_daily_log_dif',
        'US_Stocks_daily_log_dif', 'EURUSD_daily_log_dif']

df_signals = BVAR_raw_data[keep]

# signals
df_signals.columns = ['EUR_yields', 'Spreads', 'EUR_Assets', 'US_Assets', 'EURUSD']

df_signals['EUR_mp'] = np.where((df_signals['EUR_yields'] > 0) &
                                (df_signals['EUR_Assets'] < 0) &
                                (df_signals['Spreads'] > 0), 1, 0)

df_signals['Macro_EUR'] = np.where((df_signals['EUR_yields'] > 0) &
                                   (df_signals['EUR_Assets'] > 0) &
                                   (df_signals['Spreads'] > 0), 1, 0)

df_signals['US_mp'] = np.where((df_signals['EUR_yields'] > 0) &
                               (df_signals['US_Assets'] < 0) &
                               (df_signals['Spreads'] < 0), 0, 1)

df_signals['global_risk'] = np.where((df_signals['EUR_yields'] > 0) &
                                     (df_signals['EUR_Assets'] > 0) &
                                     (df_signals['US_Assets'] > 0) &
                                     (df_signals['Spreads'] < 0), 1, -1)

df_signals = df_signals.join(df_all[['rel_slope_Factor', 'Signal_MP_Trend']])

# df_signals['global_risk'].plot()

df_performance = pd.DataFrame(index=df_signals.index)
factors = ['EUR_mp', 'Macro_EUR', 'US_mp', 'global_risk']
for factor in factors:
    df_performance[factor] = performance(df_signals[factor], total_returns=df_signals['EURUSD']).cumsum()

df_str_ret = pd.DataFrame(index=df_signals.index)
for factor in factors:
    df_str_ret[factor] = performance(df_signals[factor], total_returns=df_signals['EURUSD'])

# check performance
# ======================================================================================================================
df_perf = df_performance.join(df_perf_slope).dropna()
df_corr = df_perf.drop(['Fed_Factor', 'Foreign_Factor'], axis=1).corr()


#
def corr_matrix(corr):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0, annot=True,
                square=True, ax=ax)


corr_matrix(df_corr)

df_str_ret = df_str_ret.join(df_ret)
corr_matrix(df_str_ret.drop(['Fed_Factor', 'Foreign_Factor'], axis=1).corr())

# Example of strategy
# ======================================================================================================================
df1 = df_perf[['Macro_EUR', 'rel_slope_Factor', 'global_risk']].mean(axis=1)
df2 = df_perf[['Macro_EUR', 'rel_slope_Factor']].mean(axis=1)
df_comb = pd.concat([df1, df2], axis=1)

df_comb.columns = ['Combo_EUR_NEWS_Rel_Slope_Gbl_Risk', 'Combo_EUR_NEWS_Rel_Slope']

df_combo2 = pd.Series(df_perf[['Macro_EUR', 'rel_slope_Factor','Signal_MP_Trend']].mean(axis=1), name = 'Combo_MP_Factor')
df_combo2 = pd.DataFrame(df_combo2)

df_str_perf = pd.concat([df_perf[['Macro_EUR']], df_perf[['rel_slope_Factor']], df_comb[['Combo_EUR_NEWS_Rel_Slope']], df_combo2],
                        axis=1)


df_str_perf.dropna().plot()

# put all signals
df_signals['Combo_EUR_NEWS_Rel_Slope'] = df_signals[['rel_slope_Factor', 'Macro_EUR']].mean(axis=1)

df_signals[df_str_perf.columns].plot(subplots=True)

df_str_perf.plot()

df_str_ret['Combo_EUR_NEWS_Rel_Slope'] = df_str_ret[['Macro_EUR', 'rel_slope_Factor']].mean(axis=1)

ret = df_str_ret[df_str_perf.columns]

df_stats = []
for col in df_str_perf.columns:
    temp = mainStats(ret[col])
    temp.columns = [col]
    df_stats.append(temp)

pd.concat(df_stats, axis=1).to_clipboard()

ret.to_csv('S:\Investment Solutions Group\Quant_research\Moustafa\EUR_news_signal.csv')
