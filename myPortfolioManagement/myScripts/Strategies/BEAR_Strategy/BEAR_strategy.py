import pandas as pd
import numpy as np
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import get_bbq_data, get_bbq_ccy
from myScripts.functions.download_and_prepare_data import make_sma

import matplotlib.style

matplotlib.style.use('seaborn')


def BEAR_Strategy(df_data):
    df = pd.DataFrame(index=df_data.index)

    strategy_names = ['foreign_mp', 'Macro_foreign_news', 'US_mp', 'global_risk']
    strategy_names = [x + '_signal' for x in strategy_names]

    df[strategy_names[0]] = np.where((df_data['foreign_yields'] > 0) &
                                     (df_data['Foreign_Assets'] < 0) &
                                     (df_data['Spreads'] > 0), 1, 0)

    df[strategy_names[1]] = np.where((df_data['foreign_yields'] > 0) &
                                     (df_data['Foreign_Assets'] > 0) &
                                     (df_data['Spreads'] > 0), 1, 0)

    df[strategy_names[2]] = np.where((df_data['foreign_yields'] > 0) &
                                     (df_data['US_Assets'] < 0) &
                                     (df_data['Spreads'] < 0), 0, 1)

    df[strategy_names[3]] = np.where((df_data['foreign_yields'] > 0) &
                                     (df_data['Foreign_Assets'] > 0) &
                                     (df_data['US_Assets'] > 0) &
                                     (df_data['Spreads'] < 0), 1, -1)
    return df


start_date = '2000-01-01'

# 10 year yields G9
currencies = ['GBPUSD',
              'EURUSD',
              'CADUSD',
              'JPYUSD',
              'NOKUSD',
              'SEKUSD',
              'NZDUSD',
              'CHFUSD',
              'AUDUSD']

yields_10y_bbg_tickers = ['GUKG10 Index',  # UK
                          'GDBR10 Index',  # EUR
                          'FMSTBY10 Index',  # Canada
                          'GJGB10 Index',  # Japan
                          'GTNOK10Y Govt',  # Norway
                          'GTSEK10Y Govt',  # Sweden
                          'GTNZD10Y Govt',  # New Zealand
                          'CHFI10Y Curncy',  # Swiss
                          'GACGB10 Index']  # AUS

stock_tickers_bbg = ['MCX Index',  # UK
                     'SX5E Index',  # EUR
                     'SPTSX Index',  # Canada
                     'TPX Index',  # Japan
                     'OSEBX Index',  # Norway
                     'SBX Index',  # Sweden
                     'NZSE50FF Index',  # New Zealand
                     'SMI Index',  # Swiss
                     'AS51 Index']  # AUS

# Download raw data
# ======================================================================================================================
df_perf_list = []
df_ret_list = []
for i in range(0, len(stock_tickers_bbg)):
    ccy = currencies[i]

    bbq_tickers = [yields_10y_bbg_tickers[i], 'GT10 Govt', stock_tickers_bbg[i], 'SPX Index']

    BVAR_raw_data = get_bbq_data(bbq_tickers, start_date=start_date)
    BVAR_raw_data.dropna(inplace=True)
    BVAR_raw_data.columns = ['10Y_Foreign', '10Y_US', 'Foreign_Stocks', 'US_Stocks']

    # Clean Data
    # ==================================================================================================================
    BVAR_raw_data['10Y_Foreign_daily_change'] = BVAR_raw_data['10Y_Foreign'].diff()
    BVAR_raw_data['Spread_yields'] = BVAR_raw_data['10Y_Foreign'] - BVAR_raw_data['10Y_US']
    BVAR_raw_data['Spread_yields_daily_change'] = BVAR_raw_data['Spread_yields'].diff()
    BVAR_raw_data['Foreign_Stocks_daily_log_dif'] = np.log(BVAR_raw_data['Foreign_Stocks']).diff()
    BVAR_raw_data['US_Stocks_daily_log_dif'] = np.log(BVAR_raw_data['US_Stocks']).diff()
    # BVAR_raw_data[currencies[0] + '_daily_log_dif'] = np.log(BVAR_raw_data[currencies[0]]).diff()

    # Signals
    # ======================================================================================================================
    keep = ['10Y_Foreign_daily_change', 'Spread_yields_daily_change', 'Foreign_Stocks_daily_log_dif',
            'US_Stocks_daily_log_dif']

    df_signals = BVAR_raw_data[keep]

    # signals
    col_names = ['foreign_yields', 'Spreads', 'Foreign_Assets', 'US_Assets']
    df_signals.columns = col_names

    df_ema = make_sma(df=df_signals, spans=[25, 60, 90, 120])
    df_signals_25 = df_ema.filter(like='_sm25').copy()
    df_signals_25.columns = col_names
    df_signals_25 = BEAR_Strategy(df_signals_25)
    df_signals_25.columns = [x + '_sm25' for x in df_signals_25.columns]

    df_signals_60 = df_ema.filter(like='_sm60').copy()
    df_signals_60.columns = col_names
    df_signals_60 = BEAR_Strategy(df_signals_60)
    df_signals_60.columns = [x + '_sm60' for x in df_signals_60.columns]

    df_signals_90 = df_ema.filter(like='_sm90').copy()
    df_signals_90.columns = col_names
    df_signals_90 = BEAR_Strategy(df_signals_90)
    df_signals_90.columns = [x + '_sm90' for x in df_signals_90.columns]

    df_signals_120 = df_ema.filter(like='_sm120').copy()
    df_signals_120.columns = col_names
    df_signals_120 = BEAR_Strategy(df_signals_120)
    df_signals_120.columns = [x + '_sm120' for x in df_signals_120.columns]

    # Strategy Signals
    df_mr = df_signals[col_names]
    for col in df_mr.columns:
        df_mr[col + '_cum'] = df_mr[col].cumsum()
        temp = make_sma(df=df_mr[[col]], spans=[25]).filter(like='_sm25')
        df_mr = df_mr.join(temp)
        df_mr[col + '_signal_mr'] = np.where(df_mr[col] >= df_mr[col + '_sm25'], -1, 1)  # mean reverting strategy
        df_mr[col + '_signal_mom'] = np.where(df_mr[col] >= df_mr[col + '_sm25'], 1, -1)  # mean reverting strategy

    df_mr_signals = df_mr.filter(like='signal_mr')
    df_m_signals = df_mr.filter(like='signal_mom')

    df_signal_mom = pd.concat([df_signals_25, df_signals_60, df_signals_90, df_signals_120], axis=1)
    df_signal_mom['mom_composite_all'] = df_signal_mom.mean(axis=1)

    strategy_names = ['foreign_mp', 'Macro_foreign_news', 'US_mp', 'global_risk']

    # foreign mp
    df_signal_mom['composite_' + strategy_names[0]] = df_signal_mom.filter(like=strategy_names[0]).mean(axis=1)

    # macro news
    df_signal_mom['composite_' + strategy_names[1]] = df_signal_mom.filter(like=strategy_names[1]).mean(axis=1)

    # US_mp inverse corr
    df_signal_mom['composite_' + strategy_names[2]] = df_signal_mom.filter(like=strategy_names[2]).mean(axis=1)
    df_signal_mom['composite_' + strategy_names[3]] = df_signal_mom.filter(like=strategy_names[3]).mean(axis=1)

    df_signal_mom['comp_US_MP_Global_Risk'] = df_signal_mom[['composite_' + strategy_names[2]]].join(
        df_signal_mom[['composite_' + strategy_names[3]]]).mean(axis=1)

    # macro news + US_mp_risk

    df_signals.columns = col_names
    df_str = BEAR_Strategy(df_signals)
    df_signals = df_signals.join(df_str)
    df_signals = pd.concat(
        [df_signals, df_signals_25, df_signal_mom['composite_' + strategy_names[0]],
         df_signal_mom['composite_' + strategy_names[1]],
         df_signal_mom['composite_' + strategy_names[2]], df_signal_mom['composite_' + strategy_names[3]],
         df_signal_mom[['mom_composite_all']], df_signal_mom[['comp_US_MP_Global_Risk']]], axis=1)

    spot_mb = get_bbq_ccy(ccy=ccy, start_date=start_date).dropna()
    df_all = df_signals.join(spot_mb)
    df_all.dropna(inplace=True)
    df_all = pd.concat([df_all, df_mr_signals, df_m_signals], axis=1)

    # signals = df_str.columns
    # signals = list(signals) + ['composite_' + strategy_names[0], 'composite_' + strategy_names[1],
    #                            'composite_' + strategy_names[2],
    #                            'composite_' + strategy_names[3], 'mom_composite_all']

    signals = ['Macro_foreign_news_signal_sm25', 'comp_US_MP_Global_Risk', 'composite_' + strategy_names[1]]

    df_performance = pd.DataFrame(index=df_all.index)
    df_str_ret = pd.DataFrame(index=df_all.index)
    for signal in signals:
        ccy_ret = np.log(df_all[ccy]).diff()
        df_performance[signal] = performance(df_all[signal], total_returns=ccy_ret).cumsum()
        df_str_ret[signal] = performance(df_all[signal], total_returns=ccy_ret)

    df_performance['composite'] = df_performance.mean(axis=1)

    signals2 = list(df_m_signals.columns)
    df_performance2 = pd.DataFrame(index=df_all.index)
    df_str_ret2 = pd.DataFrame(index=df_all.index)
    for signal in signals2:
        ccy_ret = np.log(df_all[ccy]).diff()
        df_performance2[signal] = performance(df_all[signal], total_returns=ccy_ret).cumsum()
        df_str_ret2[signal] = performance(df_all[signal], total_returns=ccy_ret)

    signals3 = list(df_mr_signals.columns)
    df_performance3 = pd.DataFrame(index=df_all.index)
    df_str_ret3 = pd.DataFrame(index=df_all.index)
    for signal in signals3:
        ccy_ret = np.log(df_all[ccy]).diff()
        df_performance3[signal] = performance(df_all[signal], total_returns=ccy_ret).cumsum()
        df_str_ret3[signal] = performance(df_all[signal], total_returns=ccy_ret)

    # df_performance[['composite_' + strategy_names[0], 'composite_' + strategy_names[1], 'composite_' + strategy_names[2], 'composite_' + strategy_names[3], 'mom_composite_all']].plot(title=ccy)
    # df_performance.dropna().plot(title=ccy)
    # df_performance2.dropna().plot(title=ccy + '_momentum')

    df_perf_com = pd.DataFrame(index=df_performance2.index)
    df_perf_com['Rel_equity_ts'] = df_performance2[['US_Assets_signal_mom', 'Foreign_Assets_signal_mom']].mean(axis=1)
    df_perf_com = df_perf_com.join(df_performance2[['Spreads_signal_mom']])
    df_perf_com = df_perf_com.join(df_performance2[['foreign_yields_signal_mom']])
    df_perf_com['composite'] = df_perf_com.mean(axis=1)

    df_perf_com.plot(title=ccy + '_momentum')

    # df_performance3.dropna().plot(title=ccy + 'mean_rev')

    df_performance = df_performance.dropna()
    df_performance['Currency'] = ccy
    df_perf_list.append(df_performance)

    df_str_ret['Currency'] = ccy
    df_ret_list.append(df_str_ret)
#
# df_perm_all = pd.concat(df_perf_list)
# df_ret_all = pd.concat(df_ret_list)
#
# df_stats = []
# for ccy in currencies:
#     for sig in signals:
#         temp_curr = df_ret_all[df_ret_all['Currency'] == ccy]
#         temp = mainStats(temp_curr[sig])
#         temp.columns = [sig + '_' + ccy]
#         temp = temp.drop(['mean', 'std', 'kurtosis', 'min', 'max', 'skew', 'MDD/vol'])
#         df_stats.append(temp)
#
# df_stats_all = pd.concat(df_stats, axis=1)
#
# df_stats_all.filter(like='EURUSD').plot.bar(title='EUR')
# df_stats_all.filter(like='NZDUSD').plot.bar(title='NZD')
#
# df_stats_all.columns
