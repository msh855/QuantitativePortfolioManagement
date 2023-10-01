import pandas as pd
import numpy as np
from myScripts.functions.download_and_prepare_data import get_bbq_data
from myScripts.functions.performance_stats import mainStats, performance
import os

# Download raw data
# ======================================================================================================================
bbq_tickers_policy_rates = ['GDBR10 Index', 'GT10 Govt', 'SX5E Index', 'SPX Index', 'EURUSD Curncy']
start_date = '2000-01-01'

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

df_signals.columns = ['EUR_yields', 'Spreads', 'EUR_Assets', 'US_Assets', 'EURUSD']

df_signals['EUR_mp'] = np.where((df_signals['EUR_yields'] > 0) &
                                (df_signals['EUR_Assets'] < 0) &
                                (df_signals['Spreads'] > 0), 0, 1)

df_signals['Macro_EUR'] = np.where((df_signals['EUR_yields'] > 0) &
                                   (df_signals['EUR_Assets'] > 0) &
                                   (df_signals['Spreads'] > 0), 1, 0)

df_signals['US_mp'] = np.where((df_signals['EUR_yields'] > 0) &
                               (df_signals['US_Assets'] < 0) &
                               (df_signals['Spreads'] < 0), 0, 1)

df_signals['global_risk'] = np.where((df_signals['EUR_yields'] > 0) &
                                     (df_signals['EUR_Assets'] > 0) &
                                     (df_signals['US_Assets'] > 0) &
                                     (df_signals['Spreads'] < 0), 1, 0)

df_performance = pd.DataFrame(index=df_signals.index)
df_performance['EUR_mp'] = performance(df_signals['EUR_mp'], total_returns=df_signals['EURUSD']).cumsum()
df_performance['Macro_EUR'] = performance(df_signals['Macro_EUR'], total_returns=df_signals['EURUSD']).cumsum()
df_performance['US_mp'] = performance(df_signals['US_mp'], total_returns=df_signals['EURUSD']).cumsum()
df_performance['global_risk'] = performance(df_signals['global_risk'], total_returns=df_signals['EURUSD']).cumsum()

df_performance['EUR_mp'].plot()
df_performance['Macro_EUR'].plot()
df_performance['US_mp'].plot()
df_performance['global_risk'].plot()

df_performance['composite'] = df_performance.mean(axis=1)

df_performance.plot()

df_performance['Macro_EUR_signal'] = df_signals['Macro_EUR']
