import pandas as pd
import numpy as np
from myPortfolioManagement.myData import get_stock_prices, get_US_yields, get_FX_USD_spots
from myPortfolioManagement.myDataPreparation import make_sma
from myPortfolioManagement.myBacktesting import performance
from myPortfolioManagement.myPerformanceMetrics import performance_overview
import seaborn as sns

import quantstats as qs

# download US Yields
# ======================================================================================================================
df_USyields = get_US_yields(freq='d', add_fed_rate=True)
df = df_USyields[['2Y', 'DFF']].copy()

# define slope
# ======================================================================================================================
df['Slope2Y'] = df['2Y'] - df['DFF']

# check
df.dropna().plot(secondary_y='Slope2Y')

# define momentum
# ======================================================================================================================
df_sm = make_sma(df[['Slope2Y']], spans=[50])
df_all = df.join(df_sm).dropna()

# generate signal
# ======================================================================================================================
df_all['Signal'] = np.where(df_all['Slope2Y'] > df_all['Slope2Y_sm50'], 1, -1)

# Get data trading instruments: EURUSD, Stocks, Japan
# ======================================================================================================================
df_stock = get_stock_prices(yahoo_tickers=['^IXIC'], wide_format=True)
df_stock = df_stock.rename(columns={'^IXIC': 'Nasdaq'})
df_fx = get_FX_USD_spots()

df_temp = pd.concat([df_stock, df_fx], axis=1)
df_temp.index.name = df_all.index.name

# merge all
# ======================================================================================================================
df_all = df_all.join(df_temp)

# performance
# ======================================================================================================================
df_perm = pd.DataFrame(index=df_all.index)
for col in df_temp.columns:
    # df_perm[col] = performance(signal=df_all['Signal'], returns=np.log(df_all[col]).diff()).cumsum()
    df_perm[col + '_ret'] = performance(signal=df_all['Signal'], returns=np.log(df_all[col]).diff())
    df_perm[col + '_ret_pct'] = performance(signal=df_all['Signal'], returns=df_all[col].pct_change())

# from wide to long to plot
# ======================================================================================================================
dfm = df_perm.cumsum().dropna().reset_index().melt(id_vars='Date', var_name='Stock', value_name='val')
g = sns.relplot(data=dfm, x='Date', y='val', col='Stock', col_wrap=4, kind='line')

# stats
df_perm_stats = performance_overview(df_perm, short=True).sort_values(by=['cagr'], ascending=False)

df_all['Signal'][df_all.index >= '2023-08-01']
df_all['JPYUSD'][df_all.index >= '2023-08-01'].plot()
df_perm[col + '_ret'][df_perm[col + '_ret'].index >= '2023-08-01'].cumsum().plot()
