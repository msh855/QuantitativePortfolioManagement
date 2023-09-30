import pandas as pd
import numpy as np
from myPortfolioManagement.myData import get_stock_prices, get_US_yields, get_FX_spots
from myPortfolioManagement.myDataPreparation import make_sma
from myPortfolioManagement.myBacktesting import performance
from myPortfolioManagement.myPerformanceMetrics import performance_overview
import seaborn as sns

import quantstats as qs
from myPortfolioManagement.myReturns import returns_from_prices

# download US Yields
# ======================================================================================================================
df_USyields = get_US_yields(freq='d', add_fed_rate=True)

# define slope
# ======================================================================================================================
df = df_USyields[['2Y', 'DFF', '1Y']].copy()
df['Slope2Y'] = df['DFF'] - df['2Y']  # if 2Y > DFF, means markets expect Fed to increase rates
df['Slope1Y'] = df['DFF'] - df['1Y']

# check
df.dropna().plot(secondary_y='Slope2Y')
df[['2Y', 'DFF']].dropna().fillna(method='bfill').rolling(30).mean().plot()
df[['1Y', 'DFF']].dropna().fillna(method='bfill').rolling(30).mean().plot()

# define momentum
# ======================================================================================================================
df_sm = make_sma(df[['Slope2Y', 'Slope1Y']], spans=[50])
df_all = df.join(df_sm).dropna()

# generate signal
# ======================================================================================================================
df_all['Signal'] = np.where(df_all['Slope2Y'] > df_all['Slope2Y_sm50'], 1, -1)
df_all['Signal2'] = np.where(df_all['Slope1Y'] > df_all['Slope1Y_sm50'], 1, -1)

# Get data trading instruments: EURUSD, Stocks, Japan
# ======================================================================================================================
df_stock = get_stock_prices(yahoo_tickers=['^IXIC'], wide_format=True)
df_stock = df_stock.rename(columns={'^IXIC': 'Nasdaq'})
df_fx = get_FX_spots(currencies=['EUR', 'JPY'], wide_format=True)

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
    ret_temp = returns_from_prices(df_all[col])
    df_perm[col + 'str1'] = performance(signal=df_all['Signal'], returns=ret_temp)
    # df_perm[col + 'str2'] = performance(signal=df_all['Signal2'], returns=np.log(df_all[col]).diff())

# from wide to long to plot
# ======================================================================================================================
dfm = df_perm.cumsum().dropna().reset_index().melt(id_vars='Date', var_name='Stock', value_name='val')
dfm['asset'] = [x.split('str')[0] for x in dfm.Stock]
g = sns.relplot(data=dfm, x='Date', y='val', col='Stock', col_wrap=3, kind='line')

# stats
df_perm_stats = performance_overview(df_perm, short=True).sort_values(by=['cagr'], ascending=False)

ret_bench = returns_from_prices(df_stock)
for str in ['EURUSDstr1', 'JPYUSDstr1']:
    df_temp = df_perm[[str]].join(ret_bench).dropna()
    qs.reports.html(df_temp[str], benchmark=df_temp['Nasdaq'], output='/Users/safishajjouz/Downloads',
                    download_filename=str + '.html')
