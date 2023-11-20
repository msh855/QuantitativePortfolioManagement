import pandas as pd
import numpy as np
from myPortfolioManagement.Macroeconomics.getdata import get_US_yield_spreads, get_US_yields
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myBacktesting import performance

from pypfopt.expected_returns import prices_from_returns
import quantstats as qs

# stock market
df_stock = get_stock_prices(yahoo_tickers=['^IXIC'], wide_format=True)
df_USyields = get_US_yields(freq='d', add_fed_rate=True)
df_USyields[['1M']].dropna().plot()

# get monetary signal
df_US_spreads = get_US_yield_spreads(spread_from='DFF')
df_US_spreads = df_US_spreads[['spread_2Y', 'spread_10Y']]

# define smoothing averaging
spans = [10, 30, 50, 120, 200]
df_ema_list = []

for span in spans:
    temp = df_US_spreads.ewm(span=span).mean()
    temp.columns = temp.columns + '_sm' + str(span)
    df_ema_list.append(temp)

df_ema = pd.concat(df_ema_list, axis=1)

columns_spreads = df_US_spreads.columns
columns_spreads_sm = df_ema.columns

df_all = df_US_spreads.reset_index().merge(df_ema.reset_index())
df_all = df_all.merge(df_stock.reset_index())
df_all = df_all.set_index('Date')

# create signal
df_all[['spread_2Y', 'spread_2Y_sm50']].plot()
df_all['Signal'] = np.where(df_all['spread_2Y'] > df_all['spread_2Y_sm50'], 1, -1)

df_all['Signal_pass_all'] = np.where((df_all['spread_2Y'] > df_all['spread_2Y_sm10']) &
                                     (df_all['spread_2Y'] > df_all['spread_2Y_sm30']) &
                                     (df_all['spread_2Y'] > df_all['spread_2Y_sm50']) &
                                     (df_all['spread_2Y'] > df_all['spread_2Y_sm120']), -1, 1)

df_all['Signal_pass_all'].plot()

# df_str_all = pd.DataFrame(index=df_all.index)
# for spread in columns_spreads:
#     for signal in columns_spreads_sm:
#         df_all['Signal' + signal] = np.where(df_all[spread] > df_all[signal], 1, 0)
#         factor = performance(signal=df_all['Signal' + signal], returns=np.log(df_all['^IXIC']).diff()).cumsum()
#         df_str_all['PnL' + signal] = factor

df_performance = pd.DataFrame(index=df_all.index)

for signal in ['Signal', 'Signal_pass_all']:
    df_performance[signal] = performance(signal=df_all[signal], returns=np.log(df_all['^IXIC']).diff()).cumsum()

prices_from_returns(df_performance.diff().dropna()).plot()

df_all[['spread_2Y', 'spread_10Y']].plot()

# df_all['Signal']['2020-01-01':].plot()


# df_all['Signal'] = np.where(df_all['spread_10Y'] > df_all['spread_2Y'], -1, 1)
# df_all['Signal'] = np.where(np.sign(df_all['spread_2Y']) != np.sign(df_all['spread_10Y']), 0, df_all['Signal'])

factor = performance(signal=df_all['Signal'], returns=np.log(df_all['^IXIC']).diff()).cumsum()
factor = pd.Series(factor, name='Strategy')
factor = pd.DataFrame(factor)
factor['Market'] = np.log(df_all['^IXIC']).diff().cumsum()
# factor['Excess'] = factor['Strategy'] - factor['Market']
factor.plot()

factor['Composite'] = factor.mean(axis=1)

# backtest
# =========
ret_factor = factor[['Strategy']].diff()
ret_bench = factor[['Market']].diff()
ret = ret_factor.join(ret_bench)

from myPortfolioManagement.myPortfolioOptimisation import inverse_vol_portfolio, port_CVAR

ret['Composite2'] = (0.40 * ret['Strategy'] + 0.60 * ret['Market']) / 2


qs.reports.html(ret['Composite2']['1990-01-01':], benchmark=ret['Market']['1990-01-01':],
                output='/Users/safishajjouz/GitHub')

prices_from_returns(factor.diff()).plot()

returns = qs.stats.monthly_returns(ret['Composite'].diff()) * 100
returns_mrkt = qs.stats.monthly_returns(ret['Market'].diff()) * 100
df_monthly = pd.DataFrame({'strategy': returns['EOY'], 'Market': returns_mrkt['EOY']})

df_monthly.plot.bar()
