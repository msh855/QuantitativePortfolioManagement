import pandas as pd
import numpy as np
from myScripts.functions.download_and_prepare_data import make_sma, get_historical_spots
import statsmodels.api as sm
import seaborn as sns
from myScripts.functions.performance_stats import mainStats, performance
from medh.momentum import BaseMomentum

# plotting
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('QtAgg')
import matplotlib.dates as mdates

# plt.interactive(True)
myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})


def macd_ewma(price, short_window, long_window):
    df_signal = pd.DataFrame(index=price.index, columns=price.columns)
    hl_short = np.log(0.5) / (np.log(1 - 2 / (short_window + 1)))
    hl_long = np.log(0.5) / (np.log(1 - 2 / (long_window + 1)))
    short_mavg = price.ewm(halflife=hl_short, adjust=False, min_periods=short_window).mean()
    long_mavg = price.ewm(halflife=hl_long, adjust=False, min_periods=long_window).mean()
    df_signal[short_mavg > long_mavg] = 1
    df_signal[short_mavg < long_mavg] = -1
    return df_signal


def vol_adj_spots(df_returns, vol_window=60):
    raw_vol = BaseMomentum.std_vol(df_returns, window=vol_window)
    # Vol normalise returns
    adj_return = df_returns / raw_vol * np.sqrt(252)
    adj_spot = adj_return.cumsum()

    return adj_spot, raw_vol


def mom_production(df_returns, vol_window=60, windows_lst=[(16, 64), (32, 128), (64, 256)], vol_scale=0.075):
    adj_spot, raw_vol = vol_adj_spots(df_returns, vol_window)
    signals = pd.DataFrame(0, index=adj_spot.index, columns=adj_spot.columns)
    for window in windows_lst:
        momentum_df = macd_ewma(adj_spot, window[0], window[1])
        scaled_momentum = (momentum_df / raw_vol) * vol_scale
        signals = signals + scaled_momentum

    signals = signals / len(windows_lst)

    return signals


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]

    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


# download data
# =============
df = get_historical_spots().dropna()
df['return'] = np.log(df['CHFUSD']).diff()
df['return_cum'] = df['return'].cumsum()

# performance of production signal
signals = mom_production(df[['return']])
performance(signals['return'], df['return']).cumsum().plot()

# smooth returns
ret_cum_adj, raw_vol = vol_adj_spots(df[['return']])
df_spot_sm = make_sma(df=ret_cum_adj.dropna(), spans=[16, 32, 64, 128, 256])
df_spot_sm.to_clipboard()  # from r script
#
# # dta_m = df_spot_sm.pct_change()
# dta_m = remove_outliers(df_spot_sm)
# dta_m.to_clipboard()
#
# # Construct the dynamic factor model
# '''
# The optimals factors and lags were based on the R package 'nowcasting'
# '''
#
# # transform to stationary and normalise
# df_spot_sm.dropna(inplace=True)

data = pd.read_csv('DFM_dataset.csv')
data = data.set_index('Date')
data.index.name = 'Date'
#data.to_csv('DFM_dataset.csv', index=True)

nfactors = 6
lags = 10
model = sm.tsa.DynamicFactor(endog=data, k_factors=2, factor_order=2,
                               standardize=False,
                               idiosyncratic_ar1=False)
model.summary()

# fit model (!! it is usually slow)
initial_res = model.fit(method='powell', disp=False)
results = model.fit(initial_res.params, disp=10, maxiter=2000)
results.summary()

# get factors
df_factors = results.factors.filtered
df_factors.columns = ['factor' + str(x) for x in range(1, len(df_factors.columns) + 1)]
df_factors.index = data.index

df_factors[df_factors.index >= '1980'].corr()

res = results
mod = model
spec = res.specification

design = mod.ssm['design']
transition = mod.ssm['transition']
ss_kalman_gain = res.filter_results.kalman_gain[:, :, -1]
k_states = ss_kalman_gain.shape[0]

W1 = np.linalg.inv(np.eye(k_states) - np.dot(
    np.eye(k_states) - np.dot(ss_kalman_gain, design),
    transition
)).dot(ss_kalman_gain)[0]

# Compute the factor mean vector
factor_mean = np.dot(W1, data.loc['1972-02-01':, 'return_sm16':'return_sm_prod'].mean())

# Normalize the factors
factor = res.factors.filtered[0]
dusphci = df_spot_sm.diff()[1:].values
factor *= np.std(df_spot_sm.diff()[1:]) / np.std(factor)

# Compute the coincident index
coincident_index = np.zeros(mod.nobs + 1)
# The initial value is arbitrary; here it is set to
# facilitate comparison
coincident_index[0] = usphci.iloc[0] * factor_mean / dusphci.mean()
for t in range(0, mod.nobs):
    coincident_index[t + 1] = coincident_index[t] + factor[t] + factor_mean

# Attach dates
coincident_index = pd.Series(coincident_index, index=dta.index).iloc[1:]

# Normalize to use the same base year as USPHCI
coincident_index *= (usphci.loc['1992-07-01'] / coincident_index.loc['1992-07-01'])

df_factors.cumsum().plot()

df_mom = df_spot_sm.dropna().reset_index('Date').merge(df_factors.reset_index('Date'))
df_mom = df_mom.dropna()
df_mom = df_mom.set_index('Date')

df_factors.sum(axis=1).cumsum().plot()

df_mom['ret_cum'] = df['return_cum'][df['return_cum'].index >= '1970-01-06 ']
raw_vol = raw_vol[raw_vol.index >= '1970-01-06 ']

vol_scale = 0.075

df_mom['Signal1'] = np.where(df_mom['ret_cum'] > df_mom['factor1'], 1, -1)
df_mom['Signal1'] = (df_mom['Signal1'] / raw_vol['return']) * vol_scale

df_mom['Signal2'] = np.where(df_mom['ret_cum'] > df_mom['factor2'], 1, -1)
df_mom['Signal2'] = (df_mom['Signal2'] / raw_vol['return']) * vol_scale

df_mom['Signal3'] = np.where(df_mom['ret_cum'] > df_mom['factor3'], 1, -1)
df_mom['Signal3'] = (df_mom['Signal3'] / raw_vol['return']) * vol_scale

df_mom['Signal_com'] = df_mom[['Signal1', 'Signal2', 'Signal3']].mean(axis=1)

df_mom['Spot'] = df['CHFUSD'][df['CHFUSD'].index >= '1970-01-06 ']

######

df_perf = performance(df_mom['Signal_com'], total_returns=np.log(df_mom['Spot']).diff()).cumsum()
df_perf = pd.DataFrame(df_perf)
df_perf.columns = ['DFM_mom']

df_per_prod = performance(signals['return'], df['return']).cumsum()
df_per_prod = pd.DataFrame(df_per_prod)
df_per_prod.columns = ['Prod_mom']
df_per_prod = df_per_prod[df_per_prod.index >= '1970-01-06 ']

df_perf = df_perf.reset_index().merge(df_per_prod.reset_index()).dropna()
df_perf = df_perf.set_index('Date')

df_final = df_perf[df_perf.index >= '1990-01-01']
df_final = df_final / df_final.iloc[0] * 100
df_final.plot(title='Dynamic Momentum Learning vs Production (CHF)')
plt.show()

mainStats(df_final.pct_change()['Prod_mom'])
mainStats(df_final.pct_change()['DFM_mom'])

# comparisons


####


rsquared = results.get_coefficients_of_determination(method='individual')

top_5 = []
for factor_name in rsquared.columns[:2]:
    top_factor = (rsquared[factor_name].sort_values(ascending=False)
                  .iloc[:5].round(2).reset_index())
top_factor.columns = pd.MultiIndex.from_product([
    [f'Top ten variables explained by {factor_name}'],
    ['Variable', r'$R^2$']])
top_5.append(top_factor)

pd.concat(top_5, axis=1)

with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Create point forecasts, 3 steps ahead
point_forecasts = results.forecast(steps=3)

# Print the forecasts for the first 5 observed variables
print(point_forecasts.T.head())


# =================================================


def transform(column, transforms):
    transformation = transforms[column.name]
    # For quarterly data like GDP, we will compute
    # annualized percent changes
    mult = 4 if column.index.freqstr[0] == 'Q' else 1

    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference, multiplied by 100
    #      (i.e. approximate percent change)
    #      with optional multiplier for annualization
    elif transformation == 5:
        column = np.log(column).diff() * 100 * mult
    # 6 => Log second difference, multiplied by 100
    #      with optional multiplier for annualization
    elif transformation == 6:
        column = np.log(column).diff().diff() * 100 * mult
    # 7 => Exact percent change, multiplied by 100
    #      with optional annualization
    elif transformation == 7:
        column = ((column / column.shift(1)) ** mult - 1.0) * 100

    return column


base_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv'

# 1. Download data
orig_m = pd.read_csv(base_url).dropna()

# 2. Extract transformation information
transform_m = orig_m.iloc[0, 1:]
orig_m = orig_m.iloc[1:]

# 3. Extract the date as an index
orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
orig_m.drop('sasdate', axis=1, inplace=True)

# 4. Apply the transformations
dta_m = orig_m.apply(transform, axis=0,
                     transforms=transform_m)

# 5. Remove outliers (but not in 2020)
dta_m.loc[:'2019-12'] = remove_outliers(dta_m.loc[:'2019-12'])

# - FRED-QD --------------------------------------------------------------
# 1. Download data

orig_q = pd.read_csv('https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv').dropna(how='all')

# 2. Extract factors and transformation information
factors_q = orig_q.iloc[0, 1:]
transform_q = orig_q.iloc[1, 1:]
orig_q = orig_q.iloc[2:]

# 3. Extract the date as an index
orig_q.index = pd.PeriodIndex(orig_q.sasdate.tolist(), freq='Q')
orig_q.drop('sasdate', axis=1, inplace=True)

# 4. Apply the transformations
dta_q = orig_q.apply(transform, axis=0,
                     transforms=transform_q)

# 5. Remove outliers (but not in 2020)
dta_q.loc[:'2019Q4'] = remove_outliers(dta_q.loc[:'2019Q4'])
