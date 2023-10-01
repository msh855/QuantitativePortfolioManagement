import pandas as pd
from myScripts.functions.download_and_prepare_data import get_data
import numpy as np
from sklearn.linear_model import LogisticRegression
import math
import pandas_ta as ta
from myScripts.functions.download_and_prepare_data import get_ccy10y_spreads, get_ccy2y_spreads, relative_spread


def performance(signal, total_returns, tc=.0002):
    trans_cost = tc * (signal.diff()).abs()
    excess_return = total_returns * signal.shift(1) - trans_cost

    return excess_return


# get currency
currency = 'EURUSD'
df_ccy = get_data(currency='EURUSD', get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy[['Spot']]

mytarget = 'target'
df = df_ccy.copy()
df['return'] = np.log(df['Spot']).diff()
df['Direction'] = np.where(df['return'] > 0, 1, 0)
df[mytarget] = df['Direction'].shift(-1)

# add business cycle factor

halflife = 126
# carry on long-term bond (difference between longer-term yields across currencies)
df_ccy_10y_spread = get_ccy10y_spreads(currency=currency, daily_change=False)
df_ccy_10y_spread['10y_spread_sm'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife, adjust=False).mean()
df_ccy_10y_spread = pd.DataFrame(df_ccy_10y_spread.iloc[:, 1])

df_ccy_2y_spread = get_ccy2y_spreads(currency=currency, daily_change=False)
df_ccy_2y_spread['2y_spread_sm'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife, adjust=False).mean()

# Business cycle factor (difference between the spread 2-10 years)
df_spreads = relative_spread(currency=currency)
df_spreads.index.name = 'Date'
df_spreads['rel_slope_sm'] = df_spreads.iloc[:, 0].ewm(halflife=halflife, adjust=False).mean()

# merge dataframes
df = df.reset_index().merge(df_spreads.reset_index())
df = df.merge(df_ccy_2y_spread.reset_index())
df = df.merge(df_ccy_10y_spread.reset_index())
df = df.set_index('Date')

# create time series momentum
spot = df['Spot']

# get different momentum
df['KAMA'] = ta.kama(spot)
df['mcgd'] = ta.mcgd(spot)

df['ema10'] = ta.ema(spot, length=10)
df['ema30'] = ta.ema(spot, length=30)
df['ema60'] = ta.ema(spot, length=60)
df['ema120'] = ta.ema(spot, length=120)
df['ema200'] = ta.ema(spot, length=200)

# predictive regressions
# df = df.iloc[50:]
df = df.dropna()
var_drop = ['return', 'Direction', 'Spot', 'target']
mypredictors = list(df.drop(var_drop, axis=1).columns)

moving_averages = [x for x in mypredictors if 'ema' in x]
contrls = [x for x in mypredictors if '_sm' in x]

mypredictors = moving_averages + contrls

data = df
window = 30

dataframes = []
y_pred_list = []
# expanding window
dates = data.index

pred_df = pd.DataFrame(index=data.index, columns=['yhat'])

for ix in data.index[window: -1]:
    prev_ix = data[:ix].index[-window]
    data_temp = data[prev_ix: ix]

    x = data_temp[mypredictors].to_numpy()
    y = data_temp[mytarget].to_numpy()

    if ix != data.index[window: -1][0]:
        y_pred = model.predict(x)
        pred_df.loc[ix, 'yhat'] = y_pred[-1]

    model = LogisticRegression(fit_intercept=True)
    model.fit(x, y)


temp_nan = pd.Series(np.repeat(np.nan, 30))
y_pred_list = pd.concat(y_pred_list)
signal = pd.concat([temp_nan, y_pred_list])

data['Signal_rolling'] = signal.values
data['Signal_rolling'] = pred_df.values

# regression
x = data[mypredictors].to_numpy()
y = data[mytarget].to_numpy()
model = LogisticRegression(fit_intercept=True)
model.fit(x, y)
y_pred = model.predict(x)

# drivers importance
w0 = model.intercept_[0]  # model's intercept
w = model.coef_[0]  # model coefficients

feature_importance = pd.DataFrame(mypredictors, columns=["feature"])
feature_importance["importance"] = pow(math.e, w)
feature_importance = feature_importance.sort_values(by=["importance"], ascending=False)
ax = feature_importance.plot.barh(x='feature', y='importance')

# normalise
feature_importance['weights'] = np.nan
total_weight = feature_importance['importance'].sum()

for i in range(0, len(feature_importance)):
    feature_importance['weights'].iloc[i] = feature_importance['importance'].iloc[i] / total_weight

# synthetic momentum strategy
df_dummy = df[feature_importance.feature]
df_dummy['synthetic_mom'] = np.nan
df_dummy = df_dummy.fillna(0)

for j in range(0, len(df_dummy)):
    df_dummy['synthetic_mom'].iloc[j] = np.dot(df_dummy.drop(['synthetic_mom'], axis=1).iloc[j],
                                               feature_importance['weights'])

data['synthetic_mom'] = df_dummy['synthetic_mom']

data[['Spot', 'synthetic_mom', 'ema200', 'ema10']].plot()

# performance of strategy
# =========================
bps = 2e-4
data['Signal_synthetic'] = np.where(data['Spot'] > data['synthetic_mom'], 1, -1)
pnl = performance(data['Signal_rolling'], np.log(data['Spot']).diff(), bps).cumsum()

pnl.plot()
