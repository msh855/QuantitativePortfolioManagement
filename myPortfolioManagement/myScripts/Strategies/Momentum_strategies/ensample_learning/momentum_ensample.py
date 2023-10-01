import pandas as pd
from myScripts.functions.download_and_prepare_data import get_data
import numpy as np
from myScripts.functions.performance_stats import performance
from sklearn.linear_model import LogisticRegression
from myScripts.functions.download_and_prepare_data import get_ccy10y_spreads, get_ccy2y_spreads, relative_spread

# get currency
currency = 'CHFUSD'
df_ccy = get_data(currency=currency, get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy[['Spot']]

df = df_ccy.copy()

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

# create new dataframe
mytarget = 'target'
df['return'] = np.log(df['Spot']).diff()
df['Direction'] = np.where(df['return'] > 0, 1, -1)
df[mytarget] = df['Direction'].shift(-1)  # predictive regressions

# Strategy Signals
df['return_cum'] = df['return'].cumsum()
df['return_cum_sm'] = df['return_cum'].ewm(halflife=126, adjust=False).mean()

df['Signal_mean_rev'] = np.where(df['return_cum'] >= df['return_cum_sm'], -1, 1)  # mean reverting strategy
df['Signal_mom'] = np.where(df['return_cum'] >= df['return_cum_sm'], 1, -1)  # momentum strategy

# features
spot = df['Spot']
df['ema10'] = spot.ewm(span=10, adjust=False).mean()
df['ema30'] = spot.ewm(span=30, adjust=False).mean()
df['ema60'] = spot.ewm(span=60, adjust=False).mean()
df['ema120'] = spot.ewm(span=120, adjust=False).mean()
df['ema200'] = spot.ewm(span=200, adjust=False).mean()

# Models
data = df.dropna()
data.to_clipboard()

# define controls
var_drop = ['return', 'Direction', 'Spot', 'target', 'return_cum', 'return_cum_sm', 'Signal_mean_rev', 'Signal_mom']
controls = list(data.drop(var_drop, axis=1).columns)
controls = [x for x in controls if 'ema' not in x]
controls = [x for x in controls if '_sm' in x]

# predictive correlations between momentum technicals, strategies and return direction
x_list = [x for x in data.columns if x not in controls and 'ema' in x]
x_list = x_list + ['Signal_mean_rev', 'Signal_mom']

# train model
cut_off_date = '2018-01-01'
data_training = data[data.index <= cut_off_date]
model_list = []
for xvar in x_list:
    mypredictors = [xvar] + controls
    x = data_training[mypredictors].values
    y = data_training[mytarget].values
    model = LogisticRegression(fit_intercept=True, penalty='l2', class_weight='balanced')
    model.fit(x, y)
    model_list.append(model.fit(x, y))
    # get predictions
    x_pred = data[mypredictors].values
    y_pred = model.predict(x_pred)

    data['y_pred_' + xvar] = y_pred

# combine all
mypredictors = x_list + controls

x = data_training[mypredictors].values
y = data_training[mytarget].values
model = LogisticRegression(fit_intercept=True, penalty='l2', class_weight='balanced')
model.fit(x, y)
model_list.append(model.fit(x, y))

# get predictions
x_pred = data[mypredictors].values
y_pred = model.predict(x_pred)
data['y_pred_all'] = y_pred

# Strategy performance
bps = 2e-4
df['PnL_pure_mean_rev'] = performance(data['Signal_mean_rev'], np.log(df['Spot']).diff(), bps).cumsum()
df['PnL_pure_mom'] = performance(data['Signal_mom'], np.log(df['Spot']).diff(), bps).cumsum()

for xvar in x_list + ['all']:
    df['PnL_model_' + xvar] = performance(data['y_pred_' + xvar], np.log(data['Spot']).diff(), bps).cumsum()

pnl_vars = [x for x in df.columns if 'PnL' in x]
df[pnl_vars].plot(title=currency)

# out-of-sample performance
temp = df[pnl_vars]
temp = temp[temp.index >= cut_off_date].dropna()


def prices_from_returns(returns, log_returns=False):
    """
    Calculate the pseudo-prices given returns. These are not true prices because
    the initial prices are all set to 1, but it behaves as intended when passed
    to any PyPortfolioOpt method.

    :param returns: (daily) percentage returns of the assets
    :type returns: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) pseudo-prices.
    :rtype: pd.DataFrame
    """
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = 1 + returns
    ret.iloc[0] = 1  # set first day pseudo-price
    return ret.cumprod()


prices_from_returns(temp.diff()).plot()

####################################################

from mlforecast.utils import generate_daily_series

series = generate_daily_series(
    n_series=20,
    max_length=100,
    n_static_features=1,
    static_as_categorical=False,
    with_trend=True
)
series.head()

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

models = [
    lgb.LGBMRegressor(),
    xgb.XGBRegressor(),
    RandomForestRegressor(random_state=0),
]

from mlforecast import MLForecast
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean


@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)


fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    differences=[1],
)

fcst.fit(series, id_col='index', time_col='ds', target_col='y', static_features=['static_0'])

predictions = fcst.predict(14)
predictions