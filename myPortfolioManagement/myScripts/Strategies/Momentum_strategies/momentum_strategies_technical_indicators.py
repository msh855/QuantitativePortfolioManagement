import pandas as pd
from myScripts.functions.download_and_prepare_data import get_data
import numpy as np
from myScripts.functions.performance_stats import performance
from sklearn.linear_model import LogisticRegression
from myScripts.functions.download_and_prepare_data import get_ccy10y_spreads, get_ccy2y_spreads, relative_spread
import math


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


# get currency
bps = 2e-4

currency = 'CHFUSD'
df_ccy = get_data(currency=currency, get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy[['Spot']]
df = df_ccy.copy()

macro_regime_lookback = 30

halflife1 = 10
halflife2 = 60
halflife3 = 120

# carry on long-term bond (difference between longer-term yields across currencies)
df_ccy_10y_spread = get_ccy10y_spreads(currency=currency, daily_change=False)
df_ccy_10y_spread['10y_spread_sm1'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
#df_ccy_10y_spread['10y_spread_sm2'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
#df_ccy_10y_spread['10y_spread_sm3'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

df_ccy_10y_spread = pd.DataFrame(df_ccy_10y_spread.iloc[:, 1])

df_ccy_2y_spread = get_ccy2y_spreads(currency=currency, daily_change=False)
df_ccy_2y_spread['2y_spread_sm1'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
#df_ccy_2y_spread['2y_spread_sm2'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
#df_ccy_2y_spread['2y_spread_sm3'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

# df_ccy_2y_spread = pd.DataFrame(df_ccy_2y_spread.iloc[:, 1])

# Business cycle factor (difference between the spread 2-10 years)
df_spreads = relative_spread(currency=currency)
df_spreads.index.name = 'Date'
df_spreads['rel_slope_sm1'] = df_spreads.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
#df_spreads['rel_slope_sm2'] = df_spreads.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
#df_spreads['rel_slope_sm3'] = df_spreads.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

# df_spreads = pd.DataFrame(df_spreads.iloc[:, 1])

df = df.reset_index().merge(df_spreads.reset_index())
df = df.merge(df_ccy_2y_spread.reset_index())
df = df.merge(df_ccy_10y_spread.reset_index())
df = df.set_index('Date')

# create new dataframe
mytarget = 'target'
df['return'] = np.log(df['Spot']).diff()
df['move'] = np.where(df['return'] > 0, 1, -1)
df[mytarget] = df['move'].shift(-1)

spot = df['Spot']

# technicals
# df['KAMA_short_term'] = ta.kama(spot, fast=2, slow=30)
# df['KAMA_long_term'] = ta.kama(spot, fast=5, slow=120)
# df['MCGD'] = ta.mcgd(spot)

# Strategy Signals
df['return_cum'] = df['return'].cumsum()
df['return_cum_sm'] = df['return_cum'].ewm(halflife=126, adjust=False).mean()

df['Signal_mean_rev'] = np.where(df['return_cum'] >= df['return_cum_sm'], -1, 1)
df['Signal_mom'] = np.where(df['return_cum'] >= df['return_cum_sm'], 1, -1)

df['ema10'] = spot.ewm(span=10, adjust=False).mean()
df['ema30'] = spot.ewm(span=30, adjust=False).mean()
df['ema60'] = spot.ewm(span=60, adjust=False).mean()
df['ema120'] = spot.ewm(span=120, adjust=False).mean()
df['ema200'] = spot.ewm(span=200, adjust=False).mean()

df = df.iloc[200:]

# strategies
# df['Signal_KAMA_mean_reversion'] = np.where(df['KAMA_short_term'] > df['KAMA_long_term'], -1, 1)
# df['Signal_KAMA_mom'] = np.where(df['KAMA_short_term'] > df['KAMA_long_term'], 1, -1)

# model
var_drop = ['return', 'move', 'Spot', 'target', 'return_cum', 'return_cum_sm']
mypredictors = list(df.drop(var_drop, axis=1).columns)
mypredictors = [x for x in mypredictors if currency not in x]
# mypredictors = [x for x in df.columns if 'Signal' in x]

mytarget = 'target'
data = df.dropna()

data_training = data[data.index <= '2018-01-01']

# train model
x = data_training[mypredictors].values
y = data_training[mytarget].values
model = LogisticRegression(fit_intercept=True, penalty='l2', class_weight='balanced')
model.fit(x, y)

# get predictions
x_pred = data[mypredictors].values
y_pred = model.predict(x_pred)

# performance
data['y_pred'] = y_pred

df['PnL_mean_rev'] = performance(data['Signal_mean_rev'], np.log(df['Spot']).diff(), bps).cumsum()
df['PnL_mom'] = performance(data['Signal_mom'], np.log(df['Spot']).diff(), bps).cumsum()
df['PnL_model'] = performance(data['y_pred'], np.log(data['Spot']).diff(), bps).cumsum()

pnl_vars = [x for x in df.columns if 'PnL' in x]
df[pnl_vars].plot(title=currency)

# out-of-sample
prices = df[pnl_vars][df.index >= '2018-01-01']
prices = prices_from_returns(prices.diff().dropna())
prices.plot(figsize=(12, 5))




# drivers importance
w0 = model.intercept_[0]  # model's intercept
w = model.coef_[0]        # model coefficients

feature_importance = pd.DataFrame(mypredictors, columns=["feature"])
feature_importance["importance"] = pow(math.e, w)
feature_importance = feature_importance.sort_values(by=["importance"], ascending=False)
ax = feature_importance.plot.barh(x='feature', y='importance')
