import pandas as pd
from myScripts.functions.download_and_prepare_data import get_data
import numpy as np
from myScripts.functions.performance_stats import performance
from sklearn.linear_model import LogisticRegression
from myScripts.functions.download_and_prepare_data import get_ccy10y_spreads, get_ccy2y_spreads, relative_spread

# get currency
bps = 2e-4

currency = 'CADUSD'
df_ccy = get_data(currency=currency, get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy[['Spot']]
df = df_ccy.copy()

# download controls
halflife1 = 10
halflife2 = 60
halflife3 = 120

# carry on long-term bond (difference between longer-term yields across currencies)
df_ccy_10y_spread = get_ccy10y_spreads(currency=currency, daily_change=False)
df_ccy_10y_spread['10y_spread_sm1'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
df_ccy_10y_spread['10y_spread_sm2'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
df_ccy_10y_spread['10y_spread_sm3'] = df_ccy_10y_spread.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

df_ccy_10y_spread = pd.DataFrame(df_ccy_10y_spread.iloc[:, 1])

df_ccy_2y_spread = get_ccy2y_spreads(currency=currency, daily_change=False)
df_ccy_2y_spread['2y_spread_sm1'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
df_ccy_2y_spread['2y_spread_sm2'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
df_ccy_2y_spread['2y_spread_sm3'] = df_ccy_2y_spread.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

# df_ccy_2y_spread = pd.DataFrame(df_ccy_2y_spread.iloc[:, 1])

# Business cycle factor (difference between the spread 2-10 years)
df_spreads = relative_spread(currency=currency)
df_spreads.index.name = 'Date'
df_spreads['rel_slope_sm1'] = df_spreads.iloc[:, 0].ewm(halflife=halflife1, adjust=False).mean()
df_spreads['rel_slope_sm2'] = df_spreads.iloc[:, 0].ewm(halflife=halflife2, adjust=False).mean()
df_spreads['rel_slope_sm3'] = df_spreads.iloc[:, 0].ewm(halflife=halflife3, adjust=False).mean()

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

# controls
var_drop = ['return', 'move', 'Spot', 'target']
controls = list(df.drop(var_drop, axis=1).columns)
controls = [x for x in controls if currency not in x]

# construct models
spot = df['Spot']
df['ema10'] = spot.ewm(span=10, adjust=False).mean()
df['ema30'] = spot.ewm(span=30, adjust=False).mean()
df['ema60'] = spot.ewm(span=60, adjust=False).mean()
df['ema120'] = spot.ewm(span=120, adjust=False).mean()
df['ema200'] = spot.ewm(span=200, adjust=False).mean()

# strategies
# Naive momentum signal
df['return_cum'] = df['return'].cumsum()
df['return_cum_sm'] = df['return_cum'].ewm(halflife=126, adjust=False).mean()
df['naive_mom_signal'] = np.where(df['return_cum'] > df['return_cum_sm'], 1, -1)

# Naive mean-reverting 
df['return_cum'] = df['return'].cumsum()
df['return_cum_sm'] = df['return_cum'].ewm(halflife=126, adjust=False).mean()
df['naive_mom_signal'] = np.where(df['return_cum'] > df['return_cum_sm'], 1, -1)

# momentum
var_drop = ['return', 'move', 'Spot', 'target'] + controls
momentum_vars = list(df.drop(var_drop, axis=1).columns)
momentum_vars = [x for x in momentum_vars if currency not in x]

mytarget = 'target'
data = df.dropna()

data_training = data[data.index <= '2018-01-01']
#data_training = data

# train model
for var in momentum_vars:
    predictors = [var] + controls
    x = data_training[predictors].values
    y = data_training[mytarget].values
    model = LogisticRegression(fit_intercept=True, penalty='l2', class_weight='balanced')
    model.fit(x, y)

    # get predictions
    x_pred = data[predictors].values
    y_pred = model.predict(x_pred)

    # performance
    name = 'y_pred' + var
    data[name] = y_pred

    pnl_logistic = performance(data[name], np.log(data['Spot']).diff(), bps)
    data['PnL_' + var] = pnl_logistic.cumsum()


pnl_logistic = performance(data['naive_mom_signal'], np.log(data['Spot']).diff(), bps)
data['PnL_' + 'naive'] = pnl_logistic.cumsum()

pnl_vars = [x for x in data.columns if 'PnL' in x]
data[pnl_vars].plot()




# get predictions
x_pred = data[predictors].values
y_pred = model.predict(x_pred)

# performance
data['y_pred'] = y_pred

pnl_logistic = performance(data['y_pred'], np.log(data['Spot']).diff(), bps)
pnl_logistic.cumsum().plot(title=currency)

data[['y_pred', 'move']].plot()

prob = pd.Series(model.predict_proba(x_pred).ravel())
prob.index = data.index
prob.plot()
