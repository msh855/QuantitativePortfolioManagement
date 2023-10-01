import pandas as pd
import numpy as np
from myScripts.functions.prepared_data_for_dynamic_mom_learning import \
    get_features
from myScripts.functions.download_and_prepare_data import get_data
from myScripts.functions.prepared_data_for_dynamic_mom_learning import performance, \
    get_mom_strategies
from sklearn.linear_model import LogisticRegression
from myScripts.functions.prepared_data_for_dynamic_mom_learning import train_logistic_regression
from myScripts.functions.performance_stats import mainStats

# run script to get the data
currency = 'CADUSD'

# get features
df_features = get_features(currency=currency)

# get currency data
df_ccy = get_data(currency=currency, get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy.drop(['Currency', 'Flip'], axis=1)

# get momentum and mean reverting strategies
df_strategies = get_mom_strategies(spot_rate=df_ccy['Spot'])

# get strategies
df_strategies = df_strategies.reset_index()

keep = ['Date', 'return', 'Profit_Signal_mom_ema_prod', 'Profit_Signal_mean_rev_ema_prod', 'Signal_mom_ema_prod',
        'Signal_mean_rev_ema_prod']

df_strategies = df_strategies[keep]

# merge
data = df_ccy.reset_index().merge(df_strategies)
data = data.merge(df_features.reset_index())
data = data.set_index('Date')

# Create Targets for PnL
data['target_PnL_mr'] = data['Profit_Signal_mean_rev_ema_prod'].shift(-1)
data['target_PnL_mom'] = data['Profit_Signal_mom_ema_prod'].shift(-1)

# lagged returns
data['return_lag1'] = data['return'].shift(1)
data['return_lag2'] = data['return'].shift(2)
data['return_lag3'] = data['return'].shift(3)

data['ccy'] = currency

# train model
data = data.dropna()
data.to_clipboard()
cut_off_date = '2018-01-01'

# fit model
target = 'target_PnL_mom'
mypredictors = ['US_spread_6M_sm', 'US_spread_10Y_sm', 'US_spread_30Y_sm', 'rel_slope_sm',
                'US_Curvature_sm', 'US_ShiftFactor_sm']

df_prob = train_logistic_regression(data=data, cut_off_date=cut_off_date, mypredictors=mypredictors, target=target)

data_all = data.reset_index().merge(df_prob[['Signal_prob']].reset_index())
data_all = data_all.set_index('Date')
data_all['Signal_strategy'] = data_all['Signal_mom_ema_prod'] * data_all['Signal_prob']


# performance
def performance_mod(signal, total_returns, tc=.0002):
    trans_cost = tc * (signal.diff()).abs()
    excess_return = total_returns * signal - trans_cost
    return excess_return


ret_logistic = performance(data_all['Signal_strategy'], total_returns=np.log(data_all['Spot']).diff())[
               cut_off_date:]
ret_pure = performance(data_all['Signal_mom_ema_prod'], total_returns=np.log(data_all['Spot']).diff())[cut_off_date:]

mainStats(ret_logistic)
mainStats(ret_pure)

performance(data_all['Signal_strategy'], total_returns=np.log(data_all['Spot']).diff())[cut_off_date:].cumsum().plot()
performance(data_all['Signal_mom_ema_prod'], total_returns=np.log(data_all['Spot']).diff())[cut_off_date:].cumsum().plot()

performance(data_all['Signal_strategy'], total_returns=np.log(data_all['Spot']).diff()).cumsum().plot()
performance(data_all['Signal_mom_ema_prod'], total_returns=np.log(data_all['Spot']).diff()).cumsum().plot()

performance(data_all['Signal_strategy'], total_returns=np.log(data_all['Spot']).diff()).cumsum().plot()


model_list = []

# alla sample
data = data.dropna()

df_list = []
for target in targets:
    for predictor in mypredictors:
        df_temp = pd.DataFrame(index=data.index)
        df_temp['Target'] = target
        df_temp['predictor'] = predictor
        x = data[[predictor]].values
        y = data[[target]].values
        model = LogisticRegression(fit_intercept=True, penalty='l2', class_weight='balanced')
        model.fit(x, y)
        model_list.append(model.fit(x, y))
        # get predictions
        x_pred = data[[predictor]].values
        y_pred = model.predict(x_pred)
        # data['y_pred_' + target] = y_pred
        names = model.classes_
        df_temp['prob_pred_' + str(names[0])] = model.predict_proba(x_pred)[:, 0]
        df_temp['prob_pred_' + str(names[1])] = model.predict_proba(x_pred)[:, 1]

        df_temp['prob_pred_' + str(names[0])] = round(df_temp['prob_pred_' + str(names[0])], 3)
        df_temp['prob_pred_' + str(names[1])] = round(df_temp['prob_pred_' + str(names[1])], 3)
        df_temp['coef'] = model.coef_[0][0]

        df_list.append(df_temp)

df_all = pd.concat(df_list)

# all sample
x = data[mypredictors].values
y = data[['target_PnL_mom']].values
model = LogisticRegression(fit_intercept=True)
model.fit(x, y)
model_list.append(model.fit(x, y))
# get predictions
x_pred = data[mypredictors].values
y_pred = model.predict(x_pred)

names = model.classes_
data['prob_pred_' + 'target_PnL_mom' + str(names[0])] = model.predict_proba(x_pred)[:, 0]
data['prob_pred_' + 'target_PnL_mom' + str(names[1])] = model.predict_proba(x_pred)[:, 1]

data['prob_pred_' + 'target_PnL_mom' + str(names[0])].plot()

model.predict_proba(x_pred)[:, 0]

data['total_Signal_rv'] = data['y_pred_target_PnL_mr'] * data['Signal_mean_rev']
data['total_Signal_mom'] = data['y_pred_target_PnL_mom'] * data['Signal_mom']

performance(data['total_Signal_rv'], np.log(data['Spot']).diff()).cumsum().plot()
