import pandas as pd
import numpy as np
from myScripts.functions.prepared_data_for_dynamic_mom_learning import get_data_for_dynamic_mom_learning


def performance(signal, total_returns, tc=.0002):
    trans_cost = tc * (signal.diff()).abs()
    excess_return = total_returns * signal.shift(1) - trans_cost

    return excess_return


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


# run script to get the data
data = get_data_for_dynamic_mom_learning(currency='CHFUSD')

data['ret_pure_mean_rev'] = performance(data['Signal_mean_rev'], np.log(data['Spot']).diff(), 0)
data['ret_signal_mom'] = performance(data['Signal_mom'], np.log(data['Spot']).diff(), 0)

# data.to_clipboard()

# get predecitions from R script
df_from_R = pd.read_clipboard()

df_from_R['model'] = np.where(df_from_R['model'] == 'model1', 'mean_rev', 'mom')
df_from_R['Date'] = pd.DatetimeIndex(df_from_R['Date'])

# assess performance of strategies
df = data.reset_index().merge(df_from_R)

# Strategy performance
df['Final_Signal'] = np.where(df['Predictions'] == 0, 0, df['Signal_mean_rev'])
df['Final_Signal'] = np.where((df['Predictions'] == 1) & (df['model'] == 'mom'), df['Signal_mom'], df['Final_Signal'])
df['Final_Signal'] = np.where((df['Predictions'] == 0) & (df['model'] == 'mom'), 0, df['Final_Signal'])

bps = 2e-4
df['PnL_pure_mean_rev'] = performance(df['Signal_mean_rev'], np.log(df['Spot']).diff(), bps).cumsum()
df['PnL_pure_mom'] = performance(df['Signal_mom'], np.log(df['Spot']).diff(), bps).cumsum()
df['PnL_BMA'] = performance(df['Final_Signal'], np.log(df['Spot']).diff(), bps).cumsum()

# plot performance
temp = df[['Date', 'PnL_pure_mean_rev', 'PnL_pure_mom', 'PnL_BMA']].dropna()
temp = temp.set_index('Date')
temp = temp.diff().dropna()
prices_from_returns(temp).plot()

#
# ##
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import BaggingClassifier
# from sklearn.metrics import accuracy_score
#
# # Load data
# data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# data = data.dropna()
#
# # Define features and target variable
# X = data.drop('target_variable', axis=1)
# y = data['target_variable']
#
# # Normalize data
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Define rolling window and number of splits
# window_size = 60
# num_splits = 10
#
# # Create rolling training and testing sets
# tscv = TimeSeriesSplit(n_splits=num_splits)
# X_train, y_train = [], []
# X_test, y_test = [], []
# for train_index, test_index in tscv.split(X_scaled):
#     X_train.append(X_scaled[train_index])
#     y_train.append(y[train_index])
#     X_test.append(X_scaled[test_index])
#     y_test.append(y[test_index])
#
# # Create bagging classifier with logistic regression
# bagging = BaggingClassifier(LogisticRegression(), n_estimators=10, max_samples=0.8, max_features=1.0)
#
# # Train and test model with rolling window
# accuracy_scores = []
# for i in range(num_splits):
#     X_train_window, y_train_window = X_train[i][-window_size:], y_train[i][-window_size:]
#     X_test_window, y_test_window = X_test[i][:window_size], y_test[i][:window_size]
#
#     bagging.fit(X_train_window, y_train_window)
#     y_pred = bagging.predict(X_test_window)
#     accuracy_score = accuracy_score(y_test_window, y_pred)
#     accuracy_scores.append(accuracy_score)
#
# # Calculate average accuracy score
# avg_accuracy_score = np.mean(accuracy_scores)