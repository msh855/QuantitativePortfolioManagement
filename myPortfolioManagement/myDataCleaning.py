import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import quantstats
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.outliers import OutlierTrimmer
import ffn

class RollingStandardScaler(BaseEstimator, TransformerMixin):
    """Rolling standard Scaler

    https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d


    Standardized the given data series using the mean and std
    commputed in rolling or expanding mode.

    Parameters
    ----------
    window : int
        Number of periods to compute the mean and std.
    mode : str, optional, default: 'rolling'
        Mode

    Attributes
    ----------
    pd_object : pandas.Rolling
        Pandas window object.
    w_mean : pandas.Series
        Series of mean values.
    w_std : pandas.Series
        Series of std. values.
    """

    def __init__(self, window, mode='rolling'):
        '''mode = [rolling or expanding]"'''

        self.window = window
        self.mode = mode

        # to fill in code
        self.pd_object = None
        self.w_mean = None
        self.w_std = None
        self.__fitted__ = False

    def __repr__(self):
        return f"RollingStandardScaler(window={self.window}, mode={self.mode})"

    def fit(self, X, y=None):
        """Fits.

        Computes the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_shape, n_features)
            The data used to compute the per-feature mean and std. Used for
            later scaling along the feature axis.
        y
            Ignored.
        """
        self.pd_object = getattr(X, self.mode)(self.window)
        self.w_mean = self.pd_object.mean()
        self.w_std = self.pd_object.std()
        self.__fitted__ = True

        return self

    def transform(self, X):
        """Transforms.

        Scale features of X according to the window mean and standard
        deviation.

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed data.
        """
        self._check_fitted()

        standardized = X.copy()
        return (standardized - self.w_mean) / self.w_std

    def inverse_transform(self, X):
        """Inverse transform

        Undo the transform operation

        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.

        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed (original) data.
        """
        self._check_fitted()

        unstandardized = X.copy()
        return (unstandardized * self.w_std) + self.w_mean

    def _check_fitted(self):
        """ Checks if the algorithm is fitted. """
        if not self.__fitted__:
            raise ValueError("Please, fit the algorithm first.")


def data_overview(df: pd.DataFrame,
                  my_assets_col_name: str,
                  my_date_col_name: str,
                  price_col_name: str) -> pd.DataFrame:
    """
    This function expects a dataframe in long-format and a date column
    and returns a pandas dataframe that shows the period of available data
    that are available for each asset
    ...

    Args:
          df (dataframe): A list of yahoo tickers
          my_assets_col_name(str): the name of the column of
                                   your dataframe with
                                   asset names (e.g tickers, names etc..)
          my_date_col_name (str): the name of the column of your dataframe with the dates
                                  which column in your dataframe.
                                  Can be your index name

    Returns:
      pandas Dataframe: Returns a pandas dataframe with stock prices and other info
    """

    df_overview = df

    if type(df_overview.index) == pd.DatetimeIndex:
        df_overview = df_overview.reset_index()

    df_overview[my_date_col_name] = pd.to_datetime(df_overview[my_date_col_name], format='%Y/%m/%d')

    df1 = df_overview[df_overview.groupby(my_assets_col_name).Date.transform('min') == df_overview[my_date_col_name]][
        [my_assets_col_name, my_date_col_name]]
    df2 = df_overview[df_overview.groupby(my_assets_col_name).Date.transform('max') == df_overview[my_date_col_name]][
        [my_assets_col_name, my_date_col_name]]
    df1 = df1.rename(columns={my_date_col_name: "Date_min"})
    df2 = df2.rename(columns={my_date_col_name: "Date_max"})

    # merge dataframes
    df_overview = df1.merge(df2, how="left")

    df_overview["trading_days"] = df_overview["Date_max"] - df_overview["Date_min"]
    df_overview['trading_days'] = [df_overview['trading_days'][i].days for i in range(len(df_overview['trading_days']))]
    df_overview['years_available'] = df_overview['trading_days']/365


    df_overview = df_overview.rename(columns={'Stock': my_assets_col_name})
    df_overview = df_overview.drop_duplicates()

    # from long to wide

    df.groupby(my_assets_col_name)

    df_wide = df.pivot_table(index=my_date_col_name,
                             columns=my_assets_col_name,
                             values=price_col_name)

    df_NAs = pd.DataFrame(pd.Series(df_wide.isnull().mean().round(4).mul(100).sort_values(ascending=False),
                                    name='percentage_of_NAs'))

    df_overview = df_overview.merge(df_NAs, on=my_assets_col_name)
    df_overview = df_overview.set_index(my_assets_col_name)

    return df_overview.sort_values('years_available', ascending=False)


def qs_remove_outliers(dta: pd.DataFrame or pd.Series) -> pd.DataFrame or pd.Series:
    assert isinstance(dta, pd.DataFrame or pd.Series)
    treated = quantstats.stats.remove_outliers(dta)
    return treated


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


def ts_standarise(x: pd.Series, window: int, mode='rolling') -> pd.Series:
    """
    Args:
        x (pd.Series): DESCRIPTION.
        window (int): DESCRIPTION.
        mode (TYPE, optional): DESCRIPTION. Defaults to 'rolling'. Can also be
                                            'expanding'

    Returns:
        x_scaled (TYPE): DESCRIPTION.

    """
    x = x.dropna()
    # set parameters
    xrolling = RollingStandardScaler(window=window, mode=mode)
    # fit series
    xrolling.fit(X=x)
    # transform series
    x_scaled = xrolling.transform(X=x)

    return x_scaled


def ts_standarise_df(df: pd.DataFrame, window: int, mode: str = 'rolling') -> pd.DataFrame:
    df_stz = df.apply(lambda x: ts_standarise(x, window=window, mode=mode))
    return df_stz


def adf_statistics(time_series):
    """
    Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(time_series.values)
    if result[1] < 0.0500:  # result[1] contains the p-value
        return 0  # returns 0 value if p-value of test is under 5%
    else:
        return 1


def adf_tests(df):
    """
    Augmented Dickey-Fuller test applied to every column in DataFrame
    """
    results = df.apply(adf_statistics, axis=0)  # Output is a Pandas series
    if sum(results) == 0:
        print('Null hypothesis of non-stationarity is rejected for ALL series with p-values < 5%')
    else:
        for i, v in results.items():
            if v == 1:
                print(f'Null hypothesis of non-stationarity of {i} series is NOT rejected')
            else:
                print(f'Null hypothesis of non-stationarity of {i} series is rejected')


def normalise_data(df: pd.DataFrame or pd.Series = None, **kwarg) -> pd.DataFrame:
    x = df.values  # returns a numpy array
    min_max_scaler = MinMaxScaler(**kwarg)
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalised = pd.DataFrame(x_scaled, index=df.index)
    df_normalised.columns = df.columns
    return df_normalised


def get_hurst_exponent_function(time_series: pd.Series, max_lag=20):
    """Returns the Hurst Exponent of the time series
       A value above 0.40 denotes some long-term persistance
    """
    time_series = time_series.to_numpy()
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


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


def cap_outliersTS(returns: pd.DataFrame = None, capping_method='iqr',
                   tail='both',
                   fold=5,
                   plot: bool = False, **kwargs):
    # ref: https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/OutlierTrimmer.ipynb
    # ref: https://feature-engine.trainindata.com/en/latest/user_guide/outliers/OutlierTrimmer.html
    capper = OutlierTrimmer(capping_method=capping_method,
                            tail=tail,
                            fold=fold, **kwargs)
    capper.fit(returns)
    # capper.right_tail_caps_  # outlier values
    train_t = capper.transform(returns)
    prices_nomalised = ffn.to_price_index(train_t.dropna(), start=100)
    if plot:
        prices_nomalised.plot()
    else:
        return prices_nomalised
