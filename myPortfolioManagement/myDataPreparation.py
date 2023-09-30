#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:41:20 2022

@author: safishajjouz
"""
import pandas as pd
import numpy as np

from tsmoothie.smoother import LowessSmoother
from scipy.signal import filtfilt

from sklearn.base import BaseEstimator, TransformerMixin

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter  # Christiano Fitzgerald
import pywt
from statsmodels.tsa.stattools import adfuller

from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler

import quantstats


# function that denoises
def smooth_series_LowessSmoother(df: pd.DataFrame or pd.Series,
                                 smoothing_parameter: float = 0.1) -> pd.DataFrame:
    """
    ref: https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/TimeSeries_Smoothing_Clustering/TimeSeries_Smoothing_Clustering.ipynb 

    Args:
        df (pd.DatafRame): DESCRIPTION.
        smoothing_parameter (float, optional): DESCRIPTION. Defaults to 0.1.

    Returns:
        df_sm (TYPE): DESCRIPTION.

    """

    '''
    df: Time-Series Dataframe Dates are the index of the df 
    smoothing_parameter: Hyperparameter that controls the smoothing degree. 
                         Higher values result to more smoothed series 
    '''
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna()
    df.index = pd.to_datetime(df.index, format='%Y/%m/%d')
    # df.interpolate(method='time', inplace=True)

    # smooth Series 
    data_series = df.transpose().to_numpy()
    df_smoothed = LowessSmoother(
        smooth_fraction=smoothing_parameter)  # higher values for smooth fraction more smooth (trend)
    df_smoothed.smooth(data_series)

    # Check smoothed series intervals
    # low, up = df_smoothed.get_intervals('prediction_interval')

    # generate smoothed DataFrame 
    df_sm = df.copy()
    for i in range(len(df.columns)):
        df_sm.iloc[:, i] = df_smoothed.smooth_data[i]

    df_sm.columns = df.columns + '_smooth_Lowess'
    return df_sm


def HPfilter(x: pd.Series, freq: str = 'daily', rescaled_lambda=True) -> pd.DataFrame:
    """
    # Ravn–Uhlig rule sets lambda to 1600p^4, where pq is the number of periods per quarter
    # https://home.uchicago.edu/~huhlig/papers/uhlig.ravn.res.2002.pdf 
    # https://www.stata.com/manuals/tstsfilterhp.pdf

    Args:
        x (pd.Series): DESCRIPTION.
        freq (str, optional): DESCRIPTION. Defaults to 'daily'.
        rescaled_lambda (TYPE, optional): DESCRIPTION. Defaults to True.

    Returns:
        dfx (TYPE): DESCRIPTION.

    """

    if rescaled_lambda:
        if freq == 'daily':
            Lambda = 1600 * ((365 / 4) ** 4)
        if freq == 'weekly':
            Lambda = 1600 * (12 ** 4)
        if freq == 'monthly':
            Lambda = 129600
        if freq == 'quarterly':
            Lambda = 1600
        if freq == 'yearly':
            Lambda = 6.25
    else:
        if freq == 'daily':
            Lambda = 1600 * ((365 / 4) ** 4)
        if freq == 'weekly':
            Lambda = 1600 * (12 ** 4)
        if freq == 'monthly':
            Lambda = 14400
        if freq == 'quarterly':
            Lambda = 1600
        if freq == 'yearly':
            Lambda = 100

    x_cycle = hpfilter(x.dropna(), lamb=Lambda)[0]
    x_trend = hpfilter(x.dropna(), lamb=Lambda)[1]

    dfx = pd.concat([x, x_cycle, x_trend], axis=1)

    return dfx


def CFfilter(x: pd.Series, low: int = 6, high: int = 32,
             drift: bool = True) -> pd.DataFrame:
    """
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.filters.cf_filter.cffilter.html

    Args:
        x (pd.Series): DESCRIPTION.
        low (int, optional): DESCRIPTION. Defaults to 6.
        high (int, optional): DESCRIPTION. Defaults to 32.
        drift (bool, optional): DESCRIPTION. Defaults to True.

    Returns:
        TYPE: DESCRIPTION.

    """

    x_cycle = cffilter(x.dropna(), low=low, high=high, drift=drift)[0]
    x_trend = cffilter(x.dropna(), low=low, high=high, drift=drift)[1]

    return pd.concat([x, x_cycle, x_trend], axis=1)


## function that denoises
def freq_sampling_filter(x: pd.Series or np.array,
                         n_components: float = 100) -> np.array:
    """
    Args:
        x (pd.Series or np.array): DESCRIPTION.
        n_components (float, optional): DESCRIPTION. Defaults to 100  Higher values more smoothed series 

    Returns:
        _coff (TYPE): DESCRIPTION.
    """

    if isinstance(x, type(pd.Series)) == True:
        x = x.to_numpy()

    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n

    # keep frequencies with large contributions 
    _mask = PSD > n_components
    _coff = np.fft.fftshift(np.real(np.fft.ifft(_mask)))

    return _coff


# function that denoises
def fft_denoiser(x: pd.Series or np.array,
                 n_components: float = 100) -> np.array:
    """
    This function wraps up a process to denoise time series data using the 
    fourier transform 
    
    ref: https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d 
         https://stackoverflow.com/questions/67992487/fourier-transformation-fft-for-time-series-but-both-ends-of-cleaned-data-move 

    Args:
        x (pd.Series or np.array): DESCRIPTION.
        n_components (float, optional): DESCRIPTION. Defaults to 100.
                                        Higher values more smoothed series 

    Returns:
        cleaned (TYPE): DESCRIPTION.

    """

    if isinstance(x, type(pd.Series)) == True:
        x = x.to_numpy()

    coff = freq_sampling_filter(x, n_components)
    cleaned = filtfilt(coff, 1, x, padlen=len(x) - 1, padtype='constant')

    return cleaned


# function that denoises
def denoise_series_fft(x: pd.Series, n_components: float = 100):
    """
    

    Args:
        x (pd.Series): DESCRIPTION.
        n_components (float, optional): DESCRIPTION. Defaults to 100.
                                        Higher values more smoothed series 

    Raises:
        f: DESCRIPTION.

    Returns:
        dfx (TYPE): DESCRIPTION.

    """

    if isinstance(x, pd.Series) == False:
        raise f"you passed {type(x)} instead you need pd.Series"

    x = x.dropna()
    den_noised = fft_denoiser(x, n_components=n_components)

    x_name_sm = x.name + '_smoothed_fft'
    dfx = x.to_frame()
    dfx[x_name_sm] = den_noised

    return dfx


def denoise_series_welvet(x: pd.Series,
                          smoothing_scale: float = 0.5, wavelet: str = 'db6',
                          mode: str = 'per') -> pd.DataFrame:
    """
    ref: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html 

    Args:
        x (pd.Series): DESCRIPTION.
        smoothing_scale (float, optional): DESCRIPTION. Defaults to 0.5.
        wavelet (str, optional): DESCRIPTION. Defaults to 'db6'.
        mode (str, optional): DESCRIPTION. Defaults to 'per'.

    Returns:
        dfx (TYPE): DESCRIPTION.

    """
    x = x.dropna()

    coefficients = pywt.wavedec(x, wavelet, mode='per')
    coefficients[1:] = [pywt.threshold(i, value=smoothing_scale * x.max(), mode='soft') for i in coefficients[1:]]
    reconstructed_signal = pywt.waverec(coefficients, wavelet, mode='per')

    dfx = x.to_frame()
    name = x.name + '_smooth_welvet'
    dfx[name] = reconstructed_signal

    return dfx


def denoise_series_kf(x: pd.Series) -> pd.DataFrame:
    if isinstance(x,pd.DataFrame):
        raise 'You passed a dataframe. Pass pandas series'

    x_array = x.dropna().to_numpy()

    # initial guesses 
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)
    state_means, _ = kf.filter(x_array)
    filter_series = state_means

    dfx = x.dropna().to_frame()
    name = x.name + '_smooth_KF'
    dfx[name] = filter_series

    return dfx


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
    df_overview['years_available'] = df_overview['trading_days'] / np.timedelta64(1, 'Y')

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


# def create_credit_impulse(freq="q"):
#     credit_impulse = ['CRDQXMAPABIS',  # Euro Area
#                       'QUSPAM770A',  # US
#                      # 'QCNPAM770A',  # China
#                       'QGBPAM770A',  # United Kingdom
#                       'CRDQJPAPABIS']  # Japan
#     gdp = ['GDP', # US
#            'EUNNGDP', # Euro
#           # 'MKTGDPCNA646NWDB', #China
#            'JPNNGDP', #Japan
#            'UKNGDP' ] # United Kingdom
#
#     symbols = gdp + credit_impulse
#
#     data = download_fred_data(fred_sumbol=symbols, freq=freq)
#     data.columns = ['EUR', 'US', 'CHN', 'GB', 'JP']
#     return data

# proxy fund rate data
# source: https://www.kansascityfed.org/Economic%20Review/documents/319/2016-Measuring%20the%20Stance%20of%20Monetary%20Policy%20on%20and%20off%20the%20Zero%20Lower%20Bound.pdf
#       : https://www.frbsf.org/wp-content/uploads/sites/4/el2022-30.pdf

# 1. MORTGAGE30US - weekly, 30-year fixed Mortage rate
# 2. DGS2         - daily, 2-year Treasuries
# 3. DGS5         - daily, 5-year Treasuries
# 4. DGS7         - daily, 7-year Treasuries
# 5. DGS10        - daily, 10-year Treasuries
# 6. Bond buyer index: state/local bonds, 20-year, general obligation --- N/A
# 7. AAA          - Monthly, Moody's Seasoned Aaa Corporate Bond Yield
# 8. DBAA         - Daily, Moody's Seasoned Baa Corporate Bond Yield
# Spreads: Mortgage rates and 10-year Treasury spread , Two-year and 10-year Treasury spread,
#          Aaa corporate bond yield and 10-year Treasury spread, Baa corporate bond yield and 10-year Treasury spread



def make_sma(df: pd.DataFrame = None, spans: list = [50, 60, 90, 200]) -> pd.DataFrame:
    df_ema_list = []

    for span in spans:
        temp = df.ewm(span=span).mean()
        temp.columns = temp.columns + '_sm' + str(span)
        df_ema_list.append(temp)

    df_ema = pd.concat(df_ema_list, axis=1)
    df_ema_prod = df.ewm(halflife=126, adjust=False).mean()
    df_ema_prod.columns = df_ema_prod.columns + '_sm_prod'
    df_ema = df_ema.reset_index().merge(df_ema_prod.reset_index())
    df_ema = df_ema.set_index('Date')

    return df_ema