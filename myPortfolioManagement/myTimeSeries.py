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
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter  # Christiano Fitzgerald
from myPortfolioManagement.myUtils import check_date_index, data_check_TS
import pywt
from pykalman import KalmanFilter

from aeon.forecasting.trend import TrendForecaster
import pandas as pd
from datetime import timedelta
from myPortfolioManagement.myUtils import check_date_index



# Functions to decompose TS to cycle and trend
# ======================================================================================================================

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
    x = data_check_TS(x)
    # if isinstance(x, pd.DataFrame) == True & x.shape[1] == 1:  # number of columns less than one convert to series
    #     x = x.iloc[:, 0]

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
    x = data_check_TS(x)

    # if isinstance(x, pd.DataFrame) == True & x.shape[1] == 1:  # number of columns less than one convert to series
    #     x = x.iloc[:, 0]

    x_cycle = cffilter(x.dropna(), low=low, high=high, drift=drift)[0]
    x_trend = cffilter(x.dropna(), low=low, high=high, drift=drift)[1]

    return pd.concat([x, x_cycle, x_trend], axis=1)


# function to denoise TS
# ======================================================================================================================
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
    # check if the input is in the correct format
    check_date_index(df)

    # if isinstance(df, pd.Series):
    #     df = df.to_frame()

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
    if isinstance(x, pd.DataFrame):
        raise 'You passed a dataframe. Pass pandas series'

    if isinstance(x, type(pd.Series)) == True:
        x = x.to_numpy()

    coff = freq_sampling_filter(x, n_components)
    cleaned = filtfilt(coff, 1, x, padlen=len(x) - 1, padtype='constant')

    return cleaned


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
    x = data_check_TS(x)

    # if isinstance(x, pd.DataFrame) == True & x.shape[1] == 1:  # number of columns less than one convert to series
    #     x = x.iloc[:, 0]
    #
    # if isinstance(x, pd.Series) == False:
    #     raise f"you passed {type(x)} instead you need pd.Series"

    x = x.dropna()
    den_noised = fft_denoiser(x, n_components=n_components)

    x_name_sm = x.name + '_smoothed_fft'
    dfx = x.to_frame()
    dfx[x_name_sm] = den_noised

    return dfx[[x_name_sm]]


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
    x = data_check_TS(x)

    # if isinstance(x, pd.DataFrame) == True & x.shape[1] == 1:  # number of columns less than one convert to series
    #     x = x.iloc[:, 0]
    #
    # if isinstance(x, pd.DataFrame):
    #     raise 'You passed a dataframe. Pass pandas series'

    x = x.dropna()

    coefficients = pywt.wavedec(x, wavelet, mode='per')
    coefficients[1:] = [pywt.threshold(i, value=smoothing_scale * x.max(), mode='soft') for i in coefficients[1:]]
    reconstructed_signal = pywt.waverec(coefficients, wavelet, mode=mode)

    dfx = x.to_frame()
    name = x.name + '_smooth_welvet'
    dfx[name] = reconstructed_signal

    return dfx[[name]]


def denoise_series_kf(x: pd.Series) -> pd.DataFrame:
    x = data_check_TS(x)
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

    return dfx[[name]]


def make_sma(df: pd.DataFrame, spans: list = [50, 60, 90, 200], add_prod: bool = False) -> pd.DataFrame:
    check_date_index(df)

    df_ema_list = []

    for span in spans:
        temp = df.ewm(span=span).mean()
        temp.columns = temp.columns + '_sm' + str(span)
        df_ema_list.append(temp)

    df_ema = pd.concat(df_ema_list, axis=1)
    if add_prod:
        df_ema_prod = df.ewm(halflife=126, adjust=False).mean()
        df_ema_prod.columns = df_ema_prod.columns + '_sm_prod'
        df_ema = df_ema.join(df_ema_prod)

    return df_ema


# Classes to collect functions
# ======================================================================================================================
class Trends:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        if isinstance(self.df, pd.Series):
            raise 'You passed a pandas series. You need to pass a DataFrame'

    def make_sma(self, spans: list = [50, 60, 90, 200], add_prod: bool = False) -> pd.DataFrame:
        return make_sma(self.df, spans, add_prod)

    def smooth_series_LowessSmoother(self, smoothing_parameter: float = 0.1) -> pd.DataFrame:
        return smooth_series_LowessSmoother(self.df, smoothing_parameter)

    def denoise_series_kf(self) -> pd.DataFrame:
        return denoise_series_kf(self.df)

    def denoise_series_welvet(self,
                              smoothing_scale: float = 0.5, wavelet: str = 'db6',
                              mode: str = 'per') -> pd.DataFrame:
        return denoise_series_welvet(self.df, smoothing_scale, wavelet, mode)

    def denoise_series_fft(self, n_components: float = 100):
        return denoise_series_fft(self.df, n_components)


class decomposeTS:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        if isinstance(self.df, pd.Series):
            raise 'You passed a pandas series. You need to pass a DataFrame'

    def HPfilter(self, freq: str = 'daily', rescaled_lambda=True) -> pd.DataFrame:
        if self is pd.DataFrame:
            self.df = self.iloc[:, 0]
        return HPfilter(self.df, freq, rescaled_lambda)

    def CFfilter(self, low: int = 6, high: int = 32, drift: bool = True) -> pd.DataFrame:
        return CFfilter(self.df, low, high, drift)


def forecast_trend(data: pd.DataFrame or pd.Series = None, steps: list = [1, 2, 3], feq: str = 'd'):
    y = data.to_period(feq)
    forecaster = TrendForecaster()
    forecaster.fit(y)  # fit the forecaster
    pred = forecaster.predict(fh=steps)  # predict the next value
    return pred

def look_back(data:pd.DataFrame or pd.Series, years:int = 3):
    check_date_index(data)
    # Get the date three years ago from today
    three_years_ago = data.index[-1].to_pydatetime() - timedelta(days=years * 365)
    # Select data from three years ago until today
    data_lookback = data[str(three_years_ago.date()):]
    return data_lookback

