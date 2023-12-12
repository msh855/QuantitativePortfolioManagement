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
import numpy as np
import statsmodels.tsa.stattools as ts
import math
import logging


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

    lambda_values = {
        'daily': 1600 * ((365 / 4) ** 4),
        'weekly': 1600 * (12 ** 4),
        'monthly': 14400,
        'quarterly': 1600,
        'yearly': 6.25 if rescaled_lambda else 100
    }

    Lambda = lambda_values[freq]

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


def BoostedHP(x: pd.DataFrame or pd.Series, freq: str = 'daily', iter: bool = True, stopping: str = "BIC",
              sig_p: float = 0.050,
              Max_Iter: int = 100, rescaled_lambda: bool = True):
    # ' Boosting the Hodrick-Prescott Filter
    # '
    # ' Coded by Mei Ziwei
    # ' Documented by Shi Zhentao
    # '
    # ' All in one function of conducting the boosted HP-filter.
    ##############################################################################
    # ' Parameters:
    # '
    # ' x - a raw time series to be filtered.
    # ' lam - turning parameter, default value is 1600,
    # '   as recommended by Hodrick and Prescott (1997) for quarterly data.
    # ' iter - logical, True (default) to conduct the boosted HP filter.
    # '   False does not iterated, which is exactly the original HP filter.
    # ' stopping - stopping criterion.  "BIC" (default), or  "adf", or  "nonstop"  means keeping
    # '    iteration until the maximum number of iteration, specified by Max_Iter is reached.
    # ' sig_p - a threshold of the p-value for the ADF test, with default value 0.050.
    # '    Only effective when stopping = "adf".
    # ' Max_Iter - maximal number of iterations. The default is 100.
    ##############################################################################
    # ' The function returns a dictionary containing the following items:
    # '
    # ' cycle - The cyclical component in the final iteration.
    # ' trend - The trend component in the final iteration.
    # ' trend_hist - The estimated trend in each iteration.
    # ' iter_num - The total number of iterations when it stops.
    # ' BIC_hist - The path of the BIC up to the final iterations.
    # ' adf_p_hist - The path of the ADF test p-value up to the final iteration.
    ##############################################################################
    # ' Details:
    # '
    # ' This is the main function of implementing the boosted HP filter (Phillisp and
    # ' Shi, 2019). The arguments accommendate the orginal HP filter (iter =
    # ' False), the boosted HP filter with the BIC stopping criterion (stopping = "BIC"),
    # ' or ADF test stopping criterion (stopping = "adf"),
    # ' or keep going until the maximum number of iterations is reached (stopping = "nonstop").
    # '
    # ' Either the original HP filter or the bHP filter requires lambda to
    # ' control the strength of the weak learner for in-sample fitting.
    # ' The default is lambda = 1600,
    # ' which is recommended by Hodrick and Prescott (1997) for quarterly data.
    # ' lambda should be adjusted for different frequencies.
    # ' For example, lambda = 129600 for monthly data and
    # ' lambda = 100 or 6.25 for annual data.
    # '
    # ' See the vignette with a brief introduction of the idea of bHP.
    # '
    ##############################################################################
    # ' References:
    # '
    # ' Phillips, Peter CB, and Zhentao Shi. "Boosting: Why you can use the hp
    # ' filter." arXiv: 1905.00175, Cowles Foundation Discussion Paper No.2192,
    # ' (2019).
    # '
    ##############################################################################

    lambda_values = {
        'daily': 1600 * ((365 / 4) ** 4),
        'weekly': 1600 * (12 ** 4),
        'monthly': 14400,
        'quarterly': 1600,
        'yearly': 6.25 if rescaled_lambda else 100
    }

    lam = lambda_values[freq]

    check_date_index(x)

    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)

    data_original = x.copy()

    x = np.array(x)

    ## generating trend operator matrix "S：
    raw_x = x  # save the raw data before HP
    n = len(x)  # data size

    I_n = np.eye(n)
    D_temp = np.vstack((np.zeros([1, n]), np.eye(n - 1, n)))
    D_temp = np.dot((I_n - D_temp), (I_n - D_temp))
    D = D_temp[2:n].T
    S = np.linalg.inv(I_n + lam * np.dot(D, D.T))  # Equation 4 in PJ
    mS = I_n - S

    ##########################################################################

    ## the simple HP-filter
    if not iter:
        print("Original HP filter.")
        x_f = np.dot(S, x)
        x_c = x - x_f
        result = {"cycle": x_c, "trend_hist": x_f, \
                  "stopping": "nonstop", "trend": x - x_c, "raw_data": raw_x}

    ##########################################################################

    ## The Boosted HP-filter
    if iter:
        ### ADF test as the stopping criterion
        if stopping == "adf":

            print("Boosted HP-ADF.")

            r = 1
            stationary = False
            x_c = x

            x_f = np.zeros([n, Max_Iter])
            adf_p = np.zeros([Max_Iter, 1])

            while (r <= Max_Iter) and (not stationary):

                x_c = np.dot(mS, x_c)
                x_f[:, [r - 1]] = x - x_c
                adf_p_r = ts.adfuller(x_c, maxlag=math.floor(pow(n - 1, 1 / 3)), autolag=None, \
                                      regression="ct")[1]

                # x_c is the residual after the mean and linear trend being removed by HP filter
                # we use the critical value for the ADF distribution with
                # the intercept and linear trend specification

                adf_p[[r - 1]] = adf_p_r
                stationary = adf_p_r <= sig_p

                # Truncate the storage matrix and vectors
                if stationary:
                    R = r
                    x_f = x_f[:, 0:R]
                    adf_p = adf_p[0:R]
                    break

                r += 1

            if r > Max_Iter:
                R = Max_Iter
                logging.warning("The number of iterations exceeds Max_Iter. \
                The residual cycle remains non-stationary.")

            result = {"cycle": x_c, "trend_hist": x_f, "stopping": stopping,
                      "signif_p": sig_p, "adf_p_hist": adf_p, "iter_num": R,
                      "trend": x - x_c, "raw_data": raw_x}


        else:  # either BIC or nonstopping

            # assignment
            r = 0
            x_c_r = x
            x_f = np.zeros([n, Max_Iter])
            IC = np.zeros([Max_Iter, 1])
            # IC_decrease = True

            I_S_0 = I_n - S
            c_HP = np.dot(I_S_0, x)
            I_S_r = I_S_0

            while r < Max_Iter:

                r += 1

                x_c_r = np.dot(I_S_r, x)
                x_f[:, [r - 1]] = x - x_c_r
                B_r = I_n - I_S_r
                IC[[r - 1]] = np.var(x_c_r) / np.var(c_HP) + \
                              np.log(n) / (n - np.sum(np.diag(S))) * np.sum(np.diag(B_r))

                I_S_r = np.dot(I_S_0, I_S_r)  # update for the next round

                if r >= 2 and stopping == "BIC":
                    if IC[[r - 2]] < IC[[r - 1]]:
                        break

            # final assignment
            R = r - 1
            x_f = x_f[:, list(range(0, R))]
            x_c = x - x_f[:, [R - 1]]

            if stopping == "BIC":
                print("Boosted HP-BIC.")
                # save the path of BIC till iter+1 times to keep the "turning point" of BIC history.
                result = {"cycle": x_c, "trend_hist": x_f, "stopping": stopping,
                          "BIC_hist": IC[0:(R + 1)], "iter_num": R, "trend": x - x_c, "raw_data": raw_x}

            if stopping == "nonstop":
                print('Boosted HP-BIC with stopping = "nonstop".')
                result = {"cycle": x_c, "trend_hist": x_f, "stopping": stopping,
                          "BIC_hist": IC, "iter_num": Max_Iter - 1, "trend": x - x_c, "raw_data": raw_x}

    df_results = pd.DataFrame(index=data_original.index)
    name = data_original.columns[0]
    df_results[name] = result['raw_data']
    df_results[name + '_cycle'] = result['cycle']
    df_results[name + '_trend'] = result['trend']

    return df_results


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

    def BoostedHP(self, freq: str = 'daily', iter: bool = True, stopping: str = "BIC",
                  sig_p: float = 0.050,
                  Max_Iter: int = 100, rescaled_lambda: bool = True) -> pd.DataFrame:
        if self is pd.DataFrame:
            self.df = self.iloc[:, 0]
        return BoostedHP(self.df, freq, iter, stopping, sig_p, Max_Iter, rescaled_lambda)


def forecast_trend(data: pd.DataFrame or pd.Series = None, steps: list = [1, 2, 3], feq: str = 'd'):
    y = data.to_period(feq)
    forecaster = TrendForecaster()
    forecaster.fit(y)  # fit the forecaster
    pred = forecaster.predict(fh=steps)  # predict the next value
    return pred


def look_back(data: pd.DataFrame or pd.Series, years: int = 3):
    check_date_index(data)
    # Get the date three years ago from today
    three_years_ago = data.index[-1].to_pydatetime() - timedelta(days=years * 365)
    # Select data from three years ago until today
    data_lookback = data[str(three_years_ago.date()):]
    return data_lookback
