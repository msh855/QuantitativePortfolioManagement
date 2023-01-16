#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 08:55:35 2022

@author: safishajjouz
"""
import numpy as np

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myReturns import get_stock_returns, convert_returns_freq, calculate_returns
from myPortfolioManagement.myDataPreparation import denoise_series_kf, smooth_series_LowessSmoother, ts_standarise

stocks = ['^IXIC', 'VTI', 'VGLT', 'VGIT', 'IAU', 'DJP']
df_prices = get_stock_prices(stocks, wide_format=True)

ret = get_stock_returns(stocks)
convert_returns_freq(ret, convert_to='weekly')

ret = calculate_returns(df_prices, rolling_window=6, convert_to='monthly', annualise_factor=12)
ret.plot()

ret['VTI'] = np.where(ret['VTI']==0, np.nan, ret['VTI'])

ret_denoised = denoise_series_kf(ret['VTI'].dropna())

# series
ret_denoised.plot()

# distribution
ret_denoised.plot.density()

df_prices_sm = smooth_series_LowessSmoother(df_prices, smoothing_parameter=0.01)

df_prices_sm.plot(subplots=True)

df_prices_sm.to_csv('/Users/safishajjouz/GitHub/myPortfolioManagement/df_prices_sm.csv')

prices = df_prices['VTI'].dropna()

x1 = ts_standarise(prices, window=60, mode='rolling')
x1 = x1.to_frame()
x1['VTA_ts_sm'] = ts_standarise(prices, window=120, mode='rolling')

x1.plot()

df_prices_sm.rolling(4).mean()
df_prices_sm.rolling(4).std()
df_prices_sm.rolling(4).skew()
df_prices_sm.rolling(4).kurt()

prices_ewm = df_prices_sm.ewm(50)
prices_ewm.mean().plot()
