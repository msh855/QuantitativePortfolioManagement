#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 08:55:35 2022

@author: safishajjouz
"""

cd /Users/safishajjouz/GitHub/myPythonPackages

from myPortfolioManagement.myData import * 
from myPortfolioManagement.myPortfolioOptimisation import *
from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myPortfolioSelection import *
from myPortfolioManagement.myBacktesting import *
from myPortfolioManagement.myReturns import *
from myPortfolioManagement.myReports import metrics 

from myPortfolioManagement.myUtils import * 

from myPortfolioManagement.myDataPreparation import * 



stocks = ['^IXIC', 'VTI', 'VGLT', 'VGIT', 'IAU', 'DJP']
df_prices = get_stock_prices(stocks, long_format = True)

ret = download_returns(stocks)
convert_returns_freq(ret, convert_to = 'weekly')

ret = calculate_returns(df_prices, rolling_window = 6, convert_to = 'monthly', annualise_factor = 12)
ret.plot()

denoise_series_kf(ret['VTI']).plot()
df_prices_sm = smooth_series_LowessSmoother(df_prices,  smoothing_parameter = 0.01)
df_prices_sm.plot(subplots= True)

ret_sm.to_csv('/Users/safishajjouz/GitHub/myPortfolioManagement/df_prices_sm.csv')

prices = df_prices['VTI'].dropna()


x1 = ts_standarise(prices, window = 60, mode = 'rolling')
x1 = x1.to_frame()
x1['VTA_expanding'] = ts_standarise(prices, window = 60, mode = 'expanding')

x1.plot()


ret_sm.rolling(4).mean()
ret_sm.rolling(4).std()
ret_sm.rolling(4).skew()
ret_sm.rolling(4).kurt()


