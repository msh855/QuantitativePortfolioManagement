#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:55:27 2022

@author: safishajjouz
"""


cd /Users/safishajjouz/GitHub/myPythonPackages

import warnings
warnings.filterwarnings("ignore")

from myPortfolioManagement.myData import * 
from myPortfolioManagement.myPortfolioOptimisation import *
from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myPortfolioSelection import *

from empyrical.perf_attrib import perf_attrib
from empyrical.utils import get_fama_french

import quantstats as qs
import pyfolio as pf


df= load_fidelity_prices()

drop_funds = ['HgCapital Trust Ord', 
              'WisdomTree Physical Precious Metals ETC GBP', 
              'SLI UK Real Estate Platform 1 Inc', 
              'Royal London Sustainable Div C Acc', 
              'HarbourVest Global Priv Equity Ord', 
              'M&G Emerging Markets Bond GBP I Inc', 
              'Royal London Sterl Extra Yld Bd A',
              'Royal London Sustainable Div C Acc', 
              'Vanguard Glb Bd Idx £ H Dist',
               'Allianz Strategic Bond C Inc']
           
df= df[-df['fund'].isin(drop_funds)]

df_prices = df.pivot_table(index='Date', 
                           columns='fund', 
                           values='price')

ret = calculate_returns(df_prices)
ret = ret[ret.index>='2012-12-04']


port = generate_HRP_portfolios_flex(returns = ret, 
                                    weight_max = 0.20, 
                                    weight_min = 0.02) 


port_weights  = equal_weight_portfolio(ret, 
                                       my_assets_col_name='fund')

ret_port = calculate_portfolio_returns(ret, list(port_weights.index), 
                                       [0.10, 0.05, 0.05, 0.10, 0.20,0.30,0.10, 0.10])

alpha_beta_table(ret_port, 
           returns_benchmark = ret['Vanguard FTSE Dev Wld ex-UK Eq Idx £ Acc'], 
           period = 'yearly') 


ret_bear = ret[ret['Vanguard FTSE Dev Wld ex-UK Eq Idx £ Acc'] < 0]
ret['Vanguard FTSE Dev Wld ex-UK Eq Idx £ Acc'][ret['Vanguard FTSE Dev Wld ex-UK Eq Idx £ Acc'] > 0]

 qs.reports.full(ret)

ret_port.index = ret_port.index.tz_localize('US/Eastern')

factors = get_fama_french()

ret.index = ret.index.tz_localize('US/Eastern')

fact_loadings = pd.melt(ret.reset_index(), id_vars='Date')
positions = fact_loadings.merge(port_weights, on = 'fund')

positions = positions.drop('value', axis = 1)
positions = positions.set_index(['Date', 'fund'])
positions = positions.sort_index(level=0)

fact_loadings = fact_loadings.set_index(['Date', 'fund'])
fact_loadings.index = fact_loadings.index.set_names(['Date', 'fund' ])
fact_loadings = fact_loadings.sort_index(level=0)



perf_attrib(returns = ret_port, 
            positions = positions, 
            factor_returns = factors, 
            factor_loadings= fact_loadings)

pf.create_returns_tear_sheet()

qs.utils.make_index()
qs.reports.full(ret_1970s)