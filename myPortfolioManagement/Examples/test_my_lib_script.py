#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 08:10:17 2021

@author: safishajjouz
"""

cd /Users/safishajjouz/GitHub/myPythonPackages

from myPortfolioManagement.myData import * 
from myPortfolioManagement.myPortfolioOptimisation import *
from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myPortfolioSelection import *
from pypfopt.expected_returns import returns_from_prices

import matplotlib.pyplot as plt
import seaborn as sns

#import seaborn as sns
#plt.rcParams['figure.figsize'] = (20, 18)
plt.style.use('fast')

benchmark_name = 'Invesco EQQQ NASDAQ-100 ETF GBP'


myassets = ['Scottish Mortgage Ord',                # Global Equity (focus on long-term) 
            'Invesco EQQQ NASDAQ-100 ETF GBP',      # Tech 
            'Rathbone Global Opportunities S Acc',  # Flex Global Cap 
            'Stewart Inv APAC Ldrs Sstby B GBP Acc', # Pacific Specialist 
            'BlackRock Throgmorton Trust Ord',       # UK small Cap 
            'HarbourVest Global Priv Equity Ord',    # Alternative 
            'Janus Henderson Mlt-Ast AbsRet I Acc']  # Absolute Alpha (proxy for JPM's fund due to missing data)

df_prices = load_fidelity_prices()
set(df_prices['fund'])

# drop_funds = ['Allianz Strategic Bond C Inc',
#               'HarbourVest Global Priv Equity Ord', 
#               'HgCapital Trust Ord']
#               #'Invesco EQQQ NASDAQ-100 ETF GBP'] 
#               #'Scottish Mortgage Ord']

# df_prices = df_prices[-df_prices['fund'].isin(drop_funds)]

data_overview(df_prices, my_assets_col_name = 'fund' , 
                         my_date_col_name = 'Date', 
                         price_col_name = 'price')

df_prices_wide = df_prices.pivot_table(index='Date', 
                                columns='fund', 
                                values='price')

df_prices_wide.plot(subplots=True,
                    layout=(8, 3),
                    sharex=False,
                    sharey=False,
                    # colormap='viridis',
                    fontsize=14,
                    legend=True,
                    linewidth=0.2);
plt.tight_layout();


ret = returns_from_prices(df_prices_wide)
ret = ret[ret.index>='2012-12-04']

### Adjust returns for Yields 
ret['Royal London Sterl Extra Yld Bd A']   = ret['Royal London Sterl Extra Yld Bd A'] + ffn.core.deannualize(0.05, 252)
ret['M&G Emerging Markets Bond GBP I Inc'] = ret['M&G Emerging Markets Bond GBP I Inc'] + + ffn.core.deannualize(0.05, 252)


df_alpha_beta = alpha_beta(ret, benchmark_name)



###################################################3

rolling_window     = 5
rolling_frequency  = 'Y'
my_assets_col_name = 'fund' 
my_date_col_name   = 'Date'
price_col_name = 'price'
benchmark_name = 'Invesco EQQQ NASDAQ-100 ETF GBP'


df_reward_metrics = ranking_metrics(df = df_prices.reset_index(), 
                                   rolling_window = rolling_window, 
                                   rolling_frequency = rolling_frequency,
                                   my_assets_col_name = my_assets_col_name, 
                                   my_date_col_name = my_date_col_name,
                                   price_col_name = price_col_name,
                                   benchmark_name = benchmark_name)

df_reward_metrics.sort_values('overall_score', ascending=False)


#simple returns 
ret_simple = returns_from_prices(df)

# calculate rolling mean 
df_rolling = df.resample('Y').mean().pct_change().rolling(3).mean()



df_corr = df_rolling.corr()
df_corr = df_corr[df_corr<0.4]
plt.figure(figsize=(18, 10))
heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);


# resambple to focus on trends 
df_rolling = df_prices.resample('Y').mean().pct_change().rolling(3).mean().dropna(how = 'all')
df_rolling.head()

### Quick Check on Correlations (linear and Tails)

df_corr = df_rolling.corr()
df_corr = df_corr[df_corr<0.4]
plt.figure(figsize=(18, 10))
heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

### Hierarchical Clustering 

rp.plot_clusters(returns=ret, codependence="spearman",
                      linkage='ward', max_k=10,
                      leaf_order=True, dendrogram=True, ax=None)

### Tail co-dependence 

rp.plot_clusters(returns=ret, codependence="tail",
                      linkage='ward', max_k=10,
                      leaf_order=True, dendrogram=True, ax=None)
