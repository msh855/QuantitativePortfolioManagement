"""
Created on Mon Mar 28 11:53:48 2022

@author: safishajjouz
"""
import warnings

warnings.filterwarnings('ignore')
import riskfolio as rp
from myPortfolioManagement.myPlots import *
from pyData.getdata import get_US_yields, get_USyield_curve_factors, get_fred_data, get_US_yield_spreads
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

## factors and recessions
US_yields = get_US_yields(freq='m')
df_US_yield_factors = get_USyield_curve_factors(start_date='1995-01-01',freq='m')

df_US_yield_factors.plot()
df_US_yield_factors.to_csv('yield_curve_pca.csv')

# get recessions 
US_rec = get_fred_data(['USREC'], freq='m')
df_US_yield_factors = get_USyield_curve_factors(freq='m')
df_US_yield_factors = df_US_yield_factors.merge(US_rec, left_index=True, right_index=True)
df_US_yield_factors = df_US_yield_factors.reset_index()

# create plot
fig, ax = plt.subplots()
df_US_yield_factors[['USyc_ShiftFactor', 'USyc_slope', 'USyc_curvature', 'Date']].plot.line(ax=ax, figsize=(8, 5),
                                                                                            x='Date', alpha=0.4,
                                                                                            color="blue",
                                                                                            label='Correlation')
df_US_yield_factors[['USREC', 'Date']].plot.area(ax=ax, figsize=(8, 5), x='Date', alpha=0.3, color="gray", label='Sine')
#plt.xlim(min(df.Date), max(df.Date))
plt.ylim(-1, 1.1)
plt.legend(["6 months rolling Correlations", '36 months rolling correlations', "US Recessions"], loc="lower right",
           prop={'size': 8})
ax.set_title('Rolling correlation Between 10y2y and 10y3m spreads', size=12)

yields_monthly = get_US_yield_spreads(freq='m')
df_US_yield_factors = get_USyield_curve_factors(freq='m')
df_US_yields = yields_monthly.merge(df_US_yield_factors, left_index=True, right_index=True)

correlation_matrix(df_US_yields)

# download Fred data


# compute correlations 
df_dummy = yields_monthly[['spread_3M', 'spread_2Y']].dropna()
s1 = df_dummy['spread_3M']
s2 = df_dummy['spread_2Y']
s6 = pd.Series(s1.rolling(6).corr(s2), name='spread_cor')
s12 = pd.Series(s1.rolling(12).corr(s2), name='spread_cor')
s24 = pd.Series(s1.rolling(24).corr(s2), name='spread_cor')
s36 = pd.Series(s1.rolling(36).corr(s2), name='spread_cor')

s6 = pd.DataFrame(s6).dropna()
s12 = pd.DataFrame(s12).dropna()
s24 = pd.DataFrame(s24).dropna()
s36 = pd.DataFrame(s36).dropna()

df_spread_cor = pd.concat([s6, s12, s24, s36], axis=1)
df_spread_cor.columns = ['spread_cor_6', 'spread_cor_12', 'spread_cor_24', 'spread_cor_36']

from statsmodels.tsa.stattools import grangercausalitytests

# perform Granger-Causality test
grangercausalitytests(yields_monthly[['spread_3M', 'spread_2Y']].dropna(), maxlag=[6])
grangercausalitytests(yields_monthly[['spread_2Y', 'spread_3M']].dropna(), maxlag=[6])

# get recessions 
US_rec = get_fred_data(['USREC'], freq='m')
US_rec = US_rec[US_rec.index >= min(df_spread_cor.index)]

df_spread_cor = df_spread_cor.merge(US_rec, left_index=True, right_index=True)
df = df_spread_cor.reset_index()

# create plot
fig, ax = plt.subplots()
df[['spread_cor_6', 'Date']].plot.line(ax=ax, figsize=(8, 5), x='Date', alpha=0.4, color="blue", label='Correlation')
# df[['spread_cor_12', 'Date']].plot.line(ax=ax, figsize=(8, 5), x='Date', alpha=0.6, color="blue")
# df[['spread_cor_24', 'Date']].plot.line(ax=ax, figsize=(8, 5), x='Date', alpha=0.6, color="blue")
df[['spread_cor_36', 'Date']].plot.line(ax=ax, figsize=(8, 5), x='Date', alpha=0.6, color="red")
df[['USREC', 'Date']].plot.area(ax=ax, figsize=(8, 5), x='Date', alpha=0.3, color="gray", label='Sine')
plt.xlim(min(df.Date), max(df.Date))
plt.ylim(-1, 1.1)
plt.legend(["6 months rolling Correlations", '36 months rolling correlations', "US Recessions"], loc="lower right",
           prop={'size': 8})
ax.set_title('Rolling correlation Between 10y2y and 10y3m spreads', size=12)

#
US_yields = get_US_yields(freq='m')
US_yields['spread_2Y_3M'] = US_yields['2Y'] - US_yields['3M']

US_yields = US_yields[['spread_2Y_3M']].merge(yields_monthly[['spread_10Y_3M', 'spread_10Y_2Y']], left_index=True,
                                              right_index=True)


US_yields.dropna().to_csv('spreads.csv')

US_yields.dropna().plot()

correlation_matrix(US_yields)

from datetime import datetime
from matplotlib.dates import date2num

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['spread_cor_6'])
ax.axvspan(date2num(datetime(2007, 1, 12)), date2num(datetime(2009, 6, 1)),
           label="2009 Recession", color="green", alpha=0.3)
ax.legend()
ax.set_ylabel('Number of Unemployment Claims')
ax.set_title('US Unemployment Claims Over Time', size=18)

rp.plot_clusters(returns=yields_monthly, codependence="spearman",
                 linkage='ward', leaf_order=True, dendrogram=True, ax=None)

# Standardising the data
X = pd.DataFrame(yields_monthly.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pd.DataFrame(X))
# Transformed the arrays of scaled values into a DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

hier_comp = linkage(X_scaled, method='complete', metric='euclidean')
hier_average = linkage(X_scaled, method='average', metric='euclidean')
hier_ward = linkage(X_scaled, method='ward', metric='euclidean')

plt.figure(figsize=(10, 8))
plt.title('Dendrogram of Stocks', fontsize=14)
plt.xlabel('Distance', fontsize=10)
plt.ylabel('Stock', fontsize=10)
dendrogram(
    hier_ward,
    orientation='right',
    #   leaf_rotation=90.,
    leaf_font_size=20,
    labels=X_scaled.index.values,
    color_threshold=3
)
plt.yticks(fontsize=11)
plt.show()
