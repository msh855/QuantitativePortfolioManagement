from myPortfolioManagement.myTimeSeries import Trends, decomposeTS, forecast_trend, look_back
from maData.getdata import get_US_yields
from myPortfolioManagement.myBootstrapping import bootstrappingTS
import matplotlib.pyplot as plt
import pandas as pd
from myPortfolioManagement.myBacktesting import prep_dist
import seaborn as sns
from myPortfolioManagement.myBootstrapping import BootstrapStationary

# ======================================================================================================================
# specs
# ======================================================================================================================
date_tr = '2020-07-01'
n_sample = 10000

# ======================================================================================================================
# load data
# ======================================================================================================================
df_yields = get_US_yields(freq='d')
df_yields['Spread'] = df_yields['2Y'] - df_yields['3M']

df_yields['3Y'].plot.density()

# split train and test set
df_yields_tr = df_yields[df_yields.index <= date_tr]
df_yields_out = df_yields[df_yields.index > date_tr]

# ======================================================================================================================
# boostrap
# ======================================================================================================================
boost_tep = 'sb'
blc_years = 5 * 365

boost_types = ['sb', 'mbb', 'cbb']

results_list = {}

for bst in boost_types:
    yields_boost_temp = bootstrappingTS(df_yields_tr['2Y'].dropna(), bootstrap_type=bst, block_size=blc_years,
                                        n_samples=n_sample, seed=123)
    results_list[bst] = yields_boost_temp
    if bst in ['sb', 'cbb']:
        yields_boost_temp_opt = bootstrappingTS(df_yields_tr['2Y'].dropna(), bootstrap_type=bst, optimal_block=True,
                                                n_samples=n_sample, seed=123)
        results_list[bst + '_opt'] = yields_boost_temp_opt

# Assuming 'dict_of_dfs' is your dictionary of dataframes
df_results = pd.concat([v.assign(key=k) for k, v in results_list.items()])

# get expected rates and melt data
df_results_means = df_results.groupby('key').mean()
df_results_long = pd.melt(df_results_means.reset_index(), id_vars='key')

# plot
sns.displot(data=df_results_long, hue='key', x='value',
            kind="kde", fill=False, legend=True, height=5, aspect=0.5,
            cut=0, bw_adjust=1)
# df_yields[col].plot.density()
plt.axvline(x=df_yields_tr['2Y'].mean(), color='black', label='historical (ex-ante)')
plt.axvline(x=df_yields_out['2Y'].mean(), color='red', label='Realized (ex-post)')
plt.legend()
plt.title('Ex-post Evaluation: ' + '2Y')
plt.show()

yields_boots2y = bootstrappingTS(df_yields_tr['2Y'].dropna(), bootstrap_type=boost_tep, block_size=blc_years,
                                 n_samples=n_sample)
yields_boots2y = bootstrappingTS(df_yields_tr['2Y'].dropna(), bootstrap_type=boost_tep, block_size=blc_years,
                                 n_samples=n_sample)
yields_boots2y = bootstrappingTS(df_yields_tr['2Y'].dropna(), bootstrap_type=boost_tep, block_size=blc_years,
                                 n_samples=n_sample)

yields_boots10 = bootstrappingTS(df_yields_tr['10Y'].dropna(), bootstrap_type=boost_tep, block_size=blc_years,
                                 n_samples=n_sample)
yields_bootsYC = bootstrappingTS(df_yields_tr['Spread'].dropna(), bootstrap_type=boost_tep, block_size=blc_years,
                                 n_samples=n_sample)

prob_dist = prep_dist(yields_boots2y)
prob_dist.plot()

for col, yeld in zip(['2Y'], [yields_boots2y]):
    plt.figure()
    yeld.mean().plot.density()
    # df_yields[col].plot.density()
    plt.axvline(x=df_yields_tr[col].mean(), color='black', label='historical (ex-ante)')
    plt.axvline(x=df_yields_out[col].mean(), color='red', label='Realized (ex-post)')
    plt.legend()
    plt.title('Ex-post Evaluation: ' + col)
    plt.show()
