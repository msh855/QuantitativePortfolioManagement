from myPortfolioManagement.myTimeSeries import Trends, decomposeTS, forecast_trend, look_back
from maData.getdata import get_US_yields
from myPortfolioManagement.myBootstrapping import bootstrappingTS
import matplotlib.pyplot as plt
import pandas as pd
from myPortfolioManagement.myBacktesting import prep_dist
import seaborn as sns
from myPortfolioManagement.myBootstrapping import BootstrapStationary
from myPortfolioManagement.myBacktesting import prep_dist, plot_fan_chart, sim_series, sim_paths

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

# ======================================================================================================================
# trends
# ======================================================================================================================
trneds = decomposeTS(df_yields[['2Y']].dropna())
#trends_bHP = trneds.BoostedHP(Max_Iter=10, stopping='nonstop')
trends_HP = trneds.HPfilter(freq='daily')
trends_HP.plot()

# ======================================================================================================================
# boostrap check different types
# ======================================================================================================================

sim = sim_series(df_yields[['2Y']].dropna(), n_sample=10000)
ylds = df_yields[['2Y']].dropna()

ylds.sample(frac=0.50, ignore_index=True, replace=True)
yields_paths = sim_paths(df_yields[['2Y']].dropna(), out_of_sample_date=date_tr, n_sample=10)



boost_tep = 'mbb'
blc_years = 5 * 365

# split train and test set
df_yields_tr = df_yields[df_yields.index <= date_tr]
df_yields_out = df_yields[df_yields.index > date_tr]

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

# resample [latest information]
df_results_yr = df_results.groupby(['key']).resample('Y').last()
df_results_yr = df_results_yr.drop(['key'], axis=1)

# possible historical averages [i.e. the mean of each path]
df_poss_hist_ave = df_results.groupby('key').mean()  # mean across columns / mean of each path
df_poss_hist_ave_long = pd.melt(df_poss_hist_ave.reset_index(), id_vars='key')

ave_hist = df_yields_tr['2Y'].mean()
last_info_training = df_yields_tr['2Y'].tail(1)[0]
last_info_oos = df_yields_out['2Y'].tail(1)[0]
ave_oos = round(df_yields_out['2Y'].mean(), 2)

# plot
sns.displot(data=df_poss_hist_ave_long[df_poss_hist_ave_long['key'] == 'mbb'], hue='key', x='value',
            kind="kde", fill=False, legend=True, height=5, aspect=0.5,
            cut=0, bw_adjust=1)
# df_yields[col].plot.density()
plt.axvline(x=ave_hist, color='black', label='historical (ex-ante)')
plt.axvline(x=last_info_training, color='red', label='Last information')
plt.axvline(x=ave_oos, color='blue', label='Average (ex-post')
plt.axvline(x=last_info_oos, color='green', label='Last Realized info (ex-post')
plt.title('Ex-post Evaluation: ' + '2Y')
plt.legend()
plt.show()

# Expected mean in each year
df_poss_mean_per_year = df_results.set_index('key', append=True).mean(axis=1).reset_index(
    'key')  # mean across columns / mean of each path
df_poss_mean_per_year.columns = ['key', 'rate']
df_poss_mean_per_year_long = pd.melt(df_poss_mean_per_year, id_vars='key')

df_poss_mean_per_year[df_poss_mean_per_year['key'] == 'mbb'].join(df_yields_tr['2Y']).drop('key', axis=1).plot.density()

# plot
sns.displot(data=df_poss_mean_per_year_long[df_poss_mean_per_year_long['key'] == 'mbb'], hue='key', x='value',
            kind="kde", fill=False, legend=True, height=5, aspect=0.5,
            cut=0, bw_adjust=1)
# df_yields[col].plot.density()
plt.axvline(x=ave_hist, color='black', label='historical (ex-ante)')
plt.axvline(x=last_info_training, color='red', label='Last information')
plt.axvline(x=ave_oos, color='blue', label='Average (ex-post')
plt.axvline(x=last_info_oos, color='green', label='Last Realized info (ex-post')
plt.title('Ex-post Evaluation: ' + '2Y')
plt.legend()
plt.show()

df_perc = df_results[df_results['key'] == 'mbb'].drop('key', axis=1)
df_perc = prep_dist(df_perc)

df_perc['0.95'].plot()

# ======================================================================================================================
# fan chart
# ======================================================================================================================
import numpy as np

# This is the fan part, using 'fill_between'
fig, ax = plt.subplots(figsize=(9, 5))
n_bands = int(np.floor(len(df_perc.columns) / 2))

for i in range(n_bands):
    # Choose alpha in a range of values
    alpha = 0.5 * (i + 1) / n_bands
    # Fill in colour between bands (ie between each 'fan')
    ax.fill_between(
        df_perc.index,
        # dates_to_fill,
        df_perc[df_perc.columns[i]],
        df_perc[df_perc.columns[-i - 1]],
        color="xkcd:blue",
        alpha=alpha,
        zorder=1,
    )

# Plot historical data
dates = df_perc.reset_index()['Date']
yvalues = df_perc

ax.plot(dates, yvalues, color="black", lw=1.5, zorder=3)
ax.axvspan(df_perc.index[0], df_perc.index[-1], facecolor="grey", alpha=0.2, zorder=0)
ax.grid(False, which="both")
ax.set_title('fan chart', loc="left", fontsize=12)

# df_results = df_results.droplevel('key')
# get expected rates and melt data
df_results_means = df_results.groupby('key').mean()
df_results_long = pd.melt(df_results_means.reset_index(), id_vars='key')

# plot
sns.displot(data=df_results_long, hue='key', x='value',
            kind="kde", fill=False, legend=True, height=5, aspect=0.5,
            cut=0, bw_adjust=1)
# df_yields[col].plot.density()
plt.axvline(x=df_yields_tr['2Y'].mean(), color='black', label='historical (ex-ante)')
plt.axvline(x=df_yields_out['2Y'].tail(1)[0], color='red', label='Realized (ex-post)')
plt.legend()
plt.title('Ex-post Evaluation: ' + '2Y')
plt.show()

# ======================================================================================================================
# boostrap check different windows
# ======================================================================================================================
boost_tep = 'mbb'
blc_years = [2 * 365]
results_list = {}
df_yields_2year_tr = df_yields_tr['2Y'].dropna()
df_try = df_yields_2year_tr[df_yields_2year_tr.index <= '1981-01-01']

for years in blc_years:
    yields_boost_temp = bootstrappingTS(df_yields_2year_tr, bootstrap_type=boost_tep, block_size=years,
                                        n_samples=n_sample, seed=123)
    results_list['window_' + str(years / 365)] = yields_boost_temp

# Assuming 'dict_of_dfs' is your dictionary of dataframes
df_results = pd.concat([v.assign(key=k) for k, v in results_list.items()])

# boxplots
df_results.drop(['key'], axis=1, inplace=True)
df_results['year'] = df_results.index.year
data = pd.melt(df_results, id_vars='year')

sns.boxplot(data=data, x='year', y='value')
plt.xticks(rotation=45, ha='right')

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
