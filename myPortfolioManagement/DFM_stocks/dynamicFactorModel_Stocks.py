import pandas as pd
from openbb_terminal.sdk import TerminalStyle
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myDataPreparation import remove_outliers
from myPortfolioManagement.myData import get_stock_prices_from_openBB, get_stock_info
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
import numpy as np
import os

theme = TerminalStyle("light", "light", "light")

plt.style.use('seaborn')
warnings.filterwarnings("ignore")



# import tickers
# ======================================================================================================================
wd = os.getcwd()
data_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data'
file_to_load = 'stock_screening.xlsx'
file_path = os.path.join(wd,data_path, file_to_load)


# load tickers
df_tickers = pd.read_excel(file_path)
df_tickers.dropna(inplace=True)
index_name = 'YahooTicker'
tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])

# start_date = "2005-01-01"
# prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date)

file_path_prices = os.path.join(wd,data_path, 'df_prices.csv')
df_prices = pd.read_csv(file_path_prices)
df_prices = df_prices.set_index('date')

# Group stocks for DFM
# ======================================================================================================================
df_sectors = get_stock_info(yahoo_tickers=tickers)
df_sectors.groupby('Sector').count()[['Industry']].sort_values(by=['Industry'], ascending=False)

groups = df_sectors.drop(['Currency', 'Country'], axis=1)
groups = groups.join(df_tickers[['YahooTicker', 'Company']].set_index('YahooTicker'))
groups = groups.drop(['Industry'], axis=1)
groups.columns = ['group', 'description']

# Construct the variable => list of factors dictionary
factors = {row['description']: ['Global', row['group']]
           for ix, row in groups.iterrows()}

# Create Portfolio
# ======================================================================================================================
df_prices = df_prices.reset_index().merge(groups.reset_index())
df_prices = df_prices.set_index('date')
prices = df_prices.pivot_table(index=df_prices.index, values='Adj_Close_USD', columns='description')
returns = prices.pct_change()
returns = remove_outliers(returns)

# Construct the dynamic factor model
factor_multiplicities = {'Global': 3}  # all stocks have three common factors
factor_orders = {'Global': 4}  # all global factors assume to follow a VAR(4)

# Construct the dynamic factor model
returns.index = pd.to_datetime(returns.index)
returns.index = pd.DatetimeIndex(returns.index).to_period('D')
ret_m = returns.resample('M').mean()

model = sm.tsa.DynamicFactorMQ(ret_m, factors=factors,
                               factor_orders=factor_orders,
                               factor_multiplicities=factor_multiplicities)

model.summary()
results = model.fit(disp=10)
results.summary()

results.factors.filtered  # use information up to time t
results.factors.smoothed  # use full dataset

# Get estimates of the global and labor market factors,
# conditional on the full dataset ("smoothed")
factor_names = ['Global.1', 'Global.2', 'Technology']
mean = results.factors.smoothed[factor_names]
mean = mean[mean.index>='2023-01-01']

std = pd.concat([results.factors.smoothed_cov.loc[name, name]
                 for name in factor_names], axis=1)
std  = std[std.index>='2023-01-01']
crit = norm.ppf(1 - 0.05 / 2)
lower = mean - crit * std
upper = mean + crit * std


with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 3))
    mean.plot(ax=ax)

    for name in factor_names:
        ax.fill_between(mean.index, lower[name], upper[name], alpha=0.3)

    ax.set(title='Estimated factors: smoothed estimates and 95% confidence intervals')
    fig.tight_layout()

# explanatory power of factors
rsquared = results.get_coefficients_of_determination(method='individual')

top_ten = []
for factor_name in rsquared.columns[:3]:
    top_factor = (rsquared[factor_name].sort_values(ascending=False)
                                       .iloc[:10].round(2).reset_index())
    top_factor.columns = pd.MultiIndex.from_product([
        [f'Top ten variables explained by {factor_name}'],
        ['Variable', r'$R^2$']])
    top_ten.append(top_factor)

top_ten_table = pd.concat(top_ten, axis=1)
top_ten_table.iloc[:,0:2]
top_ten_table.iloc[:,2:4]
top_ten_table.iloc[:,4:6]


with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
    fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
    fig.tight_layout(rect=[0, 0, 1, 0.95]);


group_counts = groups[['description', 'group']]
group_counts = group_counts[group_counts['description'].isin(companies)]
group_counts = group_counts.groupby('group', sort=False).count()['description'].cumsum()

with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='joint', figsize=(14, 3));

    # Add in group labels
    ax = fig.axes[0]
    ax.set_ylim(0, 1.2)
    for i in np.arange(1, len(group_counts), 2):
        start = 0 if i == 0 else group_counts[i - 1]
        end = group_counts[i] + 1
        ax.fill_between(np.arange(start, end) - 0.6, 0, 1.2, color='k', alpha=0.1)
    for i in range(len(group_counts)):
        start = 0 if i == 0 else group_counts[i - 1]
        end = group_counts[i]
        n = end - start
        text = group_counts.index[i]
        if len(text) > n:
            text = text[:n - 3] + '...'

        ax.annotate(text, (start + n / 2, 1.1), ha='center')

    # Add label for GDP
    ax.set_xlim(-1.5, model.k_endog + 0.5)
    ax.annotate('GDP', (model.k_endog - 1.1, 1.05), ha='left', rotation=90)

    fig.tight_layout();

returns.tail()

# Create point forecasts, 3 steps ahead
point_forecasts = results.forecast(steps=3)

# Print the forecasts for the first 5 observed variables
print(point_forecasts.T.head())

# Create forecasts results objects, through the end of 20201
prediction_results = results.get_prediction(start='2005-01', end='2023-10')

variables = ['ASML',
             'NVIDIA',
             'Tesla']

# The `predicted_mean` attribute gives the same
# point forecasts that would have been returned from
# using the `predict` or `forecast` methods.
point_predictions = prediction_results.predicted_mean[variables]

# We can use the `conf_int` method to get confidence
# intervals; here, the 95% confidence interval
ci = prediction_results.conf_int(alpha=0.05)
lower = ci[[f'lower {name}' for name in variables]]
upper = ci[[f'upper {name}' for name in variables]]

# Plot the forecasts and confidence intervals
with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot the in-sample predictions
    point_predictions.loc[:'2023-07'].plot(ax=ax)

    # Plot the out-of-sample forecasts
    point_predictions.loc['2023-07':].plot(ax=ax, linestyle='--',
                                           color=['C0', 'C1', 'C2'],
                                           legend=False)

    # Confidence intervals
    for name in variables:
        ax.fill_between(ci.index,
                        lower[f'lower {name}'],
                        upper[f'upper {name}'], alpha=0.1)

    # Forecast period, set title
    ylim = ax.get_ylim()
    ax.vlines('2023-07', ylim[0], ylim[1], linewidth=1)
    ax.annotate(r' Forecast $\rightarrow$', ('2020-01', -1.7))
    ax.set(title=('Treasury securities / Federal Funds Rate spreads:'
                  ' in-sample predictions and out-of-sample forecasts, with 95% confidence intervals'), ylim=ylim)

    fig.tight_layout()



# Get the titles of the variables as they appear in the dataset
unemp_description = 'ASML'
gdp_description = 'NVIDIA'

# Compute the point forecasts
fcast_m = results.forecast('2023-12')[unemp_description]
fcast_q = results.forecast('2023-12')[gdp_description]

# For more convenient plotting, combine the observed data with the forecasts
plot_m = pd.concat([ret_m[unemp_description], fcast_m])
plot_q = pd.concat([ret_m[gdp_description], fcast_q])

with sns.color_palette('deep'):
    fig, axes = plt.subplots(2, figsize=(14, 4))

    # Plot real GDP growth, data and forecasts
    plot_q.plot(ax=axes[0])
    axes[0].set(title=gdp_description)
    axes[0].hlines(0, plot_q.index[0], plot_q.index[-1], linewidth=1)

    # Plot the change in the unemployment rate, data and forecasts
    plot_m.plot(ax=axes[1])
    axes[1].set(title=unemp_description)
    axes[1].hlines(0, plot_m.index[0], plot_m.index[-1], linewidth=1)

    # Show the forecast period in each graph
    for i in range(2):
        ylim = axes[i].get_ylim()
        axes[i].fill_between(plot_q.loc['2023-07':].index,
                             ylim[0], ylim[1], alpha=0.1, color='C0')
        axes[i].annotate(r' Forecast $\rightarrow$',
                         ('2023-07', ylim[0] + 0.1 * ylim[1]))
        axes[i].set_ylim(ylim)

    # Title
    fig.suptitle('Data and forecasts',
                 fontsize=14, fontweight=600)

    fig.tight_layout(rect=[0, 0, 1, 0.95])




# Reverse the transformations

# For real GDP, we take the level in 2000Q1 from the original data,
# and then apply the growth rates to compute the remaining levels
plot_q_orig = (plot_q / 100 + 1)**0.25
plot_q_orig.loc['2000Q1'] = dta['2020-02'].orig_q.loc['2000Q1', gdp_description]
plot_q_orig = plot_q_orig.cumprod()

# For the unemployment rate, we take the level in 2000-01 from
# the original data, and then we apply the changes to compute the
# remaining levels
plot_m_orig = plot_m.copy()
plot_m_orig.loc['2000-01'] = dta['2020-02'].orig_m.loc['2000-01', unemp_description]
plot_m_orig = plot_m_orig.cumsum().plot()