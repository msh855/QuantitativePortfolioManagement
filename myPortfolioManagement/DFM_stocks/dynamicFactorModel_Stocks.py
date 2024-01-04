import pandas as pd
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myDataCleaning import remove_outliers
from myPortfolioManagement.DFM_stocks.prepare_macro_data import get_macro_data
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
import numpy as np
import os

warnings.filterwarnings("ignore")
plt.style.use('seaborn')

# load macro data
# ======================================================================================================================
monthly_macro_data, quart_macro_data, factors_macro_data = get_macro_data()

# import tickers
# ======================================================================================================================
wd = os.getcwd()
data_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data'
file_to_load = 'stock_screening.xlsx'
file_path = os.path.join(wd, data_path, file_to_load)

# load tickers
df_tickers = pd.read_excel(file_path)
df_tickers.dropna(inplace=True)
index_name = 'YahooTicker'
tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])

# start_date = "2005-01-01"
# prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date)

file_path_prices = os.path.join(wd, data_path, 'df_prices.csv')
df_prices = pd.read_csv(file_path_prices)
df_prices = df_prices.set_index('date')

# Group stocks for DFM
# ======================================================================================================================
df_sectors = pd.read_excel('/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx')
groups = df_sectors.copy()
groups = groups.set_index('YahooTicker')
#df_sectors.groupby('Sector').count()[['Industry']].sort_values(by=['Industry'], ascending=False)
groups = groups.drop(['Industry', 'adj_weight_GBP'], axis=1)
groups.columns = ['group', 'description']

# Construct the variable => list of factors dictionary
factors = {row['description']: ['Global', row['group']]
           for ix, row in groups.iterrows()}

factors.update(factors_macro_data)

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

monthly_macro_data.index.name = 'date'
data_monthly = monthly_macro_data.join(ret_m)
quart_macro_data.index.name = 'date'

model = sm.tsa.DynamicFactorMQ(data_monthly, factors=factors,
                               factor_orders=factor_orders,
                               factor_multiplicities=factor_multiplicities)

model.summary()
results = model.fit(disp=10)
results.summary()

results.factors.filtered  # use information up to time t
results.factors.smoothed  # use full dataset

# Get estimates of the global and labor market factors,
# conditional on the full dataset ("smoothed")
slice_date = '2019-01-01'
factor_names = ['Global.1', 'Global.2', 'Technology']
mean = results.factors.smoothed[factor_names]
mean = mean[mean.index >= slice_date]

std = pd.concat([results.factors.smoothed_cov.loc[name, name]
                 for name in factor_names], axis=1)
std = std[std.index >= slice_date]
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
top_ten_table.iloc[:, 0:2]
top_ten_table.iloc[:, 2:4]
top_ten_table.iloc[:, 4:6]

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
plot_q_orig = (plot_q / 100 + 1) ** 0.25
# plot_q_orig.loc['2000Q1'] = dta['2020-02'].orig_q.loc['2000Q1', gdp_description]
plot_q_orig = plot_q_orig.cumprod()

price = df_prices['Adj Close'][df_prices['description'] == unemp_description]
price.index = pd.to_datetime(price.index)
price_m = price.resample('M').mean()
price_m.plot()
# For the unemployment rate, we take the level in 2000-01 from
# the original data, and then we apply the changes to compute the
# remaining levels
plot_m_orig = plot_m.copy()
# plot_m_orig.loc['2000-01'] = dta['2020-02'].orig_m.loc['2000-01', unemp_description]
plot_m_orig = plot_m_orig.cumsum()

with sns.color_palette('deep'):
    fig, axes = plt.subplots(2, figsize=(14, 4))

    # Plot real GDP, data and forecasts
    plot_q_orig.plot(ax=axes[0])
    axes[0].set(title=('Real Gross Domestic Product'
                       ' (original scale: Billions of Chained 2012 Dollars)'))

    # Plot the unemployment rate, data and forecasts
    plot_m_orig.plot(ax=axes[1])
    axes[1].set(title='Civilian Unemployment Rate (original scale: Percent)')

    # Show the forecast period in each graph
    for i in range(2):
        ylim = axes[i].get_ylim()
        axes[i].fill_between(plot_q.loc['2020-02':].index,
                             ylim[0], ylim[1], alpha=0.1, color='C0')
        axes[i].annotate(r' Forecast $\rightarrow$',
                         ('2020-03', ylim[0] + 0.5 * (ylim[1] - ylim[0])))
        axes[i].set_ylim(ylim)

    # Title
    fig.suptitle('Data and forecasts (February 2020 vintage), original scale',
                 fontsize=14, fontweight=600)

    fig.tight_layout(rect=[0, 0, 1, 0.95]);

# impact of news

# The original point forecasts are monthly
point_forecasts_m = results.forecast()[gdp_description]

# Resample to quarterly frequency by taking the value in the last
# month of each quarter
point_forecasts_q = point_forecasts_m.resample('Q').last()
value_last = point_forecasts_q[point_forecasts_q.index[0]]

print('Baseline (February 2020) forecast for real GDP growth'
      f' in 2020Q2: {point_forecasts_q[point_forecasts_q[value_last]]:.2f}%')

###


# **Updated GDP forecast: March 2020 vintage**
vintage_results = {'2020-02': results}

# Get the updated monthly and quarterly datasets
start = '2000'
updated_endog_m = dta['2020-03'].dta_m.loc[start:, :]
gdp_description = defn_q.loc['GDPC1', 'description']
updated_endog_q = dta['2020-03'].dta_q.loc[start:, [gdp_description]]

# Get the results for March 2020 using `apply`
vintage_results['2020-03'] = results.apply(
    updated_endog_m, endog_quarterly=updated_endog_q)

# Print the updated forecast for real GDP growth in 2020Q2
updated_forecasts_q = (
    vintage_results['2020-03'].forecast('June 2020')[gdp_description]
    .resample('Q').last())

print('March 2020 forecast for real GDP growth in 2020Q2:'
      f' {updated_forecasts_q["2020Q2"]:.2f}%')

# Apply our results to the remaining vintages
for vintage in ['2020-04', '2020-05', '2020-06']:
    # Get updated data for the vintage
    updated_endog_m = dta[vintage].dta_m.loc[start:, :]
    updated_endog_q = dta[vintage].dta_q.loc[start:, [gdp_description]]

    # Get updated results for for the vintage
    vintage_results[vintage] = results.apply(
        updated_endog_m, endog_quarterly=updated_endog_q)

# Compute forecasts for each vintage
forecasts = {vintage: res.forecast('June 2020')[gdp_description]
.resample('Q').last().loc['2020Q2']
             for vintage, res in vintage_results.items()}
# Convert to a Pandas series with a date index
forecasts = pd.Series(list(forecasts.values()),
                      index=pd.PeriodIndex(forecasts.keys(), freq='M'))

# Print our forecast for 2020Q2 real GDP growth across all vintages
for vintage, value in forecasts.items():
    print(f'{vintage} forecast for real GDP growth in 2020Q2:'
          f' {value:.2f}%')

# Compute the news and impacts on the real GDP growth forecast
# for 2020Q2, between the April and March vintages
news = vintage_results['2020-04'].news(
    vintage_results['2020-03'], impact_date='2020-06',
    impacted_variable=gdp_description,
    comparison_type='previous')

# We can re-arrange the `details_by_impact` table to show the new
# observations with the top ten impacts (in absolute value)
details = news.details_by_impact
details.index = details.index.droplevel(['impact date', 'impacted variable'])
details['absolute impact'] = np.abs(details['impact'])
details = (details.sort_values('absolute impact', ascending=False)
           .drop('absolute impact', axis=1))
details.iloc[:10].round(2)

news_results = {}
vintages = ['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
impact_date = '2020-06'

for i in range(1, len(vintages)):
    vintage = vintages[i]
    prev_vintage = vintages[i - 1]

    # Notice that to get the "incremental" news, we are computing
    # the news relative to the previous vintage and not to the baseline
    # (February 2020) vintage
    news_results[vintage] = vintage_results[vintage].news(
        vintage_results[prev_vintage],
        impact_date=impact_date,
        impacted_variable=gdp_description,
        comparison_type='previous')

group_impacts = {'2020-02': None}

for vintage, news in news_results.items():
    # Start from the details by impact table
    details_by_impact = (
        news.details_by_impact.reset_index()
        .drop(['impact date', 'impacted variable'], axis=1))

    # Merge with the groups dataset, so that we can identify
    # which group each individual impact belongs to
    impacts = (pd.merge(details_by_impact, groups, how='left',
                        left_on='updated variable', right_on='description')
               .drop('description', axis=1)
               .set_index(['update date', 'updated variable']))

    # Compute impacts by group, summing across the individual impacts
    group_impacts[vintage] = impacts.groupby('group').sum()['impact']

# Add in a row of zeros for the baseline forecast
group_impacts['2020-02'] = group_impacts['2020-03'] * np.nan

# Convert into a Pandas DataFrame, and fill in missing entries
# with zeros (missing entries happen when there were no updates
# for a given group in a given vintage)
group_impacts = (
    pd.concat(group_impacts, axis=1)
    .fillna(0)
    .reindex(group_counts.index).T)
group_impacts.index = forecasts.index

# Print the table of impacts from data in each group,
# along with a row with the "Total" impact
(group_impacts.T
 .append(group_impacts.sum(axis=1).rename('Total impact on 2020Q2 forecast'))
 .round(2).iloc[:, 1:])

with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Stacked bar plot showing the impacts by group
    group_impacts.plot(kind='bar', stacked=True, width=0.3, zorder=2, ax=ax);

    # Line plot showing the forecast for real GDP growth in 2020Q2 for each vintage
    x = np.arange(len(forecasts))
    ax.plot(x, forecasts, marker='o', color='k', markersize=7, linewidth=2)
    ax.hlines(0, -1, len(group_impacts) + 1, linewidth=1)

    # x-ticks
    labels = group_impacts.index.strftime('%b')
    ax.xaxis.set_ticklabels(labels)
    ax.xaxis.set_tick_params(size=0)
    ax.xaxis.set_tick_params(labelrotation='auto', labelsize=13)

    # y-ticks
    ax.yaxis.set_tick_params(direction='in', size=0, labelsize=13)
    ax.yaxis.grid(zorder=0)

    # title, remove spines
    ax.set_title('Evolution of real GDP growth nowcast: 2020Q2', fontsize=16, fontweight=600, loc='left')
    [ax.spines[spine].set_visible(False)
     for spine in ['top', 'left', 'bottom', 'right']]

    # base forecast vs updates
    ylim = ax.get_ylim()
    ax.vlines(0.5, ylim[0], ylim[1] + 5, linestyles='--')
    ax.annotate('Base forecast', (-0.2, 22), fontsize=14)
    ax.annotate(r'Updated forecasts and impacts from the "news" $\rightarrow$', (0.65, 22), fontsize=14)

    # legend
    ax.legend(loc='upper center', ncol=4, fontsize=13, bbox_to_anchor=(0.5, -0.1), frameon=False)

    fig.tight_layout()
