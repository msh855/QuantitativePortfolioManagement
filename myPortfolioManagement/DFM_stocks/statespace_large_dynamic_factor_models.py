import types
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os


def transform(column, transforms):
    transformation = transforms[column.name]
    # For quarterly data like GDP, we will compute
    # annualized percent changes
    mult = 4 if column.index.freqstr[0] == 'Q' else 1

    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference, multiplied by 100
    #      (i.e. approximate percent change)
    #      with optional multiplier for annualization
    elif transformation == 5:
        column = np.log(column).diff() * 100 * mult
    # 6 => Log second difference, multiplied by 100
    #      with optional multiplier for annualization
    elif transformation == 6:
        column = np.log(column).diff().diff() * 100 * mult
    # 7 => Exact percent change, multiplied by 100
    #      with optional annualization
    elif transformation == 7:
        column = ((column / column.shift(1)) ** mult - 1.0) * 100

    return column


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]

    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


def load_fredmd_data(vintage):
    base_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/'

    # - FRED-MD --------------------------------------------------------------
    # 1. Download data
    orig_m = (pd.read_csv(f'{base_url}/monthly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract transformation information
    transform_m = orig_m.iloc[0, 1:]
    orig_m = orig_m.iloc[1:]

    # 3. Extract the date as an index
    orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
    orig_m.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_m = orig_m.apply(transform, axis=0,
                         transforms=transform_m)

    # 5. Remove outliers (but not in 2020)
    dta_m.loc[:'2019-12'] = remove_outliers(dta_m.loc[:'2019-12'])

    # - FRED-QD --------------------------------------------------------------
    # 1. Download data
    orig_q = (pd.read_csv(f'{base_url}/quarterly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract factors and transformation information
    factors_q = orig_q.iloc[0, 1:]
    transform_q = orig_q.iloc[1, 1:]
    orig_q = orig_q.iloc[2:]

    # 3. Extract the date as an index
    orig_q.index = pd.PeriodIndex(orig_q.sasdate.tolist(), freq='Q')
    orig_q.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_q = orig_q.apply(transform, axis=0,
                         transforms=transform_q)

    # 5. Remove outliers (but not in 2020)
    dta_q.loc[:'2019Q4'] = remove_outliers(dta_q.loc[:'2019Q4'])

    # - Output datasets ------------------------------------------------------
    return types.SimpleNamespace(
        orig_m=orig_m, orig_q=orig_q,
        dta_m=dta_m, transform_m=transform_m,
        dta_q=dta_q, transform_q=transform_q, factors_q=factors_q)


# Load the vintages of data from FRED
dta = {date: load_fredmd_data(date)
       for date in ['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']}

# Print some information about the base dataset
n, k = dta['2020-02'].dta_m.shape
start = dta['2020-02'].dta_m.index[0]
end = dta['2020-02'].dta_m.index[-1]

print(f'For vintage 2020-02, there are {k} series and {n} observations,'
      f' over the period {start} to {end}.')

with sns.color_palette('deep'):
    fig, axes = plt.subplots(3, figsize=(14, 6))

    # Plot the raw data from the February 2020 vintage, for:
    #
    vintage = '2020-02'
    variable = 'RPI'
    start = '2000-01'
    end = '2020-01'

    # 1. Plot the original dataset, for 2000-01 through 2020-01
    dta[vintage].orig_m.loc[start:end, variable].plot(ax=axes[0])
    axes[0].set(title='Original data', xlim=('2000', '2020'), ylabel='Billons of $')

    # 2. Plot the transformed data, still including outliers
    # (we only stored the transformation with outliers removed, so
    # here we'll manually perform the transformation)
    transformed = transform(dta[vintage].orig_m[variable],
                            dta[vintage].transform_m)
    transformed.loc[start:end].plot(ax=axes[1])
    mean = transformed.mean()
    iqr = transformed.quantile([0.25, 0.75]).diff().iloc[1]
    axes[1].hlines([mean - 10 * iqr, mean + 10 * iqr],
                   transformed.index[0], transformed.index[-1],
                   linestyles='--', linewidth=1)
    axes[1].set(title='Transformed data, with bands showing outliers cutoffs',
                xlim=('2000', '2020'), ylim=(mean - 15 * iqr, mean + 15 * iqr),
                ylabel='Percent')
    axes[1].annotate('Outlier', xy=('2013-01', transformed.loc['2013-01']),
                     xytext=('2014-01', -5.3), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), )

    # 3. Plot the transformed data, with outliers removed (see missing value for 2013-01)
    dta[vintage].dta_m.loc[start:end, 'RPI'].plot(ax=axes[2])
    axes[2].set(title='Transformed data, with outliers removed',
                xlim=('2000', '2020'), ylabel='Percent')
    axes[2].annotate('Missing value in place of outlier', xy=('2013-01', -1),
                     xytext=('2014-01', -2), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    fig.suptitle('Real Personal Income (RPI)',
                 fontsize=12, fontweight=600)

    fig.tight_layout(rect=[0, 0.00, 1, 0.95]);

data_path = "/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/Data"
data_des_month = os.path.join(data_path, 'fredmd_definitions.csv')
data_des_quart = os.path.join(data_path, 'fredqd_definitions.csv')

# Definitions from the Appendix for FRED-MD variables
defn_m = pd.read_csv(data_des_month)
defn_m.index = defn_m.fred

# Definitions from the Appendix for FRED-QD variables
defn_q = pd.read_csv(data_des_quart)
defn_q.index = defn_q.fred

# Example of the information in these files:
defn_m.head()

# Replace the names of the columns in each monthly and quarterly dataset
map_m = defn_m['description'].to_dict()
map_q = defn_q['description'].to_dict()
for date, value in dta.items():
    value.orig_m.columns = value.orig_m.columns.map(map_m)
    value.dta_m.columns = value.dta_m.columns.map(map_m)
    value.orig_q.columns = value.orig_q.columns.map(map_q)
    value.dta_q.columns = value.dta_q.columns.map(map_q)

# Get the mapping of variable id to group name, for monthly variables
groups = defn_m[['description', 'group']].copy()

# Re-order the variables according to the definition CSV file
# (which is ordered by group)
columns = [name for name in defn_m['description']
           if name in dta['2020-02'].dta_m.columns]
for date in dta.keys():
    dta[date].dta_m = dta[date].dta_m.reindex(columns, axis=1)

# Add real GDP (our quarterly variable) into the "Output and Income" group
gdp_description = defn_q.loc['GDPC1', 'description']
groups.loc['GDPC1'] = {'description': gdp_description, 'group': 'Output and Income'}

# Display the number of variables in each group
(groups.groupby('group', sort=False)
 .count()
 .rename({'description': '# series in group'}, axis=1))

# Construct the variable => list of factors dictionary
factors = {row['description']: ['Global', row['group']]
           for ix, row in groups.iterrows()}

# Check that we have the desired output for "Real personal income"
print(factors['Real Personal Income'])

# config
factor_multiplicities = {'Global': 2}
factor_orders = {'Global': 4}

# Get the baseline monthly and quarterly datasets
start = '2000'
endog_m = dta['2020-02'].dta_m.loc[start:, :]
gdp_description = defn_q.loc['GDPC1', 'description']
endog_q = dta['2020-02'].dta_q.loc[start:, [gdp_description]]

# Construct the dynamic factor model
model = sm.tsa.DynamicFactorMQ(
    endog_m, endog_quarterly=endog_q,
    factors=factors, factor_orders=factor_orders,
    factor_multiplicities=factor_multiplicities)

model.summary()
results = model.fit(disp=10)
results.summary()

# Get estimates of the global and labor market factors,
# conditional on the full dataset ("smoothed")
factor_names = ['Global.1', 'Global.2', 'Labor Market']
mean = results.factors.smoothed[factor_names]


# Compute 95% confidence intervals
std = pd.concat([results.factors.smoothed_cov.loc[name, name]
                 for name in factor_names], axis=1)
crit = norm.ppf(1 - 0.05 / 2)
lower = mean - crit * std
upper = mean + crit * std

with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 3))
    mean.plot(ax=ax)

    for name in factor_names:
        ax.fill_between(mean.index, lower[name], upper[name], alpha=0.3)

    ax.set(title='Estimated factors: smoothed estimates and 95% confidence intervals')
    fig.tight_layout();

# #### Explanatory power of the factors
rsquared = results.get_coefficients_of_determination(method='individual')

top_ten = []
for factor_name in rsquared.columns[:2]:
    top_factor = (rsquared[factor_name].sort_values(ascending=False)
                  .iloc[:10].round(2).reset_index())
    top_factor.columns = pd.MultiIndex.from_product([
        [f'Top ten variables explained by {factor_name}'],
        ['Variable', r'$R^2$']])
    top_ten.append(top_factor)
pd.concat(top_ten, axis=1)

# **Plotting $R^2$**
with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
    fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
    fig.tight_layout(rect=[0, 0, 1, 0.95]);

group_counts = defn_m[['description', 'group']]
group_counts = group_counts[group_counts['description'].isin(dta['2020-02'].dta_m.columns)]
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

# ### Forecasting

# Create point forecasts, 3 steps ahead
point_forecasts = results.forecast(steps=3)

# Print the forecasts for the first 5 observed variables
print(point_forecasts.T.head())

# Create forecasts results objects, through the end of 20201
prediction_results = results.get_prediction(start='2000', end='2022')

variables = ['1-Year Treasury C Minus FEDFUNDS',
             '5-Year Treasury C Minus FEDFUNDS',
             '10-Year Treasury C Minus FEDFUNDS']

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
    point_predictions.loc[:'2020-01'].plot(ax=ax)

    # Plot the out-of-sample forecasts
    point_predictions.loc['2020-01':].plot(ax=ax, linestyle='--',
                                           color=['C0', 'C1', 'C2'],
                                           legend=False)

    # Confidence intervals
    for name in variables:
        ax.fill_between(ci.index,
                        lower[f'lower {name}'],
                        upper[f'upper {name}'], alpha=0.1)

    # Forecast period, set title
    ylim = ax.get_ylim()
    ax.vlines('2020-01', ylim[0], ylim[1], linewidth=1)
    ax.annotate(r' Forecast $\rightarrow$', ('2020-01', -1.7))
    ax.set(title=('Treasury securities / Federal Funds Rate spreads:'
                  ' in-sample predictions and out-of-sample forecasts, with 95% confidence intervals'), ylim=ylim)

    fig.tight_layout()

# #### Forecasting example


# Get the titles of the variables as they appear in the dataset
unemp_description = 'Civilian Unemployment Rate'
gdp_description = 'Real Gross Domestic Product, 3 Decimal (Billions of Chained 2012 Dollars)'

# Compute the point forecasts
fcast_m = results.forecast('2021-12')[unemp_description]
fcast_q = results.forecast('2021-12')[gdp_description].resample('Q').last()

# In[25]:


# For more convenient plotting, combine the observed data with the forecasts
plot_m = pd.concat([dta['2020-02'].dta_m.loc['2000':, unemp_description], fcast_m])
plot_q = pd.concat([dta['2020-02'].dta_q.loc['2000':, gdp_description], fcast_q])

with sns.color_palette('deep'):
    fig, axes = plt.subplots(2, figsize=(14, 4))

    # Plot real GDP growth, data and forecasts
    plot_q.plot(ax=axes[0])
    axes[0].set(title='Real Gross Domestic Product (transformed: annualized growth rate)')
    axes[0].hlines(0, plot_q.index[0], plot_q.index[-1], linewidth=1)

    # Plot the change in the unemployment rate, data and forecasts
    plot_m.plot(ax=axes[1])
    axes[1].set(title='Civilian Unemployment Rate (transformed: change)')
    axes[1].hlines(0, plot_m.index[0], plot_m.index[-1], linewidth=1)

    # Show the forecast period in each graph
    for i in range(2):
        ylim = axes[i].get_ylim()
        axes[i].fill_between(plot_q.loc['2020-02':].index,
                             ylim[0], ylim[1], alpha=0.1, color='C0')
        axes[i].annotate(r' Forecast $\rightarrow$',
                         ('2020-03', ylim[0] + 0.1 * ylim[1]))
        axes[i].set_ylim(ylim)

    # Title
    fig.suptitle('Data and forecasts (February 2020 vintage), transformed scale',
                 fontsize=14, fontweight=600)

    fig.tight_layout(rect=[0, 0, 1, 0.95]);

# Reverse the transformations

# For real GDP, we take the level in 2000Q1 from the original data,
# and then apply the growth rates to compute the remaining levels
plot_q_orig = (plot_q / 100 + 1) ** 0.25
plot_q_orig.loc['2000Q1'] = dta['2020-02'].orig_q.loc['2000Q1', gdp_description]
plot_q_orig = plot_q_orig.cumprod()

# For the unemployment rate, we take the level in 2000-01 from
# the original data, and then we apply the changes to compute the
# remaining levels
plot_m_orig = plot_m.copy()
plot_m_orig.loc['2000-01'] = dta['2020-02'].orig_m.loc['2000-01', unemp_description]
plot_m_orig = plot_m_orig.cumsum()

# In[27]:
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

# ### Nowcasting GDP, real-time forecast updates, and the news

# The original point forecasts are monthly
point_forecasts_m = results.forecast('June 2020')[gdp_description]

# Resample to quarterly frequency by taking the value in the last
# month of each quarter
point_forecasts_q = point_forecasts_m.resample('Q').last()

print('Baseline (February 2020) forecast for real GDP growth'
      f' in 2020Q2: {point_forecasts_q["2020Q2"]:.2f}%')

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
