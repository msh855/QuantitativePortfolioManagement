import pandas as pd
import matplotlib.pyplot as plt
from myPortfolioManagement.myData import get_fred_data
import matplotlib.ticker as mtick

# https://www.cbo.gov/publication/58457
# ref: https://fred.stlouisfed.org/release/tables?rid=20&eid=1194154&od=#
fred_symbols = ['WALCL',  # Total Assets
                'WSHOTSL',  # US_Securities
                'WLCFLL',  # Loans
                'WSHOFADSL',  # Federal Agency Debt
                'SWPT',  # Credit Swaps
                'WSHOMCB']  # Mortage-Backed
#
# 'WSHOBL',  # Bills
# 'WSHONBNL',  # Notes and Bonds
# 'WSHONBIIL',  # Linkers

fed_assets = get_fred_data(fred_sumbol=fred_symbols, freq='a')
fed_assets = fed_assets.fillna(0)

col_names = ['Total', 'Total_US_Treasuries', 'Loans', 'Federal_Agency_Debt', 'Swaps', 'Mortage Backed']
fed_assets.columns = col_names

fed_assets['US_Treasuries (% of Total)'] = fed_assets['Total_US_Treasuries'] / fed_assets['Total']

fed_assets['MBS (% of Total)'] = fed_assets['Mortage Backed'] / fed_assets['Total']

fed_assets['Loans (% of Total)'] = fed_assets['Loans'] / fed_assets['Total']

fed_assets['Swaps (% of Total)'] = fed_assets['Swaps'] / fed_assets['Total']

fed_assets['Other (% of Total)'] = 1 - fed_assets['MBS (% of Total)'] - fed_assets['US_Treasuries (% of Total)'] - \
                                   fed_assets['Loans (% of Total)'] - fed_assets['Swaps (% of Total)']
import plotly.express as px

vars = ['US_Treasuries (% of Total)', 'MBS (% of Total)', 'Loans (% of Total)', 'Swaps (% of Total)', 'Other (% of Total)']
df = fed_assets.reset_index()

fig = px.bar(df, x="Date", y=vars,
             title="Wide-Form Input")
fig.update_yaxes(ticklabelposition="inside top", title=None)
fig.show(renderer="browser")

# convert to trillion dollars
fed_assets = fed_assets / 1000000
fed_assets = round(fed_assets, 2)
fed_assets = fed_assets.tail(-1)  # drop first row

fig, ax = plt.subplots(figsize=(15, 6))
fed_assets.drop(['Total_Bonds', 'Total'], axis=1).plot(kind='area', ax=ax, stacked=False,
                                                       ylabel='Trillion of USD dollars')
plt.show()

proportions = [fed_assets[x] / fed_assets['Total'] for x in col_names]
df_prop = pd.concat(proportions, axis=1)
df_prop.columns = col_names
df_prop = df_prop.fillna(0)
df_prop = round(df_prop, 2)

df_prop['Other'] = 1 - df_prop.drop('Total_Bonds', axis=1).sum(axis=1)

df_prop = df_prop * 100

fig, ax = plt.subplots(figsize=(15, 6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
df_prop.drop(['Total_Bonds', 'Other'], axis=1).plot(kind='area', ax=ax, stacked=False, ylabel='% of Total Assets',
                                                    title="Fed's Portfolio Composition of Asset")
plt.show()

# df_prop_temp = df_prop.drop('Total_Bonds', axis=1)
# df_long = pd.melt(df_prop_temp.reset_index(), id_vars='Date')
# df_long = df_long.set_index('Date')
