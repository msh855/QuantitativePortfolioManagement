from myPortfolioManagement.myData import load_fredmd_data, get_stock_info
import pandas as pd
import os

# prepare Portfolio dataset
# ======================================================================================================================
file_path_tickers = '/myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
df_tickers = pd.read_excel(file_path_tickers)
df_tickers.dropna(inplace=True)
index_name = 'YahooTicker'
# df_weights = df_tickers[['YahooTicker', 'Company', 'adj_weight_GBP']]
tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])

file_path = '/myPortfolioManagement/MeteoraGlobalEquityFund/Data/df_prices.csv'
df_prices = pd.read_csv(file_path)
df_prices = df_prices.set_index('date')

# Group stocks for DFM
# ======================================================================================================================
# df_sectors = get_stock_info(yahoo_tickers=tickers)
# df_portfolio_descr = df_sectors[['Sector', 'Industry']].join(df_weights.set_index(index_name))
# df_portfolio_descr.to_clipboard()
df_sectors = df_tickers[[index_name, 'Sector']]
df_sectors.groupby('Sector').count().sort_values(by=[index_name], ascending=False)

groups = df_tickers[[index_name, 'Sector', 'Company']]
groups = groups.set_index(index_name)
groups.columns = ['group', 'description']

# Construct the variable => list of factors dictionary
factors = {row['description']: ['Global', row['group']]
           for ix, row in groups.iterrows()}

# load Macro Series
# ======================================================================================================================
files_to_load = ['2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
dta = {date: load_fredmd_data(date) for date in files_to_load}


def get_fredmd_data_description(path_for_data_descrp='/Users/safishajjouz/GitHub/tsa-notebooks/data'):
    # Definitions from the Appendix for FRED-MD variables
    file_monthly = os.path.join(path_for_data_descrp, 'fredmd_definitions.csv')
    defn_m = pd.read_csv(file_monthly)
    defn_m.index = defn_m.fred

    # Definitions from the Appendix for FRED-QD variables
    file_quarterly = os.path.join(path_for_data_descrp, 'fredqd_definitions.csv')
    defn_q = pd.read_csv(file_quarterly)
    defn_q.index = defn_q.fred
    return defn_m, defn_q


defn_m, defn_q = get_fredmd_data_description()

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
columns = [name for name in defn_m['description'] if name in dta[files_to_load[0]].dta_m.columns]

dta[files_to_load[0]].dta_m.columns

for date in dta.keys():
    dta[date].dta_m = dta[date].dta_m.reindex(columns, axis=1)

from collections import Counter

mylist = [20, 30, 25, 20]
[k for k, v in Counter(list(map_m.keys())).items() if v > 1]

# Add real GDP (our quarterly variable) into the "Output and Income" group
gdp_description = defn_q.loc['GDPC1', 'description']
groups.loc['GDPC1'] = {'description': gdp_description, 'group': 'Output and Income'}

# Display the number of variables in each group
(groups.groupby('group', sort=False)
 .count()
 .rename({'description': '# series in group'}, axis=1))
