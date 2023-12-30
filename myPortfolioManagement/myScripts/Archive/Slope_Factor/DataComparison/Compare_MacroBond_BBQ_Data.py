import pandas as pd
from myScripts.functions.download_and_prepare_data import get_bbq_data, get_historical_spots, get_policy_rates, \
    get_bond_yield
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# main parameters
currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']

spots_macrobond_list = []
policy_rates_macrobond_list = []
df_bonds_list = []
start_date = '1980-01-01'

for ccy in currencies + ['US']:

    if ccy != 'US':
        # Macro Bond
        spot_mb = get_historical_spots(ccy=ccy).dropna()
        spots_macrobond_list.append(spot_mb)

    # policy rate
    # =============
    if ccy != 'US':
        country = ccy[0:3]
    else:
        country = ccy

    df_policy_rates = get_policy_rates(country=[country])
    policy_rates_macrobond_list.append(df_policy_rates)

    # bonds
    df_bonds = get_bond_yield(country=country)[['2Y']]
    df_bonds.columns = [x + '_' + country for x in df_bonds.columns]
    df_bonds_list.append(df_bonds)

# merge all dfs
# ================
spots_macrobond = pd.concat(spots_macrobond_list, axis=1)
policy_rates_macrobond = pd.concat(policy_rates_macrobond_list, axis=1)
df_bonds = pd.concat(df_bonds_list, axis=1)

policy_rates_macrobond_na_filled = policy_rates_macrobond.fillna(method='ffill')
df_bonds_na_filled = df_bonds.fillna(method='ffill')

# bbq data
bbq_tickers_policy_rates = ['FDTR Index', 'EURR002W Index', 'SZBRLOMB Index', 'BOJDPBAL Index', 'UKBRBASE Index',
                            'FMSTTGOV Index', 'NZOCRS Index', 'RBATCTR Index', 'SWBRDEP Index', 'NOBRDEPA Index']

df_bbq = get_bbq_data(bbq_tickers_policy_rates, start_date=start_date)

mapping = {'FDTR Index': 'US',
           'SZBRLOMB Index': 'CHF',
           'EURR002W Index': 'EUR',
           'BOJDPBAL Index': 'JPY',
           'UKBRBASE Index': 'GBP',
           'FMSTTGOV Index': 'CAD',
           'NZOCRS Index': 'NZD',
           'RBATCTR Index': 'AUD',
           'SWBRDEP Index': 'SEK',
           'NOBRDEPA Index': 'NOK'}

df_bbq = df_bbq.rename(columns=mapping)
df_bbq.fillna(method="ffill", inplace=True)

# 2 year Bonds bbq
# ================
bbq_tickers_bonds = ['USGG2YR Index', 'GTDEM2Y Govt', 'GTCHF2Y Govt', 'GTJPY2Y Govt', 'GTGBP2Y Govt', 'GCAN2YR Index',
                     'GNZGB2 Index',
                     'GTAUD2Y Govt', 'GTSEK2Y Govt', 'GTNOK2Y Govt']

df_bbq_2y_bonds = get_bbq_data(bbq_tickers_bonds, start_date=start_date)

mapping_bonds = {'USGG2YR Index': 'US',
                 'GTCHF2Y Govt': 'CHF',
                 'GTDEM2Y Govt': 'EUR',
                 'GTJPY2Y Govt': 'JPY',
                 'GTGBP2Y Govt': 'GBP',
                 'GCAN2YR Index': 'CAD',
                 'GNZGB2 Index': 'NZD',
                 'GTAUD2Y Govt': 'AUD',
                 'GTSEK2Y Govt': 'SEK',
                 'GTNOK2Y Govt': 'NOK'}

df_bbq_2y_bonds = df_bbq_2y_bonds.rename(columns=mapping_bonds)

df_bbq_2y_bonds.columns = ['2Y_' + x for x in df_bbq_2y_bonds.columns]

# define subplot layout
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(20, 10))
fig.suptitle('Policy Rates', fontsize=16)

temp_list_policy_rates = []

i = 0
for country in policy_rates_macrobond.columns:
    i = i + 1
    temp = df_bbq[[country]].copy()
    temp.columns = [x + '_bbq' for x in temp.columns]
    temp = temp.reset_index().merge(policy_rates_macrobond[[country]].reset_index())
    temp = temp.set_index('Date')
    temp['Macro_Bond_Diff'] = temp[country] - temp[country + '_bbq']

    temp[['Macro_Bond_Diff']].plot(ax=axes[i - 1], title=country)
    temp_list_policy_rates.append(temp)


pd.concat(temp_list_policy_rates)


# 2 year bonds bbq
# =================

# define subplot layout
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(20, 10))
fig.suptitle('Bond Rates na filled', fontsize=16)
i = 0
temp_list_bonds = []

for country in df_bonds.columns:
    i = i + 1
    temp = df_bbq_2y_bonds[[country]].copy()
    temp.columns = [x + '_bbq' for x in temp.columns]
    temp = temp.reset_index().merge(df_bonds_na_filled[[country]].reset_index())
    temp = temp.set_index('Date')
    temp['Macro_Bond_Diff'] = temp[country] - temp[country + '_bbq']
    temp[['Macro_Bond_Diff']].plot(ax=axes[i - 1], title=country)
    temp_list_bonds.append(temp)

### individual call
start_date = '1980-01-01'
end_date = '2023-04-17'

start_date = start_date.replace('-', '')
end_date = end_date.replace('-', '')

import pdblp as blp

con = blp.BCon(timeout=1000000)
con.start()
data_all = con.bdh(['RBATCTR Index', 'FDTR Index'], 'PX_LAST', start_date, end_date)
data_all.columns = data_all.columns.droplevel(1)
con.stop()

data_all.dropna().plot()

# get bbq data
# bbq_tickers = ['IOE1 A:00_0_R comdty', 'GC1 A:00_0_R comdty', 'VIX index', 'CL2 A:00_0_R comdty', 'HG1 A:00_0_R comdty']
# df_bbq = get_bbq_data(bbq_tickers)
# df_bbq.columns = ['IronOre_adj_fut', 'Gold_adj_fut', 'Vix', 'WTI_adj_fut', 'Copper_adj_fut']
