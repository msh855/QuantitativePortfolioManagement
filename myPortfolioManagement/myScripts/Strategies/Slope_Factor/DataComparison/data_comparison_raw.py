import pandas as pd
from myScripts.functions.download_and_prepare_data import get_policy_rates, get_bbq_data, get_historical_spots, \
    get_bbq_ccy, get_bond_yield

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})

currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD']
start_date = '1980-01-01'
spots_list = []
df_bonds_bbq_list = []
df_rates_list = []

# bbq data: policy rates
bbq_tickers_policy_rates = ['FDTR Index', 'EURR002W Index', 'SZBRLOMB Index', 'BOJDPBAL Index', 'UKBRBASE Index',
                            'FMSTTGOV Index', 'NZOCRS Index', 'RBATCTR Index', 'SWBRDEP Index', 'NOBRDEPA Index']

df_bbq_policy_rates = get_bbq_data(bbq_tickers_policy_rates, start_date=start_date)

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

df_bbq_policy_rates = df_bbq_policy_rates.rename(columns=mapping)
df_bbq_policy_rates.fillna(method="ffill", inplace=True)

# Yields
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

for ccy in currencies + ['US']:
    if ccy != 'US':
        country = ccy[0:3]
    else:
        country = 'US'

    if ccy != 'US':
        # spot comparison
        spot_bbq = get_bbq_ccy(ccy=ccy, start_date=start_date).dropna()
        spot_bbq = spot_bbq[spot_bbq.index >= start_date]
        spot_bbq.columns = ['spot' + '_BBQ']

        spot_mb = get_historical_spots(ccy=ccy).dropna()
        spot_mb = spot_mb[spot_mb.index >= start_date]
        spot_mb.columns = ['spot' + '_macrobond']

        spots = spot_bbq.reset_index().merge(spot_mb.reset_index())
        spots['diff'] = spots['spot' + '_BBQ'] - spots['spot' + '_macrobond']
        spots['diff_pct'] = (spots['spot' + '_BBQ'] - spots['spot' + '_macrobond']) / spots['spot' + '_macrobond']
        spots['ccy'] = country
        spots_list.append(spots)

    # bond comparison
    df_bonds = get_bond_yield(country=country)[['2Y']]
    df_bonds.columns = ['2Y' + '_macrobond']
    df_bonds_bbq = df_bbq_2y_bonds[[country]]
    df_bonds_bbq = df_bonds_bbq.reset_index().merge(df_bonds.reset_index())
    df_bonds_bbq['diff'] = df_bonds_bbq[country] - df_bonds_bbq['2Y' + '_macrobond']
    df_bonds_bbq['diff_pct'] = (df_bonds_bbq[country] - df_bonds_bbq['2Y' + '_macrobond']) / df_bonds_bbq[
        '2Y' + '_macrobond']
    df_bonds_bbq['ccy'] = country
    df_bonds_bbq_list.append(df_bonds_bbq)

    # rate comparison
    df_rates_mb = df_policy_rates = get_policy_rates(country=[country])
    df_rates_mb.columns = ['rates' + '_macrobond']

    df_rates = df_bbq_policy_rates[[country]]
    df_rates = df_rates.reset_index().merge(df_rates_mb.reset_index())
    df_rates['diff'] = df_rates[country] - df_rates['rates' + '_macrobond']
    df_rates['diff_pct'] = (df_rates[country] - df_rates['rates' + '_macrobond']) / df_rates['rates' + '_macrobond']
    df_rates['ccy'] = country
    df_rates_list.append(df_rates)


def plot_box_plot(spots_list, column_to_plot='diff'):
    spots_all = pd.concat(spots_list)
    spots_all = spots_all.set_index('Date')
    spots_all.boxplot(column=[column_to_plot], by='ccy')


fig_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Yield_Strategy_presentation_drivers_of_diff'

plot_box_plot(spots_list, column_to_plot='diff_pct')
plt.title('Diff in Spots (vs USD)')

plot_box_plot(df_bonds_bbq_list, column_to_plot='diff_pct')
plt.title('Diff in Bond Rates')

plot_box_plot(df_rates_list, column_to_plot='diff_pct')
plt.title('Diff in Policy rates')
