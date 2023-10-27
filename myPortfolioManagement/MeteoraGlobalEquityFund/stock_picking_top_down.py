import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import os
from openbb_terminal.sdk import openbb
from scipy.stats import zscore
from myPortfolioManagement.myData import  get_stock_prices_from_openBB, get_stock_prices
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myPerformanceMetrics import alpha_beta_table
from sklearn import preprocessing as pre
from myPortfolioManagement.myPerformanceMetrics import cagr
import numpy as np

from openbb_terminal.sdk import TerminalStyle
import quantstats as qs
from yahoofinancials import YahooFinancials

qs.extend_pandas()
theme = TerminalStyle("light", "light", "light")


# import tickers
# ==============
working_directory = os.getcwd()
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
path = os.path.join(working_directory, file_path)
mil_path = 'S:\Investment Solutions Group\Quant_research\Moustafa\Github\/ ' \
           'QuantitativePortfolioManagement\ /' \
           'myPortfolioManagement\MeteoraGlobalEquityFund\Data\stock_screening.xlsx'

df_tickers = pd.read_excel(path)
index_name = df_tickers.columns[0]

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "1995-01-01"


# df_prices_list = []
# for ticker, company in zip(tickers, companies):
#     data = openbb.stocks.load(ticker, start_date=start_date)
#     data[index_name] = ticker
#     data['Company'] = company
#     df_prices_list.append(data)
#
# df_prices = pd.concat(df_prices_list)
#
# prices = df_prices[['Adj Close', index_name]]
# prices = prices.pivot(columns=index_name, values='Adj Close')

prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date)

prices = get_stock_prices(yahoo_tickers=tickers, start_date=start_date)

# decompose to cyclical and trend
# ===============================
prices_temp = prices
df_decompose_list = []
for tick in tickers:
    temp_decomp = openbb.qa.decompose(data=prices_temp[tick].dropna(), multiplicative=True)
    df_decompose = pd.concat([temp_decomp[1], temp_decomp[2], prices_temp[tick].dropna()], axis=1).dropna()
    df_decompose.columns = ['trend_cycle', 'trend_trend', 'price']
    df_decompose['fitted'] = df_decompose['trend_cycle'] + df_decompose['trend_trend']
    df_decompose['residual'] = df_decompose['price'] - df_decompose['fitted']
    df_decompose['cycle_importance'] = abs(df_decompose['trend_cycle']) / (
            abs(df_decompose['trend_cycle']) + abs(df_decompose['trend_trend']))
    df_decompose['trend_importance'] = abs(df_decompose['trend_trend']) / (
            abs(df_decompose['trend_cycle']) + abs(df_decompose['trend_trend']))
    df_decompose[index_name] = tick
    df_decompose_list.append(df_decompose)

df_decompose = pd.concat(df_decompose_list)
df_decompose['Undervalued'] = np.where(df_decompose['trend_cycle'] <= -0.05, 'Yes', 'No')

# for tick, comp in zip(tickers, companies):
#     temp_dec = df_decompose[df_decompose[index_name] == tick]
#     temp_dec[['trend_cycle', 'trend_trend', 'price']].plot(title=comp, secondary_y='trend_cycle')

df_decompose['Upside_potential'] = ((abs(df_decompose['price']) - abs(df_decompose['trend_trend'])) / abs(
    df_decompose['trend_trend'])) * -1

df_temp_upside = df_decompose.set_index(index_name)[['Upside_potential']].groupby(index_name).tail(1).sort_values(
    'Upside_potential', ascending=False)
df_temp_upside.plot.bar(title='Potential Upside Relative to Trend (Fair) Price')

# for tick, comp in zip(tickers, companies):
#     temp_dec = df_decompose[df_decompose.index >= '2022-01-01']
#     temp_dec = temp_dec[temp_dec[index_name] == tick]
#     temp_dec[['trend_importance', 'price']].plot(title=comp, secondary_y='price')

# Count Likely to be overvalued vs Undervalued
df_count = df_decompose.groupby([index_name, 'Undervalued']).count()
df_count['count'] = df_count['trend_cycle']

df_count = df_count.reset_index().pivot(index=index_name, columns='Undervalued', values='count')
df_count['Undervalued_Chances'] = np.where(df_count['No'] < df_count['Yes'], 'Undervalued',
                                           'Overvalued')
df_count = df_count[['Undervalued_Chances']]

fair_values = []
for tick in tickers:
    yearly_price = df_decompose['trend_trend'][df_decompose[index_name] == tick].resample('Y').mean()
    grouped = pd.Series(yearly_price.pct_change().mean())
    grouped = pd.DataFrame(grouped)
    grouped.columns = ['Average_Growth_Fair_Price']
    grouped['average_growth'] = df_decompose['price'][df_decompose[index_name] == tick].resample(
        'Y').mean().pct_change().mean()
    grouped['cagr'] = cagr(df_decompose['price'][df_decompose[index_name] == tick])
    grouped[index_name] = tick
    fair_values.append(grouped)

df_fair_values = pd.concat(fair_values)
df_fair_values = df_fair_values.set_index(index_name)
df_fair_values['Second_Valuation_Critirion'] = np.where(
    df_fair_values['cagr'] > df_fair_values['Average_Growth_Fair_Price'], 'Overvalued', 'Undervalued')

# Valuation scores
df_valuations = df_fair_values[['Second_Valuation_Critirion']].join(df_count)

# Upside Potential
# ================
df_upside = df_decompose[['trend_cycle', 'trend_trend', 'price', index_name, 'Upside_potential']].groupby(
    index_name).tail(1)
df_upside = df_upside.set_index(index_name)
df_upside['Upside_norm'] = zscore(df_upside['Upside_potential'])
# x = df_upside['Upside_norm'].to_numpy().reshape(-1, 1)
# df_upside['Upside_norm'] = pre.MinMaxScaler().fit_transform(x)
# df_upside.sort_values('Upside_norm')

df_valuations = df_valuations.join(df_upside[['Upside_potential', 'Upside_norm']])

# encode
# ============
df_valuations = pd.get_dummies(df_valuations, columns=['Undervalued_Chances', 'Second_Valuation_Critirion'],
                               drop_first=True)
df_valuations['Valuation_Score'] = df_valuations['Undervalued_Chances_Undervalued'] * df_valuations[
    'Second_Valuation_Critirion_Undervalued']
df_valuations['Valuation'] = np.where(df_valuations['Valuation_Score'] == 1, 'Undervalued', 'Overvalued')

# keep valuation scores
# =====================
df_valuations_scores = df_valuations[['Upside_norm', 'Valuation_Score']]
df_valuations_scores['Valuation_Score_final'] = 0.20 * df_valuations['Upside_norm'] + 0.80 * df_valuations[
    'Valuation_Score']
df_valuations_scores = df_valuations_scores[['Valuation_Score_final']]
df_valuations_scores.sort_values(by='Valuation_Score_final').plot.bar()

# performance of stocks
# ======================
df_perfm = performance_overview(prices_temp, prices=True, short=False)
df_perfm.index.name = index_name
df_overview = df_tickers[[index_name, 'Company']].set_index(index_name).join(df_perfm)
df_overview['Sample_Size_years'] = (((df_overview['end'] - df_overview['start'])).dt.days) / 365

# benchmark
bench_ticker = ['SCHX', '^GSPC']
bench_ret = []

for tick in bench_ticker:
    price_sp = openbb.stocks.load(tick, start_date=start_date)
    ret_sp = price_sp['Adj Close'].pct_change()
    ret_sp.name = tick
    ret_sp = pd.DataFrame(ret_sp)
    bench_ret.append(ret_sp)

ret_sp = pd.concat(bench_ret, axis=1)

# df_prices_ewm = df_prices.ewm(span=30).mean()
# ret_ewm = df_prices.ewm(span=30).mean().pct_change()

df_main_stats = get_main_stats(prices_temp, rf=0.05)

# greeks_list = []
# for tick in tickers:
#     temp_greeks = get_rolling_greek_stats(prices_temp[[tick]].pct_change(), ret_sp[['^GSPC']], rolling_period=30)
#     temp_greeks.name = tick
#     greeks_list.append(pd.DataFrame(temp_greeks))
#
# df_beta_rolling = pd.concat(greeks_list, axis=1).transpose()[['beta']]

ret_sm = prices_temp.ewm(span=30).mean().pct_change()
bench_sm = ret_sp[['^GSPC']].ewm(span=30).mean()

df_beta_decompose = alpha_beta_table(prices_temp.pct_change(), ret_sp[['^GSPC']], rf=0.05)
df_beta_decompose = df_beta_decompose[['beta', 'beta_bull', 'beta_bear']]
df_beta_decompose['Defensive'] = np.where(df_beta_decompose['beta_bear'] < 1, 1, 0)
df_beta_decompose['Defensive_norm'] = zscore(
    df_beta_decompose['beta_bear']) * -1  # multiply with -1 to punish risky stocks

df_main_stats_sm = get_main_stats(prices_temp.ewm(span=30).mean(), rf=0.05)
df_main_stats_sm = df_main_stats_sm.drop(['max_drawdown', 'sharpe', 'probabilistic_sortino', 'probabilistic_sharpe'],
                                         axis=1)

# finalise
df_final = df_valuations_scores.join(df_beta_decompose[['Defensive_norm']])
df_final = df_final.join(df_main_stats[['max_drawdown']])
df_final = df_final.join(df_main_stats_sm)
df_final = df_final.join(df_overview[['Sample_Size_years']])

score_weights = [0.30, 0.20, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05]
df_final['ranking'] = df_final.dot(score_weights)

# normalize all values to be between 0 and 1
x = df_final['ranking'].to_numpy().reshape(-1, 1)
df_final['ranking_norm'] = pre.MinMaxScaler().fit_transform(x)
df_final['weight'] = df_final['ranking_norm'] / df_final['ranking_norm'].sum()
df_final['weight'].sort_values()

#
# ####
# prices_dummy = prices.copy()
# prices_dummy['year'] = prices_dummy.index.year
# prices_dummy['month'] = prices_dummy.index.month
#
# for i in prices_dummy['month'].unique():
#     prices_dummy[prices_dummy.month == i].fillna(method='ffill').drop_duplicates('year')['AAPL'].plot()
#
# prices_dummy.fillna(method='ffill').drop_duplicates('year')[['AAPL']].plot.bar()


# add industries
# ==============
df_sectors = openbb.stocks.ca.screener(similar=tickers, data_type="overview")
df_sectors = df_sectors[["Ticker\n\n", 'Sector', 'Industry', 'Country']]
df_sectors = df_sectors.rename(columns={"Ticker\n\n": 'Ticker'})
df_sectors.columns = df_sectors.columns[1:, ].insert(0, index_name)

# overview of Data
df_overview_temp = df_final.join(df_sectors.set_index([index_name]))

# check if there is missing info for some stocks
# ================================================
temp_missin = df_overview_temp[df_overview_temp.isna().any(axis=1)]
temp_missin = temp_missin[['Sector', 'Industry', 'Country']]

temp_missin.iloc[0, :] = ["Consumer Defensive", "Packaged Foods", "Switzerland"]  # NSRGF
temp_missin.iloc[1, :] = ["Investment Trust", "Investment Trust", "Global"]  # SMT
temp_missin.iloc[2, :] = ["Communication Services", "Telecom Services", "Global"]  # Soft Bank / 9984.T
temp_missin.iloc[3, :] = ["Healthcare", "Drug Manufacturers—General", "UK"]  # AZN / AstraZeneca
temp_missin.iloc[4, :] = ["Investment Trust", "Investment Trust", "Asia"]  # FAS.L
temp_missin.iloc[5, :] = ["Investment Trust", "Private Equity", "Global"]  # HVPE
temp_missin.iloc[6, :] = ["Industrials", "Engineering & Construction", "France"]  # DG.PA
temp_missin.iloc[7, :] = ["Consumer Cyclical", "Luxury Goods", "France"]  # MC.PA
temp_missin.iloc[8, :] = ['Technology', 'Consumer Electronics', 'South Korea']  # Samsung
temp_missin.iloc[9, :] = ["Investment Trust", "Private Equity", "Global"]  # III
temp_missin.iloc[10, :] = ["Industrials", "Railroads", "Canada"]  # CNR.TO

temp_missin = pd.merge(df_overview_temp, temp_missin, on=index_name, how='left')

temp_missin['Sector'] = np.where(temp_missin['Sector_x'].isna(), temp_missin['Sector_y'], temp_missin['Sector_x'])
temp_missin['Industry'] = np.where(temp_missin['Industry_x'].isna(), temp_missin['Industry_y'],
                                   temp_missin['Industry_x'])
temp_missin['Country'] = np.where(temp_missin['Country_x'].isna(), temp_missin['Country_y'], temp_missin['Country_x'])

temp_missin = temp_missin[df_overview_temp.columns]

#### Sectors
df_sectors_final = temp_missin[['Sector', 'Industry', 'Country']]
df_overview_temp = temp_missin.copy()

# get market shares
yahoo_financials = YahooFinancials(tickers, concurrent=True, max_workers=5)
temp_ccy = yahoo_financials.get_currency()
temp_ccy = pd.DataFrame.from_dict(temp_ccy.items())
temp_ccy.columns = [index_name, 'Currency']
temp_ccy.Currency = [x.upper() for x in temp_ccy.Currency]

temp_mrk_cap = yahoo_financials.get_market_cap()
temp_mrk_cap = pd.DataFrame.from_dict(temp_mrk_cap.items())
temp_mrk_cap.columns = [index_name, 'MarketCap']

temp_mrk_cap = temp_mrk_cap.set_index([index_name]).join(temp_ccy.set_index([index_name]))

# Get FX to convert market Cap
# =============================
fx = []
for ccy in temp_mrk_cap.Currency:
    if ccy != 'USD':
        fx_temp = openbb.forex.load(to_symbol='USD', from_symbol=ccy)
        fx_temp['FX'] = 'USD' + ccy
        fx_temp['Currency'] = ccy
        fx.append(fx_temp[['Close', 'FX', 'Currency']].tail(1))

fx = pd.concat(fx)

temp_ccy_fx = temp_ccy.set_index(['Currency']).join(fx.set_index(['Currency']))
temp_ccy_fx = temp_ccy_fx.reset_index()
temp_ccy_fx['Close'] = temp_ccy_fx['Close'].fillna(1)
temp_ccy_fx['FX'] = temp_ccy_fx['FX'].fillna('USD')

# merge with market cap data
temp = temp_mrk_cap.join(temp_ccy_fx.set_index([index_name, 'Currency']))
temp = temp.drop(['Currency'], axis=1)
temp = temp.reset_index()
temp['MarketCap_USD'] = temp['MarketCap'] * temp['Close']

total_mark_cap = temp['MarketCap_USD'].sum()
temp['weight_market_cap'] = (temp['MarketCap_USD'] / total_mark_cap) * 100
temp = temp[[index_name, 'Currency', 'weight_market_cap']]

####
df_market_cap_final = temp.drop(['Currency'], axis=1).set_index(index_name)

#######
df_final = df_final.join(df_market_cap_final)
df_final = df_final.join(df_sectors_final)

###
fig, axes = plt.subplots(1, 2, gridspec_kw={"hspace": 5}, figsize=(10, 6))
plot1 = df_final[['weight_market_cap', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[0], y='weight_market_cap',
                                                                                   cmap='rainbow_r',
                                                                                   autopct='%1.f%%', legend=False)
plot1.set_title('Market Value')
plot1.set(ylabel=None)
plot1.tick_params(labelsize=7)

plot2 = df_final[['weight', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[1], y='weight',
                                                                        cmap='rainbow_r',
                                                                        autopct='%1.f%%', legend=False)
plot2.set_title('weight')
plot2.set(ylabel=None)
plt.show()

###
group_by = 'Industry'
fig, axes = plt.subplots(2, 1, gridspec_kw={"hspace": 3}, figsize=(10, 8))
ax1 = df_final[['weight', group_by]].groupby(group_by).sum().sort_values(by='weight',
                                                                         ascending=False).plot(ax=axes[1],
                                                                                               kind='bar',
                                                                                               legend=False)
ax1.tick_params(axis='x', which='both', labelsize=6)
ax1.set_title('weight')
ax1.set(ylabel=None)

ax2 = df_final[['weight_market_cap', group_by]].groupby(group_by).sum().sort_values(by='weight_market_cap',
                                                                                    ascending=False).plot(ax=axes[0],
                                                                                                          kind='bar',
                                                                                                          legend=False)
ax2.set_title('Market Value')
ax2.set(ylabel=None)
ax2.tick_params(axis='x', which='both', labelsize=6)
plt.show()

###### export

keep = ['Company', 'Sector', 'Industry', 'weight', 'weight_market_cap', 'Valuation_Score_final', 'Defensive_norm',
        'max_drawdown',
        'calmar', 'adjusted_sortino', 'cagr']

df_company_names = df_tickers[['YahooTicker', 'Company']].set_index(index_name)
df_final = df_final.join(df_company_names)
df_final_export = df_final.drop_duplicates()
df_final_export[keep].to_clipboard()
