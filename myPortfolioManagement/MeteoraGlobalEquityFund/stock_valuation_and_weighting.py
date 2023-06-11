import matplotlib.pyplot as plt
import pandas as pd
import os
from openbb_terminal.sdk import openbb
from scipy.stats import zscore
import warnings
from myPortfolioManagement.myData import get_stock_prices_from_openBB, get_stock_info
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myPerformanceMetrics import get_main_stats, get_rolling_greek_stats
from myPortfolioManagement.myPerformanceMetrics import alpha_beta_table
from sklearn import preprocessing as pre
from myPortfolioManagement.myPerformanceMetrics import cagr
import numpy as np
import matplotlib
from openbb_terminal.sdk import TerminalStyle
import quantstats as qs
from yahoofinancials import YahooFinancials

qs.extend_pandas()
theme = TerminalStyle("light", "light", "light")
warnings.filterwarnings("ignore")

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
start_date = "2016-01-01"

prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date, base_currency='GBP',
                                      index_name=index_name, wide_format=True)

# decompose to cyclical and trend
# ===============================
prices_temp = prices.pivot(columns=index_name, values='Adj_Close_GBP')
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
df_prices_bench = get_stock_prices_from_openBB(bench_ticker, start_date=start_date, base_currency='GBP',
                                               index_name=index_name)

prices_bench = df_prices_bench.pivot(columns = index_name, values = 'Adj_Close_GBP')
ret_sp = prices_bench.pct_change()


df_main_stats = get_main_stats(prices_temp, rf=0.05)

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

df_final.columns

score_weights = [0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.10, 0.05]
df_final['ranking'] = df_final.dot(score_weights)

# normalize all values to be between 0 and 1
x = df_final['ranking'].to_numpy().reshape(-1, 1)
df_final['ranking_norm'] = pre.MinMaxScaler().fit_transform(x)
df_final['weight'] = df_final['ranking_norm'] / df_final['ranking_norm'].sum()
df_final['weight'].sort_values()

df_final[['weight']].sort_values(by = 'weight').plot.bar()

# add industries
# ==============
df_sectors = get_stock_info(yahoo_tickers=tickers)
#TODO: English Stock are quoted in pence and not in pounds. Need to be adjusted
df_market_cap = prices[['YahooTicker', 'Market_Cap_USD']]
df_market_cap = df_market_cap.pivot(columns=index_name, values='Market_Cap_USD')
df_market_cap = df_market_cap.dropna().tail(1)
df_market_cap = df_market_cap.transpose()
df_market_cap.columns = ['Market_Cap_USD']

total_mark_cap = df_market_cap['Market_Cap_USD'].sum()
df_market_cap['weight_market_cap'] = (df_market_cap['Market_Cap_USD'] / total_mark_cap)

#######
df_final = df_final.join(df_market_cap)
df_final = df_final.join(df_sectors)

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
        'max_drawdown', 'adjusted_sortino', 'cagr']

df_company_names = df_tickers[['YahooTicker', 'Company']].set_index(index_name)
df_final = df_final.join(df_company_names)
df_final_export = df_final.drop_duplicates()
df_final_export[keep].to_clipboard()
