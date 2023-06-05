import matplotlib.pyplot as plt
import pandas as pd
import os
from openbb_terminal.sdk import openbb
from scipy.stats import zscore
import warnings
from myPortfolioManagement.myPerformanceMetrics import performance_overview
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from sklearn import preprocessing as pre
from myPortfolioManagement.myPerformanceMetrics import cagr
import numpy as np

from openbb_terminal.sdk import TerminalStyle
import quantstats as qs

qs.extend_pandas()
theme = TerminalStyle("light", "light", "light")
warnings.filterwarnings("ignore")

# import tickers
# ==============
working_directory = os.getcwd()
file_path = 'myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
path = os.path.join(working_directory, file_path)
df_tickers = pd.read_excel(path)
index_name = df_tickers.columns[0]

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "1995-01-01"

df_prices_list = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data[index_name] = ticker
    data['Company'] = company
    df_prices_list.append(data)

df_prices = pd.concat(df_prices_list)
prices = df_prices[['Adj Close', index_name]]
prices = prices.pivot(columns=index_name, values='Adj Close')

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
# df_decompose['Undervalued'] = np.where(df_decompose['price'] < df_decompose['trend_trend'], 'Yes', 'No')
df_decompose['Undervalued'] = np.where(df_decompose['trend_cycle'] <= -0.10, 'Yes', 'No')

# for tick, comp in zip(tickers, companies):
#     temp_dec = df_decompose[df_decompose[index_name] == tick]
#     temp_dec[['trend_cycle', 'trend_trend', 'price']].plot(title=comp, secondary_y='trend_cycle')

df_decompose['Upside_potential'] = ((abs(df_decompose['price']) - abs(df_decompose['trend_trend'])) / abs(
    df_decompose['trend_trend'])) * -1

df_temp_upside = df_decompose.set_index(index_name)[['Upside_potential']].groupby(index_name).tail(1).sort_values(
    'Upside_potential', ascending=False)
df_temp_upside.plot.bar(title='Potential Upside Relative to Trend (Fair) Price')
#
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

df_valuations = df_valuations.join(df_upside[['Upside_norm', 'Upside_norm']])

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

df_main_stats = get_main_stats(df_prices, rf=0.05)


def get_greek_stats(ret, ret_bench, rolling_period=30):
    ret_temp = ret.join(ret_bench)
    # ret_temp.iloc[:,0].greeks(ret_temp.iloc[:,1].dropna())
    df_rolling_stats = ret_temp.iloc[:, 0].rolling_greeks(ret_temp.iloc[:, 1], periods=rolling_period).dropna()
    df_rolling_stats['alpha_beta_corr'] = df_rolling_stats.corr()['alpha'][0]
    df_rolling_stats = df_rolling_stats.mean()

    return df_rolling_stats


greeks_list = []
for tick in tickers:
    temp_greeks = get_greek_stats(df_prices.pct_change()[[tick]], ret_sp[['^GSPC']], rolling_period=30)
    temp_greeks.name = tick
    greeks_list.append(pd.DataFrame(temp_greeks))
pd.concat(greeks_list, axis=1).transpose()






prices_dummy = prices.copy()
prices_dummy['year'] = prices_dummy.index.year
prices_dummy['month'] = prices_dummy.index.month

for i in prices_dummy['month'].unique():
    prices_dummy[prices_dummy.month == i].fillna(method='ffill').drop_duplicates('year')['AAPL'].plot()

prices_dummy.fillna(method='ffill').drop_duplicates('year')[['AAPL']].plot.bar()

# portfolio selection
# ====================
keep = ['cagr', 'max_drawdown', 'calmar', 'daily_vol', 'twelve_month_win_perc']

df_rank = df_overview[keep]
df_rank['daily_vol'] = df_rank['daily_vol'] * -1  # multiplied with -1 to penalize when aggregate

# join
df_rank = df_rank.join(df_upside[['Upside_norm']])

# define ranking
df_rank = df_rank.reset_index().set_index([index_name, 'Company'])
df_rank.columns


def get_weights_ranking(weights_for_ranking=[0.20, 0.25, 0.25, 0.10, 0.10, 0.10]):
    df_rank_norm = df_rank.astype(float).apply(zscore)
    df_rank_norm['ranking'] = df_rank_norm.dot(weights_for_ranking)
    df_rank_norm = df_rank_norm.sort_values(by='ranking', ascending=False)

    x = df_rank_norm['ranking'].to_numpy().reshape(-1, 1)

    # normalize all values to be between 0 and 1
    df_rank_norm['ranking_norm'] = pre.MinMaxScaler().fit_transform(x)
    df_rank_norm['weight'] = df_rank_norm['ranking_norm'] / df_rank_norm['ranking_norm'].sum()

    return df_rank_norm


df_rank_norm = get_weights_ranking()

n = len(keep)
df_rank2 = get_weights_ranking([1 / n] * n)
df_rank2 = df_rank2[['weight']]
df_rank2.columns = ['weight_eq_zscore']
df_rank2 = df_rank2.reset_index()
df_rank2 = df_rank2.set_index(index_name)

df_rank_norm = df_rank_norm[['weight']].join(df_rank2[['weight_eq_zscore']])

df_rank_norm.sort_values('weight', ascending=False)
df_rank_norm.to_clipboard()

# add industries
# ==============

df_sectors = openbb.stocks.ca.screener(similar=tickers, data_type="overview")
df_sectors = df_sectors[["Ticker\n\n", 'Sector', 'Industry', 'Country']]
df_sectors = df_sectors.rename(columns={"Ticker\n\n":})
df_sectors.columns = df_sectors.columns[1:, ].insert(0, index_name)

# overview of Data
df_overview = df_peak_prices.set_index([index_name]).join(df_sectors.set_index([index_name]))

# check if there is missing info for some stocks
# ================================================
temp_missin = df_overview[df_overview.isna().any(axis=1)]
temp_missin = temp_missin[['Sector', 'Industry', 'Country']]

temp_missin.iloc[0, :] = ["Industrials", "Engineering & Construction", "France"]  # DG.PA
temp_missin.iloc[1, :] = ["Investment Trust", "Investment Trust", "Asia"]  # FAS.L
temp_missin.iloc[2, :] = ["Investment Trust", "Private Equity", "Global"]  # HVPE
temp_missin.iloc[3, :] = ["Investment Trust", "Private Equity", "Global"]  # III
temp_missin.iloc[4, :] = ["Consumer Cyclical", "Luxury Goods", "France"]  # MC.PA
temp_missin.iloc[5, :] = ["Consumer Defensive", "Packaged Foods", "Switzerland"]  # NSRGF
temp_missin.iloc[6, :] = ['Technology', 'Consumer Electronics', 'South Korea']  # Samsung
temp_missin.iloc[7, :] = ["Investment Trust", "Investment Trust", "Global"]  # SMT

temp_missin = pd.merge(df_overview, temp_missin, on=index_name, how='left')

temp_missin['Sector'] = np.where(temp_missin['Sector_x'].isna(), temp_missin['Sector_y'], temp_missin['Sector_x'])
temp_missin['Industry'] = np.where(temp_missin['Industry_x'].isna(), temp_missin['Industry_y'],
                                   temp_missin['Industry_x'])
temp_missin['Country'] = np.where(temp_missin['Country_x'].isna(), temp_missin['Country_y'], temp_missin['Country_x'])

temp_missin = temp_missin[df_overview.columns]
df_overview = temp_missin.copy()

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

== == == == == == == == ==

# merge with sectoral info
df_overview = df_overview.join(temp.set_index([index_name]))
df_overview = df_overview.join(df_weights.set_index([index_name]))
df_overview = df_overview.sort_values(by='weight_market_cap', ascending=False)
df_overview = df_overview.reset_index().drop_duplicates()
df_overview = df_overview.set_index(index_name)

###
fig, axes = plt.subplots(1, 2, gridspec_kw={"hspace": 5}, figsize=(10, 6))
plot1 = df_overview[['weight_market_cap', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[0], y='weight_market_cap',
                                                                                      cmap='rainbow_r',
                                                                                      autopct='%1.f%%', legend=False)
plot1.set_title('Market Value')
plot1.set(ylabel=None)
plot1.tick_params(labelsize=7)

plot2 = df_overview[['adjusted_weights', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[1], y='adjusted_weights',
                                                                                     cmap='rainbow_r',
                                                                                     autopct='%1.f%%', legend=False)
plot2.set_title('adjusted_weights')
plot2.set(ylabel=None)
plt.show()

###
group_by = 'Industry'
fig, axes = plt.subplots(2, 1, gridspec_kw={"hspace": 3}, figsize=(10, 8))
ax1 = df_overview[['adjusted_weights', group_by]].groupby(group_by).sum().sort_values(by='adjusted_weights',
                                                                                      ascending=False).plot(ax=axes[1],
                                                                                                            kind='bar',
                                                                                                            legend=False)
ax1.tick_params(axis='x', which='both', labelsize=6)
ax1.set_title('adjusted_weights')
ax1.set(ylabel=None)

ax2 = df_overview[['weight_market_cap', group_by]].groupby(group_by).sum().sort_values(by='weight_market_cap',
                                                                                       ascending=False).plot(ax=axes[0],
                                                                                                             kind='bar',
                                                                                                             legend=False)
ax2.set_title('Market Value')
ax2.set(ylabel=None)
ax2.tick_params(axis='x', which='both', labelsize=6)
plt.show()

# df_port_weights.join(df_overview[['Company', 'Average_Growth_Fair_Price', 'Second_Valuation_Critirion')


df_temp = df_fair_values.join(df_count[['Undervalued_Chances']])
df_overview = df_overview.join(
    df_temp[['Average_Growth_Fair_Price', 'Second_Valuation_Critirion', 'Undervalued_Chances']])

col1 = ['start', 'end', 'rf', 'total_return', 'cagr', 'max_drawdown',
        'calmar', 'mtd', 'three_month', 'six_month', 'ytd', 'one_year',
        'three_year', 'five_year', 'ten_year', 'incep', 'daily_sharpe',
        'daily_sortino', 'daily_mean', 'daily_vol', 'daily_skew', 'daily_kurt',
        'best_day', 'worst_day', 'monthly_sharpe', 'monthly_sortino',
        'monthly_mean', 'monthly_vol', 'monthly_skew', 'monthly_kurt',
        'best_month', 'worst_month', 'yearly_sharpe', 'yearly_sortino',
        'yearly_mean', 'yearly_vol', 'yearly_skew', 'yearly_kurt', 'best_year',
        'worst_year', 'avg_drawdown', 'avg_drawdown_days', 'avg_up_month',
        'avg_down_month', 'win_year_perc']

col_list = ['Company'] + ['Average_Growth_Fair_Price', 'Second_Valuation_Critirion', 'Undervalued_Chances'] + col1
df_overview[col_list].to_clipboard()

df_overview.to_clipboard()
