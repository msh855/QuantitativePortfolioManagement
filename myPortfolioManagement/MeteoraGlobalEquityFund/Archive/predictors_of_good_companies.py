import ffn
import pandas as pd
import os

from myPortfolioManagement.myData import get_stock_prices, get_stock_info, get_sp500_tickers, get_nasdaq_tickers
from myPortfolioManagement.myPerformanceMetrics import get_main_stats, reward_metric, performance_overview
from myPortfolioManagement.myDataCleaning import data_overview
from myPortfolioManagement.myPlots import scatter_plot_simple, correlation_matrix

from openbb_terminal.sdk import openbb
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# nasdaq = get_nasdaq_tickers()
# sp = get_sp500_tickers()
#
# keep_col = ['Ticker', 'Company']
# nasdaq = nasdaq[keep_col]
# sp = sp[keep_col]
#
# tickers_nasdaq = list(nasdaq.Ticker)
# tickers_sp = list(sp.Ticker)
#
# tickers_all = tickers_sp + tickers_nasdaq
# tickers = list(set(tickers_all))
#
# df_tickers = pd.concat([sp, nasdaq]).drop_duplicates()
# df_tickers.to_csv('Nasdaq_sp_companies.csv')
#
# df_tickers = pd.read_csv('Nasdaq_sp_companies.csv')
# df_tickers = df_tickers.drop(['Unnamed: 0'], axis=1)
# # check
# df_tickers.set_index('Ticker')
# df_tickers
#
# # df_sample = df_tickers.sample(100)
#
# tickers = list(df_tickers.Ticker)
#
# start_date = '1950-01-01'
# data_list = []
# for tik in tickers:
#     try:
#         data_temp = openbb.stocks.load(symbol=tik, start_date=start_date)
#         data_temp['ticker'] = tik
#         data_list.append(data_temp)
#     except:
#         pass
#
# df_prices = pd.concat(data_list)
# df_prices.to_csv('df_prices_bsdq.csv')
df_prices = pd.read_pickle('df_prices_nasdaq.pkl')
df_prices.set_index('Date', inplace=True)

# data overview
df_prices_wide = df_prices.pivot(columns='ticker', values='Adj Close')
df_prices_wide.index = pd.to_datetime(df_prices_wide.index)

# get main stats
df_main_stats = get_main_stats(df=df_prices_wide, rf=0.05)
df_main_stats = df_main_stats[df_main_stats.age_sample > 2]

# normalise score
df_st = df_main_stats.drop(['age_sample'], axis=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
d = scaler.fit_transform(df_st)
scaled_df = pd.DataFrame(d, columns=df_st.columns, index=df_st.index)

# get the score
df_score = pd.Series(scaled_df.sum(axis=1), name='score')
df_score = pd.DataFrame(df_score)

scaler2 = MinMaxScaler(feature_range=(0, 1))
d2 = scaler2.fit_transform(df_score)
scaled_df2 = pd.DataFrame(d2, columns=df_score.columns, index=df_score.index)

# plot top stocks
scaled_df2.sort_values('score').tail(50).plot.bar()


# correlation of two key metrics
mask = (df_main_stats['adjusted_sortino'] > 0) & (df_main_stats['adjusted_sortino'] <= 2)
df_main_stats_refined = df_main_stats.loc[mask]
df_main_stats_refined = df_main_stats_refined[df_main_stats['cagr'] > 0]

scatter_plot_simple(df_main_stats_refined, x='cagr', y='adjusted_sortino')

# plot cagr
df_main_stats_refined[['cagr']].sort_values('cagr').plot.bar()
df_main_stats_refined[['cagr']].hist(bins=30)
df_main_stats[['cagr']].hist(bins=60)

# see all information for stocks
tickers = list(df_main_stats_refined.index)
df_stock_info = get_stock_info(yahoo_tickers=tickers, data_type='all')

string_columns = df_stock_info.select_dtypes(include='object').columns
df_inf_new = df_stock_info.drop(string_columns, axis=1)
df_inf_new = df_inf_new.join(scaled_df2)

s = df_inf_new.corr().loc['score'].sort_values()
# Assuming your Series is named 's'
s = s.drop(s.index[-1])
s.plot.bar(title='Most Correlated Features with Good Companies: What Drives Performance Metrics?')

sns.regplot(x=df_inf_new['score'], y=df_inf_new['Fwd P/E'], lowess=True,
            line_kws={'color': 'red'})

