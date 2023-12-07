import warnings

warnings.filterwarnings('ignore')

import ffn
import pandas as pd
import numpy as np
from myPortfolioManagement.myData import get_stock_prices, get_sp500_tickers, get_stock_info
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.MeteoraGlobalEquityFund.Trade212_Account.get_account_info import get_pie_details, get_pies
from myPortfolioManagement.myBootstrapping import bootstrappingTS
import pickle
import quantstats as qs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
df_pie_details = get_pie_details(id='1783656')  # big 7
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]
df_pie_weights['tickers_212'] = np.where(df_pie_weights['tickers_212'] == 'FB', 'META', df_pie_weights['tickers_212'])

# download prices per date of purchase
# ====================================
ticker_col_name = 'YahooTickers'
tickers = ['MSFT', 'GOOG', 'AAPL', 'NVDA', 'TSLA', 'AMZN', 'META']
start_date = '1950-01-01'

# identify Non GBP stocks and download FX
# ======================================================================================================================
df_prices = get_stock_prices(yahoo_tickers=tickers, start_date=start_date, fix_data=True)
df_prices_wide = df_prices.pivot(columns='yahooTicker', values='adjclose')

# Backtest
# ======================================================================================================================
ret = df_prices_wide.pct_change().dropna()

# portfolio returns
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_pie_weights[['tickers_212', 'expectedShare']])

# calculate benchmark
# ===================
ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]
ret_all = ret_port.join(ret_bench).dropna()

price_index = ffn.core.to_price_index(ret_all, start=1)
price_index = ffn.core.rebase(price_index, value=1)
price_index.plot()

df_main_stats = get_main_stats(price_index, smart=True, rf=0.05).transpose()

qs.reports.basic(ret_all['myPortfolio'], benchmark=ret_all[bench_name], rf=0.05)
qs.reports.html(ret_all['myPortfolio'], benchmark=ret_all[bench_name], rf=0.05,
                output='/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/big7_port.html')

# Look at Big 7
# ======================================================================================================================
sp500 = get_sp500_tickers()
sp500['sp_weight_unadjusted_google'] = sp500['Market Cap'] / sp500['Market Cap'].sum()

# sort per weight
sp500 = sp500.sort_values(by=['sp_weight_unadjusted_google'], ascending=False)

# calculate market cap of google
google1_mark_cap = sp500[sp500['Ticker'] == 'GOOG']['Market Cap']
google2_mark_cap = sp500[sp500['Ticker'] == 'GOOGL']['Market Cap']
google_total = float(google1_mark_cap) + float(google2_mark_cap)

# remove google
sp500 = sp500[sp500['Ticker'] != 'GOOGL']
sp500['Market Cap'] = np.where(sp500['Ticker'] == 'GOOG', google_total, sp500['Market Cap'])

sp500['sp_weight'] = sp500['Market Cap'] / sp500['Market Cap'].sum()
sp500 = sp500.sort_values(by=['sp_weight'], ascending=False)

df10 = sp500.head(10)
df_big7 = df10[df10['Ticker'].isin(tickers)]
df_big7['big7_weight'] = df_big7['Market Cap'] / df_big7['Market Cap'].sum()

df_big7[['Ticker', 'big7_weight']].set_index('Ticker').plot.bar(y='big7_weight')

# yearly
# TODO: The function does not work when I resample. Need to be flexible to adjust to the frequency of the data
# df_prices_yearly = df_prices_wide.resample('Y').last()
df_main_stats_big7 = get_main_stats(df_prices_wide, smart=True, rf=0.05)

df_main_stats_big7['adjusted_sortino'].sort_values().plot.bar()
df_main_stats_big7['cagr'].sort_values().plot.bar()

# compare my weights with weight based on market value
ret_port = calculate_portfolio_returns(returns=ret, myweights=df_big7[['Ticker', 'big7_weight']],
                                       portfolio_name='port_big7')
ret_all = ret_all.join(ret_port)

get_main_stats(ret_all, smart=True, rf=0.05)

# see Expected Returns and Risks via Bootstrapping
# ======================================================================================================================
#
bootstrap_types = ['cbb', 'sb', 'nbb', 'mbb']
# results_all = {}
# results_all['boostrap_type'] = []
#
# for boost_type in bootstrap_types:
#
#     # get sample of returns
#     data = {}
#     data['results'] = []
#     data['stats'] = []
#
#     for tik in tickers:
#         if boost_type in ['sb', 'cbb']:
#             ret_boostp = bootstrappingTS(ret[tik], n_samples=10000, bootstrap_type=boost_type, optimal_block=True,
#                                          seed=123)
#         else:
#             ret_boostp = bootstrappingTS(ret[tik], n_samples=10000, bootstrap_type=boost_type, block_size=365 * 2,
#                                          seed=123)
#         stats_boostp = get_main_stats(ret_boostp)
#         data['results'].append({tik: ret_boostp})
#         data['stats'].append({tik: stats_boostp})
#
#     results_all['boostrap_type'].append(data)
#
# import pickle
#
# with open('simulate_results.pkl', 'wb') as f:
#     pickle.dump(results_all, f)

with open('/Users/safishajjouz/Library/CloudStorage/OneDrive-Personal/QuantPort_Results/simulate_results.pkl',
          'rb') as f:
    results_all = pickle.load(f)

data_lists = {}
for i, tp in enumerate(bootstrap_types):
    data_temp = results_all['boostrap_type'][i]
    data_lists[tp] = data_temp

grand_list = []
for j, dd in enumerate(data_lists):
    data = data_lists[dd]
    stats_results_list = []
    for i, tik in enumerate(tickers):
        df_temp = data['stats'][i][tik]
        df_temp['Ticker'] = tik
        stats_results_list.append(df_temp)

    df_stats = pd.concat(stats_results_list)
    df_stats['btype'] = bootstrap_types[j]
    grand_list.append(df_stats)

results_final = pd.concat(grand_list)

# plotting results
# ======================================================================================================================

metric = ['cagr', 'adjusted_sortino', 'max_drawdown'][1]
with PdfPages('ad_sortino_bootstrapping.pdf') as pdf:
    for tik in tickers:
        data_plot = results_final[results_final['Ticker'] == tik]
        data_plot = data_plot.drop(['Ticker'], axis=1)
        hist_ave = df_main_stats_big7.loc[tik][metric]
        fig1 = plt.figure()
        sns.displot(data=data_plot.reset_index(), hue='btype', x=metric, palette='Set2',
                    kind="kde", fill=False, legend=True, height=5, aspect=2,
                    cut=0, bw_adjust=1)
        plt.axvline(x=hist_ave, color='r', linestyle='-')
        plt.title(tik)
        pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
        plt.close()
        # plt.show()
        # We can also set the file's metadata via the PdfPages object:

results_final.groupby(['Ticker', 'btype']).median().plot.bar()


# Expected Best investment
# ======================================================================================================================

def greate_score(df_stats_data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    d = scaler.fit_transform(df_stats_data)
    scaled_df = pd.DataFrame(d, columns=df_stats_data.columns, index=df_stats_data.index)

    # see which is a good investment relative to others
    df_score = pd.Series(scaled_df.sum(axis=1), name='score')
    df_score = pd.DataFrame(df_score)
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    d2 = scaler2.fit_transform(df_score)
    scaled_df2 = pd.DataFrame(d2, columns=df_score.columns, index=df_score.index)
    return scaled_df2.sort_values(by=['score'], ascending=False)


df_expected_scores = []
expected_stats = results_final.groupby(['Ticker', 'btype']).median().reset_index()

for tp in bootstrap_types:
    sats = expected_stats[expected_stats['btype'] == tp]
    sats = sats.drop('btype', axis=1)
    sats = sats.set_index('Ticker')
    df_temp = greate_score(sats)
    df_temp['btype'] = tp
    df_expected_scores.append(df_temp)

df_expected_best_inv = pd.concat(df_expected_scores)

df_expected_best_inv.pivot(values='score', columns='btype').plot.bar()

temp = df_stats[['cagr', 'Ticker']].reset_index().copy()
temp = temp.drop(['index'], axis=1)
temp = temp.set_index('Ticker')

scaler = MinMaxScaler(feature_range=(0, 1))
d = scaler.fit_transform(temp)
scaled_df = pd.DataFrame(d, columns=temp.columns, index=temp.index)

sns.displot(data=scaled_df.reset_index(), hue='Ticker', x='cagr',
            kind="kde", fill=False, legend=True, height=5, aspect=0.5,
            cut=0, bw_adjust=1)

# see exposures
# ======================================================================================================================

scaled_df2 = greate_score(df_main_stats_big7)

# see all information for stocks
df_stock_info = get_stock_info(yahoo_tickers=tickers, data_type='all')
string_columns = df_stock_info.select_dtypes(include='object').columns
df_inf_new = df_stock_info.drop(string_columns, axis=1)
df_inf_new = df_inf_new.join(scaled_df2)

s = df_inf_new.corr().loc['score'].sort_values()
s = s.drop(s.index[-1])
s.plot.barh(title='Most Correlated Features with Overall Performance Metrics')

sns.regplot(x=df_inf_new['score'], y=df_inf_new['Fwd P/E'], lowess=True,
            line_kws={'color': 'red'})

# compare Margins
df_stock_info[['Profit M']].sort_values('Profit M').plot.barh()
df_inf_new[['score']].sort_values('score').plot.bar()