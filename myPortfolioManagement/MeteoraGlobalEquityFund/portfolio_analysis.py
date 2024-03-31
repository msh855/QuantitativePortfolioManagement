import ffn
from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myPerformanceMetrics import get_main_stats, reward_metric, performance_overview
import pandas as pd
import os
from myPortfolioManagement.Trade212_Account.get_account_info import get_pie_details, get_pies, \
    get_portfolio_info

from myPortfolioManagement.myDataCleaning import data_overview
from myPortfolioManagement.myPlots import scatter_plot_simple, correlation_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Trade212 Account
# ======================================================================================================================
df_pies = get_pies()  # all pies
df_pie_details = get_pie_details(id='1547833')  # global Meteora
df_trade212_all_stocks = get_portfolio_info()
df_pie_weights = df_pie_details[['tickers_212', 'expectedShare']]

# import tickers
# ======================================================================================================================
working_directory = ('/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/'
                     'MeteoraGlobalEquityFund/Data')

# get currency info
# ======================================================================================================================
file1 = os.path.join(working_directory, 'df_prices.csv')
df_currency_info = pd.read_csv(file1)
df_currency_info = df_currency_info[['YahooTicker', 'Currency']].drop_duplicates()
df_currency_info['Currency'] = [x.upper() for x in df_currency_info['Currency']]

file2 = os.path.join(working_directory, 'stock_screening.xlsx')
df_port_info = pd.read_excel(file2)
df_port_main_info = df_currency_info.merge(df_port_info)

# download prices per date of purchase
# ====================================
df_port_main_info = df_port_main_info.merge(df_trade212_all_stocks[['Date_of_Purchase', 'tickers_212']])
df_port_main_info = df_port_main_info.merge(df_pie_details)

ticker_col_name = df_port_main_info.columns[0]

# identify Non GBP stocks and download FX
# ======================================================================================================================
base_currency = 'GBP'
prices_list = []
start_date = '1950-01-01'
for item in zip(df_port_main_info[ticker_col_name], df_port_main_info['Date_of_Purchase'],
                df_port_main_info['Currency']):

    data = get_stock_prices(yahoo_tickers=[item[0]], start_date=start_date, fix_data=True)

    if item[2] != base_currency:
        fx_temp = openbb.forex.load(to_symbol=base_currency, from_symbol=item[2],
                                    start_date=start_date)
        fx_temp.index.name = 'Date'
        fx_temp = fx_temp[['Adj Close']]
        fx_temp.columns = ['Spot']

        data = data.join(fx_temp)
    else:
        data['Spot'] = 1

    prices_list.append(data)

df_prices = pd.concat(prices_list)

# data overview
df_overview = data_overview(df=df_prices.reset_index(), my_assets_col_name='yahooTicker',
                            my_date_col_name=df_prices.index.name,
                            price_col_name='price_fx_adj')

df_overview = df_overview.join(df_port_main_info[[ticker_col_name, 'Company']].set_index(ticker_col_name))

df_prices_wide = df_prices.pivot(columns='yahooTicker', values='price_fx_adj')
df_main_stats = get_main_stats(df_prices_wide, rf=0.05)

# normalise score
df_st = df_main_stats.drop(['Age(sample)'], axis=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
d = scaler.fit_transform(df_st)
scaled_df = pd.DataFrame(d, columns=df_st.columns, index=df_st.index)

# see which is a good investment relative to others
df_score = pd.Series(scaled_df.sum(axis=1), name='score')
df_score = pd.DataFrame(df_score)
scaler2 = MinMaxScaler(feature_range=(0, 1))
d2 = scaler2.fit_transform(df_score)
scaled_df2 = pd.DataFrame(d2, columns=df_score.columns, index=df_score.index)

# plot
scaled_df2.sort_values('score').plot.bar()

# plot cagr
df_main_stats[['cagr']].sort_values('cagr').plot.bar()

# correlation of two key metrics
scatter_plot_simple(df_main_stats, x='cagr', y='adjusted_sortino')

# plot prices
ffn.rebase(df_prices_wide.fillna(method='bfill'), 1)

# see all information for stocks
tickers = df_port_main_info[ticker_col_name]
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

correlation_matrix(df_inf_new, corr_limit=0.80, phi_correlation=True)

#
df_stock_info = df_stock_info.set_index('Company')
scatter_plot_simple(df_stock_info[['Beta', 'Profit M']], x='Profit M', y='Beta')

df_stock_info[['Profit M']].sort_values('Profit M').plot.barh()

# investment performance
# ======================
df_per_fx_adj = performance_overview(df_prices_wide, prices=True, short=True)

df_per_fx_adj[['cagr']].sort_values('cagr', ascending=False)

# overview
# ======================================================================================================================

df_reward_metrics = reward_metric(df_prices.reset_index(),
                                  rolling_window=5,
                                  rolling_frequency='Y',
                                  my_assets_col_name='yahooTicker',
                                  my_date_col_name='Date',
                                  price_col_name='price_fx_adj')
