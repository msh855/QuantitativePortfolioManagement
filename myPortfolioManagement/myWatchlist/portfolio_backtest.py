import pandas as pd
from openbb_terminal.sdk import openbb
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio, equal_weight_portfolio
import quantstats as qs
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myBacktesting import bootstrap_portfolio_performance
import numpy as np
from yahoofinancials import YahooFinancials
from openbb_terminal.sdk import TerminalStyle

theme = TerminalStyle("light", "light", "light")

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

# load tickers
df_tickers = pd.read_excel('/Users/safishajjouz/PycharmProjects/myWatchlist/Data/stock_screening.xlsx')
df_tickers.dropna(inplace=True)
index_name = 'YahooTicker'

tickers = list(df_tickers.YahooTicker)
companies = list(df_tickers['Company'])
start_date = "2005-01-01"

df_prices = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data[index_name] = ticker
    data['Company'] = company
    df_prices.append(data)

df_prices = pd.concat(df_prices)

prices = df_prices[['Adj Close', index_name]]
prices = prices.pivot(columns=index_name, values='Adj Close')

# Optimase Portfolio
# ===================
returns = prices.pct_change()
# returns_training = returns.dropna()
returns_training = returns.fillna(0)

df_weights = df_tickers[[index_name, 'adjusted_weights']]

# optimal weight
# ===============
inv_vol_weights = inverse_vol_portfolio(returns_training=returns_training)  # Inverse Vol
equal_weights = equal_weight_portfolio(returns_training)  # Equally Waighted
min_vol_weights = port_GMV(returns_training=returns_training)

df_port_weights = pd.concat([inv_vol_weights, equal_weights, min_vol_weights], axis=1)
df_port_weights.index.name = df_weights.columns[0]
df_port_weights = pd.concat([df_port_weights, df_weights.set_index(index_name)], axis=1)

# portfolio returns
# =================
df_portfolios = []
for col in df_port_weights.columns:
    temp_portf = calculate_portfolio_returns(returns_training, df_port_weights[[col]].reset_index(), portfolio_name=col)
    df_portfolios.append(temp_portf)

df_portfolios = pd.concat(df_portfolios, axis=1)

# benchmark
price_sp = openbb.stocks.load('^GSPC', start_date=start_date)
ret_sp = price_sp['Adj Close'].pct_change()
ret_sp.name = 'S&P'
ret_sp = pd.DataFrame(ret_sp)

# total returns
date_index_name = 'date'
ret = ret_sp.reset_index().merge(df_portfolios.reset_index())
ret = ret.set_index(date_index_name)

# check
portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

# calculate distance from Peak
# ============================
df_peak_prices = df_prices[[index_name, 'Company', 'Adj Close']].groupby(by=[index_name, 'Company']).max()
df_peak_prices.columns = ['max_price']

df_last_price = df_prices[[index_name, 'Company', 'Adj Close']].groupby(by=[index_name, 'Company']).tail(1)
df_last_price = df_last_price.reset_index().drop(['date'], axis=1).set_index([index_name, 'Company'])
df_last_price.columns = ['last_close']

df_peak_prices = df_peak_prices.reset_index().merge(df_last_price.reset_index()[[index_name, 'last_close']])
df_peak_prices['dist_from_peak'] = df_peak_prices['last_close'] / df_peak_prices['max_price'] - 1
df_peak_prices = df_peak_prices.set_index('Company')

# plot
plot_bar = df_peak_prices.sort_values(by='dist_from_peak')['dist_from_peak'].plot.bar(y='dist_from_peak', rot=90,
                                                                                      title='Distance From Recent Peak')
plot_bar.tick_params(axis='x', which='both', labelsize=6)

# get sectoral Info and merge with price info
# ==================
df_sectors = openbb.stocks.ca.screener(similar=tickers, data_type="overview")
df_sectors = df_sectors[["Ticker\n\n", 'Sector', 'Industry', 'Country']]
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

# Performance
df_portf_perfm = performance_overview(prices_from_returns(ret), prices=True, short=False)
df_portf_perfm.index.name = df_port_weights.index.name
df_portf_perfm.to_clipboard()

ret['adjusted_weights']

ret_new = ret[ret.index <= '2020-01-01']

qs.reports.html(ret_new['adjusted_weights'], ret_new['S&P'],
                output='/Users/safishajjouz/PycharmProjects/myWatchlist/Data',
                download_filename='myPie_pre_pand.html')

ret_cp = ret.copy()
ret_cp.index.name = 'Date'
out_of_sample_date = '2022-01-01'
boost_results = bootstrap_portfolio_performance(returns=ret_cp['adjusted_weights'],
                                                returns_benchmark=ret_cp['S&P'],
                                                out_of_sample_date='2022-01-01',
                                                n_sim=5000)

boost_results[0]
boost_results[1]['alpha_out_sample'].hist()

ret_test = prices_from_returns(ret_cp[ret_cp.index >= out_of_sample_date])
ret_test.plot()

# fan_chart(returns=ret['adjusted_weights'],
#           weight_period=None,
#           out_of_sample_date='2022-01-01',
#           n_sample=100,
#           chart_title='Cumulative Returns')
#
# backtest_report(ret[['adjusted_weights']], ret[['S&P']], out_of_sample_date='2022-01-01')
