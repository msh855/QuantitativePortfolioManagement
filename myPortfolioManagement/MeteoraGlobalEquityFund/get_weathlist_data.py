import pandas as pd
from openbb_terminal.sdk import openbb
from yahoofinancials import YahooFinancials
from matplotlib import pyplot as plt
from myPortfolioManagement.myReturns import calculate_portfolio_returns
from myPortfolioManagement.myPerformanceAnalytics import performance_overview
from myPortfolioManagement.myBacktesting import metrics
from myPortfolioManagement.myPortfolioOptimisation import equal_weight_portfolio
from pypfopt.expected_returns import prices_from_returns
from myPortfolioManagement.myPortfolioOptimisation import port_GMV, inverse_vol_portfolio, \
    port_target_return, port_target_volatility
import quantstats as qs
import numpy as np

plt.style.use('seaborn')
pd.options.display.float_format = '{:20,.2f}'.format

# load tickers
df_tickers = pd.read_excel('/Users/safishajjouz/PycharmProjects/myWatchlist/Data/yahoo_tickers.xlsx')
df_tickers.dropna(inplace=True)
myweights = df_tickers[['YahooTicker', 'myWeights']]
myweights.columns = ['Ticker', 'myWeights']

if myweights['myWeights'].sum() != 100:
    print('error in custom weights')

tickers = list(df_tickers.YahooTicker)
# tickers.append('^GSPC')
companies = list(df_tickers['Company'])
# companies.append('S&P')
start_date = "1990-01-01"

df_prices = []
for ticker, company in zip(tickers, companies):
    data = openbb.stocks.load(ticker, start_date=start_date)
    data['Ticker'] = ticker
    data['Company'] = company
    df_prices.append(data)

df_prices = pd.concat(df_prices)

# Optimase Portfolio
# ===================
prices = df_prices[['Adj Close', 'Ticker']]
prices = prices.pivot(columns='Ticker', values='Adj Close')
returns = prices.pct_change()

# optimise per routines
# ======================
returns_training = returns.dropna()

# optimal weight
# ===============
inv_vol_weights = inverse_vol_portfolio(returns_training=returns_training)  # Inverse Vol
equal_weights = equal_weight_portfolio(returns_training)  # Equally Waighted
min_vol_weights = port_GMV(returns_training=returns_training)
myweights['own_weights'] = myweights.myWeights / 100
df_myweights = myweights[['Ticker', 'own_weights']].copy()

df_port_weights = pd.concat([inv_vol_weights, equal_weights, min_vol_weights], axis=1)
df_port_weights.index.name = myweights.columns[0]
df_port_weights = pd.concat([df_port_weights, df_myweights.set_index('Ticker')], axis=1)

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
index_name = 'date'
ret = ret_sp.reset_index(index_name).merge(df_portfolios.reset_index(index_name))
ret = ret.set_index(index_name)

portf_prices = prices_from_returns(ret)
portf_prices.plot()
portf_prices.pct_change().corr()

index_valuation = (portf_prices.tail(1) - portf_prices.max()) / portf_prices.max()
index_valuation = index_valuation.transpose()
index_valuation.columns = ['Undervalued']

index_valuation.sort_values(by='Undervalued').plot.barh()

# calculate distance from Peak
# ============================
df_peak_prices = df_prices[['Ticker', 'Company', 'Adj Close']].groupby(by=['Ticker', 'Company']).max()
df_peak_prices.columns = ['max_price']

df_last_price = df_prices[['Ticker', 'Company', 'Adj Close']].groupby(by=['Ticker', 'Company']).tail(1)
df_last_price = df_last_price.reset_index().drop(['date'], axis=1).set_index(['Ticker', 'Company'])
df_last_price.columns = ['last_close']

df_peak_prices = df_peak_prices.reset_index().merge(df_last_price.reset_index()[['Ticker', 'last_close']])
df_peak_prices['dist_from_peak'] = df_peak_prices['last_close'] / df_peak_prices['max_price'] - 1
df_peak_prices = df_peak_prices.set_index('Company')

# plot
plot_bar = df_peak_prices.sort_values(by='dist_from_peak')['dist_from_peak'].plot.bar(y='dist_from_peak', rot=90)
plot_bar.tick_params(axis='x', which='both', labelsize=6)

# get sectoral Info and merge with price info
# ==================
df_sectors = openbb.stocks.ca.screener(similar=tickers, data_type="overview")
df_sectors = df_sectors[["Ticker\n\n", 'Sector', 'Industry', 'Country']]
df_sectors.columns = df_sectors.columns[1:, ].insert(0, 'Ticker')

# overview of Data
df_overview = df_peak_prices.set_index(['Ticker']).join(df_sectors.set_index(['Ticker']))

# check if there is missing info for some stocks
# ================================================
temp_missin = df_overview[df_overview.isna().any(axis=1)]
temp_missin = temp_missin[['Sector', 'Industry', 'Country']]

temp_missin.iloc[0, :] = ["Industrials", "Engineering & Construction", "France"]  # III
temp_missin.iloc[1, :] = ["Investment Trust", "Asia", "Asia"]  # III
temp_missin.iloc[2, :] = ["Private Equity", "Private Equity", "United Kingdom"]  # HVPE
temp_missin.iloc[3, :] = ["Private Equity", "Private Equity", "United Kingdom"]  # III
temp_missin.iloc[4, :] = ["Consumer Cyclical", "Luxury Goods", "France"]  # MC.PA
temp_missin.iloc[5, :] = ["Consumer Defensive", "Packaged Foods", "Switzerland"]  # NSRGF
temp_missin.iloc[6, :] = ['Basic Materials', 'Agricultural Inputs', 'Canada']  # NTR
temp_missin.iloc[7, :] = ['Technology', 'Consumer Electronics', 'South Korea']  # Samsung

temp_missin = pd.merge(df_overview, temp_missin, on='Ticker', how='left')

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
temp_ccy.columns = ['Ticker', 'Currency']
temp_ccy.Currency = [x.upper() for x in temp_ccy.Currency]

temp_mrk_cap = yahoo_financials.get_market_cap()
temp_mrk_cap = pd.DataFrame.from_dict(temp_mrk_cap.items())
temp_mrk_cap.columns = ['Ticker', 'MarketCap']

temp_mrk_cap = temp_mrk_cap.set_index(['Ticker']).join(temp_ccy.set_index(['Ticker']))

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
temp = temp_mrk_cap.join(temp_ccy_fx.set_index(['Ticker', 'Currency']))
temp = temp.drop(['Currency'], axis=1)
temp = temp.reset_index()
temp['MarketCap_USD'] = temp['MarketCap'] * temp['Close']

total_mark_cap = temp['MarketCap_USD'].sum()
temp['weight_market_cap'] = (temp['MarketCap_USD'] / total_mark_cap) * 100
temp = temp[['Ticker', 'Currency', 'weight_market_cap']]

# merge with sectoral info
df_overview = df_overview.join(temp.set_index(['Ticker']))
df_overview = df_overview.join(myweights.set_index(['Ticker']))
df_overview = df_overview.sort_values(by='weight_market_cap', ascending=False)
df_overview = df_overview.reset_index().drop_duplicates()
df_overview = df_overview.set_index('Ticker')

# Performance
df_performance = performance_overview(returns_training)
df_performance.index.name = df_overview.index.name
df_portf_perfm = performance_overview(prices_from_returns(ret), prices=True)
df_portf_perfm.index.name = df_overview.index.name

df_performance = pd.concat([df_portf_perfm.transpose(), df_performance.transpose()], axis=1)
df_performance = df_performance.transpose()

# merge
df_overview = df_performance.join(df_overview)
df_overview.to_clipboard()

# plot
df_overview[['weight_market_cap']].plot.bar()

ret[ret.index >= '2020-01-01'].cumsum().plot()

ret[['myPortfolio', 'S&P']].cumsum().plot()

prices_from_returns(ret[ret.index >= '2020-01-01']).plot()

###
fig, axes = plt.subplots(1, 2, gridspec_kw={"hspace": 5}, figsize=(10, 6))
plot1 = df_overview[['weight_market_cap', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[0], y='weight_market_cap',
                                                                                      cmap='rainbow_r',
                                                                                      autopct='%1.f%%', legend=False)
plot1.set_title('Market Value')
plot1.set(ylabel=None)
plot1.tick_params(labelsize=7)

plot2 = df_overview[['myWeights', 'Sector']].groupby('Sector').sum().plot.pie(ax=axes[1], y='myWeights',
                                                                              cmap='rainbow_r',
                                                                              autopct='%1.f%%', legend=False)
plot2.set_title('My Weights')
plot2.set(ylabel=None)
plot2.legend(loc=0, labels=df_overview[['myWeights', 'Sector']].groupby('Sector').sum().index)
plt.show()

# performance
performance_overview(ret).transpose()
metrics(port_returns_naive, ret_sp)

# plot monthly returns
mrh.plot(port_returns, eoy=True, title='myPortfolio')
mrh.plot(ret_sp, eoy=True, title='S&P')

# plot
ret_sp_eoy = mrh.get(ret_sp, eoy=True)[['eoy']]
ret_sp_eoy.columns = ['Market']
ret_portfolio_eoy = mrh.get(port_returns, eoy=True)[['eoy']]
ret_portfolio_eoy.columns = ['Portfolio']
ret_eoy = ret_sp_eoy.reset_index().merge(ret_portfolio_eoy.reset_index())

ret_eoy.set_index('Year').plot.bar()



#
# # compare ETF and TAWAIN
#
#
# tickers_compare = ['ITWN.L', 'TSM']
# df_prices_comp = []
# for ticker in tickers_compare:
#     data = openbb.stocks.load(ticker, start_date=start_date)
#     data['Ticker'] = ticker
#     df_prices_comp.append(data)
#
# df_prices_comp = pd.concat(df_prices_comp)
#
# prices_comp = df_prices_comp.pivot(columns='Ticker', values='Adj Close')
# returns_comp = prices_comp.pct_change()
# returns_comp = returns_comp.dropna()
#
# returns_comp.corr()
# returns_comp.resample('Y').mean().corr()


# type this command in the terminal
# jupyter nbconvert myWatchlist_performance.ipynb --to html --TagRemovePreprocessor.remove_cell_tags='{"hide_code"}'

#
# df_tickers = pd.read_csv('/Users/safishajjouz/PycharmProjects/myWatchlist/Data/myWatchlist.csv')
#
# df_tickers = df_tickers[['Name', 'Symbol', 'Last Price_Currency']]
# df_tickers.dropna(how='all', inplace=True)
# df_tickers['Tickers'] = df_tickers["Symbol"].str.split(":", expand=True)[0]
#
# tickers_usd = df_tickers[df_tickers['Last Price_Currency'] == 'USD']['Tickers'].tolist()
# tickers_London = df_tickers[df_tickers['Last Price_Currency'] != 'USD']['Tickers'].tolist()
# tickers_London = [x + '.L' for x in tickers_London]
#
# tickers_London = ['NSRGY' if item == 'NESN.L' else item for item in tickers_London]  # Nestle
# tickers_London = ['0P00012CU7.L' if item == 'LU1033663649.L' else item for item in
#                   tickers_London]  # Fidelity Global tech
# tickers_London = ['0P000023MW.L' if item == 'GB00B0CNH163.L' else item for item in tickers_London]  # Global Tech
# tickers_London = ['0P0000KSP6.L' if item == 'GB00B59G4Q73.L' else item for item in tickers_London]  # Global Equity
#
# drop_symbols = ['GB00BYVGKV59.L', 'LU0827889725.L', 'GB00B46KYQ57.L', 'GB00B87BSD02.L',
#                 'GB00B41YBW71.L', 'GB00B79LTQ12.L']
#
# for x in drop_symbols:
#     tickers_London.remove(x)
#
# tickers = tickers_usd + tickers_London
