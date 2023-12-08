import ffn
from myPortfolioManagement.myData import get_stock_prices, get_stock_info
import os
import pandas as pd
from myPortfolioManagement.myReturns import calculate_portfolio_returns, get_benchmark_porfolios
from myPortfolioManagement.myUtils import clean_stock_prices
from myPortfolioManagement.myPerformanceMetrics import get_main_stats

file_path = '/Users/safishajjouz/GitHub/myFinances'
file_name = 'portfoliols_eval_all.xlsx'
file_to_load = os.path.join(file_path, file_name)
portfolio_list = []
num_portf = 5

for i in range(num_portf):
    prt_temp = pd.read_excel(file_to_load, sheet_name=i)
    prt_temp['porfolio'] = 'portfolio' + str(i + 1)
    portfolio_list.append(prt_temp)

# Initial portfolio
start_trading_date = '2020-11-24'
ret_port_list = []
for i in range(num_portf):
    df_port = portfolio_list[i]
    tickers = list(df_port['yahooTicker'])
    df_info = get_stock_info(tickers)
    df_info = df_port.merge(df_info[['yahooTicker', 'longName']])
    df_weights = df_info[['longName', 'weight']]

    # load currencies
    prices = get_stock_prices(tickers, start_date=start_trading_date, wide_format=True)
    ret = clean_stock_prices(prices)

    ret_port = calculate_portfolio_returns(ret, df_weights, portfolio_name='portfolio' + str(i + 1))
    ret_port['portfolio' + str(i + 1)].cumsum().plot()
    ret_port_list.append(ret_port)

df_port_ret = pd.concat(ret_port_list, axis=1)
df_port_ret = df_port_ret.interpolate(method='linear', limit_direction='forward', axis=0)
df_port_ret.cumsum().plot()

# get benchmarks
tickers_bench = ['VWRP.L', 'EQQQ.L', 'VUAG.L']
prices_bench = get_stock_prices(tickers_bench, start_date=start_trading_date, wide_format=True)
ret_bench = clean_stock_prices(prices_bench)
ret_bench.columns = ['Nasdaq', 'FTSE World', 'S&P 500']
ret_bench.cumsum().plot()

# other benchmarks
ret_other_bench = get_benchmark_porfolios()

# combine
ret_all = df_port_ret.join(ret_bench)
ret_all = ret_all.join(ret_other_bench)
ret_all.cumsum().plot()

# stats
df_main_stats = get_main_stats(ret_all)


prices_all = ffn.to_price_index(ret_all)
total_ret = pd.Series(ffn.calc_total_return(prices_all), name='total_return')
df_total_ret = pd.DataFrame(total_ret)
df_main_stats = df_main_stats.join(df_total_ret)

metric = 'total_return'
df_main_stats[[metric, 'adjusted_sortino']].sort_values(by=metric).plot.barh()

# fund performance
df_funds = pd.concat(portfolio_list)
funds = df_funds['yahooTicker'].unique()
df_fund_prices = get_stock_prices(funds, wide_format=True, start_date = start_trading_date)
ret_funds = clean_stock_prices(df_fund_prices)

prices_funds = ffn.to_price_index(ret_funds)
total_ret_funds = pd.Series(ffn.calc_total_return(prices_funds), name='total_return')
total_ret_funds = pd.DataFrame(total_ret_funds)
prices_funds.plot()

df_fund_main_stats=get_main_stats(ret_funds)
df_fund_main_stats = df_fund_main_stats.join(total_ret_funds)

metric = 'total_return'
df_fund_main_stats[[metric, 'adjusted_sortino']].sort_values(by=metric).plot.barh(rot=0, fontsize=8)
