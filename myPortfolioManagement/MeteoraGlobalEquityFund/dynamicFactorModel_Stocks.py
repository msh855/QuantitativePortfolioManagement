import pandas as pd
from openbb_terminal.sdk import TerminalStyle
import warnings
from matplotlib import pyplot as plt
from myPortfolioManagement.myDataPreparation import remove_outliers
from myPortfolioManagement.myData import get_stock_prices_from_openBB
import statsmodels.api as sm

theme = TerminalStyle("light", "light", "light")

plt.style.use('seaborn')
warnings.filterwarnings("ignore")

# # import tickers
# # ==============
# file_path = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data/stock_screening.xlsx'
#
# # load tickers
# df_tickers = pd.read_excel(file_path)
# df_tickers.dropna(inplace=True)
# index_name = 'YahooTicker'
#
# tickers = list(df_tickers.YahooTicker)
# companies = list(df_tickers['Company'])
# start_date = "2005-01-01"
#
# prices = get_stock_prices_from_openBB(yahoo_tickers=tickers, start_date=start_date)
#
# df_prices = prices.pivot_table(index=prices.index, values='Adj_Close_GBP', columns='YahooTicker')

file_path = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/MeteoraGlobalEquityFund/Data/df_prices.csv'
df_prices = pd.read_csv(file_path)

# Optimase Portfolio
# ===================
returns = df_prices.pct_change()
returns = remove_outliers(returns)

# Construct the dynamic factor model
model = sm.tsa.DynamicFactorMQ(returns, factors=3)
model.summary()
results = model.fit(disp=10)
results.summary()

factor_names = ['Factor1', 'Factor2', 'Factor3']
mean = results.factors.smoothed
mean.columns = factor_names
results.factors.smoothed_cov.columns = factor_names

results.factors.smoothed_cov

# Compute 95% confidence intervals
from scipy.stats import norm

std = pd.concat([results.factors.smoothed_cov.loc[name, name] for name in factor_names], axis=1)
crit = norm.ppf(1 - 0.05 / 2)
lower = mean - crit * std
upper = mean + crit * std
