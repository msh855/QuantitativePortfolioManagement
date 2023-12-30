from myPortfolioManagement.myBootstrapping import BootstrapMovingBlock
from myPortfolioManagement.myData import *
from myPortfolioManagement.myReturns import calculate_returns
from myPortfolioManagement.myBacktesting import *

### load Data
file = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/Data/BBG_CommodityIndex.xlsx'

df_com_prices = pd.read_excel(file, sheet_name='USD_daily', skiprows=6)
df_com_prices = df_com_prices.drop(['PX_VOLUME'], axis=1)
df_com_prices = df_com_prices.set_index('Date')
df_com_prices = df_com_prices.reindex(index=df_com_prices.index[::-1])
df_com_prices.head()

ret_comm = calculate_returns(df_com_prices)

samples = 100
### create a fan chart
fan_chart(returns=ret_comm['PX_LAST'],
          weight_period=None,  ## this to bootstrap from specific regimes
          out_of_sample_date='2021-01-01',
          n_sample=samples,
          chart_title='Cumulative Returns BCOM Index: Simulations Based on 1960-2020')

### create a fan chart based on the distribution of a specific period: Here from 1970 - 1980
fan_chart(returns=ret_comm['PX_LAST'],
          weight_period=['1970-01-01', '1980-01-01'],
          out_of_sample_date='2021-01-01',
          n_sample=samples,
          chart_title='Cumulative Returns BCOM Index: Simulations Based on 1970-1980 (Stagflation Period)')



prices = get_stock_prices(yahoo_tickers=['NIO'], wide_format=True)

series = prices.iloc[:, 0]
seed = 123
n_samples = 5000

df = BootstrapMovingBlock(series=series, block_size=100, n_samples=n_samples, seed=seed)


df.iloc[0, :].plot.density()
series.plot.density()


import numpy as np

means = df.mean(0)
minX, maxX = (int(means.min()), int(means.max()))  # Specify the range of x-axis
means.plot.density(ind=np.linspace(minX, maxX))
