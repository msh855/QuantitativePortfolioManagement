from myPortfolioManagement.myTimeSeries import Trends, decomposeTS, forecast_trend, look_back
from maData.getdata import get_US_yields
from myPortfolioManagement.myBootstrapping import bootstrappingTS, BootstrapCircular, BootstrapMovingBlock, bootstrappingTS_smoothie
import matplotlib.pyplot as plt
import pandas as pd
from myPortfolioManagement.myBacktesting import prep_dist
import seaborn as sns
from myPortfolioManagement.myBootstrapping import BootstrapStationary
from myPortfolioManagement.myBacktesting import prep_dist, plot_fan_chart, sim_series, sim_paths
from myPortfolioManagement.myData import get_stock_prices
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, IIDBootstrap, MovingBlockBootstrap, \
    optimal_block_length

prices = get_stock_prices(yahoo_tickers=['NIO'], wide_format=True)

series = prices.iloc[:, 0]
seed = 123
n_samples = 5000

df = BootstrapMovingBlock(series=series, block_size=100, n_samples=n_samples, seed=seed)
df_sm = bootstrappingTS_smoothie(series, n_samples=n_samples, residual_method=False)

df.iloc[0, :].plot.density()
series.plot.density()

df_sm.iloc[0, :].plot.density()
df_sm.iloc[100, :].plot.density()
series.plot.density()

import numpy as np
means = df.mean(0)
minX, maxX = (int(means.min()), int(means.max()))  # Specify the range of x-axis
means.plot.density(ind=np.linspace(minX, maxX))