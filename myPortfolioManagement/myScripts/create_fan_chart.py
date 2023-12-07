from fanchart import load_boe_history, load_boe_parameters
from fanchart import fan
import numpy as np
import matplotlib.pyplot as plt
from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myBootstrapping import bootstrappingTS
import pyfolio as pf
import ffn
from pypfopt.expected_returns import prices_from_returns
import pandas as pd

def create_fanchart(result):
    x = np.arange(result.shape[0])
    # for the median use `np.median` and change the legend below
    mean = np.mean(result, axis=1)
    offsets = (10, 20, 30, 40)
    fig, ax = plt.subplots()
    ax.plot(mean, color='black', lw=2)
    for offset in offsets:
        low = np.percentile(result, 50 - offset, axis=1)
        high = np.percentile(result, 50 + offset, axis=1)
        # since `offset` will never be bigger than 50, do 55-offset so that
        # even for the whole range of the graph the fanchart is visible
        alpha = (55 - offset) / 100
        ax.fill_between(x, low, high, color='blue', alpha=alpha)
    ax.legend(['Mean'] + [f'Pct{2 * o}' for o in offsets])
    return fig, ax




prices = get_stock_prices(yahoo_tickers=['TSLA'], wide_format=True)
ret = prices.pct_change().dropna()
ret_boostp = bootstrappingTS(ret['TSLA'], n_samples=10000, bootstrap_type='mbb', block_size=365 * 2,
                             optimal_block=False,
                             seed=123)



S = 1
T = 180
mu = 0.15
vol = 0.05
samples = 100
result = []

for i in range(samples):
    monthly_returns = np.random.normal((1+mu)**(1/T), vol/np.sqrt(T), T)
    monthly_returns = np.hstack([1, monthly_returns])
    price_list = np.cumprod(monthly_returns) * S - 1
    result.append(price_list)

result = np.array(result).T


ret_cum = ret_boostp.cumsum()
ffn.rebase(ret_cum, 1)

price_res = prices_from_returns(ret_boostp)


# custom
param = pd.concat([price_res.median(axis=1), price_res.std(axis=1), price_res.skew(axis=1)], axis=1)
param.columns = ['Mode', 'Uncertainty', 'Skewness']
param = param.reset_index()
param


res_boost = np.array(price_res).T
fig, ax = create_fanchart(res_boost)
plt.show()






history = load_boe_history()
parameters = load_boe_parameters()


history

probs = [0.05, 0.20, 0.35, 0.65, 0.80, 0.95]
parameters

fan(pars=parameters, probs=probs, historic=history[history.Date >= '2018'])

hist = df_temp[['JPYUSD']].reset_index()
hist = hist.rename(columns={'JPYUSD': 'Inflation'})
fan(pars=param, probs=probs, historic=hist[hist.Date >= '2020'])
