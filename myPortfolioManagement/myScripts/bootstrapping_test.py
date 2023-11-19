from myPortfolioManagement.myData import get_stock_prices
import pandas as pd

from arch.bootstrap import IIDBootstrap
from arch.bootstrap import optimal_block_length
from arch.bootstrap import StationaryBootstrap

import quantstats as qs

from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import *
from tsmoothie.utils_func import _id_nb_bootstrap, _id_mb_bootstrap

ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]

btype = ['nbb', 'mbb', 'cbb', 'sb'][0]
spc = SpectralSmoother(smooth_fraction=0.18, pad_len=12)
bts = BootstrappingWrapper(spc, bootstrap_type=btype, block_length=24)
bts_samples = bts.sample(data=ret_bench.dropna(), n_samples=10)

opt = optimal_block_length(ret_bench.dropna() ** 2)
opt_block = opt['stationary'][0]

nobs = len(ret_bench)
n_samples = 10000
bootstrap_data = np.empty((n_samples, nobs))
for i in np.arange(n_samples):
    bootstrap_id = _id_mb_bootstrap(nobs, opt_block)
    bootstrap_data[i] = np.squeeze(ret_bench.iloc[bootstrap_id].values)

string_name = 'path'
cols = [string_name + str(x) for x in range(0, n_samples)]

df = pd.DataFrame(bootstrap_data.transpose(), columns=cols)
df.index = ret_bench.index
func = qs.stats.adjusted_sortino
df.apply(func=func)


# Function to compute parameters
def sharpe_ratio(x):
    mu, sigma = 12 * x.mean(), np.sqrt(12 * x.var())
    return np.array([mu, sigma, mu / sigma])


# Bootstrap confidence intervals
bs = IIDBootstrap(ret_bench)
results = bs.apply(func, n_samples)
SR = pd.DataFrame(results[:, 0], columns=[func.__name__])
fig = SR.hist(bins=40)

# stationary bootstrap
# ====================

# Initialize with entropy from random.org
entropy = [877788388, 418255226, 989657335, 69307515]
seed = np.random.default_rng(entropy)

# without optimal block
bs = StationaryBootstrap(12, ret_bench, seed=seed)
results = bs.apply(func, n_samples)
SR2 = pd.DataFrame(results[:, 0], columns=[func.__name__])
fig = SR2.hist(bins=40)

# with optimal block
bs = StationaryBootstrap(opt.loc[bench_name, "stationary"], ret_bench, seed=seed)
results = bs.apply(func, n_samples)
SR3 = pd.DataFrame(results[:, 0], columns=["SR"])
fig = SR3.hist(bins=40)

# circular bootstrap
# ===================

# non-overlapping block bootstrap
# ===============================
