import polars

from myPortfolioManagement.myData import get_stock_prices
import pandas as pd
import polars as pl

from arch.bootstrap import IIDBootstrap
from arch.bootstrap import optimal_block_length
from arch.bootstrap import StationaryBootstrap

import quantstats as qs

from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import *
from tsmoothie.utils_func import _id_nb_bootstrap, _id_mb_bootstrap, _id_cb_bootstrap

ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change()
bench_name = 'SP500'
ret_bench.columns = [bench_name]

def BootstrapIDD(series: pd.Series = None, n_samples: int = 1000):
    nobs = len(series)
    bsidd = IIDBootstrap(series)
    ps_list = []
    row_range = range(0, nobs)
    for pos_data in bsidd.bootstrap(reps=n_samples):
        new_sample = pos_data[0][0]
        new_sample.index = row_range
        ps_list.append(new_sample)

    df = pd.concat(ps_list, axis=1)

    # cleaning
    string_name = series.name
    cols = [string_name + '_path_' + str(x) for x in range(1, n_samples + 1)]
    df.columns = cols
    df.index = ret_bench.index

    return df

func = qs.stats.adjusted_sortino
df_results = BootstrapIDD(ret_bench[bench_name], n_samples=10000)
df_func = df_results.apply(func)

df_pl = pl.from_pandas(df_results)
df_pl.mean().to_pandas().hist(bins=40)



# ts smoothie bootsrapping
# ======================================================================================================================
btype = ['nbb', 'mbb', 'cbb', 'sb'][1]
spc = SpectralSmoother(smooth_fraction=0.18, pad_len=12)
bts = BootstrappingWrapper(spc, bootstrap_type=btype, block_length=24)
bts_samples = bts.sample(data=ret_bench.dropna(), n_samples=10)

opt = optimal_block_length(ret_bench.dropna() ** 2)
opt_block = opt['stationary'][0]

nobs = len(ret_bench)
n_samples = 1000
bootstrap_data = np.empty((n_samples, nobs))
for i in np.arange(n_samples):
    bootstrap_id = _id_mb_bootstrap(nobs, opt_block)
    bootstrap_data[i] = np.squeeze(ret_bench.iloc[bootstrap_id].values)

string_name = 'path'
cols = [string_name + str(x) for x in range(0, n_samples)]

df = pd.DataFrame(bootstrap_data.transpose(), columns=cols)
df.index = ret_bench.index

func = qs.stats.adjusted_sortino
df_func = df.apply(func=func)
fig = df_func.hist(bins=40)

# mini bootsrapping block
bsidd = IIDBootstrap(ret_bench)
entropy = [877788388, 418255226, 989657335, 69307515]
seed = np.random.default_rng(entropy)
bs_st = StationaryBootstrap(12, ret_bench, seed=seed)
bs_opt = StationaryBootstrap(opt.loc[bench_name, "stationary"], ret_bench, seed=seed)

ind1 = bsidd.index
ind2 = bs_st.index
ind3 = bs_opt.index

# results
func_name = func.__name__

bsidd = IIDBootstrap(ret_bench)
results = np.zeros((n_samples, nobs))
count = 0
ps_list = []
row_range = range(0, nobs)
for pos_data in bsidd.bootstrap(reps=n_samples):
    new_sample = pos_data[0][0]
    new_sample.index = row_range
    ps_list.append(pos_data[0][0])
df = pd.concat(ps_list, axis=1)
string_name = bench_name
cols = [string_name + '_path_' + str(x) for x in range(0, n_samples)]
df.columns = cols
df.index = ret_bench.index



bsidd = IIDBootstrap(ret_bench)
resultsIDD = bsidd.apply(func, n_samples)
SR_idd = pd.DataFrame(resultsIDD[:, 0], columns=[func_name])
fig = SR_idd.hist(bins=40)

# stationary bootstrap
# ====================

# Initialize with entropy from random.org
entropy = [877788388, 418255226, 989657335, 69307515]
seed = np.random.default_rng(entropy)

# without optimal block
bs_st = StationaryBootstrap(12, ret_bench, seed=seed)
results_st = bs_st.apply(func, n_samples)
SR2 = pd.DataFrame(results_st[:, 0], columns=[func_name])
fig = SR2.hist(bins=40)

# with optimal block
bs_opt = StationaryBootstrap(opt.loc[bench_name, "stationary"], ret_bench, seed=seed)
results_opt = bs_opt.apply(func, n_samples)
SR3 = pd.DataFrame(results_opt[:, 0], columns=[func_name])
fig = SR3.hist(bins=40)

# circular bootstrap
# ===================

# non-overlapping block bootstrap
# ===============================

# compare
#
df_temp = df_func.copy()
df_temp = pd.DataFrame(df_temp)
df_temp.columns = ['adjsortino0']

df_temp2 = pd.concat([SR_idd, SR2, SR3], axis=1)
df_temp2.columns = ['adjsortino1', 'adjsortino2', 'adjsortino3']
df_temp2.index = df_temp.index

df_temp3 = pd.concat([df_temp, df_temp2], axis=1)

df_temp3.plot.density()

bsidd.index
bs_st.index
bs_opt.index
