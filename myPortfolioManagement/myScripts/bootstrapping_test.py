import pandas as pd
import quantstats as qs

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myBootstrapping import BootstrapIDD, BootstrapStationary, BootstrapCircular, \
    BootstrapMovingBlock, tsmoothie_boostrapping

from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=7, disable_pr_bar=False)

ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change().dropna()
bench_name = 'SP500'
ret_bench.columns = [bench_name]

n_samples = 10000
func = qs.stats.adjusted_sortino
seed_n = 123
block_size = 365 * 2  # 2 years

df_results1 = BootstrapIDD(ret_bench[bench_name], n_samples=n_samples, seed=seed_n)

df_results2 = BootstrapStationary(ret_bench[bench_name], block_size=block_size, n_samples=n_samples, optimal_block=True,
                                  seed=seed_n)

df_results3 = BootstrapStationary(ret_bench[bench_name], block_size=block_size, n_samples=n_samples,
                                  optimal_block=False,
                                  seed=seed_n)

df_results4 = BootstrapMovingBlock(ret_bench[bench_name], block_size=block_size, n_samples=n_samples, seed=seed_n)

df_results5 = BootstrapCircular(ret_bench[bench_name], block_size=block_size, n_samples=n_samples, optimal_block=False,
                                seed=seed_n)

# df_results5 = BootstrapCircular(ret_bench[bench_name], block_size=5, n_samples=n_samples, optimal_block=True,
#                                 seed=seed_n)


# tssmoothie
# ======================================================================================================================
btypes = ['mbb', 'cbb', 'sb']

df_tsmoothie_list = []
for btype in btypes:
    df_res_temp = tsmoothie_boostrapping(series=ret_bench[bench_name], block_size=block_size, bootstrap_type=btype,
                                         n_samples=n_samples, residual_method=True)
    df_tsmoothie_list.append(df_res_temp)

results = [df_results1, df_results2, df_results3, df_results4, df_results5]
df_func_list = []
for df in results:
    df_func_temp = df.p_apply(func, raw=False, executor='processes', smart=True)
    df_func_list.append(df_func_temp)

res1 = pd.concat(df_func_list, axis=1)

df_tsmoothie_func_list = []
for df in df_tsmoothie_list:
    df_func_temp = df.p_apply(func, raw=False, executor='processes', smart=True)
    df_tsmoothie_func_list.append(df_func_temp)

res2 = pd.concat(df_tsmoothie_func_list, axis=1)
res2.columns = ['tsm0', 'tsm1', 'tsm2']

sr = round(func(ret_bench, smart=True)[0], 2)
bar_plot = pd.concat([res1, res2], axis=1).plot.density()
bar_plot.axvline(x=sr, color='green', linewidth=2)
bar_plot.axvline(x=round(res1.mean()[3], 2), color='black', linewidth=1)

bar_plot = res2.plot.density()
bar_plot.axvline(x=sr, color='green', linewidth=2)


import quantstats as qs

series = qs.utils.download_returns('META').dropna()
spc = ConvolutionSmoother(window_len=10, window_type='ones')
spc = KalmanSmoother(component=, component_noise= )


bts = BootstrappingWrapper(spc, bootstrap_type='mbb', block_length=365 * 2)
bts_samples = bts.sample(data=series.dropna(), n_samples=100)
df = pd.DataFrame(bts_samples.transpose())
