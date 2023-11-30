import pandas as pd

from myPortfolioManagement.myData import get_stock_prices
from myPortfolioManagement.myBootstrapping import bootstrappingTS, bootstrappingTS_smoothie
from myPortfolioManagement.myPerformanceMetrics import get_main_stats
from arch.bootstrap import IIDBootstrap


# load data
# ==========
ret_bench = get_stock_prices(yahoo_tickers=['^GSPC'], wide_format=True).pct_change().dropna()
bench_name = 'SP500'
ret_bench.columns = [bench_name]
weight_period = ['2018-01-01', '2023-11-18']
returns = ret_bench[bench_name]
import numpy as np

# set config for bootstrapping
n_samples = 100
seed_n = 123
block_size = 365 * 2  # 2 years
btypes = ['nbb', 'mbb', 'cbb', 'sb']

results_list = []
for type in btypes:
    resilts_temp = bootstrappingTS(series=ret_bench[bench_name], n_samples=n_samples, block_size=block_size,
                                   optimal_block=False,
                                   seed=seed_n, bootstrap_type=type)
    results_list.append(resilts_temp)

df_func_list = []
for df, type in zip(results_list, btypes):
    df_func_temp = get_main_stats(df, rf=0.05, smart=True)
    df_func_temp['btype'] = type
    df_func_list.append(df_func_temp)

res1 = pd.concat(df_func_list)

# tssmoothie
# ======================================================================================================================
df_tsmoothie_list = []
for btype in btypes:
    df_res_temp = bootstrappingTS_smoothie(series=ret_bench[bench_name], block_size=block_size, bootstrap_type=btype,
                                           n_samples=n_samples, residual_method=False)
    df_tsmoothie_list.append(df_res_temp)

df_func_list2 = []
for df, type in zip(df_tsmoothie_list, btypes):
    df_func_temp = get_main_stats(df, rf=0.05, smart=True)
    df_func_temp['btype'] = type
    df_func_list2.append(df_func_temp)

res2 = pd.concat(df_func_list2)

res2['method'] = 'smoothie'
res1['method'] = 'default'

results = pd.concat([res1, res2])

for type in btypes:
    re_sliced = results[results['btype'] == type]
    res_pl = re_sliced.pivot(values='cagr', columns='method')
    res_pl.plot.density()

import quantstats as qs
func = qs.stats.sharpe
bs = IIDBootstrap(ret_bench[bench_name], seed=seed_n)
ci = bs.conf_int(func, 10000, method="basic")
ci = pd.DataFrame(ci, index=["Lower", "Upper"], columns=[bench_name])
print(ci)

bs.conf_int()

ci = bs.conf_int(func, 1000, method="percentile", reuse=True)
ci = pd.DataFrame(ci, index=["Lower", "Upper"], columns=[bench_name])
print(ci)


sr = round(func(ret_bench, smart=True)[0], 2)
bar_plot = pd.concat([res1, res2], axis=1).plot.density()
bar_plot.axvline(x=sr, color='green', linewidth=2)
bar_plot.axvline(x=round(res1.mean()[3], 2), color='black', linewidth=1)

bar_plot = res2.plot.density()
bar_plot.axvline(x=sr, color='green', linewidth=2)

bar_plot = res1.plot.density()
bar_plot.axvline(x=sr, color='green', linewidth=2)

