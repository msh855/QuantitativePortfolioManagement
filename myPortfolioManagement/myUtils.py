import pandas as pd
import quantstats as qs

pd.options.mode.use_inf_as_na = True  # show NAs instead of inf


def balance_dates(returns, returns_benchmark):
    if isinstance(returns_benchmark, pd.DataFrame):
        returns_benchmark = pd.Series(returns_benchmark.iloc[:, 0])

    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns)

    ret_balanced_dates = []

    for col in returns.columns:
        ret, ret_bench = qs.reports._match_dates(returns[col],
                                                 returns_benchmark)
        ret_balanced_dates.append(ret)

    ret_balanced_dates = pd.concat(ret_balanced_dates, axis=1)
    ret_bench = pd.DataFrame(ret_bench)

    # this to create a df with equal length 
    ret = pd.concat([ret_balanced_dates, ret_bench], axis=1)
    ret = ret.fillna(0)

    # get the bench 
    ret_bench = ret.iloc[:, -1]
    ret_bench = pd.DataFrame(ret_bench)

    # get the returns
    ret_balanced_dates = ret.drop(ret_bench.columns, axis=1)

    return ret_balanced_dates, ret_bench



