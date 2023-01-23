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


def rebase(df):
    # Find the minimum starting date among all the series
    min_start = df.apply(lambda x: x.first_valid_index()).min()
    # Create a new dataframe with the aligned index
    aligned_df = pd.DataFrame(index=pd.date_range(start=min_start, end=df.index[-1]))
    # Interpolate missing values and fill them
    for col in df.columns:
        series = df[col].reindex(aligned_df.index).ffill()
        first_valid_idx = series.first_valid_index()
        if first_valid_idx:
            series = series / series[first_valid_idx]
        aligned_df[col] = series
    aligned_df.index.name = df.index.name
    return aligned_df

