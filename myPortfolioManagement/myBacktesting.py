"""
Backtesting Module with Automatic GPU Detection
Optimized for both CPU and GPU environments

Created on Sat Jan 22 19:23:54 2022
@author: safishajjouz
"""
from collections import OrderedDict
from typing import Tuple
import warnings
import multiprocessing

import empyrical as ep
import numpy as np
import pandas as pd
import pyfolio as pf
from numpy import ndarray
from timebudget import timebudget
from joblib import Parallel, delayed
import quantstats_lumi as qs

from myPortfolioManagement.myPlots import *
from myPortfolioManagement.myUtils import balance_dates_robust

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU acceleration available via CuPy")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. Using CPU-only mode")

# Constants
NUMERICAL_PRECISION_THRESHOLD = 1e-10  # Threshold for near-zero value detection


@timebudget
def bootstrap_stats_vectorized(returns: pd.Series,
                               returns_benchmark: pd.Series = None,
                               rf: float = 0.02,
                               periods: int = 252,
                               n_sim: int = 1000,
                               use_gpu: bool = True) -> pd.DataFrame:
    """
    Vectorized bootstrap statistics calculation with GPU support

    This version pre-generates all random samples and computes metrics
    in parallel batches for massive speedup.
    """

    if not isinstance(returns_benchmark, pd.Series):
        if returns_benchmark is None:
            returns_benchmark = pd.Series(dtype='int64')

    # Prepare data
    if not returns_benchmark.empty:
        returns, returns_benchmark = balance_dates_robust(returns, returns_benchmark)

    returns_values = returns.values
    n_obs = len(returns_values)

    # Use GPU if available
    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    # Pre-generate ALL random indices at once (HUGE speedup)
    print(f"  Generating {n_sim} random samples...")
    if use_gpu and GPU_AVAILABLE:
        all_indices = cp.random.randint(0, n_obs, size=(n_sim, n_obs))
        all_indices = cp.asnumpy(all_indices)  # Transfer back to CPU once
    else:
        all_indices = np.random.randint(0, n_obs, size=(n_sim, n_obs))

    # Define metrics to calculate
    metrics_functions = [
        ('cagr', lambda r: qs.stats.cagr(pd.Series(r))),
        ('volatility', lambda r: qs.stats.volatility(pd.Series(r), periods=periods)),
        ('sharpe', lambda r: qs.stats.sharpe(pd.Series(r), rf=rf, periods=periods)),
        ('sortino', lambda r: qs.stats.adjusted_sortino(pd.Series(r), rf=rf, periods=periods)),
    ]

    if not returns_benchmark.empty:
        bench_values = returns_benchmark.values
        metrics_functions.extend([
            ('alpha', lambda r, b: ep.alpha(pd.Series(r), pd.Series(b), risk_free=rf, annualization=periods)),
            ('beta', lambda r, b: ep.beta(pd.Series(r), pd.Series(b), risk_free=rf)),
        ])

    # Parallel computation of metrics
    def compute_sample_metrics(i):
        """Compute all metrics for sample i"""
        idx = all_indices[i]
        returns_i = returns_values[idx]

        sample_metrics = {}
        for metric_name, metric_func in metrics_functions:
            try:
                if metric_name in ['alpha', 'beta'] and not returns_benchmark.empty:
                    bench_i = bench_values[idx]
                    sample_metrics[metric_name] = metric_func(returns_i, bench_i)
                else:
                    sample_metrics[metric_name] = metric_func(returns_i)
            except:
                sample_metrics[metric_name] = np.nan

        return sample_metrics

    # Parallel execution
    print(f"  Computing metrics in parallel...")
    n_jobs = min(multiprocessing.cpu_count(), 8)  # Limit to avoid overhead

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(compute_sample_metrics)(i) for i in range(n_sim)
    )

    # Convert to DataFrame
    bootstrap_values = pd.DataFrame(results)

    # Remove outliers (top and bottom 10%)
    g = int(0.1 * len(bootstrap_values))
    if g > 0:
        bootstrap_values = bootstrap_values.apply(lambda x: x.sort_values().iloc[g:-g])

    return bootstrap_values


@timebudget
def bootstrap_stats(returns: pd.Series,
                    returns_benchmark: pd.Series = None,
                    rf: float = 0.02,
                    periods: int = 252,
                    n_sim: int = 1000,
                    use_gpu: bool = None) -> pd.DataFrame:
    """
    Bootstrap statistics with automatic GPU detection

    Args:
        returns: Portfolio returns
        returns_benchmark: Benchmark returns (optional)
        rf: Risk-free rate
        periods: Trading periods per year
        n_sim: Number of simulations
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)

    Returns:
        DataFrame with bootstrap statistics
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE

    # Use vectorized GPU version if available and requested
    if use_gpu and GPU_AVAILABLE:
        return bootstrap_stats_vectorized(returns, returns_benchmark, rf, periods, n_sim, use_gpu=True)

    # Fall back to original CPU version
    if not isinstance(returns_benchmark, pd.Series):
        if returns_benchmark is None:
            returns_benchmark = pd.Series(dtype='int64')

    # metrics to calculate
    metrics_functions = [qs.stats.cagr,
                         qs.stats.volatility,
                         qs.stats.sharpe,
                         qs.stats.adjusted_sortino]

    if not returns_benchmark.empty:
        metrics_functions = metrics_functions + [ep.alpha, ep.beta]

    bootstrap_values = OrderedDict()

    # prepare returns
    if not returns_benchmark.empty:
        returns, returns_benchmark = balance_dates_robust(returns,
                                                   returns_benchmark)

    for func in metrics_functions:
        stat_name = func.__name__
        sim = int(n_sim + 0.10 * n_sim)
        out: ndarray = np.zeros(sim)

        for i in range(sim):
            # draw random returns
            idx = np.random.randint(len(returns), size=len(returns))
            returns_i = returns.iloc[idx].reset_index(drop=False)
            returns_i = returns_i.set_index('Date')

            if func in (qs.stats.sharpe, qs.stats.adjusted_sortino):
                out[i] = func(returns_i, rf=rf, periods=periods)

            if func == qs.stats.volatility:
                out[i] = func(returns_i, periods=periods)

            if not returns_benchmark.empty:
                if func in (ep.beta, ep.alpha):
                    returns_bench_i = returns_benchmark.iloc[idx].reset_index(drop=False)
                    returns_bench_i = returns_bench_i.set_index('Date')

                if func == ep.alpha:
                    out[i] = func(returns=returns_i, factor_returns=returns_bench_i, risk_free=rf,
                                  annualization=periods)
                elif func == ep.beta:
                    out[i] = func(returns=returns_i,
                                  factor_returns=returns_bench_i,
                                  risk_free=rf)

            if func not in (qs.stats.volatility,
                            ep.beta, ep.alpha, qs.stats.sharpe,
                            qs.stats.adjusted_sortino):
                out[i] = func(returns_i)

        out = sorted(out)
        # number of elements to remove from both ends of list
        g = int(0.1 * len(out))
        # remove elements
        out = out[g:-g]
        bootstrap_values[stat_name] = out

    return pd.DataFrame(bootstrap_values)


@timebudget
def bootstrap_portfolio_performance_fast(
    returns: pd.Series,
    returns_benchmark: pd.Series = None,
    periods: int = 252,
    rf: float = 0.02,
    out_of_sample_date: str = None,
    n_sim: int = 10000,
    use_gpu: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fast GPU-accelerated portfolio performance bootstrapping

    Args:
        returns: Portfolio returns (Series or DataFrame)
        returns_benchmark: Benchmark returns (Series or DataFrame)
        periods: Trading periods per year
        rf: Risk-free rate
        out_of_sample_date: Date to split samples
        n_sim: Number of simulations
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (means, distributions, distribution_stats)
    """

    # Helper function to ensure Series
    def ensure_series(data):
        """Convert to Series if needed"""
        if data is None or (isinstance(data, pd.Series) and data.empty):
            return None
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0]
            else:
                return data.squeeze()
        return data

    # Clean and align data
    returns = ensure_series(returns)
    returns_benchmark = ensure_series(returns_benchmark)

    # Balance dates if benchmark exists
    if returns_benchmark is not None and not returns_benchmark.empty:
        # Convert to DataFrame temporarily for balance_dates
        ret_df = pd.DataFrame({'returns': returns})
        bench_df = pd.DataFrame({'benchmark': returns_benchmark})

        # Align
        df_combined = ret_df.join(bench_df, how='inner').dropna()

        returns = df_combined['returns']
        returns_benchmark = df_combined['benchmark']

    # Split into in-sample and out-of-sample
    if out_of_sample_date:
        # In-sample
        ret_insample = returns[returns.index < out_of_sample_date]
        ret_bench_insample = (returns_benchmark[returns_benchmark.index < out_of_sample_date]
                             if returns_benchmark is not None else None)

        print("Computing in-sample bootstrap...")
        bootstrap_metrics_insample = bootstrap_stats_vectorized(
            returns=ret_insample,
            returns_benchmark=ret_bench_insample,
            rf=rf,
            periods=periods,
            n_sim=n_sim,
            use_gpu=use_gpu
        )

        metrics_in_sample = pd.DataFrame(
            pd.Series(bootstrap_metrics_insample.mean(), name='in_sample')
        )

        # Out-of-sample
        ret_outsample = returns[returns.index >= out_of_sample_date]
        ret_bench_outsample = (returns_benchmark[returns_benchmark.index >= out_of_sample_date]
                              if returns_benchmark is not None else None)

        print("Computing out-of-sample bootstrap...")
        bootstrap_metrics_outsample = bootstrap_stats_vectorized(
            returns=ret_outsample,
            returns_benchmark=ret_bench_outsample,
            rf=rf,
            periods=periods,
            n_sim=n_sim,
            use_gpu=use_gpu
        )

        metrics_out_of_sample = pd.DataFrame(
            pd.Series(bootstrap_metrics_outsample.mean(), name='out_of_sample')
        )

        bootstrap_metrics_insample.columns = bootstrap_metrics_insample.columns + '_in_sample'
        bootstrap_metrics_outsample.columns = bootstrap_metrics_outsample.columns + '_out_sample'

        # Full sample
        print("Computing full sample bootstrap...")
        bootstrap_metrics = bootstrap_stats_vectorized(
            returns=returns,
            returns_benchmark=returns_benchmark,
            periods=periods,
            rf=rf,
            n_sim=n_sim,
            use_gpu=use_gpu
        )

        metrics_all = pd.DataFrame(pd.Series(bootstrap_metrics.mean(), name='all_sample'))

        results_means = pd.concat([metrics_in_sample, metrics_out_of_sample, metrics_all], axis=1)
        results_dist = pd.concat([bootstrap_metrics_insample, bootstrap_metrics_outsample, bootstrap_metrics], axis=1)
    else:
        # No out-of-sample split
        print("Computing bootstrap...")
        bootstrap_metrics = bootstrap_stats_vectorized(
            returns=returns,
            returns_benchmark=returns_benchmark,
            periods=periods,
            rf=rf,
            n_sim=n_sim,
            use_gpu=use_gpu
        )

        results_means = pd.DataFrame(pd.Series(bootstrap_metrics.mean(), name='all_sample'))
        results_dist = bootstrap_metrics

    # Calculate distribution statistics
    results_dist_stats = results_dist.describe().T

    return results_means, results_dist, results_dist_stats


@timebudget
def bootstrap_portfolio_performance(returns: pd.Series,
                                    returns_benchmark: pd.Series = None,
                                    periods: int = 252,
                                    rf: float = 0.02,
                                    out_of_sample_date: str = None,
                                    n_sim: int = 10000,
                                    use_gpu: bool = None) -> Tuple[pd.DataFrame,
pd.DataFrame,
pd.DataFrame]:
    """
    Portfolio performance bootstrap with automatic GPU detection

    Args:
        returns: Portfolio returns
        returns_benchmark: Benchmark returns (optional)
        periods: Trading periods per year
        rf: Risk-free rate
        out_of_sample_date: Date to split samples (optional)
        n_sim: Number of simulations
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)

    Returns:
        Tuple of (means, distributions, distribution_stats)
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE

    # Use fast GPU version if available and requested
    if use_gpu and GPU_AVAILABLE:
        return bootstrap_portfolio_performance_fast(
            returns, returns_benchmark, periods, rf, out_of_sample_date, n_sim, use_gpu=True
        )

    # Fall back to original CPU version
    if not isinstance(returns_benchmark, type(None)):
        returns, returns_benchmark = balance_dates_robust(pd.DataFrame(returns), pd.DataFrame(returns_benchmark))
        returns = pd.Series(returns.iloc[:, 0])
        returns_benchmark = pd.Series(returns_benchmark.iloc[:, 0])

    if out_of_sample_date:
        ret_insample = returns[returns.index < out_of_sample_date]
        if not isinstance(returns_benchmark, type(None)):
            ret_bench_insample = returns_benchmark[returns_benchmark.index < out_of_sample_date]
        else:
            ret_bench_insample = None

        ret_outsample = returns[returns.index >= out_of_sample_date]
        if not isinstance(returns_benchmark, type(None)):
            ret_bench_outsample = returns_benchmark[returns_benchmark.index >= out_of_sample_date]
        else:
            ret_bench_outsample = None

            # in-sample
        bootstrap_metrics_insample = bootstrap_stats(returns=ret_insample,
                                                     returns_benchmark=ret_bench_insample,
                                                     rf=rf,
                                                     periods=periods,
                                                     n_sim=n_sim,
                                                     use_gpu=False)

        metrics_in_sample = pd.DataFrame(pd.Series(bootstrap_metrics_insample.mean(),
                                                   name='in_sample'))

        # out-of-sample
        bootstrap_metrics_outsample = bootstrap_stats(returns=ret_outsample,
                                                      returns_benchmark=ret_bench_outsample,
                                                      rf=rf,
                                                      periods=periods,
                                                      n_sim=n_sim,
                                                      use_gpu=False)

        metrics_out_of_sample = pd.DataFrame(pd.Series(bootstrap_metrics_outsample.mean(),
                                                       name='out_of_sample'))

        bootstrap_metrics_insample.columns = bootstrap_metrics_insample.columns + '_in_sample'
        bootstrap_metrics_outsample.columns = bootstrap_metrics_outsample.columns + '_out_sample'

    bootstrap_metrics = bootstrap_stats(returns=returns,
                                        returns_benchmark=returns_benchmark,
                                        periods=periods,
                                        rf=rf,
                                        n_sim=n_sim,
                                        use_gpu=False)

    metrics_all = pd.DataFrame(pd.Series(bootstrap_metrics.mean(),
                                         name='all_sample'))

    if out_of_sample_date:
        results_means = pd.concat([metrics_in_sample, metrics_out_of_sample,
                                   metrics_all],
                                  axis=1)

        results_dist = pd.concat([bootstrap_metrics_insample, bootstrap_metrics_outsample,
                                  bootstrap_metrics],
                                 axis=1)
    else:
        results_means = metrics_all
        results_dist = bootstrap_metrics

    results_dist_stats = results_dist.apply(pf.tears.plotting.timeseries.calc_distribution_stats)

    return results_means, results_dist, results_dist_stats


def sim_series(returns: pd.Series, weight_period: list = None,
               n_sample: int = 1000,
               random_state: float = None) -> pd.DataFrame:
    """
    Args:
        returns (pd.Series): DESCRIPTION.
        weight_period (list, optional): DESCRIPTION. Defaults to None.
        n_sample (int, optional): DESCRIPTION. Defaults to 1000.
        random_state (float, optional): DESCRIPTION. Defaults to None.

    Returns:
        ret_sim (TYPE): DESCRIPTION.

    """

    if weight_period:

        # equal prob to period selected   
        period_length = len(returns[(returns.index >= weight_period[0]) &
                                    (returns.index <= weight_period[1])])
        period_weight = (1 / period_length) * 100

        weights = pd.Series(np.repeat(0, len(returns.index)),
                            name='weights',
                            index=returns.index)

        weights[(weights.index >= weight_period[0]) &
                (weights.index <= weight_period[1])] = period_weight
    else:
        weights = None

    ret_sim = returns.sample(frac=n_sample,
                             replace=True,
                             ignore_index=True,
                             weights=weights,
                             random_state=random_state)
    return ret_sim


def sim_paths(returns: pd.Series, out_of_sample_date: str = None,
              weight_period: list = None,
              n_sample: int = 1000, starting_value: float = 1) -> pd.DataFrame:
    """
    Optimized vectorized version for faster simulation of possible return paths.

    Args:
        returns (pd.Series): Historical returns series.
        out_of_sample_date (str, optional): Date to split in-sample and out-of-sample. Required.
        weight_period (list, optional): [start_date, end_date] to weight sampling period. Defaults to None.
        n_sample (int, optional): Number of simulation paths. Defaults to 1000.
        starting_value (int, optional): Starting value for cumulative returns. Defaults to 1.

    Raises:
        ValueError: If out_of_sample_date is not provided.

    Returns:
        pd.DataFrame: Cumulative returns for all simulated paths.

    """

    if out_of_sample_date is None:
        raise ValueError('Out of Sample Starting Date Missing')

    ret_in_sample = returns[returns.index < out_of_sample_date]
    out_of_sample_dates = returns[returns.index >= out_of_sample_date].index
    n_forecast_periods = len(out_of_sample_dates)

    # Prepare weights once (if needed)
    if weight_period:
        period_length = len(ret_in_sample[(ret_in_sample.index >= weight_period[0]) &
                                         (ret_in_sample.index <= weight_period[1])])
        period_weight = (1 / period_length) * 100
        weights = pd.Series(np.repeat(0, len(ret_in_sample.index)),
                           index=ret_in_sample.index)
        weights[(weights.index >= weight_period[0]) &
               (weights.index <= weight_period[1])] = period_weight
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
    else:
        weights = None

    # Vectorized sampling - sample all scenarios at once instead of looping
    # Total samples needed: n_sample paths * n_forecast_periods
    total_samples_needed = n_sample * n_forecast_periods
    all_samples = ret_in_sample.sample(
        n=total_samples_needed,
        replace=True,
        weights=weights
    ).values

    # Reshape to (n_sample rows, n_forecast_periods columns)
    # Each row is one simulation path, each column is one future date
    df_scenarios = pd.DataFrame(
        all_samples.reshape(n_sample, n_forecast_periods),
        columns=out_of_sample_dates,
        index=range(n_sample)
    )

    # Vectorized historical replication using NumPy repeat
    # Create matrix where each column is a copy of historical returns
    # Shape: (len(ret_in_sample), n_sample)
    ret_hist_values = np.asarray(ret_in_sample.values).flatten()  # Ensure 1D
    ret_hist_array = np.repeat(ret_hist_values[:, np.newaxis], n_sample, axis=1)
    ret_hist = pd.DataFrame(
        ret_hist_array,
        index=ret_in_sample.index,
        columns=range(n_sample)
    )

    # Combine historical and forecasted returns
    ret_possible_path = pd.concat([ret_hist, df_scenarios.T])
    ret_possible_path_cum = ep.cum_returns(ret_possible_path,
                                           starting_value=starting_value)

    return ret_possible_path_cum


def prep_dist(df: pd.DataFrame, name_perc: list = None) -> pd.DataFrame:
    """
    Optimized vectorized calculation of percentile distributions across scenarios.

    Computes percentiles across columns (scenarios) for each row (time period).
    Used to create fan chart bands from simulation results.

    Args:
        df (pd.DataFrame): DataFrame with simulated paths (rows=time, columns=scenarios).
        name_perc (list, optional): List of percentile strings (e.g., ['0.05', '0.95']).
                                   Defaults to ['0.05', '0.20', '0.35', '0.65', '0.80', '0.95'].

    Returns:
        pd.DataFrame: DataFrame with percentile columns, indexed by original df.index.

    """
    if name_perc is None:
        name_perc = ['0.05', '0.20', '0.35', '0.65', '0.80', '0.95']

    # Convert string percentiles to numeric values (0-100 scale)
    percentiles = [float(perc) * 100 for perc in name_perc]

    # Vectorized percentile calculation using NumPy
    # Single call instead of looping - computes all percentiles at once
    # axis=1 calculates percentiles across columns (scenarios) for each row (time)
    result = np.percentile(df.values, percentiles, axis=1).T

    # Return as DataFrame with original index and percentile column names
    return pd.DataFrame(result, index=df.index, columns=name_perc)


def plot_fan_chart(returns: pd.DataFrame, fcast: pd.DataFrame,
                   out_of_sample_date: str = '2020-01-01',
                   starting_value: float = 1,
                   chart_title: str = "Cumulative Returns (%)"):
    """
    

    Args:
        :param chart_title:
        :param fcast:
        :param returns:
        :param out_of_sample_date:
        :param starting_value:

        returns (pd.DataFrame): DESCRIPTION.
        fcast (pd.DataFrame): DESCRIPTION.
        out_of_sample_date (str, optional): DESCRIPTION. Defaults to '2020-01-01'.
        chart_title (str, optional): DESCRIPTION. Defaults to "Cumulative Returns (%)".

    Returns:
        TYPE: DESCRIPTION.

    """

    # cumulative returns to see the dynamics 
    ret_hist = ep.cum_returns(returns, starting_value=starting_value)

    # out of sample period (to create gray area) 
    from_forc = min(ret_hist[ret_hist.index >= out_of_sample_date].index)
    to_forc = ret_hist.index[-1]

    # This is the fan part, using 'fill_between'
    fig, ax = plt.subplots(figsize=(9, 5))
    n_bands = int(np.floor(len(fcast.columns) / 2))

    #dates_to_fill = ret_hist.index[(ret_hist.index>=from_forc)]

    for i in range(n_bands):
        # Choose alpha in a range of values
        alpha = 0.5 * (i + 1) / n_bands
        # Fill in colour between bands (ie between each 'fan')
        ax.fill_between(
            fcast.index,
            #dates_to_fill,
            fcast[fcast.columns[i]],
            fcast[fcast.columns[-i - 1]],
            color="xkcd:blue",
            alpha=alpha,
            zorder=1,
        )

    # Plot historical data
    dates = ret_hist.reset_index()['Date']
    yvalues = ret_hist

    ax.plot(dates, yvalues,
            color="black", lw=1.5, zorder=3)
    ax.axvspan(from_forc, to_forc, facecolor="grey", alpha=0.2, zorder=0)
    ax.grid(False, which="both")
    ax.set_title(chart_title, loc="left", fontsize=12)

    return plt.show()


def fan_chart(returns: pd.DataFrame,
              weight_period: list = None,
              out_of_sample_date: str = '2020-01-01',
              n_sample: int = 10000,
              starting_value: float = 1,
              chart_title: str = 'Cumulative Returns (%)'):
    """
    

    Args:
        returns (pd.DataFrame): DESCRIPTION.
        weight_period (list, optional): DESCRIPTION. Defaults to None.
        chart_start_date (str, optional): DESCRIPTION. Defaults to '2018-01-01'.
        out_of_sample_date (str, optional): DESCRIPTION. Defaults to '2020-01-01'.
        n_sample (int, optional): DESCRIPTION. Defaults to 10000.
        chart_title (str, optional): DESCRIPTION. Defaults to 'Cumulative Returns (%)'.

    Returns:
        None.

    """

    ret_possible_path_cum = sim_paths(returns,
                                      weight_period=weight_period,
                                      out_of_sample_date=out_of_sample_date,
                                      starting_value=starting_value,
                                      n_sample=n_sample)

    dist = prep_dist(ret_possible_path_cum)

    # Normalize the out-of-sample forecast to start from the last in-sample value
    dist_clone = dist.copy()
    
    # Get the in-sample data
    dist_clone_insample = dist_clone[dist_clone.index < out_of_sample_date]
    
    # Ensure there is at least one in-sample observation
    if len(dist_clone_insample) == 0:
        raise ValueError(f"No in-sample data found before {out_of_sample_date}. "
                        "Please check that the out_of_sample_date is not before the start of the returns data.")
    
    # Get the last in-sample value for each percentile band
    last_insample_values = dist_clone_insample.iloc[-1]
    
    # Get the forecast part
    dist_clone_fcast = dist_clone[dist_clone.index >= out_of_sample_date].copy()
    
    # Normalize each percentile band so it starts from the last in-sample value
    # by scaling the forecast to continue from the last in-sample point
    if len(dist_clone_fcast) > 0:
        first_forecast_values = dist_clone_fcast.iloc[0]
        for col in dist_clone_fcast.columns:
            # Handle edge case where first_forecast_values could be zero or very close to zero
            # to avoid division by zero or extreme scaling ratios
            if abs(first_forecast_values[col]) > NUMERICAL_PRECISION_THRESHOLD:
                # Calculate the ratio to normalize and scale the entire forecast series
                ratio = last_insample_values[col] / first_forecast_values[col]
                dist_clone_fcast[col] = dist_clone_fcast[col] * ratio
            else:
                # If first forecast value is near zero, use additive shift instead of scaling
                # This preserves the forecast's relative changes and dynamics while ensuring
                # the first value matches the last in-sample value
                dist_clone_fcast[col] = dist_clone_fcast[col] + last_insample_values[col]

    dist_clone_new = pd.concat([dist_clone_insample, dist_clone_fcast])

    # plot fan chart 

    plot_fan_chart(returns, dist_clone_new, out_of_sample_date=out_of_sample_date,
                   starting_value=starting_value,
                   chart_title=chart_title)

    return


# helper function 
def beating_probability_temp(returns: pd.DataFrame, returns_benchmark: pd.DataFrame,
                             weight_period: list = None, n_sample: int = 10000,
                             random_state: float = None):
    """
    

    Args:
        returns (TYPE, optional): DESCRIPTION. Defaults to None.
        returns_benchmark (TYPE, optional): DESCRIPTION. Defaults to None.
        weight_period (TYPE, optional): DESCRIPTION. Defaults to None.
        n_sample (TYPE, optional): DESCRIPTION. Defaults to 10000.
        random_state (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        prob (TYPE): DESCRIPTION.

    """

    returns = returns.dropna()
    returns_benchmark = returns_benchmark.dropna()
    returns, returns_benchmark = balance_dates(returns, returns_benchmark)

    ret = sim_series(returns,
                     weight_period=weight_period,
                     n_sample=n_sample,
                     random_state=random_state)

    ret_bench = sim_series(returns_benchmark,
                           weight_period=weight_period,
                           n_sample=n_sample,
                           random_state=random_state)

    ret = pd.Series(ret.iloc[:, 0])
    ret_bench = pd.Series(ret_bench.iloc[:, 0])
    prob = ret[ret > ret_bench].count() / len(ret)

    return prob


def beating_probability(returns: pd.DataFrame, returns_benchmark: pd.DataFrame,
                        weight_period: list = None, n_sample: int = 10000,
                        random_state: float = None) -> pd.DataFrame:
    """
    

    Args:
        returns (pd.DataFrame): DESCRIPTION.
        returns_benchmark (pd.DataFrame): DESCRIPTION.
        weight_period (list, optional): DESCRIPTION. Defaults to None.
        n_sample (int, optional): DESCRIPTION. Defaults to 10000.
        random_state (float, optional): DESCRIPTION. Defaults to None.

    Returns:
        probs (TYPE): DESCRIPTION.

    """

    # check if benchamrk is in dataframe 
    if returns.columns.isin(returns_benchmark.columns).any():
        returns = returns.drop(returns_benchmark.columns, axis=1)

    if returns.shape[1] > 1:
        prob_list = []
        for col in returns.columns:
            ret = returns[col]
            prob = list(map(lambda x: beating_probability_temp(returns=ret,
                                                               returns_benchmark=returns_benchmark,
                                                               weight_period=weight_period,
                                                               n_sample=n_sample), range(n_sample)))
            prob_list.append(round(np.mean(prob), 2))

    else:
        # loop over the function (Monte Carlo)
        prob_list = list(map(lambda x: beating_probability_temp(returns=returns,
                                                                returns_benchmark=returns_benchmark,
                                                                weight_period=weight_period,
                                                                n_sample=n_sample), range(n_sample)))

        # get the expected prob 
        prob_list = [round(np.mean(prob_list), 2)]

    probs = pd.DataFrame(prob_list)
    probs = probs.rename(columns={0: 'prob'})
    probs.index = returns.columns

    return probs


def tear_sheet_pyfolio(returns=None, returns_benchmark=None, **kwargs):
    if not isinstance(returns_benchmark, type(None)):
        returns, returns_benchmark = balance_dates(returns, returns_benchmark)
        bench = returns_benchmark.copy()
        if bench.index.tzinfo is None:
            bench.index = bench.index.tz_localize('utc')
            bench = bench.squeeze()
    else:
        bench = returns_benchmark

    # convert to pd.Series
    ret = returns.copy()

    if ret.index.tzinfo is None:
        ret.index = ret.index.tz_localize('utc')

    # convert to pd.Series
    ret = ret.squeeze()

    pf.create_returns_tear_sheet(returns=ret,
                                 benchmark_rets=bench,
                                 **kwargs)

    return


#
#
# def backtest_report(returns: pd.DataFrame,
#                     benchmark: pd.DataFrame = None,
#                     out_of_sample_date: str = None,
#                     n_sim: int = 100,
#                     rf=0.0, **kwargs):
#     print('[Performance Metrics]\n')
#     # metrics
#     df_metrics = metrics(returns=returns, benchmark=benchmark, rf=rf, **kwargs)
#     df_metrics
#     # # change name
#     # Risk-Free Rate = Risk-Free Rate (%)
#     # Cumulative Return*100
#     # CAGR﹪ * 100
#
#     iDisplay(df_metrics)
#
#     if not isinstance(benchmark, type(None)):
#         print('Bull and Bear Market correlations')
#         iDisplay(alpha_beta_table(returns, benchmark,
#                                   rf=rf, **kwargs))
#
#     print('Monte Carlo Simulations')
#     bootstrap_portfolio_performance_stats = bootstrap_portfolio_performance(returns=returns,
#                                                                             returns_benchmark=benchmark,
#                                                                             periods=252,
#                                                                             rf=rf,
#                                                                             out_of_sample_date=out_of_sample_date,
#                                                                             n_sim=n_sim)
#     if out_of_sample_date:
#
#         cols = bootstrap_portfolio_performance_stats[2].columns
#         df_temp = bootstrap_portfolio_performance_stats[2]
#
#         # get in- and out- sample columns
#         cols_in_sample = cols[cols.str.contains('in_sample', regex=False)]
#         cols_out_sample = cols[cols.str.contains('out_sample', regex=False)]
#
#         # col_full_sample = list(set(cols) - set(cols_in_sample) - set(cols_out_sample))
#
#         # stats for in- and out- sample
#         df_out_of_sample = df_temp[cols_out_sample]
#         df_in_sample = df_temp[cols_in_sample]
#
#         df_out_of_sample.columns = df_out_of_sample.columns.str.replace("_out_sample", "")
#         df_in_sample.columns = df_in_sample.columns.str.replace("_in_sample", "")
#
#         # columns of either in- or out- of sample df should now be the same
#         # with the full sample
#         df_full_sample = df_temp[df_in_sample.columns]
#
#         # add caption
#         df1 = df_in_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption('In-Sample')
#         df2 = df_out_of_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption(
#             'Out Of Sample')
#         df3 = df_full_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption('Full Sample')
#
#         iDisplay(df1)
#         iDisplay(df2)
#         iDisplay(df3)
#
#     else:
#
#         iDisplay(bootstrap_portfolio_performance_stats[2].transpose())
#
#     # performance stats according to ffn
#
#     # rets_dummy = returns.copy()
#     # if isinstance(benchmark, type(None)) == False:
#     #     rets_dummy = pd.concat([returns, benchmark], axis = 1)
#
#     # iDisplay(performance_overview(rets_dummy).transpose())
#
#     # price_index = ffn.core.to_price_index(returns, start=100)
#     # perf = price_index.calc_stats()
#     # perf[0].display_monthly_returns()
#
#     # Monthly Returns
#     print("--------------------------------------------")
#     print(" Monthly Returns (%) ")
#
#     # produce fan chart
#     if out_of_sample_date is None:
#         out_of_sample_date = '2020-01-01'
#
#     fan_chart(returns=returns,
#               weight_period=None,
#               out_of_sample_date=out_of_sample_date,
#               n_sample=n_sim,
#               chart_title='Cumulative Returns')
#
#     # Monthly Returns
#     print("--------------------------------------------")
#     print(" [Monthly Returns] \n ")
#
#     iDisplay(monthly_heatmap(returns.squeeze(), figsize=(8, 16),
#                              cbar=True, eoy=True))
#
#     # df_monthly_returns = qs.stats.monthly_returns(returns)
#     # df_monthly_returns.style.background_gradient(cmap='Blues' , cbar = True)
#
#     return


def performance(signal: pd.Series = None, returns: pd.Series = None, bps: float = 2e-4):
    tc = (signal.diff().abs()) * bps
    ret = returns * signal.shift(1) - tc
    return ret
