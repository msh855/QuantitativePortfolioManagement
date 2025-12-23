"""
GPU-Accelerated Backtesting Module
Optimized for parallel Monte Carlo simulations
"""
import numpy as np
import pandas as pd
import empyrical as ep
import quantstats_lumi as qs
from collections import OrderedDict
from typing import Tuple
from joblib import Parallel, delayed
import multiprocessing
from timebudget import timebudget

# Try GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

from myPortfolioManagement.myUtils import balance_dates_robust


@timebudget
def bootstrap_stats_vectorized(returns: pd.Series,
                               returns_benchmark: pd.Series = None,
                               rf: float = 0.02,
                               periods: int = 252,
                               n_sim: int = 1000,
                               use_gpu: bool = True) -> pd.DataFrame:
    """
    Vectorized bootstrap statistics calculation
    
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
