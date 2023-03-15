"""
Created on Sat Jan 22 19:23:54 2022

@author: safishajjouz
"""
from collections import OrderedDict
from typing import Tuple

import empyrical as ep
import pandas as pd
import pyfolio as pf
from IPython.core.display import display as iDisplay
from numpy import ndarray
from timebudget import timebudget

from myPortfolioManagement.myPerformanceAnalytics import *
from myPortfolioManagement.myPlots import *
from myPortfolioManagement.myReports import metrics
from myPortfolioManagement.myReturns import average_returns


@timebudget
def bootstrap_stats(returns: pd.Series,
                    returns_benchmark: pd.Series = None,
                    rf: float = 0.02,
                    periods: int = 252,
                    n_sim: int = 10000) -> pd.DataFrame:
    """
    

    Args:
        returns (pd.Series): DESCRIPTION.
        returns_benchmark (pd.Series, optional): DESCRIPTION. Defaults to None.
        rf (float, optional): DESCRIPTION. Defaults to 0.02.
        periods (int, optional): DESCRIPTION. Defaults to 252.
        n_sim (int, optional): DESCRIPTION. Defaults to 10000.

    Returns:
        TYPE: DESCRIPTION.

    """

    if not isinstance(returns_benchmark, pd.Series):
        if returns_benchmark is None:
            returns_benchmark = pd.Series(dtype='int64')

    # metrics to calculate 
    metrics_functions = [
        # qs.stats.cagr,
        # qs.stats.geometric_mean,
        average_returns,
        qs.stats.volatility,
        qs.stats.sharpe,
        qs.stats.sortino,
        qs.stats.skew,
        # qs.stats.calmar,
        qs.stats.comp]

    if not returns_benchmark.empty:
        metrics_functions = metrics_functions + [ep.alpha, ep.beta]

    bootstrap_values = OrderedDict()

    # prepare returns 
    if not returns_benchmark.empty:
        returns, returns_benchmark = balance_dates(returns,
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

            if func in (average_returns, qs.stats.sharpe, qs.stats.sortino):
                out[i] = func(returns_i, rf=rf, periods=periods)

            if func == qs.stats.volatility:
                out[i] = func(returns_i, periods=periods)

            if not returns_benchmark.empty:
                if func in (ep.beta, ep.alpha):
                    returns_bench_i = returns_benchmark.iloc[idx].reset_index(drop=False)
                    returns_bench_i = returns_bench_i.set_index('Date')

                if func == ep.alpha:
                    out[i] = func(returns=returns_i, factor_returns=returns_bench_i,
                                  risk_free=rf,
                                  annualization=periods)
                elif func == ep.beta:
                    out[i] = func(returns=returns_i,
                                  factor_returns=returns_bench_i,
                                  risk_free=rf)

            if func not in (average_returns, qs.stats.volatility,
                            ep.beta, ep.alpha, qs.stats.sharpe,
                            qs.stats.sortino):
                out[i] = func(returns_i)

        out = sorted(out)
        # number of elements to remove from both ends of list
        g = int(0.1 * len(out))
        # remove elements
        out = out[g:-g]
        bootstrap_values[stat_name] = out

    return pd.DataFrame(bootstrap_values)


@timebudget
def bootstrap_portfolio_performance(returns: pd.Series,
                                    returns_benchmark: pd.Series = None,
                                    periods: int = 252,
                                    rf: float = 0.02,
                                    out_of_sample_date: str = None,
                                    n_sim: int = 10000) -> Tuple[pd.DataFrame,
pd.DataFrame,
pd.DataFrame]:
    """
    

    Args:
        returns (pd.Series): DESCRIPTION.
        returns_benchmark (pd.Series, optional): DESCRIPTION. Defaults to None.
        periods (int, optional): DESCRIPTION. Defaults to 252.
        rf (float, optional): DESCRIPTION. Defaults to 0.02.
        out_of_sample_date (str, optional): DESCRIPTION. Defaults to None.
        n_sim (int, optional): DESCRIPTION. Defaults to 10000.

    Returns:
        results_means (TYPE): DESCRIPTION.
        results_dist (TYPE): DESCRIPTION.
        results_dist_stats (TYPE): DESCRIPTION.

    """

    if not isinstance(returns_benchmark, type(None)):
        returns, returns_benchmark = balance_dates(pd.DataFrame(returns), pd.DataFrame(returns_benchmark))
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
                                                     n_sim=n_sim)

        metrics_in_sample = pd.DataFrame(pd.Series(bootstrap_metrics_insample.mean(),
                                                   name='in_sample'))

        # out-of-sample 
        bootstrap_metrics_outsample = bootstrap_stats(returns=ret_outsample,
                                                      returns_benchmark=ret_bench_outsample,
                                                      rf=rf,
                                                      periods=periods,
                                                      n_sim=n_sim)

        metrics_out_of_sample = pd.DataFrame(pd.Series(bootstrap_metrics_outsample.mean(),
                                                       name='out_of_sample'))

        bootstrap_metrics_insample.columns = bootstrap_metrics_insample.columns + '_in_sample'
        bootstrap_metrics_outsample.columns = bootstrap_metrics_outsample.columns + '_out_sample'

    bootstrap_metrics = bootstrap_stats(returns=returns,
                                        returns_benchmark=returns_benchmark,
                                        periods=periods,
                                        rf=rf,
                                        n_sim=n_sim)

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

    ret_sim = returns.sample(n=n_sample,
                             replace=True,
                             ignore_index=True,
                             weights=weights,
                             random_state=random_state)
    return ret_sim


def sim_paths(returns: pd.Series, out_of_sample_date: str = None,
              weight_period: list = None,
              n_sample: int = 1000, starting_value: float = 1) -> pd.DataFrame:
    """
    

    Args:
        returns (pd.Series): DESCRIPTION.
        out_of_sample_start_date (str, optional): DESCRIPTION. Defaults to None.
        weight_period (list, optional): DESCRIPTION. Defaults to None.
        n_sample (int, optional): DESCRIPTION. Defaults to 1000.
        starting_value (int, optional): DESCRIPTION. Defaults to 1.

    Raises:
        ValueError: DESCRIPTION.

    Returns:
        ret_possible_path_cum (TYPE): DESCRIPTION.

    """

    if out_of_sample_date is None:
        raise ValueError('Out of Sample Starting Date Missing')

    ret_in_sample = returns[returns.index < out_of_sample_date]

    out_of_sample_dates = returns[returns.index >= out_of_sample_date].index

    # dataframe where for each date has the possinle
    df_scenarios = pd.DataFrame(columns=out_of_sample_dates,
                                index=range(n_sample))

    for col in df_scenarios.columns:
        df_scenarios[col] = sim_series(ret_in_sample,
                                       weight_period=weight_period,
                                       n_sample=n_sample)

    ret_hist = pd.concat([ret_in_sample] * (n_sample), axis=1, ignore_index=True)
    ret_possible_path = pd.concat([ret_hist, df_scenarios.T])
    ret_possible_path_cum = ep.cum_returns(ret_possible_path,
                                           starting_value=starting_value)

    return ret_possible_path_cum


def prep_dist(x: pd.DataFrame, name_perc: list = None) -> pd.DataFrame:
    """
    

    Args:
        x (TYPE): DESCRIPTION.
        name_perc (TYPE, optional): DESCRIPTION. Defaults to ['0.05', '0.20','0.35','0.65','0.80', '0.95'].

    Returns:
        x (TYPE): DESCRIPTION.

    """

    if name_perc is None:
        name_perc = ['0.05', '0.20', '0.35', '0.65', '0.80', '0.95']
    for perc in name_perc:
        x[perc] = np.percentile(x, float(perc) * 100, axis=1)

    # x['0.05'] =  np.percentile(x, 5, axis = 1)
    # x['0.20'] =  np.percentile(x, 20, axis = 1)
    # x['0.35'] =  np.percentile(x, 35, axis = 1)
    # x['0.65'] =  np.percentile(x, 65, axis = 1)
    # x['0.80'] =  np.percentile(x, 80, axis = 1)
    # x['0.95'] =  np.percentile(x, 95, axis = 1)

    x = x[name_perc]
    # x = x[x.index>='2020-01-01']
    return x


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

    # dates_to_fill = ret_hist.index[(ret_hist.index>=from_forc)]

    for i in range(n_bands):
        # Choose alpha in a range of values
        alpha = 0.5 * (i + 1) / n_bands
        # Fill in colour between bands (ie between each 'fan')
        ax.fill_between(
            fcast.index,
            # dates_to_fill,
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

    # replaces the dist
    dist_clone = dist.copy()

    dist_clone_fcast = dist_clone[dist_clone.index >= out_of_sample_date]

    for col in dist_clone_fcast.columns[0:3]:
        dist_clone_fcast[col] = min(dist_clone_fcast[col])

    for col in dist_clone_fcast.columns[3:6]:
        dist_clone_fcast[col] = max(dist_clone_fcast[col])

    dist_clone_insample = dist_clone[dist_clone.index < out_of_sample_date]

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


def backtest_report(returns: pd.DataFrame,
                    benchmark: pd.DataFrame = None,
                    out_of_sample_date: str = None,
                    n_sim: int = 100,
                    rf=0.0, **kwargs):
    print('[Performance Metrics]\n')
    # metrics 
    df_metrics = metrics(returns=returns, benchmark=benchmark, rf=rf, **kwargs)
    df_metrics
    # # change name 
    # Risk-Free Rate = Risk-Free Rate (%)
    # Cumulative Return*100 
    # CAGR﹪ * 100 

    iDisplay(df_metrics)

    if not isinstance(benchmark, type(None)):
        print('Bull and Bear Market correlations')
        iDisplay(alpha_beta_table(returns, benchmark,
                                  rf=rf, **kwargs))

    print('Monte Carlo Simulations')
    bootstrap_portfolio_performance_stats = bootstrap_portfolio_performance(returns=returns,
                                                                            returns_benchmark=benchmark,
                                                                            periods=252,
                                                                            rf=rf,
                                                                            out_of_sample_date=out_of_sample_date,
                                                                            n_sim=n_sim)
    if out_of_sample_date:

        cols = bootstrap_portfolio_performance_stats[2].columns
        df_temp = bootstrap_portfolio_performance_stats[2]

        # get in- and out- sample columns 
        cols_in_sample = cols[cols.str.contains('in_sample', regex=False)]
        cols_out_sample = cols[cols.str.contains('out_sample', regex=False)]

        # col_full_sample = list(set(cols) - set(cols_in_sample) - set(cols_out_sample))

        # stats for in- and out- sample 
        df_out_of_sample = df_temp[cols_out_sample]
        df_in_sample = df_temp[cols_in_sample]

        df_out_of_sample.columns = df_out_of_sample.columns.str.replace("_out_sample", "")
        df_in_sample.columns = df_in_sample.columns.str.replace("_in_sample", "")

        # columns of either in- or out- of sample df should now be the same
        # with the full sample 
        df_full_sample = df_temp[df_in_sample.columns]

        # add caption 
        df1 = df_in_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption('In-Sample')
        df2 = df_out_of_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption(
            'Out Of Sample')
        df3 = df_full_sample.transpose().style.set_table_attributes("style='display:inline'").set_caption('Full Sample')

        iDisplay(df1)
        iDisplay(df2)
        iDisplay(df3)

    else:

        iDisplay(bootstrap_portfolio_performance_stats[2].transpose())

    # performance stats according to ffn 

    # rets_dummy = returns.copy() 
    # if isinstance(benchmark, type(None)) == False:
    #     rets_dummy = pd.concat([returns, benchmark], axis = 1)

    # iDisplay(performance_overview(rets_dummy).transpose())

    # price_index = ffn.core.to_price_index(returns, start=100) 
    # perf = price_index.calc_stats()
    # perf[0].display_monthly_returns()

    # Monthly Returns 
    print("--------------------------------------------")
    print(" Monthly Returns (%) ")

    # produce fan chart 
    if out_of_sample_date is None:
        out_of_sample_date = '2020-01-01'

    fan_chart(returns=returns,
              weight_period=None,
              out_of_sample_date=out_of_sample_date,
              n_sample=n_sim,
              chart_title='Cumulative Returns')

    # Monthly Returns 
    print("--------------------------------------------")
    print(" [Monthly Returns] \n ")

    iDisplay(monthly_heatmap(returns.squeeze(), figsize=(8, 16),
                             cbar=True, eoy=True))

    # df_monthly_returns = qs.stats.monthly_returns(returns)
    # df_monthly_returns.style.background_gradient(cmap='Blues' , cbar = True)

    return


def performance(signal: pd.Series = None, returns: pd.Series = None, bps: float = 2e-4):
    tc = (signal.diff().abs()) * bps
    ret = returns * signal.shift(1) - tc
    return ret
