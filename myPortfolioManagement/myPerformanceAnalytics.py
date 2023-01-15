#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:56:13 2022

@author: safishajjouz
"""

from empyrical.stats import alpha_beta_aligned, up_alpha_beta, down_alpha_beta
import pandas as pd

import numpy as np
from myPortfolioManagement.myData import data_overview
from myPortfolioManagement.myUtils import balance_dates
from sklearn.preprocessing import minmax_scale

import ffn
from pypfopt.expected_returns import returns_from_prices

from scipy.stats import skew

import riskfolio.RiskFunctions as rk
import riskfolio as rp
import matplotlib.pyplot as plt
import quantstats as qs


def alpha_beta_table(returns, returns_benchmark, my_date_col_name=None,
                     my_assets_col_name='asset', rf=0.02,
                     period='daily'):
    """
    # TODO Alpha's and Beta's are often calculated with OLS. For a more general Treatment
          that account for outliers you can use humbert regressions

    # TODO this whole function needs redesign

    Args:
        returns (TYPE): DESCRIPTION.
        returns_benchmark (TYPE): DESCRIPTION.
        my_date_col_name (TYPE, optional): DESCRIPTION. Defaults to None.
        my_assets_col_name (TYPE, optional): DESCRIPTION. Defaults to 'asset'.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        period (TYPE, optional): DESCRIPTION. Defaults to 'daily'.

    Returns:
        TYPE: DESCRIPTION.

    """

    # Create a balanced panel 
    returns, returns_benchmark = balance_dates(returns,
                                               returns_benchmark)

    # sanity checks 
    if isinstance(returns, pd.DataFrame) == False:
        returns = pd.DataFrame(returns)

    if isinstance(returns_benchmark, pd.DataFrame) == False:
        returns_benchmark = pd.DataFrame(returns_benchmark)

    myassets = returns.columns.to_list()
    bench_col = returns_benchmark.columns.to_list()[0]
    alpha_beta = []

    for col in myassets:
        temp = alpha_beta_aligned(returns[col],
                                  returns_benchmark[bench_col],
                                  risk_free=rf, period=period)
        alpha_beta.append(temp)

    temp = pd.DataFrame(alpha_beta)
    temp[my_assets_col_name] = myassets
    temp.columns = ['alpha', 'beta', my_assets_col_name]
    temp = temp.set_index(my_assets_col_name)

    temp_bull = alpha_beta_bull(returns, returns_benchmark,
                                my_assets_col_name=my_assets_col_name,
                                period=period)
    temp_bear = alpha_beta_bear(returns, returns_benchmark,
                                my_assets_col_name=my_assets_col_name,
                                period=period)

    temp = pd.concat([temp, temp_bull, temp_bear], axis=1)

    temp = temp[['alpha', 'alpha_bull', 'alpha_bear',
                 'beta',
                 'beta_bull', 'beta_bear']]

    return round(temp, 2)


def alpha_beta_bull(returns, returns_benchmark, my_date_col_name=None,
                    my_assets_col_name='asset', rf=0.02, period='daily'):
    returns, returns_benchmark = balance_dates(returns,
                                               returns_benchmark)

    if isinstance(returns, pd.DataFrame) == False:
        returns = pd.DataFrame(returns)

    if isinstance(returns_benchmark, pd.DataFrame) == False:
        returns_benchmark = pd.DataFrame(returns_benchmark)

    myassets = returns.columns.to_list()
    bench_col = returns_benchmark.columns.to_list()[0]

    alpha_beta = []

    for col in myassets:
        temp = up_alpha_beta(returns[col],
                             returns_benchmark[bench_col],
                             risk_free=rf, period=period)
        alpha_beta.append(temp)

    temp = pd.DataFrame(alpha_beta)
    temp[my_assets_col_name] = myassets
    temp.columns = ['alpha_bull', 'beta_bull', my_assets_col_name]
    temp = temp.set_index(my_assets_col_name)

    return round(temp, 2)


def alpha_beta_bear(returns, returns_benchmark, my_date_col_name=None,
                    my_assets_col_name='asset', rf=0.02, period='daily'):
    returns, returns_benchmark = balance_dates(returns,
                                               returns_benchmark)

    if isinstance(returns, pd.DataFrame) == False:
        returns = pd.DataFrame(returns)

    if isinstance(returns_benchmark, pd.DataFrame) == False:
        returns_benchmark = pd.DataFrame(returns_benchmark)

    # myassets = returns.drop([benchmark], axis = 1).columns.to_list()
    myassets = returns.columns.to_list()
    bench_col = returns_benchmark.columns.to_list()[0]

    alpha_beta = []

    for col in myassets:
        temp = down_alpha_beta(returns[col],
                               returns_benchmark[bench_col],
                               risk_free=rf, period=period)
        alpha_beta.append(temp)

    temp = pd.DataFrame(alpha_beta)
    temp[my_assets_col_name] = myassets
    temp.columns = ['alpha_bear', 'beta_bear', my_assets_col_name]
    temp = temp.set_index(my_assets_col_name)

    return round(temp, 2)


def performance_overview(df, prices=False, short=True):
    '''
    df: a wide dataframe of either returns or prices  
    '''

    if prices:
        df = returns_from_prices(df)

    columns = df.columns.to_list()

    df_stats = []
    for col in columns:
        ret_temp = df[[col]].dropna()
        price_index = ffn.core.to_price_index(ret_temp, start=100)
        df_stats_temp = ffn.core.GroupStats(price_index).stats
        df_stats_temp = df_stats_temp.transpose()
        df_stats.append(df_stats_temp)

    df_perf = pd.concat(df_stats)

    if short:
        keep = ['start', 'end', 'rf', 'total_return', 'cagr', 'max_drawdown', 'yearly_sharpe', 'yearly_sortino',
                'calmar']

        df_perf = df_perf[keep]

    return df_perf


##############################################################################
def returns_centered(returns):
    return returns - returns.mean()


def centered_moment(Rb, power=2):
    out = returns_centered(Rb) ** power
    out = out.mean()
    return out


def centered_co_moments(returns, returns_benchmark, p1=1, p2=2,
                        normalize=True):
    out = (returns_centered(returns) ** p1) * (returns_centered(returns_benchmark) ** p2)
    out = out.mean()

    if normalize:
        out = out / (centered_moment(returns_benchmark, power=(p1 + p2)))

    return out


def beta_Co_Moments(returns, returns_benchmark, p1=1, p2=2):
    '''
    returns(df): a dataframes with returns 
    returns_benchmark: A pandas series of the benchmark returns 
    '''
    # sanity checks 
    if isinstance(returns_benchmark, pd.DataFrame):
        raise ValueError('''you passed a Pandas DataFrame to benchmark Returns.
                          You need too pass pandas series''')

    skewness = skew(returns_benchmark)

    if (skewness > -0.05 and skewness < 0.05):
        raise Warning('''skewness is close to zero. 
                           The classical definition of the coskewness statistic 
                           is not applicable and one should normalize using 
                           the comoment without standardization''')

    temp = [centered_co_moments(returns=returns[x],
                                returns_benchmark=returns_benchmark, p1=p1, p2=p2)
            for x in returns.columns]
    if p2 == 1:
        beta_name = 'BetaCoVariance'
    elif p2 == 3:
        beta_name = 'BetaCoKurtosis'
    elif p2 == 2:
        beta_name = 'BetaCoSkewness'

    temp = pd.Series(temp, name=beta_name)
    tem_names = pd.Series(returns.columns, name='asset')
    return pd.concat([tem_names, temp], axis=1)


def beta_Co_Moments_table(returns, returns_benchmark):
    # beta CoVariance
    beta_co_var = beta_Co_Moments(returns, returns_benchmark, p2=1)

    # beta CoSkeweness
    beta_co_sk = beta_Co_Moments(returns, returns_benchmark, p2=2)

    # beta CoKurtosis
    beta_co_kr = beta_Co_Moments(returns, returns_benchmark, p2=3)

    beta_co_moments_table = pd.concat([beta_co_var.set_index('asset'),
                                       beta_co_sk.set_index('asset'),
                                       beta_co_kr.set_index('asset')],
                                      axis=1)

    return beta_co_moments_table


def ranking_metrics(df, rolling_window=3,
                    rolling_frequency='Y',
                    benchmark_name=None,
                    my_assets_col_name=None,
                    my_date_col_name=None,
                    price_col_name=None):
    df_const = consistency_score(df, rolling_window=rolling_window,
                                 rolling_frequency=rolling_frequency,
                                 benchmark_name=benchmark_name,
                                 my_assets_col_name=my_assets_col_name,
                                 my_date_col_name=my_date_col_name,
                                 price_col_name=price_col_name)

    df_reward_metric = reward_metric(df, rolling_window=rolling_window,
                                     rolling_frequency=rolling_frequency,
                                     benchmark_name=benchmark_name,
                                     my_assets_col_name=my_assets_col_name,
                                     my_date_col_name=my_date_col_name,
                                     price_col_name=price_col_name)

    df_reward_metrics = df_const.merge(df_reward_metric, on=my_assets_col_name)

    colNames_reward = df_reward_metric.columns[df_reward_metric.columns.str.contains(pat='reward')][0]
    colNames_cons_score = df_const.columns[df_const.columns.str.contains(pat='cons')][0]
    colNames_cons_alpha = df_const.columns[df_const.columns.str.contains(pat='alpha')][0]

    df_reward_metrics['overall_score'] = df_reward_metrics[colNames_cons_score] + df_reward_metrics[
        colNames_cons_alpha] + df_reward_metrics[colNames_reward]
    # df_reward_metrics = df_reward_metrics.drop(benchmark_name)

    return df_reward_metrics


def consistency_score(df, rolling_window=3,
                      rolling_frequency='Y',
                      benchmark_name=None,
                      my_assets_col_name=None,
                      my_date_col_name=None,
                      price_col_name=None):
    df = df[[my_date_col_name, my_assets_col_name, price_col_name]]

    # from long to wide 
    df = df.pivot_table(index=my_date_col_name,
                        columns=my_assets_col_name,
                        values=price_col_name)

    # calculate rolling mean 
    df_rolling = df.resample(rolling_frequency).mean().pct_change().rolling(rolling_window).mean()

    # check how many times a fund beat the index 
    df_index = df_rolling[[benchmark_name]]

    df_index = pd.melt(df_index.reset_index(), id_vars=my_date_col_name)

    df_temp = pd.melt(df_rolling.reset_index(), id_vars=my_date_col_name, value_name='rolling_returns')

    df_index = df_index.merge(df_temp, on=my_date_col_name)
    df_index['beat_index'] = df_index['rolling_returns'] - df_index['value']

    col_alpha = 'average_alpha_' + 'rolling_' + str(rolling_window) + rolling_frequency
    df_index[col_alpha] = df_index['rolling_returns'] - df_index['value']

    df_index = df_index.drop([my_assets_col_name + '_x', 'value'], axis=1)
    df_index = df_index.sort_values([my_assets_col_name + '_y', 'Date']).dropna()

    df_index = df_index.rename(columns={my_assets_col_name + '_y': my_assets_col_name})

    df_index['beat_index_sign'] = np.sign(df_index.beat_index)

    # df_index = df_index[df_index.fund != benchmark_name]

    # calcuate consistency score 
    df_consistency_score = df_index.groupby(my_assets_col_name).beat_index_sign.value_counts().unstack()

    df_consistency_score['total_number'] = df_consistency_score.sum(axis=1)

    col_name = 'consistency_score_' + 'rolling_' + str(rolling_window) + rolling_frequency

    df_consistency_score[col_name] = df_consistency_score[1] / df_consistency_score['total_number']
    df_consistency_score = df_consistency_score.fillna(0)

    # outputing results 
    df_index = df_index[[my_assets_col_name, col_alpha]].groupby(my_assets_col_name).mean()
    df_consistency_score = df_consistency_score.merge(df_index, on=my_assets_col_name)

    return df_consistency_score[[col_name, col_alpha]].sort_values(col_alpha, ascending=False)


def consistency_rank(df, my_period_weights=[0, 0.8, 0, 0.2],
                     benchmark_name=None,
                     my_assets_col_name=None,
                     my_date_col_name=None,
                     price_col_name=None):
    if sum(my_period_weights) > 1 or sum(my_period_weights) < 1:
        raise 'weights inconsistent. Must add up to 1'

    # loop over different horizons 
    df_const = []
    years = [1, 3, 5, 10]
    for i in years:
        temp = consistency_score(df, rolling_window=i,
                                 rolling_frequency='Y',
                                 benchmark_name=benchmark_name,
                                 my_assets_col_name=my_assets_col_name,
                                 my_date_col_name=my_date_col_name,
                                 price_col_name=price_col_name)
        df_const.append(temp)

    df_const = pd.concat(df_const, axis=1)

    colNames = df_const.columns[df_const.columns.str.contains(pat='alpha')]

    # calculate weighted average

    df_const_alpha = df_const[colNames]

    weight_alpha = pd.DataFrame(pd.Series(my_period_weights, index=df_const_alpha.columns.to_list(), name=0))
    df_const_alpha['weighted_sum_alpha'] = df_const_alpha.fillna(0).dot(weight_alpha) / len(years)

    df_const_returns = df_const.drop(colNames, axis=1)
    weight_returns = pd.DataFrame(pd.Series(my_period_weights, index=df_const_returns.columns.to_list(), name=0))
    df_const_returns['weighted_sum_returns'] = df_const_returns.fillna(0).dot(weight_returns) / len(years)

    df_const_returns = df_const_returns.merge(df_const_alpha, on=my_assets_col_name)
    df_const_returns = df_const_returns[['weighted_sum_returns', 'weighted_sum_alpha']]

    df_const_returns['weighted_sum'] = df_const_returns['weighted_sum_returns'] + df_const_returns['weighted_sum_alpha']

    df_const = df_const_returns['weighted_sum']

    # get trading days to adjust 
    df_data_overview = data_overview(df,
                                     my_assets_col_name=my_assets_col_name,
                                     my_date_col_name=my_date_col_name,
                                     price_col_name=price_col_name)

    df_const = df_data_overview.merge(df_const.reset_index())

    df_const = df_const[[my_assets_col_name, 'weighted_sum', 'years_available']]
    df_const['consistency_rank'] = minmax_scale(df_const['weighted_sum'] * df_const['years_available'])
    df_const = df_const[[my_assets_col_name, 'consistency_rank']].sort_values('consistency_rank', ascending=False)
    return df_const


def reward_metric(df, rolling_window=3,
                  rolling_frequency='Y',
                  my_assets_col_name=None,
                  my_date_col_name=None,
                  price_col_name=None,
                  benchmark_name=None):
    '''
    df_temp2 = consistency_rank(df, my_period_weights = my_period_weights, 
                                         my_assets_col_name = my_assets_col_name, 
                                         my_date_col_name = my_date_col_name,
                                         price_col_name = price_col_name,
                                         benchmark_name = benchmark_name)
    '''

    ## trading weeks 
    # get trading days to adjust 
    df_data_overview = data_overview(df,
                                     my_assets_col_name=my_assets_col_name,
                                     my_date_col_name=my_date_col_name,
                                     price_col_name=price_col_name)

    # from long to wide 
    df = df.pivot_table(index=my_date_col_name,
                        columns=my_assets_col_name,
                        values=price_col_name)

    # calculate rolling mean 
    df_rolling_returns = df.resample(rolling_frequency).mean().pct_change().rolling(rolling_window).mean()

    '''
    # adjust returns 
    if risk_measure == 'smv':
        d = {'adjusted_returns': df_rolling_returns.mean()/df_rolling_returns[df_rolling_returns<0].std()}
    elif risk_measure == 'std':
        d = {'adjusted_returns': df_rolling_returns.mean()/df_rolling_returns.std()}
    elif risk_measure == 'max_drowdown':
        qs.extend_pandas()
        d = {'adjusted_returns': df_rolling_returns.mean()/-df_rolling_returns.max_drawdown()}  
    '''

    d = {'adjusted_returns': df_rolling_returns.mean() / df_rolling_returns.std()}
    df_temp = pd.DataFrame(d)

    df_temp = df_data_overview.reset_index().merge(df_temp.reset_index())
    df_temp = df_temp[[my_assets_col_name, 'adjusted_returns', 'years_available']]

    # df_temp = df_temp.merge(df_temp2, on = my_assets_col_name)

    col_name = 'reward_metric_' + str(rolling_window) + rolling_frequency

    df_temp[col_name] = minmax_scale(df_temp['adjusted_returns'] *
                                     df_temp['years_available'],
                                     feature_range=(0, 1),
                                     axis=0, copy=True)

    df_temp = df_temp.sort_values(col_name, ascending=False).drop_duplicates()
    df_temp = df_temp[[my_assets_col_name, col_name]]

    return df_temp


def risk_contributions(port_weights=None, returns=None,
                       risk_measure='MV',
                       plot=True):
    '''
    port_weights (pd.DataFrame): dataframe of asset weights. Assets are the index 
    returns (pd.DataFrame): Returns of your assets. Date is an index 
    risk_measure (str): string that defines the risk measure: MV for Variance and MSV for semi-Variance 
    plot(boolonean): True to plot the risk contributions, otherwise returns just risk contributions 
    
    retunrs:
        a plot of risk contributions or an array of risk contributions 
    '''

    cov = returns.cov()
    if plot:
        ax = rp.plot_risk_con(port_weights, cov=cov, returns=returns, rm=risk_measure,
                              color="tab:blue", height=6, width=10, ax=None)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        return ax
    else:
        risk_cont = rk.Risk_Contribution(port_weights, cov=cov, returns=returns,
                                         rm=risk_measure)
        return np.round(risk_cont, 3)
