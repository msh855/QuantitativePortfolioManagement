#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 07:29:35 2021

@author: safishajjouz
"""

import riskfolio as rp
import pandas as pd
import ffn
import quantstats as qs

# for portfolio optimisation
from pypfopt import risk_models
from pypfopt import EfficientFrontier

from pypfopt import objective_functions
from pypfopt import EfficientCVaR

import ray  # to parallelise
from timebudget import timebudget  # to time functions
import itertools
from myPortfolioManagement.myReturns import average_returns


# Hierarchical Risk Parity Default option
def HRP(model='HRP',
        returns_training=None,
        covariance="hist",
        codependence='pearson',
        rm='MV',
        linkage='single',
        weight_max=None,
        weight_min=None,
        leaf_order=False, **kwargs):
    '''

    model : str, can be {'HRP', 'HERC' or 'HERC2'}
        The hierarchical cluster portfolio model used for optimize the
        portfolio. The default is 'HRP'. Posible values are:

        - 'HRP': Hierarchical Risk Parity.
        - 'HERC': Hierarchical Equal Risk Contribution.
        - 'HERC2': HERC but splitting weights equally within clusters.
        - 'NCO': Nested Clustered Optimization.

    codependence : str, optional
        The codependence or similarity matrix used to build the distance
        metric and clusters. The default is 'pearson'. Posible values are:

        - 'pearson': pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{pearson}_{i,j})}`.
        - 'spearman': spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{spearman}_{i,j})}`.
        - 'abs_pearson': absolute value pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho^{pearson}_{i,j}|)}`.
        - 'abs_spearman': absolute value spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho^{spearman}_{i,j}|)}`.
        - 'distance': distance correlation matrix. Distance formula :math:`D_{i,j} = \sqrt{(1-\rho^{distance}_{i,j})}`.
        - 'mutual_info': mutual information matrix. Distance used is variation information matrix.
        - 'tail': lower tail dependence index matrix. Dissimilarity formula :math:`D_{i,j} = -\log{\lambda_{i,j}}`.
        - 'custom_cov': use custom correlation matrix based on the custom_cov parameter. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{pearson}_{i,j})}`.

    covariance : str, optional
        The method used to estimate the covariance matrix:
        The default is 'hist'. Posible values are:

        - 'hist': use historical estimates.
        - 'ewma1'': use ewma with adjust=True, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
        - 'ewma2': use ewma with adjust=False, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
        - 'ledoit': use the Ledoit and Wolf Shrinkage method.
        - 'oas': use the Oracle Approximation Shrinkage method.
        - 'shrunk': use the basic Shrunk Covariance method.
        - 'gl': use the basic Graphical Lasso Covariance method.
        - 'jlogo': use the j-LoGo Covariance method. For more information see: :cite:`c-jLogo`.
        - 'fixed': denoise using fixed method. For more information see chapter 2 of :cite:`c-MLforAM`.
        - 'spectral': denoise using spectral method. For more information see chapter 2 of :cite:`c-MLforAM`.
        - 'shrink': denoise using shrink method. For more information see chapter 2 of :cite:`c-MLforAM`.
        - 'custom_cov': use custom covariance matrix.

    obj : str can be {'MinRisk', 'Utility', 'Sharpe' or 'ERC'}.
        Objective function used by the NCO model.
        The default is 'MinRisk'. Posible values are:

        - 'MinRisk': Minimize the selected risk measure.
        - 'Utility': Maximize the Utility function :math:`\mu w - l \phi_{i}(w)`.
        - 'Sharpe': Maximize the risk adjusted return ratio based on the selected risk measure.
        - 'ERC': Equally risk contribution portfolio of the selected risk measure.

    rm : str, optional
        The risk measure used to optimze the portfolio. If model is 'NCO',
        the risk measures available depends on the objective functon.
        The default is 'MV'. Posible values are:

        - 'equal': Equally weighted.
        - 'vol': Standard Deviation.
        - 'MV': Variance.
        - 'MAD': Mean Absolute Deviation.
        - 'MSV': Semi Standard Deviation.
        - 'FLPM': First Lower Partial Moment (Omega Ratio).
        - 'SLPM': Second Lower Partial Moment (Sortino Ratio).
        - 'VaR': Value at Risk.
        - 'CVaR': Conditional Value at Risk.
        - 'TG': Tail Gini.
        - 'EVaR': Entropic Value at Risk.
        - 'WR': Worst Realization (Minimax).
        - 'RG': Range of returns.
        - 'CVRG': CVaR range of returns.
        - 'TGRG': Tail Gini range of returns.
        - 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
        - 'ADD': Average Drawdown of uncompounded cumulative returns.
        - 'DaR': Drawdown at Risk of uncompounded cumulative returns.
        - 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
        - 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
        - 'UCI': Ulcer Index of uncompounded cumulative returns.
        - 'MDD_Rel': Maximum Drawdown of compounded cumulative returns (Calmar Ratio).
        - 'ADD_Rel': Average Drawdown of compounded cumulative returns.
        - 'DaR_Rel': Drawdown at Risk of compounded cumulative returns.
        - 'CDaR_Rel': Conditional Drawdown at Risk of compounded cumulative returns.
        - 'EDaR_Rel': Entropic Drawdown at Risk of compounded cumulative returns.
        - 'UCI_Rel': Ulcer Index of compounded cumulative returns.

    rf : float, optional
        Risk free rate, must be in the same period of assets returns.
        The default is 0.

    custom_cov : DataFrame or None, optional
        Custom covariance matrix, used when codependence or covariance
        parameters have value 'custom_cov'. The default is None.
    custom_mu : DataFrame or None, optional
        Custom mean vector when NCO objective is 'Utility' or 'Sharpe'.
        The default is None.
    linkage : string, optional
        Linkage method of hierarchical clustering, see `linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage>`_ for more details.
        The default is 'single'. Posible values are:

        - 'single'.
        - 'complete'.
        - 'average'.
        - 'weighted'.
        - 'centroid'.
        - 'median'.
        - 'ward'.
        - 'DBHT': Direct Bubble Hierarchical Tree.

    leaf_order : bool, optional
        Indicates if the cluster are ordered so that the distance between

    **kwargs:
        Other variables related to portfolio optimisation. See Risk-Folio lib

    Returns
    -------
    w : DataFrame
        The weights of optimal portfolio.

    '''

    # sanity checks
    if not isinstance(returns_training, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    port = rp.HCPortfolio(returns=returns_training)

    if weight_max is not None or weight_min is not None:
        # impose constraints
        asset_classes = {'Assets': returns_training.columns.to_list()}
        asset_classes = pd.DataFrame(asset_classes)
        asset_classes = asset_classes.sort_values(by=['Assets'])

        constraints = {'Disabled': [False, False],
                       'Type': ['All Assets', 'All Assets'],
                       'Set': ['', ''],
                       'Position': ['', ''],
                       'Sign': ['<=', '>='],
                       'Weight': [weight_max, weight_min]}

        constraints = pd.DataFrame(constraints)

        w_max, w_min = rp.hrp_constraints(constraints, asset_classes)
        port.w_max = w_max
        port.w_min = w_min

    # HRP Default
    weights = port.optimization(model=model,
                                codependence=codependence,
                                covariance=covariance,
                                rm=rm,
                                linkage=linkage,
                                leaf_order=leaf_order,
                                **kwargs)

    weights = weights.rename(columns={'weights': 'port_weight'})

    if model != 'HRP' and weight_max is not None:
        temp = clean_limit_weights(weights[['port_weight']],
                                   portfolio_name='port_weight',
                                   weight_max=weight_max)
        weights[['port_weight']] = temp

    return weights


def generate_rp_portfolios(returns_training=None,
                           rf=0.02, risk_measure=[],
                           weight_max=[]):
    """
    Args:
        returns_training (TYPE, optional): DESCRIPTION. Defaults to None.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        risk_measure (TYPE, optional): DESCRIPTION. Defaults to [].
        weight_max (TYPE, optional): DESCRIPTION. Defaults to [].

    Returns:
        df_rp (TYPE): DESCRIPTION.

    """

    # tutorial
    # https://nbviewer.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2032.ipynb

    # Building the portfolio object
    portRP = rp.Portfolio(returns=returns_training)

    # Select method and estimate input parameters:

    method_mu = 'hist'
    method_cov = 'ledoit'

    portRP.assets_stats(method_mu=method_mu, method_cov=method_cov)
    model = 'Classic'
    risk_measures = ['MV', 'MSV', 'FLPM', 'SLPM']
    df_rp = []
    for rm in risk_measures:
        w = portRP.rp_optimization(model=model, rm=rm, rf=rf)
        w = round(w, 2)
        w.columns = ['por_rp' + '_' + rm]
        df_rp.append(w)

    df_rp = pd.concat(df_rp, axis=1)

    if weight_max:
        for col in df_rp.columns.to_list():
            temp = clean_limit_weights(df_rp[[col]],
                                       portfolio_name=col,
                                       weight_max=weight_max)
            df_rp[[col]] = temp

    if risk_measure:
        df_rp = df_rp[['por_rp' + '_' + risk_measure]]

    return df_rp


def inverse_vol_portfolio(returns_training,
                          my_assets_col_name=[],
                          weight_max=[]):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        my_assets_col_name (TYPE, optional): DESCRIPTION. Defaults to [].
        weight_max (TYPE, optional): DESCRIPTION. Defaults to [].

    Returns:
        df_inverse_vol (TYPE): DESCRIPTION.

    """

    # get inverse vol weights
    weights = pd.Series(ffn.core.calc_inv_vol_weights(returns_training),
                        name='port_inverse_vol')

    # clean
    df_inverse_vol = pd.DataFrame(weights)

    df_inverse_vol.index = df_inverse_vol.index.set_names(['asset'])

    # df_inverse_vol = df_inverse_vol.reset_index()

    if weight_max:
        df_inverse_vol = clean_limit_weights(df_inverse_vol,
                                             portfolio_name=df_inverse_vol.columns[0],
                                             weight_max=weight_max)
    if my_assets_col_name:
        df_inverse_vol.index = df_inverse_vol.index.set_names([my_assets_col_name])

    return df_inverse_vol


def equal_weight_portfolio(returns_training, my_assets_col_name=[]):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        my_assets_col_name (TYPE, optional): DESCRIPTION. Defaults to [].

    Returns:
        df_naive (TYPE): DESCRIPTION.

    """

    len_assets = len(returns_training.columns)
    naive_weights = [1 / len_assets] * len_assets  # copy the equal weight to a list

    if my_assets_col_name:
        d = {my_assets_col_name: returns_training.columns.to_list(),
             'port_naive': naive_weights}
        df_naive = pd.DataFrame(d).set_index(my_assets_col_name)

    else:
        d = {'assets': returns_training.columns.to_list(),
             'port_naive': naive_weights}
        df_naive = pd.DataFrame(d).set_index('assets')

    return df_naive


def clean_limit_weights(df_weights, portfolio_name: str, weight_max: float):
    '''
    df_weights: dataframe with index set as the names of the assets
                columns is the weight of the assets
    '''
    if weight_max >= 1:
        raise ValueError('Limit cannot be more than 1')

    dict_df = df_weights.to_dict()
    weights = dict_df[list(dict_df.keys())[0]]
    weights = ffn.core.limit_weights(weights, limit=weight_max)

    weights = pd.Series(weights, name=portfolio_name)
    df_weights = pd.DataFrame(weights)
    df_weights.index = df_weights.index.set_names(['asset'])

    return df_weights


# Global Minimum Variance
def port_GMV(returns_training=None, S=None, periods=252, weight_min=0.02,
             weight_max=0.4):
    """

    '''
    This portfolio optimisation routine minimises the global volatility of the
    portfolio. No other portfolio with the chosen asset space is expected
    to have lower volatility than this one. For the calculation of weights
    does not require the use of expected returns. Returns are passed here
    to calculate the variance when S = None
    '''

    Args:
        returns_training (TYPE, optional): DESCRIPTION. Defaults to None.
        S (TYPE, optional): DESCRIPTION. Defaults to None.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 0.4.

    Returns:
        weights (TYPE): DESCRIPTION.

    """

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training,
                                            returns_data=True,
                                            frequency=periods).ledoit_wolf()

    ef = EfficientFrontier(None, S, weight_bounds=(weight_min, weight_max))
    ef.min_volatility()
    weights = ef.clean_weights()
    weights = pd.DataFrame(weights, index=[0])

    weights = pd.melt(weights, var_name='asset',
                      value_name='port_min_vol')
    return weights


# Maximise returns_training for a given level of volatility
def port_target_volatility(returns_training, market_returns=None,
                           S=None,
                           average_returns_method='hist',
                           target_volatility=None,
                           rf=0.02,
                           span=500,
                           periods=252,
                           weight_min=0.02,
                           weight_max=0.4):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        market_returns (TYPE, optional): DESCRIPTION. Defaults to None.
        S (TYPE, optional): DESCRIPTION. Defaults to None.
        average_returns_method (TYPE, optional): DESCRIPTION. Defaults to 'hist'.
        target_volatility (TYPE, optional): DESCRIPTION. Defaults to None.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        span (TYPE, optional): DESCRIPTION. Defaults to 500.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 0.4.

    Returns:
        weights (TYPE): DESCRIPTION.

    """

    '''
    This portfolio optimisation routine maximises returns 
    for a given level of volatility 
    
    '''

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training,
                                            returns_data=True,
                                            frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(returns_training,
                         method=average_returns_method,
                         benchmark_returns=market_returns,
                         span=span,
                         periods=periods, rf=rf,
                         log_returns=False)

    ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))
    ef.efficient_risk(target_volatility=target_volatility)
    weights = ef.clean_weights()
    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name='asset',
                      value_name='port_target_vol')
    return weights


# Maximise returns_training for a given level of volatility
def port_target_return(returns_training, market_returns=None,
                       average_returns_method='hist',
                       S=None,
                       target_return=None,
                       rf=0.02,
                       span=500,
                       periods=252,
                       weight_min=0.02,
                       weight_max=0.4):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        market_returns (TYPE, optional): DESCRIPTION. Defaults to None.
        average_returns_method (TYPE, optional): DESCRIPTION. Defaults to 'hist'.
        S (TYPE, optional): DESCRIPTION. Defaults to None.
        target_return (TYPE, optional): DESCRIPTION. Defaults to None.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        span (TYPE, optional): DESCRIPTION. Defaults to 500.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 0.4.

    Returns:
        weights (TYPE): DESCRIPTION.

    """

    '''
    This portfolio optimisation routine minimise volatility for a given 
    level of return the user will choose 
    '''

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True,
                                            frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(returns_training,
                         method=average_returns_method,
                         benchmark_returns=market_returns,
                         span=span,
                         periods=periods, rf=rf,
                         log_returns=False)

    ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))
    ef.add_objective(objective_functions.L2_reg)
    ef.efficient_return(target_return=target_return, market_neutral=False)
    weights = ef.clean_weights()
    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name='asset',
                      value_name='port_target_returns')
    return weights


# Maximise Sharpe
def port_max_sharpe(returns_training, market_returns=None,
                    S=None,
                    average_returns_method='hist',
                    target_volatility=None,
                    rf=0.02,
                    span=500,
                    periods=252,
                    weight_min=0.02,
                    weight_max=0.4):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        market_returns (TYPE, optional): DESCRIPTION. Defaults to None.
        S (TYPE, optional): DESCRIPTION. Defaults to None.
        average_returns_method (TYPE, optional): DESCRIPTION. Defaults to 'hist'.
        target_volatility (TYPE, optional): DESCRIPTION. Defaults to None.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        span (TYPE, optional): DESCRIPTION. Defaults to 500.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 0.4.

    Returns:
        weights (TYPE): DESCRIPTION.

    """

    '''
    This portfolio optimisation routine maximised the Sharpe ratio 
    of portoflio 
    '''

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True,
                                            frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(returns_training,
                         method=average_returns_method,
                         benchmark_returns=market_returns,
                         span=span,
                         periods=periods, rf=rf,
                         log_returns=False)

    ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))
    ef.max_sharpe()
    weights = ef.clean_weights()
    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name='asset',
                      value_name='port_max_Sharpe')
    return weights


# Port CVAR
def port_CVAR(returns_training, market_returns=None,
              average_returns_method='hist',
              target_CVAR=None,
              confidence_interval=0.95,
              rf=0.02,
              span=500,
              periods=252,
              weight_min=0.02,
              weight_max=0.4):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        market_returns (TYPE, optional): DESCRIPTION. Defaults to None.
        average_returns_method (TYPE, optional): DESCRIPTION. Defaults to 'hist'.
        target_CVAR (TYPE, optional): DESCRIPTION. Defaults to None.
        confidence_interval (TYPE, optional): DESCRIPTION. Defaults to 0.95.
        rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        span (TYPE, optional): DESCRIPTION. Defaults to 500.
        periods (TYPE, optional): DESCRIPTION. Defaults to 252.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.02.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 0.4.

    Returns:
        weights (TYPE): DESCRIPTION.

    """

    '''
    This value of the CVaR means that our average loss on the worst 5% of days 
    will be -3.35%. Let's say that this were beyond our comfort zone 
    (for a $100,000 portfolio, this would mean losing \$3350 in a day).
    
    The algorithm with maximise retunrs for a given Target of CVAR 
    '''

    # calculate expected returns_training
    mu = average_returns(returns_training, method=average_returns_method,
                         benchmark_returns=market_returns,
                         span=span,
                         periods=periods, rf=rf,
                         log_returns=False)

    ec = EfficientCVaR(mu, returns_training, beta=confidence_interval,
                       weight_bounds=(weight_min, weight_max))

    if target_CVAR is None:
        ec.min_cvar()
    else:
        ec.efficient_risk(target_cvar=target_CVAR)

    weights = ec.clean_weights()
    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name='asset',
                      value_name='port_target_CVAR')
    return weights


@ray.remote
def HRP_ray(model: str = 'HRP',
            returns_training: pd.DataFrame = None,
            covariance: str = "hist",
            codependence: str = 'pearson',
            rm: str = 'MV',
            linkage: str = 'single',
            weight_max: float = None,
            weight_min: float = None,
            leaf_order: bool = False, **kwargs):
    # sanity checks
    if not isinstance(returns_training, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    port = rp.HCPortfolio(returns=returns_training)

    if weight_max is not None or weight_min is not None:
        # impose constraints
        asset_classes = {'Assets': returns_training.columns.to_list()}
        asset_classes = pd.DataFrame(asset_classes)
        asset_classes = asset_classes.sort_values(by=['Assets'])

        constraints = {'Disabled': [False, False],
                       'Type': ['All Assets', 'All Assets'],
                       'Set': ['', ''],
                       'Position': ['', ''],
                       'Sign': ['<=', '>='],
                       'Weight': [weight_max, weight_min]}

        constraints = pd.DataFrame(constraints)

        w_max, w_min = rp.hrp_constraints(constraints, asset_classes)
        port.w_max = w_max
        port.w_min = w_min

    # HRP Default
    weights = port.optimization(model=model,
                                codependence=codependence,
                                covariance=covariance,
                                rm=rm,
                                linkage=linkage,
                                leaf_order=leaf_order,
                                **kwargs)

    weights = weights.rename(columns={'weights': 'port_weight'})

    if model != 'HRP' and weight_max is not None:
        temp = clean_limit_weights(weights[['port_weight']],
                                   portfolio_name='port_weight',
                                   weight_max=weight_max)
        weights[['port_weight']] = temp

    return weights


@timebudget
def generate_HRP_portfolios(returns=None, weight_max=None,
                            weight_min=None,
                            rf=0.02, num_cpus=1):
    models = ['HRP', 'HERC', 'HERC2']

    codependences = ['pearson', 'spearman',
                     # 'abs_pearson', 'abs_spearman',
                     'tail',
                     # 'distance',
                     'mutual_info']

    covariances = ['hist',
                   # 'ewma1','ewma2',
                   'ledoit',
                   'oas',
                   'shrunk',
                   'gl',
                   'jlogo', 'fixed', 'spectral', 'shrink']

    linkages = [  # 'single',
        # 'complete',
        # 'average',
        # 'weighted',
        # 'centroid','median',
        'ward']
    # 'DBHT']

    risk_measures = [  # 'equal',
        # 'vol',
        'MV',
        # 'MAD',
        'MSV',
        #  'FLPM',
        #  'SLPM',
        # 'VaR',
        'CVaR']
    # 'TG',
    # 'EVaR',
    #  'WR',
    #   'RG'
    # 'CVRG',
    #  'TGRG']
    # 'MDD',
    # 'ADD',
    # 'DaR',
    # 'CDaR',
    # 'EDaR',
    # 'UCI',
    # 'MDD_Rel',
    # 'ADD_Rel',
    # 'DaR_Rel',
    # 'CDaR_Rel',
    # 'EDaR_Rel',
    # 'UCI_Rel']

    objects = itertools.product(models, covariances, codependences,
                                risk_measures, linkages)

    # ray.shutdown()
    # ray.init(ignore_reinit_error=True,
    #          num_cpus=num_cpus)
    # portfolios = ray.get([HRP_ray.remote(model = obj[0],
    #                                         returns_training = returns,
    #                                         covariance=obj[1],
    #                                         codependence =obj[2],
    #                                         rm=obj[3],
    #                                         linkage = obj[4],
    #                                         rf = rf,
    #                                         weight_max = weight_max,
    #                                         weight_min = weight_min)
    #                      for obj in objects])
    # ray.shutdown()

    portfolios = []
    for obj in objects:
        try:
            portf_temp = HRP(model=obj[0],
                             returns_training=returns,
                             covariance=obj[1],
                             codependence=obj[2],
                             rm=obj[3],
                             linkage=obj[4],
                             rf=rf,
                             weight_max=weight_max,
                             weight_min=weight_min)
        except:
            next

        portfolios.append(portf_temp)

    portfolios = pd.concat(portfolios, axis=1)

    all_nums_iter = itertools.product(models, covariances,
                                      codependences,
                                      risk_measures, linkages)
    names = []
    for name in all_nums_iter:
        names.append('port_weights_' + '_'.join(name))

    portfolios.columns = names

    return portfolios


# def get_portfolios(returns, training_start_date=None,
#                      training_end_date=None,
#                      weight_max = [], weight_min = [], rf = 0.02):
#     """


#     Args:
#         returns (TYPE): DESCRIPTION.
#         training_start_date (TYPE, optional): DESCRIPTION. Defaults to None.
#         training_end_date (TYPE, optional): DESCRIPTION. Defaults to None.
#         weight_max (TYPE, optional): DESCRIPTION. Defaults to [].
#         weight_min (TYPE, optional): DESCRIPTION. Defaults to [].
#         rf (TYPE, optional): DESCRIPTION. Defaults to 0.02.

#     Returns:
#         df_all_weights (TYPE): DESCRIPTION.

#     """

#     # training period for portfolio optimisation

#     if (training_start_date==None) & (training_end_date!=None):
#         ret_training = returns[returns.index<=training_end_date]
#     else:
#         ret_training = returns[(returns.index>=training_start_date) &
#                                (returns.index<=training_end_date)]

#     # naive or equal weight portfolio allocation
#     df_naive = equal_weight_portfolio(ret_training)

#     # inverse volatility
#     df_inverse_vol = inverse_vol_portfolio(ret_training, limit = weight_max)

#     # Risk Parity
#     df_rp = generate_rp_portfolios(ret_training.fillna(0), limit = weight_max, rf = rf)

#     # HRP portfolios
#     df_HRP_portfolios = generate_HRP_portfolios(returns_training = ret_training.fillna(0),
#                             weight_max = weight_max,
#                             weight_min = weight_min,
#                             rf = rf)

#     df_all_weights = pd.concat([df_inverse_vol, df_naive,
#                                 df_HRP_portfolios, df_rp],axis = 1)

#     df_all_weights = df_all_weights.dropna(axis = 1, how ='any')


#     return df_all_weights


def make_standard_portfolios(returns_training: pd.DataFrame, target_return: float = None,
                             target_volatility: float = None,
                             weight_min: float = 0, weight_max: float = 1,
                             rebalance: str = None):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        target_return (TYPE, optional): DESCRIPTION. Defaults to None.
        target_volatility (TYPE, optional): DESCRIPTION. Defaults to None.
        weight_min (TYPE, optional): DESCRIPTION. Defaults to 0.
        weight_max (TYPE, optional): DESCRIPTION. Defaults to 1.
        rebalance (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        list: DESCRIPTION.

    """

    # naive or equal weight portfolio allocation
    df_naive = equal_weight_portfolio(returns_training)
    port_naive_w = df_naive.iloc[:, 0].to_dict()
    portfolio_naive = qs.utils.make_index(ticker_weights=port_naive_w,
                                          rebalance=rebalance,
                                          period='max',
                                          returns=returns_training,
                                          match_dates=False)

    # Risk Parity
    df_rp = generate_rp_portfolios(returns_training.fillna(0),
                                   weight_max=0.99)
    df_rp = df_rp[['por_rp_MV']]
    port_rp_w = df_rp.iloc[:, 0].to_dict()
    portfolio_rp = qs.utils.make_index(ticker_weights=port_rp_w,
                                       rebalance=rebalance,
                                       period='max',
                                       returns=returns_training,
                                       match_dates=False)

    # inverse_volatility
    df_inverse_vol = inverse_vol_portfolio(returns_training, weight_max=0.99)
    port_inverse_vol_w = df_inverse_vol.iloc[:, 0].to_dict()
    portfolio_inverse_vol = qs.utils.make_index(ticker_weights=port_inverse_vol_w,
                                                rebalance=rebalance,
                                                period='max',
                                                returns=returns_training,
                                                match_dates=False)

    # Hierarchical Risks Parity
    port_HRP_w = HRP(returns_training=returns_training.fillna(0),
                     weight_max=weight_max,
                     weight_min=weight_min)

    port_HRP_w.index = port_HRP_w.index.set_names('asset')
    weight_HRP = port_HRP_w

    # port_HRP_w = port_HRP_w.set_index('asset')
    port_HRP_w = port_HRP_w.iloc[:, 0].to_dict()
    portfolio_HRP = qs.utils.make_index(ticker_weights=port_HRP_w,
                                        rebalance=rebalance,
                                        period='max',
                                        returns=returns_training,
                                        match_dates=False)

    # min vol portfolio
    port_min_vol_w = port_GMV(returns_training, S=None, weight_min=weight_min,
                              weight_max=weight_max)

    port_min_vol_w = port_min_vol_w.set_index('asset')
    weight_min_vol = port_min_vol_w

    port_min_vol_w = port_min_vol_w.iloc[:, 0].to_dict()

    portfolio_mv = qs.utils.make_index(ticker_weights=port_min_vol_w,
                                       rebalance=rebalance,
                                       period='max',
                                       returns=returns_training,
                                       match_dates=False)

    # target returns_training
    if target_return != None:
        port_target_returns_w = port_target_return(returns_training, target_return=target_return,
                                                   weight_min=weight_min, weight_max=weight_max)

        port_target_returns_w = port_target_returns_w.set_index('asset')
        weight_target_returns = port_target_returns_w

        port_target_returns_w = port_target_returns_w.iloc[:, 0].to_dict()

        portfolio_tr = qs.utils.make_index(ticker_weights=port_target_returns_w,
                                           rebalance=rebalance,
                                           period='max',
                                           returns=returns_training,
                                           match_dates=False)

    # target volatility
    if target_volatility != None:
        port_target_vol_w = port_target_volatility(returns_training,
                                                   target_volatility=target_volatility)

        port_target_vol_w = port_target_vol_w.set_index('asset')
        weight_target_vol = port_target_vol_w

        port_target_vol_w = port_target_vol_w.iloc[:, 0].to_dict()
        portfolio_tv = qs.utils.make_index(ticker_weights=port_target_vol_w,
                                           rebalance=rebalance,
                                           period='max',
                                           returns=returns_training,
                                           match_dates=False)

    # max sharpe
    max_sharpe = port_max_sharpe(returns_training, weight_min=weight_min,
                                 weight_max=weight_max)

    max_sharpe = max_sharpe.set_index('asset')
    weight_max_sharpe = max_sharpe
    max_sharpe = max_sharpe.iloc[:, 0].to_dict()
    portfolio_max_sharpe = qs.utils.make_index(ticker_weights=max_sharpe,
                                               rebalance=rebalance,
                                               period='max',
                                               returns=returns_training,
                                               match_dates=False)

    # CVar
    cvar = port_CVAR(returns_training, weight_min=weight_min,
                     weight_max=weight_max)
    cvar = cvar.set_index('asset')
    weight_cvar = cvar
    cvar_weights = cvar.iloc[:, 0].to_dict()
    portfolio_cvar = qs.utils.make_index(ticker_weights=cvar_weights,
                                         rebalance=rebalance,
                                         period='max',
                                         returns=returns_training,
                                         match_dates=False)
    frame = {'port_GMV': portfolio_mv,
             'port_HRP': portfolio_HRP,
             'port_max_Sharpe': portfolio_max_sharpe,
             'port_min_CVaR': portfolio_cvar,
             'port_inverse_vol': portfolio_inverse_vol,
             'port_naive': portfolio_naive,
             'port_rp': portfolio_rp}

    df_weights_all = pd.concat([weight_HRP, df_naive,
                                df_rp, df_inverse_vol,
                                weight_min_vol,
                                weight_cvar,
                                weight_max_sharpe], axis=1)

    # collect
    if target_return != None and target_volatility != None:
        frame_add = {'port_Max_Returns': portfolio_tr,
                     'port_targ_vol': portfolio_tv}

        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all,
                                    weight_target_returns,
                                    weight_target_vol], axis=1)

    elif target_return != None and target_volatility == None:
        frame_add = {'port_Max_Returns': portfolio_tr}
        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all,
                                    weight_target_returns], axis=1)

    elif target_return == None and target_volatility != None:
        frame_add = {'port_Targ_Vol': portfolio_tv}
        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all,
                                    weight_target_vol], axis=1)

    portfolios = pd.DataFrame(frame)

    return [portfolios, df_weights_all]
