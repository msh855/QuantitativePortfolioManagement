#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 07:29:35 2021

@author: safishajjouz
"""

import ffn
import numpy as np
import pandas as pd
import quantstats_lumi as qs
import riskfolio as rp

# for portfolio optimisation
from pypfopt import EfficientCVaR, EfficientFrontier, objective_functions, risk_models

# skfolio for Schur Complementary Allocation and advanced methods
try:
    import skfolio
    from skfolio.optimization import (
        SchurComplementary,
        HierarchicalRiskParity as SKF_HRP,
        HierarchicalEqualRiskContribution as SKF_HERC,
        NestedClustersOptimization as SKF_NCO,
        MeanRisk,
        MaximumDiversification,
        DistributionallyRobustCVaR,
    )
    from skfolio.prior import EmpiricalPrior
    from skfolio.moments import LedoitWolf, ShrunkCovariance, EmpiricalCovariance
    from skfolio.distance import PearsonDistance, KendallDistance, SpearmanDistance
    from skfolio.cluster import HierarchicalClustering, LinkageMethod
    from skfolio.preprocessing import prices_to_returns
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
import itertools

import matplotlib.pyplot as plt
from timebudget import timebudget  # to time functions

from myPortfolioManagement.myReturns import average_returns


def _enforce_weight_constraints(weights_dict, weight_min, n_assets):
    """
    Helper function to enforce weight constraints when optimization constraints are infeasible.

    If n_assets * weight_min > 1.0, this function allows unselected assets to have zero weight
    while ensuring selected assets meet the minimum weight requirement.

    Args:
        weights_dict: Dictionary of asset weights from optimizer
        weight_min: Minimum weight constraint
        n_assets: Number of assets

    Returns:
        Dictionary of adjusted weights that sum to 1.0
    """
    if weight_min is None or n_assets * weight_min <= 1.0:
        # Constraints are feasible, return as-is
        return weights_dict

    # Constraints are infeasible - enforce weight_min only on selected assets
    weights_filtered = {k: max(v, weight_min) if v > 1e-4 else 0 for k, v in weights_dict.items()}

    # Renormalize to sum to 1
    total = sum(weights_filtered.values())
    if total > 0:
        return {k: v / total for k, v in weights_filtered.items()}
    else:
        return weights_filtered


# Hierarchical Risk Parity Default option
def HRP(
    model: str = "HRP",
    returns_training: str = None,
    covariance: str = "hist",
    codependence: str = "pearson",
    rm: str = "MV",
    linkage: str = "single",
    weight_max: float = None,
    weight_min: float = None,
    leaf_order: bool = False,
    **kwargs
):
    """

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

        - 'pearson': pearson correlation matrix. Distance formula: :math:`D_{i,j} = \\sqrt{0.5(1-\\rho^{pearson}_{i,j})}`.
        - 'spearman': spearman correlation matrix. Distance formula: :math:`D_{i,j} = \\sqrt{0.5(1-\\rho^{spearman}_{i,j})}`.
        - 'abs_pearson': absolute value pearson correlation matrix. Distance formula: :math:`D_{i,j} = \\sqrt{(1-|\\rho^{pearson}_{i,j}|)}`.
        - 'abs_spearman': absolute value spearman correlation matrix. Distance formula: :math:`D_{i,j} = \\sqrt{(1-|\\rho^{spearman}_{i,j}|)}`.
        - 'distance': distance correlation matrix. Distance formula :math:`D_{i,j} = \\sqrt{(1-\\rho^{distance}_{i,j})}`.
        - 'mutual_info': mutual information matrix. Distance used is variation information matrix.
        - 'tail': lower tail dependence index matrix. Dissimilarity formula :math:`D_{i,j} = -\\log{\\lambda_{i,j}}`.
        - 'custom_cov': use custom correlation matrix based on the custom_cov parameter. Distance formula: :math:`D_{i,j} = \\sqrt{0.5(1-\\rho^{pearson}_{i,j})}`.

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
        - 'Utility': Maximize the Utility function :math:`\\mu w - l \\phi_{i}(w)`.
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

    """

    # sanity checks
    if not isinstance(returns_training, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    port = rp.HCPortfolio(returns=returns_training)

    if weight_max is not None or weight_min is not None:
        # impose constraints
        asset_classes = {"Assets": returns_training.columns.to_list()}
        asset_classes = pd.DataFrame(asset_classes)
        asset_classes = asset_classes.sort_values(by=["Assets"])

        constraints = {
            "Disabled": [False, False],
            "Type": ["All Assets", "All Assets"],
            "Set": ["", ""],
            "Position": ["", ""],
            "Sign": ["<=", ">="],
            "Weight": [weight_max, weight_min],
        }

        constraints = pd.DataFrame(constraints)

        w_max, w_min = rp.hrp_constraints(constraints, asset_classes)
        port.w_max = w_max
        port.w_min = w_min

    # HRP Default
    weights = port.optimization(
        model=model,
        codependence=codependence,
        method_cov=covariance,
        rm=rm,
        linkage=linkage,
        leaf_order=leaf_order,
        **kwargs
    )

    weights = weights.rename(columns={"weights": "port_weight"})
    weights.index.name = "asset"

    if model != "HRP" and weight_max is not None:
        # Normalize weights to sum to 1.0 before applying additional constraints
        weights["port_weight"] = weights["port_weight"] / weights["port_weight"].sum()
        temp = clean_limit_weights(weights[["port_weight"]], portfolio_name="port_weight", weight_max=weight_max)
        weights[["port_weight"]] = temp

    return weights


def generate_rp_portfolios(returns_training=None, rf=0.02, risk_measure=[], weight_max=[]):
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

    method_mu = "hist"
    method_cov = "ledoit"

    portRP.assets_stats(method_mu=method_mu, method_cov=method_cov)
    model = "Classic"
    risk_measures = ["MV", "MSV", "FLPM", "SLPM"]
    df_rp = []
    for rm in risk_measures:
        w = portRP.rp_optimization(model=model, rm=rm, rf=rf)
        w = round(w, 2)
        w.columns = ["por_rp" + "_" + rm]
        df_rp.append(w)

    df_rp = pd.concat(df_rp, axis=1)

    if weight_max:
        for col in df_rp.columns.to_list():
            temp = clean_limit_weights(df_rp[[col]], portfolio_name=col, weight_max=weight_max)
            df_rp[[col]] = temp

    if risk_measure:
        df_rp = df_rp[["por_rp" + "_" + risk_measure]]

    return df_rp


def inverse_vol_portfolio(returns_training, my_assets_col_name=[], weight_max=[]):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        my_assets_col_name (TYPE, optional): DESCRIPTION. Defaults to [].
        weight_max (TYPE, optional): DESCRIPTION. Defaults to [].

    Returns:
        pd.DataFrame: Portfolio weights.

    """

    # get inverse vol weights
    weights = pd.Series(ffn.core.calc_inv_vol_weights(returns_training), name="port_inverse_vol")

    # Set index name
    index_name = my_assets_col_name if my_assets_col_name else "asset"
    weights.index.name = index_name

    # Create DataFrame from Series
    df_inverse_vol = weights.to_frame()

    if weight_max:
        # Apply weight limits
        df_inverse_vol = clean_limit_weights(
            df_inverse_vol, portfolio_name=df_inverse_vol.columns[0], weight_max=weight_max
        )

    return df_inverse_vol


def equal_weight_portfolio(returns_training, my_assets_col_name=[]):
    """


    Args:
        returns_training (TYPE): DESCRIPTION.
        my_assets_col_name (TYPE, optional): DESCRIPTION. Defaults to [].

    Returns:
        pd.DataFrame: Portfolio weights.

    """

    len_assets = len(returns_training.columns)
    naive_weights = [1 / len_assets] * len_assets  # copy the equal weight to a list

    # Create Series with asset names as index
    index_name = my_assets_col_name if my_assets_col_name else "asset"
    weights_series = pd.Series(naive_weights, index=returns_training.columns, name="port_naive")
    weights_series.index.name = index_name

    # Create DataFrame from Series
    df_equal_weight = weights_series.to_frame()

    return df_equal_weight


def clean_limit_weights(df_weights, portfolio_name: str, weight_max: float):
    """
    df_weights: dataframe with index set as the names of the assets
                columns is the weight of the assets
    """
    if weight_max >= 1:
        raise ValueError("Limit cannot be more than 1")

    dict_df = df_weights.to_dict()
    weights = dict_df[list(dict_df.keys())[0]]
    weights = ffn.core.limit_weights(weights, limit=weight_max)

    weights = pd.Series(weights, name=portfolio_name)
    df_weights = pd.DataFrame(weights)
    df_weights.index = df_weights.index.set_names(["asset"])

    return df_weights


# Global Minimum Variance
def port_GMV(returns_training=None, S=None, periods=252, weight_min=0.02, weight_max=0.4):
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
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True, frequency=periods).ledoit_wolf()

    # Check if constraints are feasible (n_assets * weight_min <= 1.0)
    n_assets = len(S)
    if weight_min is not None and n_assets * weight_min > 1.0:
        # Constraints are infeasible - allow zero weights for unselected assets
        ef = EfficientFrontier(None, S, weight_bounds=(0, weight_max))
    else:
        # Constraints are feasible - use original approach
        ef = EfficientFrontier(None, S, weight_bounds=(weight_min, weight_max))

    ef.min_volatility()
    weights = ef.clean_weights()

    # Apply weight constraints post-optimization if needed
    weights = _enforce_weight_constraints(weights, weight_min, n_assets)

    weights = pd.DataFrame(weights, index=[0])

    weights = pd.melt(weights, var_name="asset", value_name="port_min_vol")
    weights = weights.set_index("asset")
    return weights


# Maximise returns_training for a given level of volatility
def port_target_volatility(
    returns_training,
    market_returns=None,
    S=None,
    average_returns_method="hist",
    target_volatility=None,
    rf=0.02,
    span=500,
    periods=252,
    weight_min=0.02,
    weight_max=0.4,
):
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

    """
    This portfolio optimisation routine maximises returns
    for a given level of volatility

    """

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True, frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(
        returns_training,
        method=average_returns_method,
        benchmark_returns=market_returns,
        span=span,
        periods=periods,
        rf=rf,
        log_returns=False,
    )

    # Check if constraints are feasible
    n_assets = len(S)
    if weight_min is not None and n_assets * weight_min > 1.0:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, weight_max))
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))

    ef.efficient_risk(target_volatility=target_volatility)
    weights = ef.clean_weights()

    # Apply weight constraints post-optimization if needed
    weights = _enforce_weight_constraints(weights, weight_min, n_assets)

    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name="asset", value_name="port_target_vol")
    return weights


# Maximise returns_training for a given level of volatility
def port_target_return(
    returns_training,
    market_returns=None,
    average_returns_method="hist",
    S=None,
    target_return=None,
    rf=0.02,
    span=500,
    periods=252,
    weight_min=0.02,
    weight_max=0.4,
):
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

    """
    This portfolio optimisation routine minimise volatility for a given
    level of return the user will choose
    """

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True, frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(
        returns_training,
        method=average_returns_method,
        benchmark_returns=market_returns,
        span=span,
        periods=periods,
        rf=rf,
        log_returns=False,
    )

    # Check if constraints are feasible
    n_assets = len(S)
    if weight_min is not None and n_assets * weight_min > 1.0:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, weight_max))
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))

    ef.add_objective(objective_functions.L2_reg)
    ef.efficient_return(target_return=target_return, market_neutral=False)
    weights = ef.clean_weights()

    # Apply weight constraints post-optimization if needed
    weights = _enforce_weight_constraints(weights, weight_min, n_assets)

    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name="asset", value_name="port_target_returns")
    return weights


# Maximise Sharpe
def port_max_sharpe(
    returns_training,
    market_returns=None,
    S=None,
    average_returns_method="hist",
    target_volatility=None,
    rf=0.02,
    span=500,
    periods=252,
    weight_min=0.02,
    weight_max=0.4,
):
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

    """
    This portfolio optimisation routine maximised the Sharpe ratio
    of portoflio
    """

    if S is None:
        S = risk_models.CovarianceShrinkage(returns_training, returns_data=True, frequency=periods).ledoit_wolf()

    # calculate expected returns_training
    mu = average_returns(
        returns_training,
        method=average_returns_method,
        benchmark_returns=market_returns,
        span=span,
        periods=periods,
        rf=rf,
        log_returns=False,
    )

    # Check if constraints are feasible
    n_assets = len(S)
    if weight_min is not None and n_assets * weight_min > 1.0:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, weight_max))
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(weight_min, weight_max))

    ef.max_sharpe()
    weights = ef.clean_weights()

    # Apply weight constraints post-optimization if needed
    weights = _enforce_weight_constraints(weights, weight_min, n_assets)

    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name="asset", value_name="port_max_Sharpe")
    weights = weights.set_index("asset")
    return weights


# Port CVAR
def port_CVAR(
    returns_training,
    market_returns=None,
    average_returns_method="hist",
    target_CVAR=None,
    confidence_interval=0.95,
    rf=0.02,
    span=500,
    periods=252,
    weight_min=0.02,
    weight_max=0.4,
):
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

    """
    This value of the CVaR means that our average loss on the worst 5% of days
    will be -3.35%. Let's say that this were beyond our comfort zone
    (for a $100,000 portfolio, this would mean losing $3350 in a day).

    The algorithm with maximise retunrs for a given Target of CVAR
    """

    # calculate expected returns_training
    mu = average_returns(
        returns_training,
        method=average_returns_method,
        benchmark_returns=market_returns,
        span=span,
        periods=periods,
        rf=rf,
        log_returns=False,
    )

    # Check if constraints are feasible
    n_assets = len(returns_training.columns)
    if weight_min is not None and n_assets * weight_min > 1.0:
        ec = EfficientCVaR(mu, returns_training, beta=confidence_interval, weight_bounds=(0, weight_max))
    else:
        ec = EfficientCVaR(mu, returns_training, beta=confidence_interval, weight_bounds=(weight_min, weight_max))

    if target_CVAR is None:
        ec.min_cvar()
    else:
        ec.efficient_risk(target_cvar=target_CVAR)

    weights = ec.clean_weights()

    # Apply weight constraints post-optimization if needed
    weights = _enforce_weight_constraints(weights, weight_min, n_assets)

    weights = pd.DataFrame(weights, index=[0])
    weights = pd.melt(weights, var_name="asset", value_name="port_target_CVAR")
    weights = weights.set_index("asset")
    return weights


@ray.remote
def HRP_ray(
    model: str = "HRP",
    returns_training: pd.DataFrame = None,
    covariance: str = "hist",
    codependence: str = "pearson",
    rm: str = "MV",
    linkage: str = "single",
    weight_max: float = None,
    weight_min: float = None,
    leaf_order: bool = False,
    **kwargs
):
    # sanity checks
    if not isinstance(returns_training, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    port = rp.HCPortfolio(returns=returns_training)

    if weight_max is not None or weight_min is not None:
        # impose constraints
        asset_classes = {"Assets": returns_training.columns.to_list()}
        asset_classes = pd.DataFrame(asset_classes)
        asset_classes = asset_classes.sort_values(by=["Assets"])

        constraints = {
            "Disabled": [False, False],
            "Type": ["All Assets", "All Assets"],
            "Set": ["", ""],
            "Position": ["", ""],
            "Sign": ["<=", ">="],
            "Weight": [weight_max, weight_min],
        }

        constraints = pd.DataFrame(constraints)

        w_max, w_min = rp.hrp_constraints(constraints, asset_classes)
        port.w_max = w_max
        port.w_min = w_min

    # HRP Default
    weights = port.optimization(
        model=model,
        codependence=codependence,
        method_cov=covariance,
        rm=rm,
        linkage=linkage,
        leaf_order=leaf_order,
        **kwargs
    )

    weights = weights.rename(columns={"weights": "port_weight"})

    if model != "HRP" and weight_max is not None:
        temp = clean_limit_weights(weights[["port_weight"]], portfolio_name="port_weight", weight_max=weight_max)
        weights[["port_weight"]] = temp

    return weights


@timebudget
def generate_HRP_portfolios(returns=None, weight_max=None, weight_min=None, rf=0.02, num_cpus=1):
    models = ["HRP", "HERC", "HERC2"]

    codependences = [
        "pearson",
        "spearman",
        # 'abs_pearson', 'abs_spearman',
        "tail",
        # 'distance',
        "mutual_info",
    ]

    covariances = [
        "hist",
        # 'ewma1','ewma2',
        "ledoit",
        "oas",
        "shrunk",
        "gl",
        "jlogo",
        "fixed",
        "spectral",
        "shrink",
    ]

    linkages = [  # 'single',
        # 'complete',
        # 'average',
        # 'weighted',
        # 'centroid','median',
        "ward"
    ]
    # 'DBHT']

    risk_measures = [  # 'equal',
        # 'vol',
        "MV",
        # 'MAD',
        "MSV",
        #  'FLPM',
        #  'SLPM',
        # 'VaR',
        "CVaR",
    ]
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

    objects = itertools.product(models, covariances, codependences, risk_measures, linkages)

    portfolios = []
    for obj in objects:
        try:
            portf_temp = HRP(
                model=obj[0],
                returns_training=returns,
                covariance=obj[1],
                codependence=obj[2],
                rm=obj[3],
                linkage=obj[4],
                rf=rf,
                weight_max=weight_max,
                weight_min=weight_min,
            )
            portfolios.append(portf_temp)
        except Exception:
            continue

    portfolios = pd.concat(portfolios, axis=1)

    all_nums_iter = itertools.product(models, covariances, codependences, risk_measures, linkages)
    names = []
    for name in all_nums_iter:
        names.append("port_weights_" + "_".join(name))

    portfolios.columns = names

    return portfolios


def make_standard_portfolios(
    returns_training: pd.DataFrame,
    target_return: float = None,
    target_volatility: float = None,
    weight_min: float = 0,
    weight_max: float = 1,
    rebalance: str = None,
):
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
    portfolio_naive = qs.utils.make_index(
        ticker_weights=port_naive_w, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )

    # Risk Parity
    df_rp = generate_rp_portfolios(returns_training.fillna(0), weight_max=0.99)
    df_rp = df_rp[["por_rp_MV"]]
    port_rp_w = df_rp.iloc[:, 0].to_dict()
    portfolio_rp = qs.utils.make_index(
        ticker_weights=port_rp_w, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )

    # inverse_volatility
    df_inverse_vol = inverse_vol_portfolio(returns_training, weight_max=0.99)
    port_inverse_vol_w = df_inverse_vol.iloc[:, 0].to_dict()
    portfolio_inverse_vol = qs.utils.make_index(
        ticker_weights=port_inverse_vol_w,
        rebalance=rebalance,
        period="max",
        returns=returns_training,
        match_dates=False,
    )

    # Hierarchical Risks Parity
    port_HRP_w = HRP(returns_training=returns_training.fillna(0), weight_max=weight_max, weight_min=weight_min)

    port_HRP_w.index = port_HRP_w.index.set_names("asset")
    weight_HRP = port_HRP_w

    # port_HRP_w = port_HRP_w.set_index('asset')
    port_HRP_w = port_HRP_w.iloc[:, 0].to_dict()
    portfolio_HRP = qs.utils.make_index(
        ticker_weights=port_HRP_w, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )

    # min vol portfolio
    port_min_vol_w = port_GMV(returns_training, S=None, weight_min=weight_min, weight_max=weight_max)

    # port_GMV already returns DataFrame with 'asset' as index
    weight_min_vol = port_min_vol_w

    port_min_vol_w = port_min_vol_w.iloc[:, 0].to_dict()

    portfolio_mv = qs.utils.make_index(
        ticker_weights=port_min_vol_w, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )

    # target returns_training
    if target_return is not None:
        port_target_returns_w = port_target_return(
            returns_training, target_return=target_return, weight_min=weight_min, weight_max=weight_max
        )

        port_target_returns_w = port_target_returns_w.set_index("asset")
        weight_target_returns = port_target_returns_w

        port_target_returns_w = port_target_returns_w.iloc[:, 0].to_dict()

        portfolio_tr = qs.utils.make_index(
            ticker_weights=port_target_returns_w,
            rebalance=rebalance,
            period="max",
            returns=returns_training,
            match_dates=False,
        )

    # target volatility
    if target_volatility is not None:
        port_target_vol_w = port_target_volatility(returns_training, target_volatility=target_volatility)

        port_target_vol_w = port_target_vol_w.set_index("asset")
        weight_target_vol = port_target_vol_w

        port_target_vol_w = port_target_vol_w.iloc[:, 0].to_dict()
        portfolio_tv = qs.utils.make_index(
            ticker_weights=port_target_vol_w,
            rebalance=rebalance,
            period="max",
            returns=returns_training,
            match_dates=False,
        )

    # max sharpe
    max_sharpe = port_max_sharpe(returns_training, weight_min=weight_min, weight_max=weight_max)

    # port_max_sharpe already returns DataFrame with 'asset' as index
    weight_max_sharpe = max_sharpe
    max_sharpe = max_sharpe.iloc[:, 0].to_dict()
    portfolio_max_sharpe = qs.utils.make_index(
        ticker_weights=max_sharpe, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )

    # CVar
    cvar = port_CVAR(returns_training, weight_min=weight_min, weight_max=weight_max)
    # port_CVAR already returns DataFrame with 'asset' as index
    weight_cvar = cvar
    cvar_weights = cvar.iloc[:, 0].to_dict()
    portfolio_cvar = qs.utils.make_index(
        ticker_weights=cvar_weights, rebalance=rebalance, period="max", returns=returns_training, match_dates=False
    )
    frame = {
        "port_GMV": portfolio_mv,
        "port_HRP": portfolio_HRP,
        "port_max_Sharpe": portfolio_max_sharpe,
        "port_min_CVaR": portfolio_cvar,
        "port_inverse_vol": portfolio_inverse_vol,
        "port_naive": portfolio_naive,
        "port_rp": portfolio_rp,
    }

    df_weights_all = pd.concat(
        [weight_HRP, df_naive, df_rp, df_inverse_vol, weight_min_vol, weight_cvar, weight_max_sharpe], axis=1
    )

    # collect
    if target_return is not None and target_volatility is not None:
        frame_add = {"port_Max_Returns": portfolio_tr, "port_targ_vol": portfolio_tv}

        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all, weight_target_returns, weight_target_vol], axis=1)

    elif target_return is not None and target_volatility is None:
        frame_add = {"port_Max_Returns": portfolio_tr}
        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all, weight_target_returns], axis=1)

    elif target_return is None and target_volatility is not None:
        frame_add = {"port_Targ_Vol": portfolio_tv}
        frame.update(frame_add)

        df_weights_all = pd.concat([df_weights_all, weight_target_vol], axis=1)

    portfolios = pd.DataFrame(frame)

    return [portfolios, df_weights_all]


def risk_contributions(port_weights=None, returns=None, risk_measure="MV", plot=True):
    """
    port_weights (pd.DataFrame): dataframe of asset weights. Assets are the index
    returns (pd.DataFrame): Returns of your assets. Date is an index
    risk_measure (str): string that defines the risk measure: MV for Variance and MSV for semi-Variance
    plot(boolonean): True to plot the risk contributions, otherwise returns just risk contributions

    retunrs:
        a plot of risk contributions or an array of risk contributions
    """

    cov = returns.cov()
    if plot:
        ax = rp.plot_risk_con(
            port_weights, cov=cov, returns=returns, rm=risk_measure, color="tab:blue", height=6, width=10, ax=None
        )
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
        return ax
    else:
        risk_cont = rp.Risk_Contribution(port_weights, cov=cov, returns=returns, rm=risk_measure)
        return np.round(risk_cont, 3)


# =============================================================================
# SKFOLIO INTEGRATION: Schur Complementary Allocation
# =============================================================================

def schur_complementary(
    returns_training: pd.DataFrame,
    gamma: float = 0.5,
    covariance: str = "ledoit",
    distance: str = "pearson",
    linkage: str = "ward",
    weight_min: float = 0.0,
    weight_max: float = 1.0,
    keep_monotonic: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Schur Complementary Allocation - Unifying HRP and Minimum Variance.

    This method uses Schur-complement-inspired augmentation of sub-covariance
    matrices, revealing a link between Hierarchical Risk Parity (HRP) and
    minimum-variance portfolios (MVP).

    By tuning the regularization factor `gamma`, the method smoothly
    interpolates from HRP (gamma=0) to MVP (gamma->1).

    Based on Peter Cotton's 2024 paper: "Schur Complementary Allocation:
    A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios"

    Parameters
    ----------
    returns_training : pd.DataFrame
        DataFrame of asset returns with assets as columns and dates as index.

    gamma : float, default=0.5
        Regularization factor in [0, 1].
        - gamma = 0: equivalent to HRP (no off-diagonal information)
        - gamma -> 1: approaches minimum variance solution
        Higher gamma uses more correlation information but requires better
        conditioned covariance matrix.

    covariance : str, default='ledoit'
        Covariance estimation method:
        - 'empirical': Historical covariance
        - 'ledoit': Ledoit-Wolf shrinkage (recommended)
        - 'shrunk': Basic shrinkage

    distance : str, default='pearson'
        Distance measure for hierarchical clustering:
        - 'pearson': Pearson correlation distance
        - 'spearman': Spearman rank correlation distance
        - 'kendall': Kendall tau distance

    linkage : str, default='ward'
        Linkage method for hierarchical clustering:
        - 'single', 'complete', 'average', 'ward', 'weighted', 'centroid', 'median'

    weight_min : float, default=0.0
        Minimum weight constraint for each asset.

    weight_max : float, default=1.0
        Maximum weight constraint for each asset.

    keep_monotonic : bool, default=True
        If True, ensures portfolio variance decreases monotonically with gamma.
        This guarantees variance(Schur) <= variance(HRP).

    **kwargs
        Additional arguments passed to SchurComplementary.

    Returns
    -------
    pd.DataFrame
        DataFrame with asset weights, index is asset names, column is 'port_weight'.

    Raises
    ------
    ImportError
        If skfolio is not installed.

    Examples
    --------
    >>> # Basic usage
    >>> weights = schur_complementary(returns, gamma=0.5)
    >>>
    >>> # More aggressive (closer to MVP)
    >>> weights = schur_complementary(returns, gamma=0.8, covariance='ledoit')
    >>>
    >>> # Conservative (closer to HRP)
    >>> weights = schur_complementary(returns, gamma=0.2)
    """
    if not SKFOLIO_AVAILABLE:
        raise ImportError(
            "skfolio is required for Schur Complementary Allocation. "
            "Install it with: pip install skfolio"
        )

    if not isinstance(returns_training, pd.DataFrame):
        raise ValueError("returns_training must be a pandas DataFrame")

    # Map covariance estimator
    cov_estimators = {
        'empirical': EmpiricalCovariance(),
        'ledoit': LedoitWolf(),
        'shrunk': ShrunkCovariance(),
    }
    cov_est = cov_estimators.get(covariance.lower(), LedoitWolf())

    # Map distance estimator
    distance_estimators = {
        'pearson': PearsonDistance(),
        'spearman': SpearmanDistance(),
        'kendall': KendallDistance(),
    }
    dist_est = distance_estimators.get(distance.lower(), PearsonDistance())

    # Map linkage method
    linkage_methods = {
        'single': LinkageMethod.SINGLE,
        'complete': LinkageMethod.COMPLETE,
        'average': LinkageMethod.AVERAGE,
        'ward': LinkageMethod.WARD,
        'weighted': LinkageMethod.WEIGHTED,
        'centroid': LinkageMethod.CENTROID,
        'median': LinkageMethod.MEDIAN,
    }
    link_method = linkage_methods.get(linkage.lower(), LinkageMethod.WARD)

    # Build the model
    model = SchurComplementary(
        gamma=gamma,
        keep_monotonic=keep_monotonic,
        prior_estimator=EmpiricalPrior(covariance_estimator=cov_est),
        distance_estimator=dist_est,
        hierarchical_clustering_estimator=HierarchicalClustering(linkage_method=link_method),
        min_weights=weight_min,
        max_weights=weight_max,
        **kwargs
    )

    # Fit the model
    model.fit(returns_training)

    # Extract weights
    weights = pd.DataFrame(
        model.weights_,
        index=returns_training.columns,
        columns=['port_weight']
    )
    weights.index.name = 'asset'

    return weights


def conditional_covariance(
    covariance_matrix: pd.DataFrame,
    core_assets: list,
    satellite_assets: list = None,
) -> pd.DataFrame:
    """
    Compute the Schur complement (conditional covariance) of satellite assets
    given core assets.

    The Schur complement mathematically isolates what satellite assets contribute
    CONDITIONAL on the core holdings, removing what is already explained by
    correlations with the core.

    For a covariance matrix partitioned as:
        Σ = [Σ_CC    Σ_CS ]
            [Σ_SC    Σ_SS ]

    The Schur complement is:
        S = Σ_SS - Σ_SC @ inv(Σ_CC) @ Σ_CS

    This S represents the residual covariance of satellites after conditioning
    on the core.

    Parameters
    ----------
    covariance_matrix : pd.DataFrame
        Full covariance matrix with asset names as index and columns.

    core_assets : list
        List of asset names that form the "core" holdings (e.g., equity funds).

    satellite_assets : list, optional
        List of asset names that are potential diversifiers. If None, uses all
        non-core assets.

    Returns
    -------
    pd.DataFrame
        Conditional covariance matrix of satellite assets given core.

    Examples
    --------
    >>> # Define core (equity) and satellite (diversifiers)
    >>> core = ['SPY', 'QQQ', 'IWM']
    >>> satellites = ['AGG', 'GLD', 'TIP', 'VNQ']
    >>>
    >>> # Get conditional covariance
    >>> cond_cov = conditional_covariance(cov_matrix, core, satellites)
    >>>
    >>> # Assets with high conditional variance provide TRUE diversification
    >>> # Assets with low conditional variance are redundant given the core
    """
    all_assets = covariance_matrix.index.tolist()

    if satellite_assets is None:
        satellite_assets = [a for a in all_assets if a not in core_assets]

    # Validate assets exist
    missing_core = [a for a in core_assets if a not in all_assets]
    missing_sat = [a for a in satellite_assets if a not in all_assets]
    if missing_core:
        raise ValueError(f"Core assets not in covariance matrix: {missing_core}")
    if missing_sat:
        raise ValueError(f"Satellite assets not in covariance matrix: {missing_sat}")

    # Extract blocks
    Sigma_CC = covariance_matrix.loc[core_assets, core_assets].values
    Sigma_SS = covariance_matrix.loc[satellite_assets, satellite_assets].values
    Sigma_CS = covariance_matrix.loc[core_assets, satellite_assets].values
    Sigma_SC = covariance_matrix.loc[satellite_assets, core_assets].values

    # Compute Schur complement: S = Σ_SS - Σ_SC @ inv(Σ_CC) @ Σ_CS
    try:
        Sigma_CC_inv = np.linalg.inv(Sigma_CC)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        Sigma_CC_inv = np.linalg.pinv(Sigma_CC)

    schur_complement = Sigma_SS - Sigma_SC @ Sigma_CC_inv @ Sigma_CS

    return pd.DataFrame(
        schur_complement,
        index=satellite_assets,
        columns=satellite_assets
    )


def redundancy_analysis(
    returns_training: pd.DataFrame,
    core_assets: list,
    satellite_assets: list = None,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Analyze which satellite assets provide TRUE diversification vs being
    mathematically redundant given the core holdings.

    Uses the Schur complement to measure the conditional variance of each
    satellite asset after removing what is explained by correlations with
    the core.

    Parameters
    ----------
    returns_training : pd.DataFrame
        DataFrame of asset returns.

    core_assets : list
        List of asset names forming the core holdings (e.g., equity exposure).

    satellite_assets : list, optional
        List of potential diversifier assets. If None, uses all non-core assets.

    threshold : float, default=0.1
        Variance retention threshold. Assets retaining less than this fraction
        of their original variance (after conditioning on core) are flagged
        as potentially redundant.

    Returns
    -------
    pd.DataFrame
        Analysis results with columns:
        - 'unconditional_var': Original variance of the asset
        - 'conditional_var': Variance after conditioning on core
        - 'variance_retained': Fraction of variance retained (conditional/unconditional)
        - 'is_redundant': Boolean flag if variance_retained < threshold
        - 'diversification_value': Score indicating true diversification benefit

    Examples
    --------
    >>> # Define your core equity holdings
    >>> core = ['MSFT', 'NVDA', 'CRWD', 'V', 'JPM']
    >>>
    >>> # Analyze which diversifiers are truly independent
    >>> analysis = redundancy_analysis(returns, core)
    >>>
    >>> # Show which assets are redundant
    >>> print(analysis[analysis['is_redundant']])
    >>>
    >>> # Show best diversifiers (highest variance retained)
    >>> print(analysis.sort_values('variance_retained', ascending=False))
    """
    all_assets = returns_training.columns.tolist()

    if satellite_assets is None:
        satellite_assets = [a for a in all_assets if a not in core_assets]

    if not satellite_assets:
        raise ValueError("No satellite assets to analyze. All assets are in core.")

    # Compute covariance matrix
    cov_matrix = returns_training.cov()

    # Get conditional covariance
    cond_cov = conditional_covariance(cov_matrix, core_assets, satellite_assets)

    # Extract diagonal (variances)
    unconditional_var = pd.Series(
        np.diag(cov_matrix.loc[satellite_assets, satellite_assets].values),
        index=satellite_assets
    )
    conditional_var = pd.Series(
        np.diag(cond_cov.values),
        index=satellite_assets
    )

    # Handle numerical issues (small negative values)
    conditional_var = conditional_var.clip(lower=0)

    # Compute variance retained ratio
    variance_retained = conditional_var / unconditional_var
    variance_retained = variance_retained.clip(lower=0, upper=1)

    # Build result DataFrame
    result = pd.DataFrame({
        'unconditional_var': unconditional_var,
        'conditional_var': conditional_var,
        'variance_retained': variance_retained,
        'is_redundant': variance_retained < threshold,
        'diversification_value': variance_retained,  # Higher = better diversifier
    })

    result.index.name = 'asset'
    result = result.sort_values('diversification_value', ascending=False)

    return result


def compare_schur_vs_hrp(
    returns_training: pd.DataFrame,
    gamma_values: list = None,
    covariance: str = "ledoit",
    weight_min: float = 0.0,
    weight_max: float = 1.0,
) -> pd.DataFrame:
    """
    Compare portfolio weights across different gamma values, showing the
    transition from HRP (gamma=0) to MVP (gamma->1).

    Parameters
    ----------
    returns_training : pd.DataFrame
        DataFrame of asset returns.

    gamma_values : list, optional
        List of gamma values to compare. Default: [0, 0.25, 0.5, 0.75, 1.0]

    covariance : str, default='ledoit'
        Covariance estimation method.

    weight_min : float, default=0.0
        Minimum weight constraint.

    weight_max : float, default=1.0
        Maximum weight constraint.

    Returns
    -------
    pd.DataFrame
        Comparison of weights across gamma values. Columns are gamma values,
        rows are assets.

    Examples
    --------
    >>> comparison = compare_schur_vs_hrp(returns)
    >>> print(comparison)
    >>>
    >>> # See how gold allocation changes from HRP to MVP
    >>> print(comparison.loc['GLD'])
    """
    if not SKFOLIO_AVAILABLE:
        raise ImportError("skfolio is required. Install with: pip install skfolio")

    if gamma_values is None:
        gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {}
    for gamma in gamma_values:
        weights = schur_complementary(
            returns_training,
            gamma=gamma,
            covariance=covariance,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        col_name = f"gamma={gamma}" if gamma < 1 else "gamma≈1 (MVP)"
        if gamma == 0:
            col_name = "gamma=0 (HRP)"
        results[col_name] = weights['port_weight']

    comparison = pd.DataFrame(results)
    comparison.index.name = 'asset'

    return comparison
