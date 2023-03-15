import pandas as pd
import ffn
import quantstats as qs
from scipy import stats as scipy_stats
import numpy as np


def probabilistic_sharpe_ratio(returns, returns_bench=None, sr_benchmark=False):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.
    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """

    if sr_benchmark:
        if type(returns_bench) == pd.DataFrame:
            sr_benchmark = pd.Series(returns_bench.mean() / returns_bench.std(ddof=1), index=returns_bench.columns)
        elif type(returns_bench) not in (float, np.float64):
            sr_benchmark = pd.Series(returns_bench.mean() / returns_bench.std(ddof=1))[0]
    else:
        sr_benchmark = 0

    T = len(returns)
    kr = pd.Series(scipy_stats.kurtosis(returns, fisher=False), index=returns.columns)
    sk = pd.Series(scipy_stats.skew(returns), index=returns.columns)  # this should be set to zero for small samples
    sr = pd.Series(returns.mean() / returns.std(ddof=1), index=returns.columns)  # sharp non-annualized
    numer = (sr - sr_benchmark) * ((T - 1) ** 0.5)
    den = (1 - sk * sr + ((kr - 1) * sr ** 2) / 4) ** 0.5
    psr = scipy_stats.norm.cdf(numer / den)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def cagr(df_prices: pd.DataFrame or pd.Series):
    if isinstance(df_prices, pd.Series):
        return ffn.core.calc_cagr(df_prices.dropna())
    else:
        if isinstance(df_prices.index, pd.DatetimeIndex) == False:
            raise ValueError("Index is not a date")
        cagr_list = []
        for series in df_prices.columns:
            cagr_temp = pd.Series(ffn.core.calc_cagr(df_prices[series].dropna()))
            cagr_list.append(cagr_temp)
        cagrs = pd.concat(cagr_list)
        cagrs.index = list(df_prices.columns)
        return cagrs


def drawdown_details(prices: pd.Series, top_drawdowns: int = 5):
    prices = pd.Series(prices)
    if isinstance(prices.index, pd.DatetimeIndex) == False:
        raise ValueError("Index is not a date")
    # calculate drawdown details
    drawdown_details = qs.stats.drawdown_details(qs.stats.to_drawdown_series(prices.pct_change()))
    # calculate the max and min prices in each drawdown case
    peak_price_dates = drawdown_details['start'].to_list()
    bottom_price_dates = drawdown_details['valley'].to_list()
    # peak price
    max_prices = prices[prices.index.isin(peak_price_dates)]

    # bottom price
    min_prices = prices[prices.index.isin(bottom_price_dates)]

    drawdown_details['Peak Price'] = max_prices.values
    drawdown_details['Bottom Price'] = min_prices.values

    # Inserting the column after days column
    new_col = round(drawdown_details['days'] / 365, 0).astype('int')
    drawdown_details.insert(loc=4,
                            column='Years',
                            value=new_col)

    drawdown_details = round(drawdown_details)

    return drawdown_details.drop(['99% max drawdown'], axis=1).sort_values('max drawdown', ascending=True).head(
        top_drawdowns)


def assets_drawdown_details(df_prices: pd.DataFrame, **kwargs):
    drawdown_details_list = []
    for series in df_prices.columns:
        temp = df_prices[series]
        temp2 = drawdown_details(temp, **kwargs)
        temp2['asset'] = series
        temp2 = temp2.set_index('asset')
        drawdown_details_list.append(temp2)

    return pd.concat(drawdown_details_list)


def information_ratio(returns, benchmark):
    asset_col_name = 'assets'
    columns = returns.columns.to_list()

    info_ratio = []
    for col in columns:
        inf_r = qs.stats.information_ratio(returns[col], benchmark)
        info_ratio.append(inf_r)

    # cleaning
    temp = pd.DataFrame(info_ratio)
    temp[asset_col_name] = columns
    temp.columns = ['information_ratio', asset_col_name]
    temp = temp.set_index(asset_col_name)
    return temp
