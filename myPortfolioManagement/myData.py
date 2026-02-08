import multiprocessing as mp

import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from timebudget import timebudget
from tqdm import tqdm

try:
    from finvizfinance.screener.overview import Overview
except ModuleNotFoundError:  # optional dependency
    Overview = None
from myPortfolioManagement.myUtils import _load_stock, _load_fx, _add_stock_info


def func_adj_fx(prices: pd.DataFrame, yahoo_tickers: list, base_currency: str = "GBP"):
    col_order_original = prices.columns
    names = "longName"

    # find foreign stocks stocks
    df_stock_main_info = get_stock_info(yahoo_tickers)
    df_stock_main_info = df_stock_main_info[["yahooTicker", names, "currency"]]
    df_temp = df_stock_main_info[df_stock_main_info["currency"] != base_currency]

    # If no FX adjustment needed, return original prices
    if df_temp.empty:
        return prices

    which_com_to_adjust = list(df_temp[names])

    prices_mini = prices[which_com_to_adjust]
    price_adj_fx = prices_mini.copy()
    srt_date = price_adj_fx.index[0]
    df_lists = []
    for company, cur in zip(df_temp[names], df_temp["currency"]):
        fx = cur + base_currency
        df_fx_temp = _load_fx(fx, start_date=srt_date)
        df_adj_temp = pd.concat([prices_mini[company], df_fx_temp], axis=1)
        df_adj_temp[company + "_fx_adj"] = df_adj_temp[company] * df_adj_temp[fx]
        df_lists.append(df_adj_temp)

    df_adj_prices = pd.concat(df_lists, axis=1)
    df_adj_prices = df_adj_prices.filter(like="_fx_adj")
    df_adj_prices.columns = which_com_to_adjust
    priced_non_adj = prices.drop(which_com_to_adjust, axis=1)
    prices_adj = pd.concat([priced_non_adj, df_adj_prices], axis=1)
    prices_adj = prices_adj[col_order_original]

    return prices_adj


# function to get prices for a list of stocks
@timebudget
def get_stock_prices(
    yahoo_tickers: list = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = "daily",
    fix_data: bool = False,
    auto_adjust: bool = False,
    adj_fx: bool = False,
    base_currency: str = "GBP",
    wide_format=False,
    **kwarg
) -> pd.DataFrame:
    """
    :param yahoo_tickers: list of yahoo tickers. Defaults to None
    :param start_date: (str) of start date in YYYY-MM-DD format. For example, 2020-01-20. None assumes max sample
    :param end_date: (str) of start date in YYYY-MM-DD format. For example, 2020-01-20. None assumes latest available price
    :param freq: str 'daily', 'monthly', 'quarterly'
    :param fix_data: (bool) Detect currency unit 100x mixups and attempt repair. Default is False
    :param auto_adjust: (bool) Adjust all OHLC automatically? Default is True
    :param adj_fx: (bool) adjust prices to foreign currencies, so all prices are quoted on the same (base) currency
    :param base_currency: (str) the base currency chosen if adj_fix is set to True
    :param wide_format: (bool) the format of the output. This can be long or wide format. long format returns info of the stocks
    :param kwarg: any other parameters. See qs.stock.history(
    :return: (dataframe)
    """
    if not yahoo_tickers:
        raise ValueError("yahoo_tickers must be a non-empty list of ticker symbols")

    ncpus = max(mp.cpu_count() - 1, 1)
    results_temp = Parallel(n_jobs=ncpus, prefer="threads")(
        delayed(_load_stock)(
            yahoo_ticker=tic,
            start_date=start_date,
            end_date=end_date,
            time_interval=freq,
            fix_data=fix_data,
            auto_adjust=auto_adjust,
            **kwarg
        )
        for tic in tqdm(yahoo_tickers)
    )

    # collect dataframes
    results = pd.concat(results_temp)
    df = get_stock_info(yahoo_tickers)
    df_all = results.reset_index().merge(df, on="yahooTicker")
    df_all = df_all.set_index("Date")

    # clean columns
    # df_all = df_all.rename(columns={'longName': 'name'})

    df_all.columns = [x.lower() for x in df_all.columns]
    df_all.columns = [x.replace(" ", "") for x in df_all.columns]

    if adj_fx:
        wide_format = False
        prices = df_all.pivot(values="adjclose", columns="longname")
        df_all = func_adj_fx(prices=prices, yahoo_tickers=yahoo_tickers, base_currency=base_currency)

    if wide_format:
        df_all = df_all.pivot(values="adjclose", columns="longname")

    return df_all


def get_sp500_tickers() -> pd.DataFrame:
    """
    Returns:
        df (TYPE): DESCRIPTION.
    """

    if Overview is None:
        raise ModuleNotFoundError(
            "finvizfinance is required for get_sp500_tickers(). Install it with: pip install finvizfinance"
        )

    # for filtering: https://finviz.com/screener.ashx
    foverview = Overview()
    filters_dict = {"Index": "S&P 500"}
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    return df


def get_nasdaq_tickers() -> pd.DataFrame:
    """
    Returns:
        df (TYPE): DESCRIPTION.
    """

    if Overview is None:
        raise ModuleNotFoundError(
            "finvizfinance is required for get_nasdaq_tickers(). Install it with: pip install finvizfinance"
        )

    # for filtering: https://finviz.com/screener.ashx
    foverview = Overview()
    filters_dict = {"Exchange": "NASDAQ"}
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    return df


def get_stock_info(yahoo_tickers: list = None):
    # Reduce parallel jobs to avoid rate limiting
    ncpus = min(2, max(mp.cpu_count() - 1, 1))
    results = Parallel(n_jobs=ncpus, prefer="threads")(
        delayed(_add_stock_info)(yahoo_ticker=tic) for tic in tqdm(yahoo_tickers)
    )
    # Filter out empty DataFrames
    results = [df for df in results if not df.empty]
    if not results:
        print("Warning: No stock info could be retrieved")
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def get_option_exp_dates(yahoo_ticker: str):
    stock_info = yf.Ticker(yahoo_ticker)
    df_op_exp = pd.DataFrame(stock_info.options)
    df_op_exp.columns = ["Exp_Date"]
    df_op_exp["Stock"] = yahoo_ticker
    return df_op_exp
