import pandas as pd
import multiprocessing as mp

from datetime import date, timedelta
from timebudget import timebudget
from joblib import Parallel, delayed
from tqdm import tqdm

from finvizfinance.screener.overview import Overview
from myPortfolioManagement.base import _add_stock_main_info, _load_stock, _load_fx

from openbb import obb


# obb.user.credentials.fmp_api_key = 'eb50221eaef20292fe4b57f675be8b23'

def func_adj_fx(prices: pd.DataFrame, yahoo_tickers: list, base_currency: str = 'GBP'):
    col_order_original = prices.columns

    # find foreign stocks stocks
    df_stock_main_info = get_stock_main_info(yahoo_tickers)
    df_stock_main_info = df_stock_main_info[['yahooTicker', 'longName', 'currency']]
    df_temp = df_stock_main_info[df_stock_main_info['currency'] != base_currency]

    which_com_to_adjust = list(df_temp['longName'])
    prices_mini = prices[which_com_to_adjust]
    price_adj_fx = prices_mini.copy()
    srt_date = price_adj_fx.index[0]
    df_lists = []
    for company, cur in zip(df_temp['longName'], df_temp['currency']):
        fx = cur + base_currency
        df_fx_temp = _load_fx(fx, start_date=srt_date)
        df_adj_temp = pd.concat([prices_mini[company], df_fx_temp], axis=1)
        df_adj_temp[company + '_fx_adj'] = df_adj_temp[company] * df_adj_temp[fx]
        df_lists.append(df_adj_temp)

    df_adj_prices = pd.concat(df_lists, axis=1)
    df_adj_prices = df_adj_prices.filter(like='_fx_adj')
    df_adj_prices.columns = which_com_to_adjust
    priced_non_adj = prices.drop(which_com_to_adjust, axis=1)
    prices_adj = pd.concat([priced_non_adj, df_adj_prices], axis=1)
    prices_adj = prices_adj[col_order_original]

    return prices_adj


# function to get prices for a list of stocks
@timebudget
def get_stock_prices(yahoo_tickers: list, start_date: str = None,
                     end_date: str = None,
                     time_interval: str = 'daily', fix_data: bool = False, auto_adjust: bool = False,
                     adj_fx: bool = False, base_currency: str = 'GBP',
                     wide_format=False,
                     **kwarg):
    ncpus = max(mp.cpu_count() - 1, 1)
    results_temp = Parallel(n_jobs=ncpus, prefer="threads")(
        delayed(_load_stock)(yahoo_ticker=tic, start_date=start_date,
                             end_date=end_date,
                             time_interval=time_interval,
                             fix_data=fix_data, auto_adjust=auto_adjust, **kwarg) for tic in
        tqdm(yahoo_tickers))

    # collect dataframes
    results = pd.concat(results_temp)

    # get additional info and merge
    df = get_stock_main_info(yahoo_tickers)
    df_all = results.reset_index().merge(df, on='yahooTicker')
    df_all = df_all.set_index('Date')

    # clean columns
    df_all.columns = [x.lower() for x in df_all.columns]
    df_all.columns = [x.replace(" ", "") for x in df_all.columns]
    df_all = df_all.rename(columns={'longname': 'name'})

    if adj_fx:
        prices = df_all.pivot(values='adjclose', columns='name')
        df_all = func_adj_fx(prices=prices, yahoo_tickers=yahoo_tickers, base_currency=base_currency)

    if wide_format:
        df_all = df_all.pivot(values='adjclose', columns='name')

    return df_all


def get_sp500_tickers() -> pd.DataFrame:
    """
    Returns:
        df (TYPE): DESCRIPTION.
    """

    # for filtering: https://finviz.com/screener.ashx
    foverview = Overview()
    filters_dict = {'Index': 'S&P 500'}
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    return df


def get_nasdaq_tickers() -> pd.DataFrame:
    """
    Returns:
        df (TYPE): DESCRIPTION.
    """

    # for filtering: https://finviz.com/screener.ashx
    foverview = Overview()
    filters_dict = {'Exchange': 'NASDAQ'}
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    return df


#
# @timebudget
# def get_stock_info(yahoo_tickers: list = None, data_type: str = 'overview'):
#     if len(yahoo_tickers) > 800:
#         chunk_size = 200
#         chunks_list = list(chunk_the_list(yahoo_tickers, n=chunk_size))
#         df_temp_list = [_helper_get_stock_info(yahoo_tickers=ticks, data_type=data_type) for ticks in chunks_list]
#         df = pd.concat(df_temp_list)
#     else:
#         df = _helper_get_stock_info(yahoo_tickers=yahoo_tickers, data_type=data_type)
#
#     return df


def get_stock_main_info(yahoo_tickers: list = None):
    ncpus = max(mp.cpu_count() - 1, 1)
    results = Parallel(n_jobs=ncpus, prefer="threads")(
        delayed(_add_stock_main_info)(yahoo_ticker=tic) for tic in tqdm(yahoo_tickers))
    return pd.concat(results, ignore_index=True)
