import pandas as pd
import quantstats as qs
from openbb_terminal.sdk import openbb
import ffn
pd.options.mode.use_inf_as_na = True  # show NAs instead of inf


def balance_dates(returns, returns_benchmark):
    """

    :param returns:
    :param returns_benchmark:
    :return:
    """
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


def check_date_index(df):
    """

    :param df:
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a date index")


def data_check_TS(x) -> pd.DataFrame or pd.Series:
    # check if dataframe has a date index
    check_date_index(x)

    if isinstance(x, pd.DataFrame):
        if isinstance(x, pd.DataFrame) == True & x.shape[1] == 1:  # number of columns less than one convert to series
            x = x.iloc[:, 0]
            return x
        else:
            raise 'You passed a dataframe with one column. Pass pandas series'


def chunk_the_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def _helper_get_stock_info(yahoo_tickers: list = None, data_type: str = 'overview') -> pd.DataFrame:
    """

    :param yahoo_tickers:
    :param data_type:
    :return:
    """
    if data_type == 'all':
        funs = openbb.stocks.ca.screener
        datatypes = ['overview', 'valuation', 'financial', 'ownership', 'performance', 'technical']
        df_list = [funs(similar=yahoo_tickers, data_type=datatype) for datatype in datatypes]
        df = pd.concat(df_list, axis=1)
        df = ffn.drop_duplicate_cols(df)
    else:
        df = openbb.stocks.ca.screener(similar=yahoo_tickers, data_type=data_type)

    df.set_index('Ticker', inplace=True)
    df = _fix_missing(df)
    return df


def _fix_missing(df):
    df['Sector'].fillna('Unclassified', inplace=True)
    df['Country'].fillna('Unclassified', inplace=True)
    df['Industry'].fillna('Unclassified', inplace=True)
    return df
