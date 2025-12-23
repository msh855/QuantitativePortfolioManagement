import pandas as pd
from feature_engine.outliers import OutlierTrimmer
import ffn
import quantstats_lumi as qs

import yfinance as yf
from operator import itemgetter
from openbb import obb
import time
import json


def data_overview(df: pd.DataFrame,
                  my_assets_col_name: str,
                  my_date_col_name: str,
                  price_col_name: str) -> pd.DataFrame:
    """
    This function expects a dataframe in long-format and a date column
    and returns a pandas dataframe that shows the period of available data
    that are available for each asset
    ...

    Args:
          df (dataframe): A list of yahoo tickers
          my_assets_col_name(str): the name of the column of
                                   your dataframe with
                                   asset names (e.g tickers, names etc..)
          my_date_col_name (str): the name of the column of your dataframe with the dates
                                  which column in your dataframe.
                                  Can be your index name

    Returns:
      pandas Dataframe: Returns a pandas dataframe with stock prices and other info
    """

    df_overview = df

    if type(df_overview.index) == pd.DatetimeIndex:
        df_overview = df_overview.reset_index()

    df_overview[my_date_col_name] = pd.to_datetime(df_overview[my_date_col_name], format='%Y/%m/%d')

    df1 = df_overview[df_overview.groupby(my_assets_col_name).Date.transform('min') == df_overview[my_date_col_name]][
        [my_assets_col_name, my_date_col_name]]
    df2 = df_overview[df_overview.groupby(my_assets_col_name).Date.transform('max') == df_overview[my_date_col_name]][
        [my_assets_col_name, my_date_col_name]]
    df1 = df1.rename(columns={my_date_col_name: "Date_min"})
    df2 = df2.rename(columns={my_date_col_name: "Date_max"})

    # merge dataframes
    df_overview = df1.merge(df2, how="left")

    df_overview["trading_days"] = df_overview["Date_max"] - df_overview["Date_min"]
    df_overview['trading_days'] = [df_overview['trading_days'][i].days for i in range(len(df_overview['trading_days']))]
    df_overview['years_available'] = df_overview['trading_days']/365


    df_overview = df_overview.rename(columns={'Stock': my_assets_col_name})
    df_overview = df_overview.drop_duplicates()

    # from long to wide

    df.groupby(my_assets_col_name)

    df_wide = df.pivot_table(index=my_date_col_name,
                             columns=my_assets_col_name,
                             values=price_col_name)

    df_NAs = pd.DataFrame(pd.Series(df_wide.isnull().mean().round(4).mul(100).sort_values(ascending=False),
                                    name='percentage_of_NAs'))

    df_overview = df_overview.merge(df_NAs, on=my_assets_col_name)
    df_overview = df_overview.set_index(my_assets_col_name)

    return df_overview.sort_values('years_available', ascending=False)


def cap_outliersTS(returns: pd.DataFrame = None, capping_method='iqr',
                   tail='both',
                   fold=5,
                   plot: bool = False, **kwargs):
    # ref: https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/OutlierTrimmer.ipynb
    # ref: https://feature-engine.trainindata.com/en/latest/user_guide/outliers/OutlierTrimmer.html
    capper = OutlierTrimmer(capping_method=capping_method,
                            tail=tail,
                            fold=fold, **kwargs)
    capper.fit(returns)
    # capper.right_tail_caps_  # outlier values
    train_t = capper.transform(returns)
    prices_nomalised = ffn.to_price_index(train_t.dropna(), start=100)
    if plot:
        prices_nomalised.plot()
    else:
        return prices_nomalised


def convert_date_index(df):
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Format the datetime index
    df.index = df.index.strftime('%Y-%m-%d')

    df.index = pd.to_datetime(df.index)

    return df


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

def balance_dates_robust(returns, returns_benchmark):
    """
    Robust date balancing with proper NaN handling
    
    Args:
        returns: Return series or DataFrame
        returns_benchmark: Benchmark return series
    
    Returns:
        Tuple of (returns_balanced, benchmark_balanced)
    """
    # Convert to Series if needed
    if isinstance(returns_benchmark, pd.DataFrame):
        returns_benchmark = returns_benchmark.squeeze()
    
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.squeeze()
    
    # Remove NaNs first
    returns = returns.dropna()
    returns_benchmark = returns_benchmark.dropna()
    
    # Find common dates using inner join
    df_combined = pd.DataFrame({
        'returns': returns,
        'benchmark': returns_benchmark
    }).dropna()  # Remove any remaining NaNs
    
    returns_balanced = df_combined['returns']
    benchmark_balanced = df_combined['benchmark']
    
    return returns_balanced, benchmark_balanced
# cleaning
def _helper(df: pd.DataFrame = None, series: pd.Series = None, n_samples: int = None) -> pd.DataFrame:
    #string_name = series.name
    #cols = [string_name + '_path' + str(x) for x in range(1, n_samples + 1)]
    cols = ['path' + str(x) for x in range(1, n_samples + 1)]
    df.columns = cols
    df.index = series.index

    return df



def check_date_index(df):
    """

    :param df:
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a date index")



#
# def _helper_get_stock_info(yahoo_tickers: list = None, data_type: str = 'overview') -> pd.DataFrame:
#     """
#
#     :param yahoo_tickers:
#     :param data_type:
#     :return:
#     """
#     if data_type == 'all':
#         funs = openbb.stocks.ca.screener
#         datatypes = ['overview', 'valuation', 'financial', 'ownership', 'performance', 'technical']
#         df_list = [funs(similar=yahoo_tickers, data_type=datatype) for datatype in datatypes]
#         df = pd.concat(df_list, axis=1)
#         df = ffn.drop_duplicate_cols(df)
#     else:
#         df = openbb.stocks.ca.screener(similar=yahoo_tickers, data_type=data_type)
#
#     df.set_index('Ticker', inplace=True)
#     df = _fix_missing(df)
#     return df


def _fix_missing(df):
    df['Sector'].fillna('Unclassified', inplace=True)
    df['Country'].fillna('Unclassified', inplace=True)
    df['Industry'].fillna('Unclassified', inplace=True)
    return df


def clean_stock_prices(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ret_raw = prices.pct_change()
    ret_raw.dropna(inplace=True)
    prices_capped = cap_outliersTS(ret_raw, **kwargs)
    ret_bench = prices_capped.pct_change()
    return ret_bench


def _add_stock_sectors(yahoo_ticker: str = None) -> pd.DataFrame:
    try:
        stock_info = yf.Ticker(yahoo_ticker)
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        d = stock_info.info
        
        # Check if info is empty
        if not d:
            print(f"Warning: Empty info for {yahoo_ticker}")
            return pd.DataFrame()

        mykeys_exp = ['industry', 'sector', 'country']

        if not d.keys() & {'industry', 'sector', 'country'}:
            str_other = 'Other'
            data = {'industry': [str_other],
                    'sector': [str_other],
                    'country': [str_other]}
            df_dv = pd.DataFrame(data)

        else:
            dv = itemgetter('industry', 'sector', 'country')(
                d)
            df_dv = pd.DataFrame([dv], columns=mykeys_exp)

        df_dv['yahooTicker'] = yahoo_ticker
        order = ['yahooTicker'] + mykeys_exp
        return df_dv[order]
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Could not fetch sector info for {yahoo_ticker}: {type(e).__name__}")
        return pd.DataFrame()


def _add_stock_types(yahoo_ticker: str = None) -> pd.DataFrame:
    try:
        stock_info = yf.Ticker(yahoo_ticker)
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        d = stock_info.info
        
        # Check if info is empty
        if not d:
            print(f"Warning: Empty info for {yahoo_ticker}")
            return pd.DataFrame()
            
        mykeys_l = ['type', 'longName', 'exchange', 'currency']
        dv = itemgetter('quoteType', 'longName', 'exchange', 'currency')(
            d)
        df_dv = pd.DataFrame([dv], columns=mykeys_l)
       #df_dv.rename(columns={'longName': "name"})

        df_dv['currency'] = [x.upper() for x in df_dv['currency']]
        df_dv['yahooTicker'] = yahoo_ticker
        order = ['yahooTicker'] + mykeys_l
        return df_dv[order]
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Could not fetch type info for {yahoo_ticker}: {type(e).__name__}")
        return pd.DataFrame()


def _add_stock_info(yahoo_ticker: str = None) -> pd.DataFrame:
    try:
        df1 = _add_stock_sectors(yahoo_ticker)
        df2 = _add_stock_types(yahoo_ticker)
        
        # Check if either dataframe is empty
        if df1.empty or df2.empty:
            return pd.DataFrame()
            
        df_info = df2.merge(df1, on='yahooTicker')
        return df_info
    except Exception as e:
        print(f"Warning: Could not process {yahoo_ticker}: {type(e).__name__}")
        return pd.DataFrame()


def _load_stock(yahoo_ticker: str, period: str = 'max', start_date: str = None,
                end_date: str = None, time_interval: str = 'daily', fix_data: bool = False, **kwarg):
    time_interval_mapping = {
        'daily': '1d',
        'monthly': '1M',
        'quarterly': '3M'
    }

    time_interval_temp = time_interval_mapping[time_interval]

    stock = yf.Ticker(yahoo_ticker)
    prices = stock.history(period=period, interval=time_interval_temp,
                           start=start_date, end=end_date, repair=fix_data, **kwarg)
    prices = convert_date_index(prices)
    prices['yahooTicker'] = yahoo_ticker
    return prices


def _load_fx(cross: str = "EURUSD", start_date: str = '1950-01-01', end_date: str = None) -> pd.DataFrame:
    df_fx = obb.currency.price.historical(symbol=cross, start_date=start_date, end_date=end_date,
                                          provider='yfinance').to_df()
    df_fx = df_fx[['close']]
    df_fx.index.name = 'Date'
    df_fx.columns = [cross]
    return df_fx
