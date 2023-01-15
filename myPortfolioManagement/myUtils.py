import pandas as pd
import numpy as np
import investpy
import quantstats as qs

pd.options.mode.use_inf_as_na = True  # show NAs instead of inf


def balance_dates(returns, returns_benchmark):
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


def which_missing_in_investpy(list_to_check: list, what_to_check='isin'):
    """
    This function checks which ISIN numbers or Symbols are missing 
    based on all available stocks, etfs, trusts and mutual funds that 
    that are available in investpy library. 
    Returns a list with the missing assets 
    ...
    
    Args:
          list_to_check (list): A list containing either the isin number 
                                or symbols 
                                
           what_to_check(string): string can be either 
                                  'isin' (default) or 'symbol'
               
    Returns:
      list: Returns a list with the isin or symbols of the missing assets 
    """

    # load all available data 
    funds_all = investpy.get_funds(country=None)
    stocks_all = investpy.get_stocks(country=None)
    etfs_all = investpy.etfs.get_etfs(country=None)

    # clear None and NAs 
    funds_all = funds_all.fillna(value=np.nan).dropna(subset=[what_to_check])
    funds_isin_list = funds_all[what_to_check].to_list()

    # clear None and NAs 
    stocks_all = stocks_all.fillna(value=np.nan).dropna(subset=[what_to_check])
    stock_list = stocks_all[what_to_check].to_list()

    # clear None and NAs 
    etfs_all = etfs_all.fillna(value=np.nan).dropna(subset=[what_to_check])
    etfs_list = etfs_all[what_to_check].to_list()

    all_isin_available = funds_isin_list + stock_list + etfs_list

    not_in_mylist = list(set(list_to_check) - set(all_isin_available))

    return not_in_mylist


def which_in_investpy(list_to_check: list, what_to_check='isin'):
    """
    This function checks which ISIN numbers or Symbols exist
    in the available database
    Returns a pandas dataframe with the available assets.  
    ...
    
    Args:
          list_to_check (list): A list containing either the isin number 
                                or symbols 
                                
           what_to_check(string): string can be either 
                                  'isin' (default) or 'symbol'
               
    Returns:
      list: Returns a list with the isin or symbols of the missing assets 
    """

    keep_columns = ['name', 'country', 'symbol', 'isin', 'asset_class']

    # load all available data 
    funds_all = investpy.get_funds(country=None)
    etfs_all = investpy.etfs.get_etfs(country=None)
    stocks_all = investpy.get_stocks(country=None)

    stocks_all['asset_class'] = 'No_info_from_Yahoo'

    # drop columns 
    funds_all = funds_all[keep_columns]
    etfs_all = etfs_all[keep_columns]
    stocks_all = stocks_all[keep_columns]

    assets_all = pd.concat([funds_all, etfs_all, stocks_all])

    assets_all = assets_all[assets_all[what_to_check].isin(list_to_check)]

    return assets_all


#
# def rebase(df_prices: pd.DataFrame or pd.Series, initial_value=1):
#     if not isinstance(df_prices.index, pd.DatetimeIndex):
#         raise ValueError('Index not a date')
#     df_prices = df_prices.interpolate(method='time', limit_direction='both')
#     return ffn.core.rebase(df_prices, value=initial_value)


def rebase(df):
    # Find the minimum starting date among all the series
    min_start = df.apply(lambda x: x.first_valid_index()).min()
    # Create a new dataframe with the aligned index
    aligned_df = pd.DataFrame(index=pd.date_range(start=min_start, end=df.index[-1]))
    # Interpolate missing values and fill them
    for col in df.columns:
        series = df[col].reindex(aligned_df.index).ffill()
        first_valid_idx = series.first_valid_index()
        if first_valid_idx:
            series = series / series[first_valid_idx]
        aligned_df[col] = series
    aligned_df.index.name = df.index.name
    return aligned_df


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

    # merge datafranes
    df_overview = df1.merge(df2, how="left")

    df_overview["trading_days"] = df_overview["Date_max"] - df_overview["Date_min"]
    df_overview['years_available'] = df_overview['trading_days'] / np.timedelta64(1, 'Y')

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
