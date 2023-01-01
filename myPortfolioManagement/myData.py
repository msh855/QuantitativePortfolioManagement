import pandas as pd  # to work datafranes

import numpy as np  # to work with vectors

import datetime as dt  # to handle dates
from datetime import date, timedelta

from timebudget import timebudget  # to time functions

import ray  # to parallelise

import investpy  # to download mutual fund and trust prices
from yahoofinancials import YahooFinancials  # to download prices from Yahoo

from os import path
import glob

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from myPortfolioManagement.myUtils import which_in_investpy  # from my library

from finvizfinance.screener.overview import Overview

from fredapi import Fred

pd.options.mode.use_inf_as_na = True  # this is instead of inf to show NAs


@ray.remote
def download_mutual_funds_prices(mutual_fund_name: str, country: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
    """
    This is a wrapper function around `investpy.get_fund_historical_data`
    from the investpy library. 
    See this link: https://investpy.readthedocs.io/ 
    ...
    
    Args:
      fund_name (string): the name of the mutual fund. 
        country (string): The country of the mutual fund 
      start_date(string): Date that prices start. Is a string with this '%Y-%m-%d' date format 
        end_date(string): Date that prices end. Is a string with this '%Y-%m-%d' date format
     
    Returns:
      pandas Dataframe: Returns a pandas dataframe
    """

    # convert to date objects 
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

    # set correct date format for investpy library 
    start_date = start_date.strftime("%d/%m/%Y")
    end_date = end_date.strftime("%d/%m/%Y")

    # download the data from investpy 
    try:
        fund_prices = investpy.get_fund_historical_data(fund=mutual_fund_name,
                                                        country=country,
                                                        from_date=start_date,
                                                        to_date=end_date)
        # add a column with funds name 
        fund_prices["fund"] = mutual_fund_name

        # convert column to lower case (this to integrate with other dataframe)
        fund_prices.columns = fund_prices.columns.str.lower()

        # rename columns 
        fund_prices = fund_prices.rename(columns={'close': 'adjclose'})

        return fund_prices
    except:
        return print('Fund not fund or API problems with Investpy')


@ray.remote
def download_etf_prices(etf_name: str, country: str,
                        start_date: str, end_date: str) -> pd.DataFrame:
    """
    This is a wrapper function around `investpy.get_etf_historical_data`
    from the investpy library. 
    See this link: https://investpy.readthedocs.io/ 
    ...
    
    Args:
      fund_name (string): the name of the mutual fund. 
        country (string): The country of the mutual fund 
      start_date(string): Date that prices start. Is a string with this '%Y-%m-%d' date format 
        end_date(string): Date that prices end. Is a string with this '%Y-%m-%d' date format
     
    Returns:
      pandas Dataframe: Returns a pandas dataframe
    """

    # convert to date objects 
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

    # set correct date format for investpy library 
    start_date = start_date.strftime("%d/%m/%Y")
    end_date = end_date.strftime("%d/%m/%Y")

    # download the data from investpy 
    etf_prices = investpy.get_etf_historical_data(etf=etf_name,
                                                  country=country,
                                                  from_date=start_date,
                                                  to_date=end_date,
                                                  as_json=False,
                                                  order='ascending')

    # add a column with funds name 
    etf_prices["ETF"] = etf_name
    return etf_prices


# function to get prices for a list of stocks
@timebudget
def get_stock_prices(yahoo_tickers: list, start_date: str = '1950-01-01',
                     end_date: str = None,
                     time_interval: str = 'daily', wide_format: bool = False, fix_data: bool = False) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")

    # removes duplicates
    yahoo_tickers = list(set(yahoo_tickers))

    # loads price data
    yahoo_financials = YahooFinancials(yahoo_tickers)
    data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                      end_date=end_date,
                                                      time_interval=time_interval)

    df = pd.DataFrame(
        [{**p, 'yahooTicker': k, 'instrumentType': v['instrumentType']} for k, v in data.items() for p in
         v['prices']]).drop(columns=['date']).set_index('formatted_date')

    if fix_data:
        df_temp = df.copy()

        # fix faulty yahoo data that jumps 100x
        df_temp = df_temp.dropna(subset=['adjclose'])
        df_temp = df_temp[df_temp['adjclose'] != 0]

        jumps_up = df_temp['adjclose'] / df_temp['adjclose'].shift() > 50
        jumps_down = df_temp['adjclose'] / df_temp['adjclose'].shift() < .02
        correction_factor = 100. ** (jumps_down.cumsum() - jumps_up.cumsum())
        df_temp['adjclose'] *= correction_factor
        df = df_temp

    # rename columns
    df.index.name = 'Date'
    # Convert to date object
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    if wide_format:
        df = df.pivot_table(index='Date',
                            columns='yahooTicker',
                            values='adjclose')

    return df


# function to ger prices for a list of mutual funds 
@timebudget
def get_mutual_funds_prices(mutual_fund_isin: list, start_date: str,
                            end_date: str = None, num_cpus: int = 1) -> pd.DataFrame:
    """
    This function downloads mutual fund prices 
    is an investpy wrapper
    ...
    
    Args:
          mutual_fund_isin (list): a list of ISIN strings 
         
           
           start_date(string): Date that prices start. 
                               Is a string with this '%Y-%m-%d' date format 
             end_date(string): Date that prices end. 
                               Is a string with this '%Y-%m-%d' date format
                                  
          num_cpus(int): The number of CPUs you want to use 
      
    Returns:
      pandas Dataframe: Returns a pandas dataframe 
    """
    if end_date == None:
        end_date = end_date = date.today() - timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")

    funds = which_in_investpy(mutual_fund_isin, what_to_check='isin')

    if len(funds.index) == 0:
        raise ValueError(' None of these ISINs exist in the database')

    # create a zip object to speed up looping below 
    fund_zipped = zip(funds.name, funds.country)

    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
    mydata = ray.get([download_mutual_funds_prices.remote(fund_name,
                                                          country_name,
                                                          start_date=start_date,
                                                          end_date=end_date)
                      for fund_name, country_name in fund_zipped])
    ray.shutdown()

    # collect results and clean     
    df_funds = pd.concat(mydata)

    return df_funds


# function to get prices for a list of ETFs
@timebudget
def get_etf_prices(etf_isins: list, start_date: str, end_date: str = None,
                   num_cpus: int = 1) -> pd.DataFrame:
    """
    This function downloads ETF prices 
    is an investpy wrapper
    ...
    
    Args:
          mutual_fund_isin (list): a list of ISIN strings 
         
           
           start_date(string): Date that prices start. 
                               Is a string with this '%Y-%m-%d' date format 
             end_date(string): Date that prices end. 
                               Is a string with this '%Y-%m-%d' date format
                                  
          num_cpus(int): The number of CPUs you want to use 
      
    Returns:
      pandas Dataframe: Returns a pandas dataframe 
    """
    if end_date == None:
        end_date = end_date = date.today() - timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")

        # search ETFs
    df_etfs = investpy.etfs.get_etfs(country=None)

    # slice according to your ETFs 
    df_etfs = df_etfs[df_etfs['symbol'].isin(etf_isins)]

    if df_etfs.empty:
        raise ValueError('No ETFs in the database')

    # create a zip object to speed up looping below 
    etf_zipped = zip(df_etfs.name, df_etfs.country)

    # loop over ETFs
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
    myetfs_data = ray.get([download_etf_prices.remote(item1, item2,
                                                      start_date,
                                                      end_date)
                           for item1, item2 in etf_zipped])
    ray.shutdown()

    return myetfs_data


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
        # my_date_col_name = df_overview.index.name
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


def load_fidelity_prices(filter_date: str = '2000-01-01') -> pd.DataFrame:
    file_type = 'csv'
    seperator = ','

    path_to_funds = '/Users/safishajjouz/GitHub/myPortfolioManagement/files/FidelityPrices/funds'
    path_to_etf_trusts = '/Users/safishajjouz/GitHub/myPortfolioManagement/files/FidelityPrices/trusts_etfs'

    # asset classes 
    subfolder_class_equity = 'Equity'
    subfolder_class_Absolute_Alpha = 'AbsoluteAlpha'
    subfolder_class_Bonds = 'Bonds'
    subfolder_class_volatility_managed = 'VolatilityManaged'
    subfolder_class_commodities = 'Commodities'
    subfolder_class_alternatives = 'Alternatives'

    # pathsto funds 
    folder_name_equity = path.join(path_to_funds, subfolder_class_equity)
    folder_name_Absolute_Alpha = path.join(path_to_funds, subfolder_class_Absolute_Alpha)
    folder_name_Bonds = path.join(path_to_funds, subfolder_class_Bonds)
    folder_name_vol_managed = path.join(path_to_funds, subfolder_class_volatility_managed)
    folder_name_commodities_funds = path.join(path_to_funds, subfolder_class_commodities)
    folder_name_commodities = path.join(path_to_etf_trusts, subfolder_class_commodities)

    # load equity funds
    dataframe_equity = pd.concat([pd.read_csv(f)
                                  for f in glob.glob(folder_name_equity + "/*." + file_type)],
                                 ignore_index=False)

    dataframe_equity['Asset_Class'] = 'Equity'

    # load Bond funds
    dataframe_Bonds = pd.concat([pd.read_csv(f, sep=seperator)
                                 for f in glob.glob(folder_name_Bonds + "/*." + file_type)],
                                ignore_index=False)

    dataframe_Bonds['Asset_Class'] = 'Bonds'

    # load Absolute Alpha funds
    dataframe_absolute_alpha = pd.concat([pd.read_csv(f)
                                          for f in glob.glob(folder_name_Absolute_Alpha + "/*." + file_type)],
                                         ignore_index=False)

    dataframe_absolute_alpha['Asset_Class'] = 'Absolute_Alpha'

    # load volatility managed funds
    dataframe_market_neutral = pd.concat([pd.read_csv(f, sep=seperator)
                                          for f in glob.glob(folder_name_vol_managed + "/*." + file_type)],
                                         ignore_index=False)

    dataframe_market_neutral['Asset_Class'] = 'Vol_managed'

    # DataFrame Commodities 
    dataframe_commodities_funds = pd.concat([pd.read_csv(f, sep=seperator)
                                             for f in glob.glob(folder_name_commodities_funds + "/*." + file_type)],
                                            ignore_index=False)

    dataframe_commodities_funds['Asset_Class'] = 'Commodities'

    dataframe1 = pd.concat([dataframe_equity, dataframe_Bonds,
                            dataframe_absolute_alpha,
                            dataframe_market_neutral, dataframe_commodities_funds])

    # clean dataframe 
    dataframe1 = dataframe1.rename(columns={'Name': 'fund', 'NAV': 'price'})
    dataframe1['Date'] = pd.to_datetime(dataframe1['Date'], format='%m/%d/%Y')
    dataframe1 = dataframe1[(dataframe1['Date'] >= filter_date)]

    # dataframe1 = dataframe1.drop(['NAV'], axis = 1)
    dataframe1 = dataframe1.set_index('Date')

    # load trusts 

    # paths to ETFs and Trusts 
    folder_name_equity = path.join(path_to_etf_trusts, subfolder_class_equity)
    folder_name_commodities = path.join(path_to_etf_trusts, subfolder_class_commodities)
    folder_name_alternatives = path.join(path_to_etf_trusts, subfolder_class_alternatives)

    dataframe_equity = pd.concat([pd.read_csv(f, sep=seperator)
                                  for f in glob.glob(folder_name_equity + "/*." + file_type)],
                                 ignore_index=False)

    dataframe_equity['Asset_Class'] = 'Equity'

    dataframe_alternatives = pd.concat([pd.read_csv(f, sep=seperator)
                                        for f in glob.glob(folder_name_alternatives + "/*." + file_type)],
                                       ignore_index=False)

    dataframe_alternatives['Asset_Class'] = 'Alternatives'

    dataframe_commodities = pd.concat([pd.read_csv(f, sep=seperator)
                                       for f in glob.glob(folder_name_commodities + "/*." + file_type)],
                                      ignore_index=False)

    dataframe_commodities['Asset_Class'] = 'Commoditities'

    dataframe2 = pd.concat([dataframe_equity,
                            dataframe_alternatives,
                            dataframe_commodities])

    # clean dataframe 
    dataframe2 = dataframe2.rename(columns={'Name': 'fund', 'Close': 'price'})
    dataframe2 = dataframe2.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
    dataframe2['Date'] = pd.to_datetime(dataframe2['Date'], format='%m/%d/%Y')
    dataframe2 = dataframe2[(dataframe2['Date'] >= filter_date)]

    dataframe2 = dataframe2.set_index('Date')

    dataframe = pd.concat([dataframe1, dataframe2])

    return dataframe


def get_US_yields(freq: str = 'd', add_fed_rate: bool = False) -> pd.DataFrame:
    """
    Args:
        freq (TYPE, optional): DESCRIPTION. Defaults to 'd'.

    Yields:
        df_USyields (TYPE): DESCRIPTION.
        :param freq:
        :param add_fed_rate:

    """
    treasuries = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3',
                  'DGS5', 'DGS7', 'DGS10',
                  'DGS20', 'DGS30']

    renames = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y',
               '10Y', '20Y', '30Y']

    # Fred Codes for the US Treasuries 
    if add_fed_rate:
        treasuries = treasuries + ['EFFR']
        renames = renames + ['EFFR']

    # yield spreads 
    df_USyields = download_fred_data(treasuries, freq=freq)
    df_USyields.columns = renames

    return df_USyields


def get_US_yield_spreads(freq: str = 'd', add_fed_rate: bool = False) -> pd.DataFrame:
    """
    

    Args:
        freq (str, optional): DESCRIPTION. Defaults to 'd'.
        add_fed_rate (bool, optional): DESCRIPTION. Defaults to False.

    Returns:
        df_USyields_spreads (TYPE): DESCRIPTION.

    """

    # fred = ['d', 'm', 'q', 'a'][0]
    df_USyields = get_US_yields(freq=freq, add_fed_rate=add_fed_rate)
    yields = df_USyields.columns

    # get spread with fed rate 
    spreads = ['T10Y2Y', 'T10Y3M', 'T10YFF'][2]
    df_10y_fed_rate = download_fred_data([spreads], freq=freq)

    # get spreads  
    for col in list(df_USyields.columns.drop("10Y")):
        df_USyields["spread_10Y_" + col] = df_USyields["10Y"] - df_USyields[col]

    df_USyields_spreads = df_USyields.drop(yields, axis=1)
    df_USyields_spreads = df_USyields_spreads.merge(df_10y_fed_rate, right_index=True, left_index=True)

    return df_USyields_spreads


def get_USyield_curve_factors(start_date: str = None, freq: str = 'd') -> pd.DataFrame:
    """
    

    Args:
        start_date (str, optional): DESCRIPTION. Defaults to None.
        freq (str, optional): DESCRIPTION. Defaults to 'd'.

    Returns:
        principalDf (TYPE): DESCRIPTION.

    """
    # PCA 
    df_USyields = get_US_yields(freq=freq)

    if start_date:
        df_USyields = df_USyields[df_USyields.index >= start_date]

    rem_col = df_USyields.columns

    df_USyields = df_USyields.dropna()

    # PCA 
    pca_USyield = PCA(n_components=3)

    # Standardizing the features
    x = StandardScaler().fit_transform(df_USyields)
    principalComponents = pca_USyield.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['USyc_ShiftFactor',
                                        'USyc_slope',
                                        'USyc_curvature'])
    df_USyields = df_USyields.reset_index()
    principalDf = pd.merge(principalDf, df_USyields,
                           left_index=True, right_index=True)

    principalDf = principalDf.set_index('Date')

    principalDf = principalDf.drop(rem_col, axis=1)

    return principalDf


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


def download_fred_data(fred_sumbol: list, freq: str, print_info: bool = False,
                       my_fred_API: str = 'cc628b51e21828ae6b98c06f4eef6714') -> pd.DataFrame:
    """
    

    Args:
        fred_sumbol (list): DESCRIPTION.
        freq (str): DESCRIPTION. # fred = ['d', 'm', 'q', 'a'][0]
        print_info (bool, optional): DESCRIPTION. Defaults to False.
        my_fred_API (str, optional): DESCRIPTION. Defaults to 'cc628b51e21828ae6b98c06f4eef6714'.

    Returns:
        df (TYPE): DESCRIPTION.

    """

    fred = Fred(api_key=my_fred_API)

    series_to_download = fred_sumbol

    df = {}
    for series_id in series_to_download:
        info = fred.get_series_info(series_id)['title']
        if print_info:
            print(info)
        df[series_id] = fred.get_series(series_id, frequency=freq)
    df = pd.DataFrame(df)
    df.index.names = ['Date']
    df.index = pd.DatetimeIndex(df.index)
    return df


def get_yield_curve_factors(df: pd.DataFrame = None,
                            date_col_name: str = 'Date',
                            prefix: str = None) -> pd.DataFrame:
    """
    

    Args:
        start_date (str, optional): DESCRIPTION. Defaults to None.
        freq (str, optional): DESCRIPTION. Defaults to 'd'.

    Returns:
        principalDf (TYPE): DESCRIPTION.

    """

    df = df.dropna()
    if date_col_name in df.columns:
        df = df.set_index(date_col_name)

    rem_col = df.columns

    # PCA 
    pca_yield = PCA(n_components=3)

    # Standardizing the features
    x = StandardScaler().fit_transform(df)
    principalComponents = pca_yield.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['_ShiftFactor',
                                        '_Slope',
                                        '_Curvature'])
    df = df.reset_index()
    principalDf = pd.merge(principalDf, df,
                           left_index=True, right_index=True)

    principalDf = principalDf.set_index(date_col_name)

    principalDf = principalDf.drop(rem_col, axis=1)

    if prefix:
        principalDf.columns = prefix + principalDf.columns

    return principalDf
