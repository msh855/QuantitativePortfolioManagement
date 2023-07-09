import pandas as pd
from openbb_terminal.sdk import openbb, TerminalStyle
from datetime import date, timedelta
from timebudget import timebudget
from yahoofinancials import YahooFinancials
from os import path
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from finvizfinance.screener.overview import Overview
from fredapi import Fred
import numpy as np
import quantstats as qs

qs.extend_pandas()
pd.options.mode.use_inf_as_na = True
theme = TerminalStyle("light", "light", "light")


@timebudget
def get_stock_prices_from_openBB(yahoo_tickers: list = None, start_date: str = "1995-01-01", base_currency: str = 'GBP',
                                 index_name: str = 'YahooTicker'):
    data_list = []

    for ticker in yahoo_tickers:
        data = openbb.stocks.load(ticker, start_date=start_date)
        keep = list(data.columns)
        data[index_name] = ticker
        yahoo_financials = YahooFinancials(ticker, concurrent=True, max_workers=5)
        temp_ccy = yahoo_financials.get_currency()
        data['Currency'] = temp_ccy

        # get base rate currency
        fx_temp_base_currency = openbb.forex.load(to_symbol='USD', from_symbol=base_currency, start_date=start_date)
        fx_temp_base_currency = fx_temp_base_currency[['Adj Close']]
        fx_temp_base_currency.columns = ['Spot_' + base_currency]

        if temp_ccy != 'USD':
            fx_temp = openbb.forex.load(to_symbol='USD', from_symbol=temp_ccy, start_date=start_date)
            fx_temp = fx_temp[['Adj Close']]
            fx_temp.columns = ['Spot_USD']
            data = data.join(fx_temp)
            data['Adj_Close_USD'] = data['Adj Close'] * data['Spot_USD']
            data = data.join(fx_temp_base_currency)
            data['Adj_Close_' + base_currency] = data['Adj_Close_USD'] / data['Spot_' + base_currency]
        else:
            data['Adj_Close_USD'] = data['Adj Close']
            data = data.join(fx_temp_base_currency)
            data['Adj_Close_' + base_currency] = data['Adj_Close_USD'] / data['Spot_' + base_currency]

        name_temp = 'Adj_Close_' + base_currency
        keep_final = keep + [index_name] + ['Currency'] + ['Adj_Close_USD', name_temp]
        data = data[keep_final]
        data_list.append(data)

    df_prices = pd.concat(data_list)
    df_prices['Market_Cap_USD'] = df_prices['Volume'] * df_prices['Adj_Close_USD']

    return df_prices


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


def get_fidelity_prices(filter_date: str = '2000-01-01') -> pd.DataFrame:
    file_type = 'csv'
    seperator = ','

    path_to_funds = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/Data/FidelityPrices/funds'
    path_to_etf_trusts = '/Users/safishajjouz/GitHub/QuantitativePortfolioManagement/myPortfolioManagement/Data/FidelityPrices/trusts_etfs'

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
        treasuries = treasuries + ['DFF']
        renames = renames + ['DFF']

    # yield spreads 
    df_USyields = get_fred_data(treasuries, freq=freq)
    df_USyields.columns = renames

    return df_USyields


def get_US_yield_spreads(freq: str = 'd', spread_from='EFFR', add_fed_rate: bool = True) -> pd.DataFrame:
    """

    Args:
        spread_from:
        freq (str, optional): DESCRIPTION. Defaults to 'd'.
        add_fed_rate (bool, optional): DESCRIPTION. Defaults to False.

    Returns:
        df_USyields_spreads (TYPE): DESCRIPTION.

    """

    # fred = ['d', 'm', 'q', 'a'][0]
    df_USyields = get_US_yields(freq=freq, add_fed_rate=add_fed_rate)
    yields = df_USyields.columns

    # # get spread with fed rate
    # spreads = ['T10Y2Y', 'T10Y3M', 'T10YFF'][2]
    # df_10y_fed_rate = get_fred_data([spreads], freq=freq)

    # get spreads  
    for col in list(df_USyields.columns.drop(spread_from)):
        df_USyields["spread_" + col] = df_USyields[spread_from] - df_USyields[col]

    df_USyields_spreads = df_USyields.drop(yields, axis=1)
    # df_USyields_spreads = df_USyields_spreads.merge(df_10y_fed_rate, right_index=True, left_index=True)

    return df_USyields_spreads.dropna()


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


def get_fred_data(fred_sumbol: list, freq: str = 'm', print_info: bool = False,
                  my_fred_API: str = 'cc628b51e21828ae6b98c06f4eef6714') -> pd.DataFrame:
    """
    

    Args:
        fred_sumbol (list): DESCRIPTION.
        freq (str): options a string from ['d', 'm', 'q', 'a'].
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
                               columns=['ShiftFactor',
                                        'Slope',
                                        'Curvature'])
    df = df.reset_index()
    principalDf = pd.merge(principalDf, df,
                           left_index=True, right_index=True)

    principalDf = principalDf.set_index(date_col_name)

    principalDf = principalDf.drop(rem_col, axis=1)

    if prefix:
        principalDf.columns = prefix + principalDf.columns

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


def get_stock_info(yahoo_tickers: list = None):
    # add industries
    # ==============
    index_name = 'YahooTicker'

    df_sectors = openbb.stocks.ca.screener(similar=yahoo_tickers, data_type="overview")
    df_sectors = df_sectors[["Ticker\n\n", 'Sector', 'Industry', 'Country']]
    df_sectors = df_sectors.rename(columns={"Ticker\n\n": index_name})
    df_sectors.columns = df_sectors.columns[1:, ].insert(0, index_name)
    df_sectors = df_sectors.set_index(index_name)

    # get market shares
    yahoo_financials = YahooFinancials(yahoo_tickers, concurrent=True, max_workers=5)
    temp_ccy = yahoo_financials.get_currency()
    temp_ccy = pd.DataFrame.from_dict(temp_ccy.items())
    temp_ccy.columns = [index_name, 'Currency']
    temp_ccy.Currency = [x.upper() for x in temp_ccy.Currency]
    temp_ccy = temp_ccy.set_index(index_name)

    if temp_ccy.shape[0] >= df_sectors.shape[0]:
        df_info = temp_ccy.join(df_sectors)
    else:
        df_info = df_sectors.join(temp_ccy)

    df_info = df_info[list(df_sectors.columns) + list(temp_ccy.columns)]

    if df_info['Sector'].any():
        df_info['Sector'] = df_info['Sector'].replace(np.nan, 'Unclassified')

    if df_info['Country'].any():
        df_info['Country'] = df_info['Country'].replace(np.nan, 'Unclassified')

    if df_info['Industry'].any():
        df_info['Industry'] = df_info['Industry'].replace(np.nan, 'Unclassified')

    return df_info


#
# def get_stock_prices_from_openBB(yahoo_tickers: list = None, start_date="1995-01-01", wide_format=False):
#     index_name = 'YahooTicker'
#     df_prices = []
#     for ticker in yahoo_tickers:
#         data = openbb.stocks.load(ticker, start_date=start_date)
#         data[index_name] = ticker
#         df_prices.append(data)
#     df_prices = pd.concat(df_prices)
#     prices = df_prices[['Adj Close', index_name]]
#
#     if wide_format:
#         prices = prices.pivot(columns=index_name, values='Adj Close')
#     return prices


def get_FX_spots(currencies: list = None, start_date='1995-01-01', wide_format=False):
    fx = []
    for ccy in currencies:
        if ccy != 'USD':
            fx_temp = openbb.forex.load(to_symbol='USD', from_symbol=ccy, start_date=start_date)
            fx_temp['FX'] = 'USD' + ccy
            fx_temp['Currency'] = ccy
            fx.append(fx_temp)

    fx = pd.concat(fx)

    fx = fx[['Adj Close', 'FX', 'Currency']]
    fx = fx.rename(columns={'Adj Close': 'Spot'})

    if wide_format:
        fx = fx.pivot(columns='FX', values='Spot')

    return fx
