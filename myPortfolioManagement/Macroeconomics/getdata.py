from myPortfolioManagement.myDataAnalysis import transform
from myPortfolioManagement.myDataAnalysis import remove_outliers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import types
from fredapi import Fred
import pandas as pd
from openbb_terminal.sdk import openbb


def load_fredmd_data(vintage):
    base_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/'

    # - FRED-MD --------------------------------------------------------------
    # 1. Download data
    orig_m = (pd.read_csv(f'{base_url}/monthly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract transformation information
    transform_m = orig_m.iloc[0, 1:]
    orig_m = orig_m.iloc[1:]

    # 3. Extract the date as an index
    orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
    orig_m.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_m = orig_m.apply(transform, axis=0,
                         transforms=transform_m)

    # 5. Remove outliers (but not in 2020)
    dta_m.loc[:'2019-12'] = remove_outliers(dta_m.loc[:'2019-12'])

    # - FRED-QD --------------------------------------------------------------
    # 1. Download data
    orig_q = (pd.read_csv(f'{base_url}/quarterly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract factors and transformation information
    factors_q = orig_q.iloc[0, 1:]
    transform_q = orig_q.iloc[1, 1:]
    orig_q = orig_q.iloc[2:]

    # 3. Extract the date as an index
    orig_q.index = pd.PeriodIndex(orig_q.sasdate.tolist(), freq='Q')
    orig_q.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_q = orig_q.apply(transform, axis=0,
                         transforms=transform_q)

    # 5. Remove outliers (but not in 2020)
    dta_q.loc[:'2019Q4'] = remove_outliers(dta_q.loc[:'2019Q4'])

    # - Output datasets ------------------------------------------------------
    return types.SimpleNamespace(
        orig_m=orig_m, orig_q=orig_q,
        dta_m=dta_m, transform_m=transform_m,
        dta_q=dta_q, transform_q=transform_q, factors_q=factors_q)


def get_FX_spots(currencies: list = None, start_date='1995-01-01', base_currency ='USD', wide_format=False):
    fx = []
    for ccy in currencies:
            fx_temp = openbb.forex.load(to_symbol=base_currency, from_symbol=ccy, start_date=start_date)
            fx_temp['FX'] = ccy + base_currency
            fx_temp['Currency'] = ccy
            fx.append(fx_temp)

    fx = pd.concat(fx)

    fx = fx[['Adj Close', 'FX', 'Currency']]
    fx = fx.rename(columns={'Adj Close': 'Spot'})

    if wide_format:
        fx = fx.pivot(columns='FX', values='Spot')

    return fx


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

# def create_credit_impulse(freq="q"):
#     credit_impulse = ['CRDQXMAPABIS',  # Euro Area
#                       'QUSPAM770A',  # US
#                      # 'QCNPAM770A',  # China
#                       'QGBPAM770A',  # United Kingdom
#                       'CRDQJPAPABIS']  # Japan
#     gdp = ['GDP', # US
#            'EUNNGDP', # Euro
#           # 'MKTGDPCNA646NWDB', #China
#            'JPNNGDP', #Japan
#            'UKNGDP' ] # United Kingdom
#
#     symbols = gdp + credit_impulse
#
#     data = download_fred_data(fred_sumbol=symbols, freq=freq)
#     data.columns = ['EUR', 'US', 'CHN', 'GB', 'JP']
#     return data

# proxy fund rate data
# source: https://www.kansascityfed.org/Economic%20Review/documents/319/2016-Measuring%20the%20Stance%20of%20Monetary%20Policy%20on%20and%20off%20the%20Zero%20Lower%20Bound.pdf
#       : https://www.frbsf.org/wp-content/uploads/sites/4/el2022-30.pdf

# 1. MORTGAGE30US - weekly, 30-year fixed Mortage rate
# 2. DGS2         - daily, 2-year Treasuries
# 3. DGS5         - daily, 5-year Treasuries
# 4. DGS7         - daily, 7-year Treasuries
# 5. DGS10        - daily, 10-year Treasuries
# 6. Bond buyer index: state/local bonds, 20-year, general obligation --- N/A
# 7. AAA          - Monthly, Moody's Seasoned Aaa Corporate Bond Yield
# 8. DBAA         - Daily, Moody's Seasoned Baa Corporate Bond Yield
# Spreads: Mortgage rates and 10-year Treasury spread , Two-year and 10-year Treasury spread,
#          Aaa corporate bond yield and 10-year Treasury spread, Baa corporate bond yield and 10-year Treasury spread
