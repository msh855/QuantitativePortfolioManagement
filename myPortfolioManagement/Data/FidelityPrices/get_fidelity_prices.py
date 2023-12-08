import pandas as pd
from os import path
import glob
import quantstats as qs

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

