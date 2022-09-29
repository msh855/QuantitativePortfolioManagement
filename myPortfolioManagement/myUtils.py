import pandas as pd                         # to work datafranes 
pd.options.mode.use_inf_as_na = True        # show NAs instead of inf 
import numpy as np                          # work with vectors 

import investpy                             # Mutual Fund info 
import quantstats as qs
from statsmodels.tsa.stattools import adfuller



def balance_dates(returns, returns_benchmark):
    
    if isinstance(returns_benchmark,pd.DataFrame):
       returns_benchmark = pd.Series(returns_benchmark.iloc[:,0])
    
    if isinstance(returns,pd.Series):
       returns = pd.DataFrame(returns)
    
    
    ret_balanced_dates = []
    
    for col in returns.columns:
       ret , ret_bench = qs.reports._match_dates(returns[col], 
                                                 returns_benchmark)
       ret_balanced_dates.append(ret)
    
    ret_balanced_dates = pd.concat(ret_balanced_dates , axis = 1)
    ret_bench = pd.DataFrame(ret_bench)
    
    # this to create a df with equal length 
    ret = pd.concat([ret_balanced_dates, ret_bench], axis = 1)
    ret = ret.fillna(0)
    
    # get the bench 
    ret_bench = ret.iloc[:,-1]
    ret_bench = pd.DataFrame(ret_bench)
  
    
    # get the returns
    ret_balanced_dates = ret.drop(ret_bench.columns, axis = 1)
    
    return ret_balanced_dates, ret_bench


def which_missing_in_investpy(list_to_check:list, what_to_check = 'isin'):
    
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
    funds_all  = investpy.get_funds(country=None)
    stocks_all = investpy.get_stocks(country=None) 
    etfs_all   = investpy.etfs.get_etfs(country=None)
    
    # clear None and NAs 
    funds_all = funds_all.fillna(value=np.nan).dropna(subset = [what_to_check] )
    funds_isin_list =  funds_all[what_to_check].to_list()
    
    # clear None and NAs 
    stocks_all = stocks_all.fillna(value=np.nan).dropna(subset = [what_to_check] )
    stock_list =  stocks_all[what_to_check].to_list()

    # clear None and NAs 
    etfs_all = etfs_all.fillna(value=np.nan).dropna(subset = [what_to_check] )
    etfs_list =  etfs_all[what_to_check].to_list()
    
    all_isin_available = funds_isin_list + stock_list + etfs_list
    
    not_in_mylist = list(set(list_to_check) - set(all_isin_available))
    
    
    return not_in_mylist


def which_in_investpy(list_to_check:list, what_to_check = 'isin'):
    
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
    funds_all  = investpy.get_funds(country=None)
    etfs_all   = investpy.etfs.get_etfs(country=None)
    stocks_all = investpy.get_stocks(country=None) 
    
    stocks_all['asset_class'] = 'No_info_from_Yahoo'
    
    # drop columns 
    funds_all = funds_all[keep_columns]
    etfs_all = etfs_all[keep_columns]
    stocks_all = stocks_all[keep_columns]
    
    assets_all = pd.concat([funds_all, etfs_all,stocks_all])
    
    assets_all = assets_all[assets_all[what_to_check].isin(list_to_check)]
    
    
    return assets_all


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]
    
    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated

def adf_statistics(time_series):
    """
    Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(time_series.values)
    if result[1] < 0.0500:             # result[1] contains the p-value
        return 0                       # returns 0 value if p-value of test is under 5%
    else:
        return 1

def adf_tests(df):
    """
    Augmented Dickey-Fuller test applied to every column in DataFrame
    """
    results = df.apply(adf_statistics, axis=0) # Output is a Pandas series
    if sum(results)==0:
        print('Null hypothesis of non-stationarity is rejected for ALL series with p-values < 5%')
    else:
        for i, v in results.items():
            if v == 1:
                print(f'Null hypothesis of non-stationarity of {i} series is NOT rejected')
            else:
                print(f'Null hypothesis of non-stationarity of {i} series is rejected')    
